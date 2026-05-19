#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/early_warning/plot_robustness_metrics.py

Generate layered robustness-metric plots from a precomputed metrics CSV file.

Function
--------
This script reads a metrics CSV file and produces publication-style robustness
plots for:
1. Baseline algorithms (DC, BC)
2. Other algorithms in two multi-panel batches

Each algorithm panel is composed of five stacked metric layers:
- LCC
- Natural connectivity
- R(rand)
- R(DCR) / R(BCR)
- Predicted collapse distance

Inputs
------
- `--csv_file`: path to the metrics CSV file
- `--output_dir`: directory for output figures
- `--collapse_target`: LCC threshold shown as a horizontal reference line

Outputs
-------
For each generated figure, the script saves:
- PDF
- SVG
- PNG

Usage
-----
Example:
    python experiments/early_warning/plot_robustness_metrics.py \
        --csv_file path/to/metrics.csv \
        --output_dir path/to/output \
        --collapse_target 0.5
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator


# ==============================================================================
# Global display switches
# ==============================================================================
SHOW_LCC = True
SHOW_TGT = True
SHOW_NAT = True
SHOW_R_RAND = True
SHOW_R_DCR = True
SHOW_R_BCR = False
SHOW_PRED_DC = True
SHOW_EST_DC = False


# ==============================================================================
# Algorithm order and style configuration
# ==============================================================================
ORDERED_ALGOS = [
    'DC', 'BC', 'R1', 'DCR', 'BCR', 'R2',
    'DomiRank', 'FINDER', 'CoreHD', 'GDM', 'GDMR', 'CoreGDM',
    'MS', 'MSR', 'CI1', 'CI2', 'CI3', 'EGND',
    'EIs1', 'EIs2', 'GND', 'GNDR', 'VE', 'VER',
    'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR'
]

ALGO_DISPLAY_NAME = {
    'CI1': r'CI $\ell$-1',
    'CI2': r'CI $\ell$-2',
    'CI3': r'CI $\ell$-3',
    'EIs1': r'EI ${\sigma_1}$',
    'EI_s1': r'EI ${\sigma_1}$',
    'EIs2': r'EI ${\sigma_2}$',
    'EI_s2': r'EI ${\sigma_2}$',
}

try:
    COLORS = sns.color_palette("Paired", 12)
except Exception:
    COLORS = plt.get_cmap('tab20').colors

C_LCC = '#1f77b4'
C_NAT = '#2ca02c'
C_RRAND = '#9467bd'
C_RDEG = '#ff7f0e'
C_RBCR = '#e377c2'
C_TIGIN = COLORS[5]
C_EST = '#d62728'
C_TGT = '0.55'

STYLE_LCC_LINE = dict(color=C_LCC, lw=1.5, ls='-', alpha=0.95, label='LCC size')
STYLE_LCC_PT = dict(color=C_LCC, lw=0, ls='', marker=None, ms=0, zorder=10)

STYLE_NAT = dict(color=C_NAT, lw=1.3, ls='-', marker=None, alpha=0.9, label='Natural connectivity')
STYLE_RRAND = dict(color=C_RRAND, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(rand)')
STYLE_RDCR = dict(color=C_RDEG, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(DCR)')
STYLE_RBCR = dict(color=C_RBCR, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(BCR)')

STYLE_PRED = dict(color=C_TIGIN, lw=1.1, ls='-', marker=None, label='Collapse distance')
STYLE_EST = dict(color=C_EST, lw=1.3, ls='-', marker=None, label='Est DC (BCR)')


# ==============================================================================
# Helper functions
# ==============================================================================
def get_algo_sort_key(name):
    if name in ORDERED_ALGOS:
        return ORDERED_ALGOS.index(name)
    return 999


def get_subplot_label(idx):
    if idx < 26:
        return chr(ord('a') + idx)
    first = idx // 26 - 1
    second = idx % 26
    return f"{chr(ord('a') + first)}{chr(ord('a') + second)}"


def beautify_axis_basic(ax):
    """Apply basic axis styling."""
    ax.grid(False)
    ax.tick_params(direction='in', which='both', labelsize=6, length=2.0, width=0.6, pad=1.5)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)


def set_clean_yticks(ax, policy='mid'):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span < 1e-9:
        span = 1.0

    is_integer_mode = False

    if ymax > 10 and span > 1.0:
        fmt_str = "{:.0f}"
        steps = [1, 2, 5, 10]
        is_integer_mode = True
    elif span < 0.05:
        fmt_str = "{:.3f}"
        steps = [1, 2, 5, 10]
    else:
        fmt_str = "{:.2f}"
        steps = [1, 2, 5, 10]

    loc = MaxNLocator(nbins=7, steps=steps, integer=is_integer_mode)
    candidates = loc.tick_values(ymin, ymax)

    margin = 0.15 * span
    exclude_bottom_limit = ymin + margin
    exclude_top_limit = ymax - margin

    valid_ticks = []
    seen_labels = set()

    for t in candidates:
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue

        is_bad = False
        if policy in ['top', 'mid'] and t <= exclude_bottom_limit:
            is_bad = True
        if policy in ['bottom', 'mid'] and t >= exclude_top_limit:
            is_bad = True

        label_str = fmt_str.format(t)
        if label_str in seen_labels:
            is_bad = True

        if not is_bad:
            valid_ticks.append(t)
            seen_labels.add(label_str)

    if len(valid_ticks) < 2:
        center = (ymin + ymax) / 2
        if is_integer_mode:
            t1 = np.floor(center)
            t2 = np.ceil(center)
            if t1 == t2:
                t2 += 1
            valid_ticks = [t1, t2]
        elif span < 0.05:
            step_val = 0.002
            valid_ticks = [center - step_val, center + step_val]
        else:
            step_val = 0.01
            valid_ticks = [center - step_val, center + step_val]

    final_ticks = [t for t in valid_ticks if ymin - 0.01 * span <= t <= ymax + 0.01 * span]
    if not final_ticks:
        final_ticks = valid_ticks

    ax.set_yticks(final_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt_str.replace("{:", "%").replace("}", "")))


def get_local_ylim(series, padding=0.1, min_val=-0.001):
    series = series.dropna()
    if series.empty:
        return (0, 1)

    ymin, ymax = series.min(), series.max()
    span = ymax - ymin

    if span < 0.05:
        target_min_span = 0.005
    else:
        target_min_span = 0.04

    if span < target_min_span:
        center = (ymin + ymax) / 2.0
        if center == 0:
            lower, upper = 0, target_min_span
        else:
            lower = center - target_min_span / 2.0
            upper = center + target_min_span / 2.0
    else:
        lower = ymin - span * padding
        upper = ymax + span * padding

    if min_val is not None:
        if lower < min_val:
            lower = min_val
        if upper - lower < target_min_span:
            upper = lower + target_min_span

    return (lower, upper)


# ==============================================================================
# Core plotting function
# ==============================================================================
def plot_grid(df, algos_to_plot, n_rows, n_cols, start_idx,
              out_path, filename_prefix, args,
              is_first_part_of_sequence=False):

    AX_WIDTH = 1.70
    AX_HEIGHT = 2.70

    GAP_W = 0.35
    GAP_H = 0.32

    MARGIN_LEFT = 0.55
    MARGIN_RIGHT = 0.15
    MARGIN_BOTTOM = 0.60
    MARGIN_TOP = 0.30
    LEGEND_SPACE = 0.45

    top_margin_actual = MARGIN_TOP
    if is_first_part_of_sequence:
        top_margin_actual += LEGEND_SPACE

    fig_width = MARGIN_LEFT + (n_cols * AX_WIDTH) + ((n_cols - 1) * GAP_W) + MARGIN_RIGHT
    fig_height = MARGIN_BOTTOM + (n_rows * AX_HEIGHT) + ((n_rows - 1) * GAP_H) + top_margin_actual

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs_left = MARGIN_LEFT / fig_width
    gs_right = 1.0 - (MARGIN_RIGHT / fig_width)
    gs_bottom = MARGIN_BOTTOM / fig_height
    gs_top = 1.0 - (top_margin_actual / fig_height)

    outer_gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        left=gs_left,
        right=gs_right,
        bottom=gs_bottom,
        top=gs_top,
        wspace=GAP_W / AX_WIDTH,
        hspace=GAP_H / AX_HEIGHT
    )

    legend_handles = []
    legend_labels = []

    for i, algo in enumerate(algos_to_plot):
        r, c = i // n_cols, i % n_cols
        sub_gs = outer_gs[r, c].subgridspec(5, 1, hspace=0.00)

        df_algo = df[df['algorithm'] == algo].sort_values('step')
        df_files = df_algo[~df_algo['filename'].astype(str).str.contains('_-1_edges', na=False)]
        df_files = df_files[df_files['filename'] != 'step=-1']

        label_str = get_subplot_label(start_idx + i)
        display_name = ALGO_DISPLAY_NAME.get(algo, algo)

        ax1 = fig.add_subplot(sub_gs[0])
        h_tgt = None
        if SHOW_TGT:
            h_tgt = ax1.axhline(args.collapse_target, color=C_TGT, ls='--', lw=1.0)
        if SHOW_LCC:
            l1, = ax1.plot(df_algo['step'], df_algo['LCC'], **STYLE_LCC_LINE)
            if i == 0:
                if h_tgt is not None:
                    legend_handles.append(h_tgt)
                    legend_labels.append('Collapse target')
                legend_handles.append(l1)
                legend_labels.append('LCC size')

        ax1.set_title(f"({label_str}) {display_name} attack", fontsize=8, fontweight='bold', loc='left', pad=3)
        ax1.set_ylim(get_local_ylim(df_algo['LCC'], min_val=0))

        beautify_axis_basic(ax1)
        set_clean_yticks(ax1, policy='top')
        ax1.tick_params(labelbottom=False)

        ax2 = fig.add_subplot(sub_gs[1], sharex=ax1)
        if SHOW_NAT and not df_files.empty and 'natural_connectivity' in df_files:
            tmp = df_files.dropna(subset=['natural_connectivity'])
            if not tmp.empty:
                l, = ax2.plot(tmp['step'], tmp['natural_connectivity'], **STYLE_NAT)
                if i == 0:
                    legend_handles.append(l)
                    legend_labels.append(STYLE_NAT['label'])
                ax2.set_ylim(get_local_ylim(tmp['natural_connectivity']))

        beautify_axis_basic(ax2)
        set_clean_yticks(ax2, policy='mid')
        ax2.tick_params(labelbottom=False)

        ax3 = fig.add_subplot(sub_gs[2], sharex=ax1)
        if SHOW_R_RAND and not df_files.empty and 'R(rand)' in df_files:
            tmp = df_files.dropna(subset=['R(rand)'])
            if not tmp.empty:
                l, = ax3.plot(tmp['step'], tmp['R(rand)'], **STYLE_RRAND)
                if i == 0:
                    legend_handles.append(l)
                    legend_labels.append(STYLE_RRAND['label'])
                ax3.set_ylim(get_local_ylim(tmp['R(rand)']))

        beautify_axis_basic(ax3)
        set_clean_yticks(ax3, policy='mid')
        ax3.tick_params(labelbottom=False)

        ax4 = fig.add_subplot(sub_gs[3], sharex=ax1)
        vals_layer4 = []
        if not df_files.empty:
            if SHOW_R_DCR and 'R(DCR)' in df_files:
                tmp = df_files.dropna(subset=['R(DCR)'])
                if not tmp.empty:
                    l, = ax4.plot(tmp['step'], tmp['R(DCR)'], **STYLE_RDCR)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_RDCR['label'])
                    vals_layer4.append(tmp['R(DCR)'])

            if SHOW_R_BCR and 'R(BCR)' in df_files:
                tmp = df_files.dropna(subset=['R(BCR)'])
                if not tmp.empty:
                    l, = ax4.plot(tmp['step'], tmp['R(BCR)'], **STYLE_RBCR)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_RBCR['label'])
                    vals_layer4.append(tmp['R(BCR)'])

        if vals_layer4:
            ax4.set_ylim(get_local_ylim(pd.concat(vals_layer4)))

        beautify_axis_basic(ax4)
        set_clean_yticks(ax4, policy='mid')
        ax4.tick_params(labelbottom=False)

        ax5 = fig.add_subplot(sub_gs[4], sharex=ax1)
        vals_layer5 = []
        if not df_files.empty:
            if SHOW_PRED_DC and 'Predicted DC' in df_files:
                tmp = df_files.dropna(subset=['Predicted DC'])
                if not tmp.empty:
                    l, = ax5.plot(tmp['step'], tmp['Predicted DC'], **STYLE_PRED)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_PRED['label'])
                    vals_layer5.append(tmp['Predicted DC'])

            if SHOW_EST_DC and 'Estimated DC (BCR)' in df_files:
                tmp = df_files.dropna(subset=['Estimated DC (BCR)'])
                if not tmp.empty:
                    l, = ax5.plot(tmp['step'], tmp['Estimated DC (BCR)'], **STYLE_EST)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_EST['label'])
                    vals_layer5.append(tmp['Estimated DC (BCR)'])

        if vals_layer5:
            ax5.set_ylim(get_local_ylim(pd.concat(vals_layer5)))

        beautify_axis_basic(ax5)
        set_clean_yticks(ax5, policy='bottom')

        max_step = int(df_algo['step'].max()) if not df_algo.empty else 0
        if max_step <= 20:
            step_interval = 5
        elif max_step <= 50:
            step_interval = 10
        elif max_step <= 100:
            step_interval = 20
        elif max_step <= 200:
            step_interval = 40
        else:
            step_interval = 50

        ax5.xaxis.set_major_locator(MultipleLocator(step_interval))
        ax5.set_xlim(left=0, right=max_step * 1.05 if max_step > 0 else 1.0)
        ax5.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    fig.supxlabel('Attack step', fontsize=9, y=0.02)

    if is_first_part_of_sequence and legend_handles:
        center_x = (gs_left + gs_right) / 2.0
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(center_x, gs_top + 0.015),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=7,
            columnspacing=1.2,
            bbox_transform=fig.transFigure
        )

    for ext in ['pdf', 'svg', 'png']:
        save_to = out_path / f"{filename_prefix}-new.{ext}"
        plt.savefig(save_to, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {save_to}")

    plt.close()


# ==============================================================================
# Main entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate layered robustness-metric plots from a metrics CSV file."
    )
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the metrics CSV file.")
    parser.add_argument('--output_dir', type=str, default=".", help="Directory for saving output figures.")
    parser.add_argument('--collapse_target', type=float, default=0.5, help="Collapse target shown as a reference line.")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv_file)
    dataset_name = Path(args.csv_file).name.replace('_metrics.csv', '')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'legend.fontsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    baseline_algos = ['DC', 'BC']
    other_algos = [x for x in df['algorithm'].unique() if x not in baseline_algos and x in ORDERED_ALGOS]
    other_algos = sorted(other_algos, key=get_algo_sort_key)

    print("Plotting baseline algorithms (DC, BC)...")
    plot_grid(
        df,
        baseline_algos,
        n_rows=1,
        n_cols=2,
        start_idx=0,
        out_path=out_path,
        filename_prefix=f"{dataset_name}_baseline",
        args=args,
        is_first_part_of_sequence=True
    )

    chunk1 = other_algos[:12]
    print("Plotting other algorithms, part 1 (3x4)...")
    plot_grid(
        df,
        chunk1,
        n_rows=3,
        n_cols=4,
        start_idx=0,
        out_path=out_path,
        filename_prefix=f"{dataset_name}_others_part1",
        args=args,
        is_first_part_of_sequence=True
    )

    chunk2 = other_algos[12:]
    print("Plotting other algorithms, part 2 (4x4)...")
    plot_grid(
        df,
        chunk2,
        n_rows=4,
        n_cols=4,
        start_idx=12,
        out_path=out_path,
        filename_prefix=f"{dataset_name}_others_part2",
        args=args,
        is_first_part_of_sequence=False
    )


if __name__ == "__main__":
    main()
