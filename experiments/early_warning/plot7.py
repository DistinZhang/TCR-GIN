#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/early_warning/plot7.py

Generate the main robustness-metrics figure as an 8-row × 5-column A4 layout.

Function
--------
This script reads precomputed robustness-metrics CSV files for a fixed set of
scenarios and generates a publication-style multi-panel figure.

Each row corresponds to one scenario, and each column corresponds to one metric:
1. LCC
2. Natural connectivity
3. R(rand)
4. R(DCR)
5. Predicted collapse distance

Design choices
--------------
- No row or column labels
- Sequential subplot letters
- Tick labels shown on all x-axes
- Compact y-axis tick selection
- A4-sized figure layout

Inputs
------
- Precomputed CSV files listed in `SCENARIOS`

Outputs
-------
- `robustness_metrics_main.pdf`
- `robustness_metrics_main.svg`
- `robustness_metrics_main.png`

Usage
-----
From the repository root:
    python experiments/early_warning/plot_robustness_metrics-main.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator


# ==============================================================================
# Paths
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "robustness_metrics"


# ==============================================================================
# Scenario definitions
# ==============================================================================
SCENARIOS = [
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/transport-111/transport_111_metrics.csv',
        'algo': 'DC',
        'target': 0.3
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/transport-111/transport_111_metrics.csv',
        'algo': 'BC',
        'target': 0.3
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/transport-185/transport_185_metrics.csv',
        'algo': 'DC',
        'target': 0.5
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/transport-185/transport_185_metrics.csv',
        'algo': 'BC',
        'target': 0.5
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/Power-353/power_353_metrics.csv',
        'algo': 'DC',
        'target': 0.3
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/Power-353/power_353_metrics.csv',
        'algo': 'BC',
        'target': 0.3
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/Power-588/power_588_metrics.csv',
        'algo': 'DC',
        'target': 0.5
    },
    {
        'csv': SCRIPT_DIR / 'results/robustness_metrics/Power-588/power_588_metrics.csv',
        'algo': 'BC',
        'target': 0.5
    },
]


# ==============================================================================
# Column definitions
# ==============================================================================
COLUMNS = [
    ('LCC', True),
    ('natural_connectivity', False),
    ('R(rand)', False),
    ('R(DCR)', False),
    ('Predicted DC', False),
]

# Fixed decimal precision per column.
# Use -1 for adaptive formatting.
COL_DECIMALS = [1, -1, 2, 3, 3]

# Columns that use data-driven y-tick placement.
DATA_DRIVEN_COLS = {1, 2, 3}


# ==============================================================================
# Colors and line styles
# ==============================================================================
C_LCC = '#1f77b4'
C_NAT = '#2ca02c'
C_RRAND = '#9467bd'
C_RDEG = '#ff7f0e'
C_TGT = '0.55'

try:
    import seaborn as sns
    _COLORS = sns.color_palette("Paired", 12)
except Exception:
    _COLORS = plt.get_cmap('tab20').colors

C_PRED = _COLORS[5]

LINE_STYLES = {
    'LCC': dict(color=C_LCC, lw=1.5, ls='-', alpha=0.95),
    'natural_connectivity': dict(color=C_NAT, lw=1.3, ls='-', alpha=0.9),
    'R(rand)': dict(color=C_RRAND, lw=1.3, ls='-', alpha=0.9),
    'R(DCR)': dict(color=C_RDEG, lw=1.3, ls='-', alpha=0.9),
    'Predicted DC': dict(color=C_PRED, lw=1.1, ls='-'),
}


# ==============================================================================
# Typography
# ==============================================================================
TICK_FS = 11
TEXT_FS = 12


# ==============================================================================
# Helper functions
# ==============================================================================
def get_subplot_label(idx):
    if idx < 26:
        return chr(ord('a') + idx)
    first = idx // 26 - 1
    second = idx % 26
    return f"{chr(ord('a') + first)}{chr(ord('a') + second)}"


def beautify(ax):
    ax.grid(False)
    ax.tick_params(
        direction='in',
        which='both',
        labelsize=TICK_FS,
        length=2.5,
        width=0.6,
        pad=2
    )
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)


def get_local_ylim(series, padding=0.15, min_val=-0.001):
    series = series.dropna()
    if series.empty:
        return (0, 1)

    ymin, ymax = series.min(), series.max()
    span = ymax - ymin
    target_min = 0.005 if span < 0.05 else 0.04

    if span < target_min:
        c = (ymin + ymax) / 2.0
        if c == 0:
            lower, upper = 0, target_min
        else:
            lower, upper = c - target_min / 2, c + target_min / 2
    else:
        lower = ymin - span * padding
        upper = ymax + span * padding

    if min_val is not None and lower < min_val:
        lower = min_val
        if upper - lower < target_min:
            upper = lower + target_min

    return (lower, upper)


def nice_x_interval(max_step):
    """Return a tick interval that yields about 3-4 ticks over the x-range."""
    if max_step <= 0:
        return 5

    xlim = max_step * 1.05
    raw = xlim / 2.5
    mag = 10 ** np.floor(np.log10(max(raw, 1e-9)))
    r = raw / mag

    if r <= 1.5:
        nice = 1
    elif r <= 3.5:
        nice = 2
    elif r <= 7.5:
        nice = 5
    else:
        nice = 10

    iv = max(1, int(nice * mag))
    n_ticks = int(np.floor(xlim / iv)) + 1

    if n_ticks > 5:
        iv = iv * 2
    elif n_ticks < 3:
        iv = max(1, iv // 2)

    return iv


def set_yticks_fixed(ax, decimals):
    """
    Set 2-3 clean, non-overlapping y-ticks with a fixed number of decimals.
    """
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span < 1e-9:
        span = 1.0

    fmt = f"{{:.{decimals}f}}"
    is_int = (decimals == 0)

    loc = MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=is_int)
    cands = loc.tick_values(ymin, ymax)

    ticks = []
    seen = set()
    for t in cands:
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue
        lab = fmt.format(t)
        if lab not in seen:
            ticks.append(t)
            seen.add(lab)

    while len(ticks) > 3:
        d_first = abs(ticks[0] - ymin)
        d_last = abs(ticks[-1] - ymax)
        if d_first <= d_last:
            ticks.pop(0)
        else:
            ticks.pop()

    if len(ticks) < 2:
        t1 = ymin + 0.25 * span
        t2 = ymin + 0.50 * span
        t3 = ymin + 0.75 * span
        ticks = []
        seen = set()
        for t in [t1, t2, t3]:
            lab = fmt.format(t)
            if lab not in seen:
                ticks.append(t)
                seen.add(lab)
        if len(ticks) < 2:
            ticks = [ymin + 0.50 * span]

    ax.set_yticks(ticks)
    mpl_fmt = fmt.replace("{:", "%").replace("}", "")
    ax.yaxis.set_major_formatter(FormatStrFormatter(mpl_fmt))


def _add_midpoint(ticks, fmt):
    """Insert a midpoint into the largest adjacent gap."""
    if len(ticks) < 2:
        return ticks

    best_i, best_gap = 0, 0
    for i in range(len(ticks) - 1):
        gap = ticks[i + 1] - ticks[i]
        if gap > best_gap:
            best_gap = gap
            best_i = i

    mid = (ticks[best_i] + ticks[best_i + 1]) / 2.0
    mid = max(mid, 0.0)
    mid_lab = fmt.format(mid)
    existing = {fmt.format(t) for t in ticks}

    if mid_lab not in existing:
        ticks.append(mid)
        ticks.sort()

    return ticks


def set_yticks_data_driven(ax, src_df, col_key, decimals):
    """
    Set y-ticks using key data values:
    - value at the maximum step
    - global maximum
    - global minimum

    If only two unique ticks remain after deduplication, insert a midpoint.
    """
    ymin_ax, ymax_ax = ax.get_ylim()
    fmt = f"{{:.{decimals}f}}"

    tmp = src_df.dropna(subset=[col_key]).sort_values('step')
    if tmp.empty:
        set_yticks_fixed(ax, decimals)
        return

    val_stepmax = max(tmp.iloc[-1][col_key], 0.0)
    val_min = max(tmp[col_key].min(), 0.0)
    val_max = max(tmp[col_key].max(), 0.0)

    ticks = []
    seen = set()
    for v in [val_stepmax, val_max, val_min]:
        lab = fmt.format(v)
        if lab not in seen:
            ticks.append(v)
            seen.add(lab)
    ticks.sort()

    ticks = [t for t in ticks if ymin_ax - 1e-9 <= t <= ymax_ax + 1e-9]

    if len(ticks) == 2:
        ticks = _add_midpoint(ticks, fmt)

    if len(ticks) == 1:
        c = ticks[0]
        lo = max((ymin_ax + c) / 2.0, 0.0)
        hi = (ymax_ax + c) / 2.0
        for v in [lo, hi]:
            lab = fmt.format(v)
            if lab not in {fmt.format(t) for t in ticks}:
                ticks.append(v)
        ticks.sort()
        if len(ticks) == 2:
            ticks = _add_midpoint(ticks, fmt)

    if len(ticks) < 2:
        set_yticks_fixed(ax, decimals)
        return

    ax.set_yticks(ticks)
    mpl_fmt = fmt.replace("{:", "%").replace("}", "")
    ax.yaxis.set_major_formatter(FormatStrFormatter(mpl_fmt))


# ==============================================================================
# Main plotting routine
# ==============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': TEXT_FS,
        'axes.titlesize': TEXT_FS,
        'axes.labelsize': TEXT_FS,
        'xtick.labelsize': TICK_FS,
        'ytick.labelsize': TICK_FS,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    N_ROWS = len(SCENARIOS)
    N_COLS = len(COLUMNS)

    FIG_W = 8.27
    FIG_H = 11.69

    MARGIN_L = 0.55
    MARGIN_R = 0.10
    MARGIN_B = 0.42
    MARGIN_T = 0.12

    GAP_W = 0.42
    GAP_H = 0.50

    AX_W = (FIG_W - MARGIN_L - MARGIN_R - (N_COLS - 1) * GAP_W) / N_COLS
    AX_H = (FIG_H - MARGIN_T - MARGIN_B - (N_ROWS - 1) * GAP_H) / N_ROWS

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    gs = fig.add_gridspec(
        nrows=N_ROWS,
        ncols=N_COLS,
        left=MARGIN_L / FIG_W,
        right=1.0 - MARGIN_R / FIG_W,
        bottom=MARGIN_B / FIG_H,
        top=1.0 - MARGIN_T / FIG_H,
        wspace=GAP_W / AX_W,
        hspace=GAP_H / AX_H,
    )

    csv_cache = {}

    for row_i, scen in enumerate(SCENARIOS):
        csv_key = str(scen['csv'])
        if csv_key not in csv_cache:
            csv_cache[csv_key] = pd.read_csv(scen['csv'])
        df_all = csv_cache[csv_key]

        algo = scen['algo']
        target = scen['target']

        df_algo = df_all[df_all['algorithm'] == algo].sort_values('step')
        df_files = df_algo[~df_algo['filename'].astype(str).str.contains('_-1_edges', na=False)]
        df_files = df_files[df_files['filename'] != 'step=-1']

        max_step = int(df_algo['step'].max()) if not df_algo.empty else 0
        step_iv = nice_x_interval(max_step)

        for col_j, (col_key, show_target) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[row_i, col_j])
            idx = row_i * N_COLS + col_j
            letter = get_subplot_label(idx)

            src = df_algo if col_key == 'LCC' else df_files

            if col_key in src.columns:
                tmp = src.dropna(subset=[col_key])
                if not tmp.empty:
                    style = LINE_STYLES.get(col_key, {})
                    ax.plot(tmp['step'], tmp[col_key], **style)

                    if col_key == 'LCC':
                        raw_lower = tmp[col_key].min() - (tmp[col_key].max() - tmp[col_key].min()) * 0.15
                        ylim = (max(raw_lower, 0), 1.01)
                    else:
                        ylim = get_local_ylim(tmp[col_key])
                    ax.set_ylim(ylim)

            if show_target:
                ax.axhline(target, color=C_TGT, ls='--', lw=1.0, zorder=5)
                ax.text(
                    0.03,
                    target,
                    f'target = {target}',
                    transform=ax.get_yaxis_transform(),
                    fontsize=TICK_FS,
                    color=C_TGT,
                    va='bottom',
                    ha='left',
                    fontstyle='italic',
                    zorder=10,
                )

            ax.set_title(letter, fontsize=TEXT_FS, fontweight='bold', loc='left', pad=3)

            beautify(ax)

            dec = COL_DECIMALS[col_j]
            if dec == -1:
                _, ym = ax.get_ylim()
                dec = 1 if ym >= 10 else 2

            if col_j in DATA_DRIVEN_COLS:
                set_yticks_data_driven(ax, src, col_key, dec)
            else:
                set_yticks_fixed(ax, dec)

            ax.xaxis.set_major_locator(MultipleLocator(step_iv))
            ax.set_xlim(left=0, right=max_step * 1.05 if max_step > 0 else 1.0)
            ax.tick_params(axis='x', labelbottom=True)

    fig.supxlabel('Attack step', fontsize=TEXT_FS, fontweight='normal', y=0.005)

    prefix = OUTPUT_DIR / 'fig7'
    for ext in ['pdf', 'svg', 'png']:
        save_path = prefix.with_suffix(f'.{ext}')
        fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.06)
        print(f'Saved: {save_path}')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
