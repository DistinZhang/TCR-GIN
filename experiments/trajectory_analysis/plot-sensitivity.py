#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/trajectory_analysis/plot-sensitivity.py

Generate publication-style sensitivity-analysis figures for TCR-GIN.

This script visualizes how model performance changes under three sensitivity
settings:
1. consistency_lambda
2. piss_k
3. feature_dim

For each selected parameter, the script generates a 4x3 panel figure:
- Attack-dataset panels:
  show MAE, A-MAE, M, and S with dual y-axes.
- Static-dataset panels:
  show MAE only.
- Time-cost panel:
  optionally show runtime curves for the feature-dimension setting.

The script supports two input modes:
1. Embedded demo data (default)
2. External CSV/TSV files

Output
------
For each requested figure, the script exports:
- PDF
- SVG
- PNG

into the specified output directory.

Usage
-----
Use embedded demo data:
    python experiments/trajectory_analysis/plot-sensitivity.py

Use external files:
    python experiments/trajectory_analysis/plot-sensitivity.py \
        --static_file path/to/static_results.csv \
        --attack_file path/to/attack_results.csv

Custom output directory:
    python experiments/trajectory_analysis/plot-sensitivity.py \
        --output_dir experiments/trajectory_analysis/results

Notes
-----
- Vector outputs use Type 42 fonts for editability in Illustrator-like tools.
- The plotting style follows a compact publication-oriented layout.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

# =============================================================================
# Section 0. Embedded demo data
# =============================================================================

STATIC_DATA_STR = """
Dataset,piss_k,consistency_lambda,feature_dim,MAE,MAE_std,Time
Small,1,0.2,3,0.011,0.0002,0.013
Small,3,0.2,3,0.011,0.0002,0.013
Small,5,0.2,3,0.011,0.0005,0.013
Small,1,0,3,0.011,0.0003,0.013
Small,1,0.5,3,0.011,0.0002,0.013
Small,1,0.8,3,0.011,0.0001,0.013
Small,1,0.2,5,0.011,0.0007,0.016
Small,1,0.2,7,0.011,0.0002,0.046
Medium,1,0.2,3,0.008,0.0005,0.037
Medium,3,0.2,3,0.007,0.0007,0.042
Medium,5,0.2,3,0.008,0.0003,0.042
Medium,1,0,3,0.008,0.0001,0.037
Medium,1,0.5,3,0.007,0.0002,0.042
Medium,1,0.8,3,0.007,0.0004,0.042
Medium,1,0.2,5,0.007,0.0003,0.051
Medium,1,0.2,7,0.007,0.0002,0.167
Large,1,0.2,3,0.005,0.0007,0.074
Large,3,0.2,3,0.005,0.0011,0.080
Large,5,0.2,3,0.005,0.0010,0.081
Large,1,0,3,0.005,0.0008,0.074
Large,1,0.5,3,0.005,0.0008,0.080
Large,1,0.8,3,0.006,0.0004,0.080
Large,1,0.2,5,0.004,0.0002,0.100
Large,1,0.2,7,0.005,0.0002,0.348
Huge,1,0.2,3,0.005,0.0010,0.128
Huge,3,0.2,3,0.004,0.0008,0.139
Huge,5,0.2,3,0.004,0.0005,0.140
Huge,1,0,3,0.004,0.0008,0.128
Huge,1,0.5,3,0.005,0.0010,0.138
Huge,1,0.8,3,0.005,0.0011,0.138
Huge,1,0.2,5,0.004,0.0002,0.173
Huge,1,0.2,7,0.004,0.0003,0.647
REDDIT,1,0.2,3,0.008,0.0009,0.031
REDDIT,3,0.2,3,0.007,0.0005,0.032
REDDIT,5,0.2,3,0.007,0.0007,0.032
REDDIT,1,0,3,0.007,0.0009,0.031
REDDIT,1,0.5,3,0.007,0.0009,0.030
REDDIT,1,0.8,3,0.007,0.0005,0.029
REDDIT,1,0.2,5,0.010,0.0022,0.035
REDDIT,1,0.2,7,0.009,0.0013,0.105
"""

ATTACK_DATA_STR = """
Dataset,piss_k,consistency_lambda,feature_dim,MAE,MAE_std,M,M_std,S,S_std,Add,Add_std
BA100,1,0.2,3,0.019,0.0016,0.0,0.0,0.7,1.0,0.000,0.0002
BA100,3,0.2,3,0.020,0.0015,0.000,0.0000,0.943,0.8611,0.000,0.0003
BA100,5,0.2,3,0.021,0.0012,0.000,0.0000,1.337,1.2730,0.000,0.0003
BA100,1,0,3,0.020,0.0009,0.000,0,0.400,0.3,0.000,0.0001
BA100,1,0.5,3,0.019,0.0013,0.000,0.0000,0.499,0.3681,0.000,0.0002
BA100,1,0.8,3,0.021,0.0010,0.000,0.0000,1.488,1.2517,0.001,0.0003
BA100,1,0.2,5,0.019,0.0023,0.000,0.0000,0.908,0.5706,0.000,0.0002
BA100,1,0.2,7,0.018,0.0013,0.081,0.1799,0.461,0.5315,0.001,0.0005
LFR100,1,0.2,3,0.014,0.0005,0.0,0.0,0.5,0.1,0.001,0.0002
LFR100,3,0.2,3,0.013,0.0005,0.000,0.0000,0.526,0.1779,0.000,0.0003
LFR100,5,0.2,3,0.014,0.0006,0.000,0.0000,0.695,0.3966,0.000,0.0003
LFR100,1,0,3,0.013,0.0003,0.000,0,0.500,0.2,0.000,0.0001
LFR100,1,0.5,3,0.013,0.0002,0.000,0.0000,0.590,0.2093,0.000,0.0002
LFR100,1,0.8,3,0.014,0.0005,0.000,0.0000,1.205,0.8074,0.001,0.0004
LFR100,1,0.2,5,0.014,0.0010,0.014,0.0304,0.574,0.3625,0.000,0.0002
LFR100,1,0.2,7,0.014,0.0013,0.027,0.0608,0.963,0.9966,0.001,0.0005
ER2000,1,0.2,3,0.015,0.0033,0.0,0.0,2.0,1.3,0.005,0.0008
ER2000,3,0.2,3,0.018,0.0033,0.761,1.0429,1.672,0.8540,0.006,0.0013
ER2000,5,0.2,3,0.017,0.0051,0.092,0.1016,1.989,1.2297,0.005,0.0007
ER2000,1,0,3,0.018,0.0067,0.400,0.6,1.900,0.7,0.005,0.0008
ER2000,1,0.5,3,0.018,0.0024,0.000,0.0000,1.434,1.0451,0.004,0.0010
ER2000,1,0.8,3,0.018,0.0031,0.000,0.0000,2.674,0.4772,0.006,0.0012
ER2000,1,0.2,5,0.020,0.0027,0.000,0.0000,2.754,0.7874,0.009,0.0018
ER2000,1,0.2,7,0.024,0.0037,0.071,0.1594,2.286,0.4213,0.011,0.0038
WS2000,1,0.2,3,0.012,0.0020,0.0,0.0,0.3,0.1,0.001,0.0001
WS2000,3,0.2,3,0.011,0.0020,0.000,0.0000,0.951,1.6092,0.001,0.0003
WS2000,5,0.2,3,0.011,0.0013,0.000,0.0000,0.815,0.8595,0.001,0.0001
WS2000,1,0,3,0.012,0.0025,0.000,0,0.900,1.4,0.001,0.0001
WS2000,1,0.5,3,0.011,0.0026,0.000,0.0000,0.635,0.6304,0.001,0.0001
WS2000,1,0.8,3,0.012,0.0012,0.000,0.0000,0.371,0.2956,0.001,0.0002
WS2000,1,0.2,5,0.009,0.0005,0.000,0.0000,0.217,0.1754,0.001,0.0002
WS2000,1,0.2,7,0.010,0.0008,0.000,0.0000,0.227,0.1423,0.001,0.0003
London,1,0.2,3,0.010,0.0006,0.0,0.0,0.5,0.2,0.005,0.0022
London,3,0.2,3,0.014,0.0032,0.000,0.0000,1.303,1.0981,0.003,0.0022
London,5,0.2,3,0.012,0.0028,0.000,0.0000,0.882,1.0394,0.003,0.0010
London,1,0,3,0.011,0.0033,0.000,0,0.700,1,0.004,0.0017
London,1,0.5,3,0.012,0.0015,0.000,0.0000,0.757,0.6983,0.004,0.0026
London,1,0.8,3,0.012,0.0016,0.000,0.0000,0.873,0.8062,0.004,0.0023
London,1,0.2,5,0.011,0.0015,0.000,0.0000,0.022,0.0481,0.002,0.0020
London,1,0.2,7,0.012,0.0017,0.000,0.0000,0.443,0.5607,0.005,0.0033
Power,1,0.2,3,0.029,0.0105,1.6,0.9,2.8,1.5,0.037,0.0151
Power,3,0.2,3,0.043,0.0281,2.660,1.0529,3.441,1.9625,0.039,0.0220
Power,5,0.2,3,0.029,0.0076,2.099,0.8573,3.118,2.2021,0.036,0.0151
Power,1,0,3,0.033,0.0151,2.100,0.7,3.300,0.8,0.038,0.0079
Power,1,0.5,3,0.026,0.0045,2.443,0.8592,3.227,1.2560,0.043,0.0091
Power,1,0.8,3,0.039,0.0171,2.110,0.6699,3.467,1.4146,0.046,0.0138
Power,1,0.2,5,0.026,0.0071,1.745,0.3637,2.585,0.3243,0.039,0.0026
Power,1,0.2,7,0.042,0.0364,2.324,2.0236,2.691,1.0610,0.050,0.0252
"""

# =============================================================================
# Section 1. Global style
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

rcParams["font.size"] = 7
rcParams["axes.titlesize"] = 8
rcParams["axes.labelsize"] = 7
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6

rcParams["axes.linewidth"] = 0.5
rcParams["lines.linewidth"] = 0.75
rcParams["lines.markersize"] = 3.0
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5
rcParams["xtick.major.size"] = 2.0
rcParams["ytick.major.size"] = 2.0
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = False
rcParams["ytick.right"] = False

# =============================================================================
# Section 2. CLI and data loading
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TCR-GIN sensitivity-analysis figures.")
    parser.add_argument(
        "--static_file",
        type=str,
        default=None,
        help="Path to static-dataset CSV/TSV file. If omitted, embedded demo data is used.",
    )
    parser.add_argument(
        "--attack_file",
        type=str,
        default=None,
        help="Path to attack-dataset CSV/TSV file. If omitted, embedded demo data is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save exported figures. Default: <script_dir>/results",
    )
    return parser.parse_args()


def load_table(file_path: str | None, embedded_csv: str) -> pd.DataFrame:
    if file_path is None:
        return pd.read_csv(io.StringIO(embedded_csv.strip()))

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


# =============================================================================
# Section 3. Plotting helpers
# =============================================================================

ATTACK_DATASETS = ["BA100", "LFR100", "ER2000", "WS2000", "London", "Power"]
STATIC_DATASETS = ["Small", "Medium", "Large", "Huge", "REDDIT"]
PANEL_LETTERS = "abcdefghijkl"
DEFAULTS = {
    "piss_k": 1,
    "consistency_lambda": 0.2,
    "feature_dim": 3,
}


def get_subset(df: pd.DataFrame, ds_name: str, param_col: str) -> pd.DataFrame:
    subset = df[df["Dataset"] == ds_name]
    condition = pd.Series(True, index=subset.index)

    for col, val in DEFAULTS.items():
        if col != param_col:
            condition &= subset[col] == val

    return subset[condition].sort_values(by=param_col)


def left_axis_formatter(x, pos):
    return "" if x < 0 else f"{x:.3f}"


def right_axis_formatter(x, pos):
    return "" if x < 0 else f"{x:.1f}"


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = filename.rsplit(".", 1)[0]
    base_path = output_dir / base_name

    fig.savefig(base_path.with_suffix(".pdf"), format="pdf", transparent=True, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), format="svg", transparent=True, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), format="png", dpi=600, transparent=True, bbox_inches="tight")
    return base_path


# =============================================================================
# Section 4. Core plotting
# =============================================================================

def plot_sensitivity_figure(
    df_static: pd.DataFrame,
    df_attack: pd.DataFrame,
    param_name: str,
    param_col: str,
    param_values: List[float],
    filename: str,
    x_label: str,
    output_dir: Path,
    extra_plot_time: bool = False,
) -> None:
    fig = plt.figure(figsize=(7.08, 8.5))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.06, wspace=0.5, hspace=0.25)

    x_indices = np.arange(len(param_values))
    x_tick_labels = [str(v) for v in param_values]

    # -------------------------------------------------------------------------
    # Part 1: Attack datasets (1-6) - dual y-axis
    # -------------------------------------------------------------------------
    for i, ds in enumerate(ATTACK_DATASETS):
        ax = fig.add_subplot(4, 3, i + 1)
        data = get_subset(df_attack, ds, param_col)

        y_mae, y_mae_std = [], []
        y_add, y_add_std = [], []
        y_m, y_m_std = [], []
        y_s, y_s_std = [], []
        valid_indices = []

        for idx, val in enumerate(param_values):
            row = data[np.isclose(data[param_col], val)]
            if not row.empty:
                valid_indices.append(idx)
                y_mae.append(row["MAE"].values[0])
                y_mae_std.append(row["MAE_std"].values[0])
                y_add.append(row["Add"].values[0])
                y_add_std.append(row["Add_std"].values[0])
                y_m.append(row["M"].values[0])
                y_m_std.append(row["M_std"].values[0])
                y_s.append(row["S"].values[0])
                y_s_std.append(row["S_std"].values[0])

        ax.plot(valid_indices, y_mae, "o-", color="#1f77b4", label="MAE", markersize=3, linewidth=0.75)
        ax.fill_between(
            valid_indices,
            np.array(y_mae) - np.array(y_mae_std),
            np.array(y_mae) + np.array(y_mae_std),
            color="#1f77b4",
            alpha=0.2,
        )
        ax.plot(valid_indices, y_add, "s--", color="#ff7f0e", label="A-MAE", markersize=3, linewidth=0.75)
        ax.fill_between(
            valid_indices,
            np.array(y_add) - np.array(y_add_std),
            np.array(y_add) + np.array(y_add_std),
            color="#ff7f0e",
            alpha=0.2,
        )

        ax.set_ylabel("MAE / A-MAE", fontsize=6)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_tick_labels, fontsize=6)

        y1_max = max(
            np.max(np.array(y_mae) + np.array(y_mae_std)) if valid_indices else 0.0,
            np.max(np.array(y_add) + np.array(y_add_std)) if valid_indices else 0.0,
        )
        ax.set_ylim(bottom=-0.001, top=y1_max * 1.15 + 0.002 if y1_max > 0 else 0.1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(left_axis_formatter))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

        ax2 = ax.twinx()
        ax2.plot(valid_indices, y_m, "^-", color="#2ca02c", label="Monotonicity (M)", markersize=3, linewidth=0.75)
        ax2.fill_between(
            valid_indices,
            np.array(y_m) - np.array(y_m_std),
            np.array(y_m) + np.array(y_m_std),
            color="#2ca02c",
            alpha=0.2,
        )
        ax2.plot(valid_indices, y_s, "d--", color="#d62728", label="Smoothness (S)", markersize=3, linewidth=0.75)
        ax2.fill_between(
            valid_indices,
            np.array(y_s) - np.array(y_s_std),
            np.array(y_s) + np.array(y_s_std),
            color="#d62728",
            alpha=0.2,
        )

        ax2.set_ylabel(r"M / S $(\times 10^{2})$", fontsize=6)
        y2_max = max(
            np.max(np.array(y_m) + np.array(y_m_std)) if valid_indices else 0.0,
            np.max(np.array(y_s) + np.array(y_s_std)) if valid_indices else 0.0,
        )
        ax2.set_ylim(bottom=-0.1, top=y2_max * 1.3 + 0.05 if y2_max > 0 else 1.0)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(right_axis_formatter))
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax2.tick_params(right=True)

        ax.text(
            0.0, 1.05, f"({PANEL_LETTERS[i]}) {ds}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.grid(False)
        ax2.grid(False)

    # -------------------------------------------------------------------------
    # Part 2: Static datasets (7-11) - single y-axis
    # -------------------------------------------------------------------------
    for i, ds in enumerate(STATIC_DATASETS):
        panel_idx = 6 + i
        ax = fig.add_subplot(4, 3, panel_idx + 1)
        data = get_subset(df_static, ds, param_col)

        y_mae, y_mae_std = [], []
        valid_indices = []

        for idx_val, val in enumerate(param_values):
            row = data[np.isclose(data[param_col], val)]
            if not row.empty:
                valid_indices.append(idx_val)
                y_mae.append(row["MAE"].values[0])
                y_mae_std.append(row["MAE_std"].values[0])

        ax.plot(valid_indices, y_mae, "o-", color="#1f77b4", label="MAE", markersize=3, linewidth=0.75)
        ax.fill_between(
            valid_indices,
            np.array(y_mae) - np.array(y_mae_std),
            np.array(y_mae) + np.array(y_mae_std),
            color="#1f77b4",
            alpha=0.2,
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_tick_labels, fontsize=6)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel("MAE", fontsize=6)

        y_max = np.max(np.array(y_mae) + np.array(y_mae_std)) if valid_indices else 0.0
        ax.set_ylim(bottom=-0.001, top=y_max * 1.2 + 0.002 if y_max > 0 else 0.05)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(left_axis_formatter))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.tick_params(right=False, labelright=False)

        ax.text(
            0.0, 1.05, f"({PANEL_LETTERS[panel_idx]}) {ds}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.grid(False)

    # -------------------------------------------------------------------------
    # Part 3: Time cost panel (12)
    # -------------------------------------------------------------------------
    ax_12 = fig.add_subplot(4, 3, 12)
    if extra_plot_time:
        time_datasets = ["Small", "Medium", "Large", "Huge"]
        time_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

        for i, ds in enumerate(time_datasets):
            subset = df_static[df_static["Dataset"] == ds]
            cond = (subset["piss_k"] == 1) & (subset["consistency_lambda"] == 0.2)
            data = subset[cond].sort_values(by="feature_dim")

            y_time = []
            valid_indices = []
            for idx_val, val in enumerate(param_values):
                row = data[np.isclose(data["feature_dim"], val)]
                if not row.empty:
                    valid_indices.append(idx_val)
                    y_time.append(row["Time"].values[0])

            ax_12.plot(valid_indices, y_time, "o-", color=time_colors[i], label=ds, linewidth=0.75, markersize=3)

        ax_12.set_xticks(x_indices)
        ax_12.set_xticklabels(x_tick_labels, fontsize=6)
        ax_12.set_ylabel("Time (s)", fontsize=6)
        ax_12.set_xlabel(x_label, fontsize=7)
        ax_12.text(
            0.0, 1.05, "(l) Time Cost",
            transform=ax_12.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax_12.legend(fontsize=5, frameon=False, loc="upper left")
        ax_12.set_ylim(bottom=0)
        ax_12.grid(False)
        ax_12.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_12.tick_params(right=False)
    else:
        ax_12.axis("off")

    # -------------------------------------------------------------------------
    # Part 4: Global legend
    # -------------------------------------------------------------------------
    legend_elements = [
        mlines.Line2D([], [], color="#1f77b4", marker="o", markersize=5, label="MAE", linestyle="-", linewidth=0.75),
        mlines.Line2D([], [], color="#ff7f0e", marker="s", markersize=5, label="A-MAE", linestyle="--", linewidth=0.75),
        mlines.Line2D([], [], color="#2ca02c", marker="^", markersize=5, label="M", linestyle="-", linewidth=0.75),
        mlines.Line2D([], [], color="#d62728", marker="d", markersize=5, label="S", linestyle="--", linewidth=0.75),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.5,
        columnspacing=1.0,
    )

    base_path = save_figure(fig, output_dir, filename)
    print(f"Figures saved to: {base_path}.[pdf/svg/png]")
    plt.close(fig)


# =============================================================================
# Section 5. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_DIR / "results"
    )

    df_static = load_table(args.static_file, STATIC_DATA_STR)
    df_attack = load_table(args.attack_file, ATTACK_DATA_STR)

    plot_sensitivity_figure(
        df_static=df_static,
        df_attack=df_attack,
        param_name="consistency_lambda",
        param_col="consistency_lambda",
        param_values=[0, 0.2, 0.5, 0.8],
        filename="sensitivity_lambda.png",
        x_label=r"$\lambda$",
        output_dir=output_dir,
        extra_plot_time=False,
    )

    plot_sensitivity_figure(
        df_static=df_static,
        df_attack=df_attack,
        param_name="piss_k",
        param_col="piss_k",
        param_values=[1, 3, 5],
        filename="sensitivity_k.png",
        x_label=r"$K$",
        output_dir=output_dir,
        extra_plot_time=False,
    )

    plot_sensitivity_figure(
        df_static=df_static,
        df_attack=df_attack,
        param_name="feature_dim",
        param_col="feature_dim",
        param_values=[3, 5, 7],
        filename="sensitivity_dim.png",
        x_label=r"$d_{in}$",
        output_dir=output_dir,
        extra_plot_time=True,
    )


if __name__ == "__main__":
    main()
