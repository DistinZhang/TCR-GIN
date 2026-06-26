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
  show MAE, M, and S with dual y-axes.
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
Small,1,0.2,3,0.0112,0.0001,0.0165
Small,1,0,3,0.0112,0.0003,0.0061
Small,1,0.5,3,0.0112,0.0002,0.0176
Small,1,0.8,3,0.0113,0.0001,0.0173
Small,1,0.2,5,0.0114,0.0007,0.0409
Small,1,0.2,7,0.0108,0.0002,0.0578
Small,3,0.2,3,0.0114,0.0002,0.0175
Small,5,0.2,3,0.0113,0.0005,0.0394
Medium,1,0.2,3,0.0076,0.0005,0.0434
Medium,1,0,3,0.0076,0.0001,0.0168
Medium,1,0.5,3,0.0072,0.0002,0.044
Medium,1,0.8,3,0.0073,0.0004,0.0438
Medium,1,0.2,5,0.0065,0.0003,0.1129
Medium,1,0.2,7,0.0067,0.0002,0.1815
Medium,3,0.2,3,0.0071,0.0007,0.1039
Medium,5,0.2,3,0.0076,0.0003,0.0442
Large,1,0.2,3,0.0052,0.0007,0.1814
Large,1,0,3,0.0054,0.0008,0.0348
Large,1,0.5,3,0.0049,0.0008,0.2958
Large,1,0.8,3,0.0057,0.0004,0.0809
Large,1,0.2,5,0.0044,0.0002,0.328
Large,1,0.2,7,0.0045,0.0002,0.649
Large,3,0.2,3,0.0053,0.0011,0.2206
Large,5,0.2,3,0.0050,0.0010,0.2354
Extra-large,1,0.2,3,0.0050,0.0009,0.189
Extra-large,1,0,3,0.0041,0.0008,0.0611
Extra-large,1,0.5,3,0.0047,0.0010,0.4258
Extra-large,1,0.8,3,0.0053,0.0011,0.2063
Extra-large,1,0.2,5,0.0040,0.0002,0.5189
Extra-large,1,0.2,7,0.0042,0.0003,1.1146
Extra-large,3,0.2,3,0.0039,0.0008,0.5973
Extra-large,5,0.2,3,0.0043,0.0005,0.816
REDDIT,1,0.2,3,0.0076,0.0008,0.0274
REDDIT,1,0,3,0.0073,0.0011,0.01
REDDIT,1,0.5,3,0.0074,0.0009,0.0323
REDDIT,1,0.8,3,0.0069,0.0005,0.0311
REDDIT,1,0.2,5,0.0103,0.0022,0.0359
REDDIT,1,0.2,7,0.0092,0.0013,0.0895
REDDIT,3,0.2,3,0.0072,0.0005,0.0339
REDDIT,5,0.2,3,0.0074,0.0007,0.034
"""

ATTACK_DATA_STR = """
Dataset,piss_k,consistency_lambda,feature_dim,MAE,MAE_std,M,M_std,S,S_std
BA100,1,0.2,3,0.019345,0.001555,0,0,0.006554,0.010205
BA100,3,0.2,3,0.020369,0.001514,0,0,0.009426,0.008611
BA100,5,0.2,3,0.021344,0.001234,0,0,0.013368,0.012730
BA100,1,0,3,0.020233,0.000914,0,0,0.003888,0.003240
BA100,1,0.5,3,0.019199,0.001255,0,0,0.004994,0.003681
BA100,1,0.8,3,0.020820,0.000980,0,0,0.014882,0.012517
BA100,1,0.2,5,0.018920,0.002317,0,0,0.009078,0.005706
BA100,1,0.2,7,0.017534,0.001303,0.000805,0.001799,0.004611,0.005315
LFR100,1,0.2,3,0.013577,0.000506,0,0,0.005398,0.001304
LFR100,3,0.2,3,0.013416,0.000460,0,0,0.005257,0.001779
LFR100,5,0.2,3,0.013680,0.000568,0,0,0.006948,0.003966
LFR100,1,0,3,0.013150,0.000349,0,0,0.005286,0.002321
LFR100,1,0.5,3,0.013160,0.000245,0,0,0.005903,0.002093
LFR100,1,0.8,3,0.013661,0.000456,0,0,0.012047,0.008074
LFR100,1,0.2,5,0.014249,0.001001,0.000136,0.000304,0.005743,0.003625
LFR100,1,0.2,7,0.013760,0.001292,0.000272,0.000608,0.009634,0.009966
ER2000,1,0.2,3,0.015223,0.003268,0,0,0.020397,0.013351
ER2000,3,0.2,3,0.018324,0.003264,0.007612,0.010429,0.016722,0.008540
ER2000,5,0.2,3,0.017339,0.005116,0.000923,0.001016,0.019890,0.012297
ER2000,1,0,3,0.018309,0.006685,0.004306,0.006334,0.019094,0.007364
ER2000,1,0.5,3,0.018369,0.002441,0,0,0.014340,0.010451
ER2000,1,0.8,3,0.017879,0.003124,0,0,0.026739,0.004772
ER2000,1,0.2,5,0.020270,0.002731,0,0,0.027540,0.007874
ER2000,1,0.2,7,0.023568,0.003658,0.000713,0.001594,0.022856,0.004213
WS2000,1,0.2,3,0.011723,0.001967,0,0,0.002759,0.001195
WS2000,3,0.2,3,0.010894,0.001974,0,0,0.009509,0.016092
WS2000,5,0.2,3,0.011235,0.001301,0,0,0.008154,0.008595
WS2000,1,0,3,0.011865,0.002487,0,0,0.008511,0.013712
WS2000,1,0.5,3,0.011232,0.002647,0,0,0.006347,0.006304
WS2000,1,0.8,3,0.012175,0.001203,0,0,0.003708,0.002956
WS2000,1,0.2,5,0.008610,0.000499,0,0,0.002165,0.001754
WS2000,1,0.2,7,0.009517,0.000847,0,0,0.002268,0.001423
transport,1,0.2,3,0.009923,0.000579,0,0,0.004686,0.002452
transport,3,0.2,3,0.009568,0.000702,0,0,0.005872,0.004171
transport,5,0.2,3,0.009216,0.000225,0,0,0.004846,0.004803
transport,1,0,3,0.009589,0.000360,0,0,0.001852,0.002165
transport,1,0.5,3,0.009836,0.000681,0,0,0.007919,0.006371
transport,1,0.8,3,0.011071,0.001280,0,0,0.003297,0.001571
transport,1,0.2,5,0.010313,0.001440,0,0,0.001832,0.002433
transport,1,0.2,7,0.012760,0.002429,0,0,0.011687,0.014317
Power,1,0.2,3,0.028703,0.010592,0.015992,0.008534,0.028314,0.015160
Power,3,0.2,3,0.037442,0.023039,0.025837,0.009911,0.032723,0.017210
Power,5,0.2,3,0.027524,0.005563,0.020357,0.007247,0.028945,0.017241
Power,1,0,3,0.033171,0.014805,0.020459,0.006773,0.032817,0.007758
Power,1,0.5,3,0.026083,0.003868,0.024434,0.008592,0.032272,0.012560
Power,1,0.8,3,0.038800,0.017139,0.021097,0.006699,0.034674,0.014146
Power,1,0.2,5,0.024389,0.005343,0.017452,0.003637,0.026008,0.003123
Power,1,0.2,7,0.031249,0.019376,0.017244,0.007114,0.026341,0.010023
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

ATTACK_DATASETS = ["BA100", "LFR100", "ER2000", "WS2000", "transport", "Power"]
STATIC_DATASETS = ["Small", "Medium", "Large", "Extra-large", "REDDIT"]
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
        y_m, y_m_std = [], []
        y_s, y_s_std = [], []
        valid_indices = []

        for idx, val in enumerate(param_values):
            row = data[np.isclose(data[param_col], val)]
            if not row.empty:
                valid_indices.append(idx)
                y_mae.append(row["MAE"].values[0])
                y_mae_std.append(row["MAE_std"].values[0])
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

        ax.set_ylabel("MAE", fontsize=6)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_tick_labels, fontsize=6)

        y1_max = np.max(np.array(y_mae) + np.array(y_mae_std)) if valid_indices else 0.0
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
        time_datasets = ["Small", "Medium", "Large", "Extra-large"]
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
        mlines.Line2D([], [], color="#2ca02c", marker="^", markersize=5, label="M", linestyle="-", linewidth=0.75),
        mlines.Line2D([], [], color="#d62728", marker="d", markersize=5, label="S", linestyle="--", linewidth=0.75),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
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
