#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/baseline_comparison/plot3.py

Generate Fig. 3 (baseline comparison), including panels:
  a-e: MAE vs Time on five datasets
  f  : exact-solution analysis with inset zoom
  g  : true-MAE vs observed-MAE comparison

Usage
-----
python experiments/baseline_comparison/plot.py

Optional arguments
------------------
python experiments/baseline_comparison/plot3.py \
    --exact_results_dir experiments/baseline_comparison/exact_comparison/results \
    --output_dir experiments/baseline_comparison/results \
    --output_name fig3 \
    --formats pdf svg png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import BboxConnector, BboxPatch, TransformedBbox

warnings.filterwarnings("ignore")


# =============================================================================
# Section 0. Paths and Global Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Keep this style exactly as requested (global text 10pt, tick labels 9pt)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 600,
})

TCR_GIN_COLOR = "#FF0000"

ALGO_ORDER = [
    "TCR-GIN", "DC", "DCR", "BC", "BCR", "DomiRank", "FINDER", "GDM", "GDMR",
    "CoreGDM", "MS", "MSR", "GND", "GNDR", "CI_l1", "CI_l2", "CI_l3", "CoreHD",
    "EGND", "EI_s1", "EI_s2", "NES", "NESR", "NEM", "NEMR", "NEL", "NELR", "VE", "VER"
]

KEEP_ALGOS = ["TCR-GIN", "DC", "DCR", "BC", "BCR", "CoreHD", "MS", "MSR", "CoreGDM"]

ALGO_PAIRS = [
    ("DC", "DCR"), ("BC", "BCR"), ("GDM", "GDMR"), ("MS", "MSR"), ("GND", "GNDR"),
    ("NES", "NESR"), ("NEM", "NEMR"), ("NEL", "NELR"), ("VE", "VER")
]

COL_MAP = {
    "CollectiveInfluenceL1": "CI_l1",
    "CollectiveInfluenceL2": "CI_l2",
    "CollectiveInfluenceL3": "CI_l3",
    "CoreGDM": "CoreGDM",
    "CoreHD": "CoreHD",
    "Domirank": "DomiRank",
    "EGND": "EGND",
    "EI_s1": "EI_s1",
    "EI_s2": "EI_s2",
    "FINDER_CN": "FINDER",
    "GDM": "GDM",
    "GDMR": "GDMR",
    "GND": "GND",
    "GNDR": "GNDR",
    "MS": "MS",
    "MSR": "MSR",
    "betweenness_centrality_F": "BC",
    "betweenness_centrality_T": "BCR",
    "degree_F": "DC",
    "degree_T": "DCR",
    "network_entanglement_large": "NEL",
    "network_entanglement_large_reinsertion": "NELR",
    "network_entanglement_mid": "NEM",
    "network_entanglement_mid_reinsertion": "NEMR",
    "network_entanglement_small": "NES",
    "network_entanglement_small_reinsertion": "NESR",
    "vertex_entanglement": "VE",
    "vertex_entanglement_reinsertion": "VER",
}


# =============================================================================
# Section 1. Static Data (Panels a-e)
# =============================================================================

data_mae = {
    'Small': {'TCR-GIN': 0.011, 'DC': 0.190, 'DCR': 0.044, 'BC': 0.167, 'BCR': 0.027, 'DomiRank': 0.120, 'FINDER': 0.072, 'GDM': 0.034, 'GDMR': 0.013, 'CoreGDM': 0.008, 'MS': 0.018, 'MSR': 0.021, 'GND': 0.115, 'GNDR': 0.058, 'CI_l1': 0.024, 'CI_l2': 0.098, 'CI_l3': 0.160, 'CoreHD': 0.011, 'EGND': 0.097, 'EI_s1': 0.077, 'EI_s2': 0.419, 'NES': 0.215, 'NESR': 0.050, 'NEM': 0.176, 'NEMR': 0.045, 'NEL': 0.181, 'NELR': 0.046, 'VE': 0.157, 'VER': 0.044},
    'Medium': {'TCR-GIN': 0.008, 'DC': 0.282, 'DCR': 0.067, 'BC': 0.292, 'BCR': 0.030, 'DomiRank': 0.185, 'FINDER': 0.088, 'GDM': 0.038, 'GDMR': 0.015, 'CoreGDM': 0.010, 'MS': 0.072, 'MSR': 0.017, 'GND': 0.152, 'GNDR': 0.057, 'CI_l1': 0.029, 'CI_l2': 0.140, 'CI_l3': 0.175, 'CoreHD': 0.005, 'EGND': 0.098, 'EI_s1': 0.111, 'EI_s2': 0.1175, 'NES': 31.8652, 'NESR': 31.9972, 'NEM': 30.578, 'NEMR': 30.7031, 'NEL': 35.9983, 'NELR': 36.1698, 'VE': 8.9721, 'VER': 14.6033},
    'Large': {'TCR-GIN': 0.005, 'DC': 0.267, 'DCR': 0.059, 'BC': 0.252, 'BCR': 0.034, 'DomiRank': 0.195, 'FINDER': 0.125, 'GDM': 0.051, 'GDMR': 0.020, 'CoreGDM': 0.014, 'MS': 0.007, 'MSR': 0.004, 'GND': 0.169, 'GNDR': 0.062, 'CI_l1': 0.033, 'CI_l2': 0.128, 'CI_l3': 0.147, 'CoreHD': 0.006, 'EGND': 0.113, 'EI_s1': 0.138, 'EI_s2': 0.2641, 'NES': 100.3697, 'NESR': 100.5408, 'NEM': 109.6003, 'NEMR': 109.7611, 'NEL': 104.8616, 'NELR': 105.0416, 'VE': 26.2423, 'VER': 31.7096},
    'Huge': {'TCR-GIN': 0.005, 'DC': 0.276, 'DCR': 0.062, 'BC': 0.266, 'BCR': 0.035, 'DomiRank': 0.206, 'FINDER': 0.131, 'GDM': 0.052, 'GDMR': 0.020, 'CoreGDM': 0.015, 'MS': 0.007, 'MSR': 0.003, 'GND': 0.186, 'GNDR': 0.061, 'CI_l1': 0.042, 'CI_l2': 0.133, 'CI_l3': 0.151, 'CoreHD': 0.007, 'EGND': 0.114, 'EI_s1': 0.152, 'EI_s2': 0.460, 'NES': 0.297, 'NESR': 0.050, 'NEM': 0.265, 'NEMR': 0.044, 'NEL': 0.243, 'NELR': 0.043, 'VE': 0.259, 'VER': 0.047},
    'REDDIT': {'TCR-GIN': 0.008, 'DC': 0.064, 'DCR': 0.029, 'BC': 0.129, 'BCR': 0.002, 'DomiRank': 0.051, 'FINDER': 0.005, 'GDM': 0.004, 'GDMR': 0.001, 'CoreGDM': 0.001, 'MS': 0.139, 'MSR': 0.023, 'GND': 0.170, 'GNDR': 0.021, 'CI_l1': 0.104, 'CI_l2': 0.167, 'CI_l3': 0.170, 'CoreHD': 0.001, 'EGND': 0.023, 'EI_s1': 0.104, 'EI_s2': 0.1097, 'NES': 28.9222, 'NESR': 29.0085, 'NEM': 34.0616, 'NEMR': 34.174, 'NEL': 46.9117, 'NELR': 47.0033, 'VE': 8.5124, 'VER': 14.0638}
}

data_std = {'Small': 0.0001, 'Medium': 0.0005, 'Large': 0.0007, 'Huge': 0.0009, 'REDDIT': 0.0008}

data_time = {
    'Small': {'TCR-GIN': 0.0165, 'DC': 0.0018, 'DCR': 0.0047, 'BC': 0.0022, 'BCR': 0.0237, 'DomiRank': 2.1872, 'FINDER': 0.2725, 'GDM': 5.0745, 'GDMR': 20.4437, 'CoreGDM': 36.2579, 'MS': 2.3876, 'MSR': 2.4932, 'GND': 0.1097, 'GNDR': 0.1041, 'CI_l1': 0.419, 'CI_l2': 0.4189, 'CI_l3': 0.4322, 'CoreHD': 0.3362, 'EGND': 5.3954, 'EI_s1': 0.1067, 'EI_s2': 0.1043, 'NES': 0.7237, 'NESR': 0.0892, 'NEM': 0.7054, 'NEMR': 0.0963, 'NEL': 0.7276, 'NELR': 0.0914, 'VE': 0.7562, 'VER': 4.1086},
    'Medium': {'TCR-GIN': 0.0434, 'DC': 0.0075, 'DCR': 0.0235, 'BC': 0.011, 'BCR': 0.2955, 'DomiRank': 3.8319, 'FINDER': 0.7638, 'GDM': 5.436, 'GDMR': 22.391, 'CoreGDM': 50.9918, 'MS': 9.3396, 'MSR': 9.4666, 'GND': 0.1665, 'GNDR': 0.1249, 'CI_l1': 0.4794, 'CI_l2': 0.4913, 'CI_l3': 0.5726, 'CoreHD': 0.3624, 'EGND': 38.8013, 'EI_s1': 0.1489, 'EI_s2': 0.1502, 'NES': 14.8874, 'NESR': 0.1321, 'NEM': 15.1317, 'NEMR': 0.1402, 'NEL': 15.9606, 'NELR': 0.1353, 'VE': 6.7132, 'VER': 4.1354},
    'Large': { 'TCR-GIN': 0.1814, 'DC': 0.0237, 'DCR': 0.0654, 'BC': 0.0363, 'BCR': 1.574, 'DomiRank': 6.9533, 'FINDER': 1.788, 'GDM': 6.3918, 'GDMR': 25.7004, 'CoreGDM': 77.4105, 'MS': 17.0699, 'MSR': 17.2048, 'GND': 0.2631, 'GNDR': 0.1656, 'CI_l1': 0.5925, 'CI_l2': 0.6609, 'CI_l3': 1.3519, 'CoreHD': 0.4159, 'EGND': 129.2289, 'EI_s1': 0.2399, 'EI_s2': 0.272, 'NES': 58.7194, 'NESR': 0.1639, 'NEM': 58.1428, 'NEMR': 0.1776, 'NEL': 57.4584, 'NELR': 0.1613, 'VE': 19.4635, 'VER': 4.1646},
    'Huge': {'TCR-GIN': 0.1890, 'DC': 0.0598, 'DCR': 0.1536, 'BC': 0.09, 'BCR': 9.2372, 'DomiRank': 14.7349, 'FINDER': 2.8037, 'GDM': 8.635, 'GDMR': 31.0541, 'CoreGDM': 118.4571, 'MS': 29.7292, 'MSR': 29.8856, 'GND': 0.5057, 'GNDR': 0.2489, 'CI_l1': 0.7105, 'CI_l2': 0.8237, 'CI_l3': 1.5341, 'CoreHD': 0.4741, 'EGND': 337.0414, 'EI_s1': 0.4159, 'EI_s2': 0.4939, 'NES': 237.7763, 'NESR': 0.1935, 'NEM': 241.7978, 'NEMR': 0.2071, 'NEL': 242.5119, 'NELR': 0.2571, 'VE': 53.5046, 'VER': 4.3348
    },
    'REDDIT': {'TCR-GIN': 0.0274, 'DC': 0.0022, 'DCR': 0.0056, 'BC': 0.0073, 'BCR': 0.026, 'DomiRank': 3.1331, 'FINDER': 0.2228, 'GDM': 5.219, 'GDMR': 20.5209, 'CoreGDM': 39.2343, 'MS': 7.8986, 'MSR': 7.9442, 'GND': 0.1616, 'GNDR': 0.2768, 'CI_l1': 0.3758, 'CI_l2': 0.3744, 'CI_l3': 0.3843, 'CoreHD': 0.2968, 'EGND': 10.0787, 'EI_s1': 0.0815, 'EI_s2': 0.0856, 'NES': 11.2076, 'NESR': 11.3304, 'NEM': 11.4341, 'NEMR': 11.5634, 'NEL': 11.2453, 'NELR': 11.3772, 'VE': 6.4368, 'VER': 10.6235}
}

df_mae = pd.DataFrame(data_mae).T
df_time = pd.DataFrame(data_time).T


# =============================================================================
# Section 2. CLI / Paths / Style Map
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline comparison figure (Fig. 3).")
    parser.add_argument("--exact_results_dir", type=str, default=None, help="Directory for exact/observed CSV files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--output_name", type=str, default="fig3", help="Output base filename.")
    parser.add_argument("--formats", nargs="+", default=["pdf"], help="Save formats, e.g. pdf svg png.")
    return parser.parse_args()


def resolve_exact_results_dir(user_dir: str | None) -> Path:
    if user_dir:
        p = Path(user_dir).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    candidates = [
        SCRIPT_DIR / "exact_comparison" / "results",
        PROJECT_ROOT / "experiments" / "baseline_comparison" / "exact_comparison" / "results",
        PROJECT_ROOT / "experiments" / "3_baseline_comparison" / "3_2_exact_comparison" / "results",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_output_dir(user_dir: str | None) -> Path:
    if user_dir:
        p = Path(user_dir).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    return SCRIPT_DIR / "results"


def build_style_map() -> Dict[str, Dict[str, Any]]:
    style_map: Dict[str, Dict[str, Any]] = {}
    cmap = plt.get_cmap("tab10")

    keep_colors = {
        "TCR-GIN": "#FF0000",
        "DC": cmap(0), "DCR": cmap(0),
        "BC": cmap(2), "BCR": cmap(2),
        "MS": cmap(1), "MSR": cmap(1),
        "CoreHD": cmap(4),
        "CoreGDM": cmap(3),
    }

    marker_rules: Dict[str, str] = {}
    for orig, reins in ALGO_PAIRS:
        marker_rules[orig] = "^"
        marker_rules[reins] = "s"

    marker_rules.update({"CI_l1": "^", "CI_l2": "v", "CI_l3": "<", "EI_s1": "^", "EI_s2": "s"})
    for algo in ["DomiRank", "FINDER", "CoreGDM", "CoreHD", "EGND"]:
        marker_rules[algo] = "^"
    marker_rules["TCR-GIN"] = "*"

    for algo in ALGO_ORDER:
        m = marker_rules.get(algo, "o")
        if algo in KEEP_ALGOS:
            style_map[algo] = {"color": keep_colors[algo], "marker": m, "label": algo}
        else:
            style_map[algo] = {"color": "#B0B0B0", "marker": m, "label": "Others"}

    return style_map


# =============================================================================
# Section 3. Data Loading for Panels f/g
# =============================================================================

def parse_mae_value(v: Any) -> float:
    s = str(v).strip()
    if "±" in s:
        s = s.split("±")[0].strip()
    return float(s)


def load_exact_observed_data(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Load data for panels (f) and (g).
    Falls back to dummy data if files are missing.
    """
    try:
        df_exact_detail = pd.read_csv(base_path / "result_exact_detailed.csv")
        df_observ_detail = pd.read_csv(base_path / "result_observ_detailed.csv")
        df_exact_summary = pd.read_csv(base_path / "result_exact.csv")
        df_observ_summary = pd.read_csv(base_path / "result_observ.csv")

        df_exact_detail.columns = df_exact_detail.columns.str.strip()
        df_observ_detail.columns = df_observ_detail.columns.str.strip()

        df_exact_detail.rename(columns={"TI-GIN": "TCR-GIN", "TI-GIN (Std)": "TCR-GIN (Std)"}, inplace=True)
        df_observ_detail.rename(columns={"TI-GIN": "TCR-GIN", "TI-GIN (Std)": "TCR-GIN (Std)"}, inplace=True)

        if "Label" in df_observ_detail.columns:
            df_observ_detail = df_observ_detail.rename(columns={"Label": "Label_observ"})
        observ_keep_cols = ["network", "Label_observ"]
        df_observ_detail_clean = df_observ_detail[
            [c for c in observ_keep_cols if c in df_observ_detail.columns]
        ]
        df_merged = pd.merge(df_exact_detail, df_observ_detail_clean, on="network")
        df_merged = df_merged.sort_values("Label").reset_index(drop=True)

        # Build data for panel (g)
        mae_data: List[Dict[str, Any]] = []

        tcr_mae_true = parse_mae_value(df_exact_summary["MAE"].iloc[0])
        tcr_mae_obs = parse_mae_value(df_observ_summary["MAE"].iloc[0])
        mae_data.append({"Algorithm": "TCR-GIN", "True_MAE": tcr_mae_true, "Observed_MAE": tcr_mae_obs})

        for col_detail, algo_name in COL_MAP.items():
            if col_detail not in df_merged.columns:
                continue

            preds = df_merged[col_detail]
            trues = df_merged["Label"]
            mask = preds.notna() & trues.notna()
            if mask.sum() == 0:
                continue

            mae_true = float(np.mean(np.abs(preds[mask] - trues[mask])))
            obs_col = next((c for c in df_observ_summary.columns if col_detail in c and "(MAE)" in c), None)
            if obs_col:
                mae_obs = float(df_observ_summary[obs_col].iloc[0])
                mae_data.append({"Algorithm": algo_name, "True_MAE": mae_true, "Observed_MAE": mae_obs})

        df_mae_plot = pd.DataFrame(mae_data)
        return df_merged, df_mae_plot, COL_MAP

    except Exception as e:
        print(f"[WARN] Failed to load exact/observed files. Using dummy data. Reason: {e}")
        n = 400
        df_merged = pd.DataFrame({
            "Label": np.random.rand(n),
            "Label_observ": np.random.rand(n),
            "TCR-GIN": np.random.rand(n),
            "TCR-GIN (Std)": 0.01 * np.ones(n),
        })
        df_mae_plot = pd.DataFrame({
            "Algorithm": ["TCR-GIN", "DC"],
            "True_MAE": [0.01, 0.2],
            "Observed_MAE": [0.02, 0.25],
        })
        return df_merged, df_mae_plot, {}


# =============================================================================
# Section 4. Plotting Functions
# =============================================================================

def plot_panels_a_to_e(fig: plt.Figure, gs_row1, gs_row2, style_map: Dict[str, Dict[str, Any]]) -> None:
    dataset_names = ["Small", "Medium", "Large", "Huge", "REDDIT"]
    titles = ["a", "b", "c", "d", "e"]

    for i in range(5):
        ax = fig.add_subplot(gs_row1[i]) if i < 3 else fig.add_subplot(gs_row2[i - 3])
        ds_name = dataset_names[i]

        pareto_points = []

        for algo in ALGO_ORDER:
            if algo not in df_mae.columns:
                continue

            x = df_time.loc[ds_name, algo]
            y = df_mae.loc[ds_name, algo]
            if pd.isna(x) or pd.isna(y):
                continue

            style = style_map[algo]
            pareto_points.append((float(x), float(y)))

            if algo == "TCR-GIN":
                y_err = data_std.get(ds_name, 0.0)
                ax.errorbar(
                    x, y, yerr=y_err, fmt=style["marker"], color=style["color"],
                    ecolor="black", elinewidth=0.8, capsize=2, capthick=0.8,
                    markersize=10, zorder=200, markeredgecolor="black", markeredgewidth=0.5
                )
            else:
                ms = 5 if style["marker"] == "s" else 6
                alpha_val = 0.85 if algo in KEEP_ALGOS else 0.5
                z_ord = 100 if algo in KEEP_ALGOS else 10
                ax.scatter(
                    x, y, color=style["color"], marker=style["marker"], s=ms ** 2 * 1.5,
                    zorder=z_ord, alpha=alpha_val, edgecolors="white", linewidth=0.3
                )

        # Pareto front
        pareto_points.sort(key=lambda p: p[0])
        pareto_front = []
        min_mae = float("inf")
        for t, m in pareto_points:
            if m < min_mae:
                pareto_front.append((t, m))
                min_mae = m

        if pareto_front:
            pt_time, pt_mae = zip(*pareto_front)
            ax.plot(pt_time, pt_mae, color="gray", linestyle="--", linewidth=1.2, zorder=1)

        # REDDIT panel: generalization marker
        if i == 4:
            gen_x = df_time.loc["REDDIT", "TCR-GIN"]
            ax.plot(
                gen_x, 0.012, marker="*", color="white", markeredgecolor="#FF0000",
                markersize=9, markeredgewidth=1.2, zorder=210, linestyle="None",
                label="Generalization"
            )
            leg_gen = ax.legend(loc="upper right", frameon=True, framealpha=0.7, edgecolor="lightgray", fontsize=10)
            leg_gen.get_frame().set_linewidth(0.4)
            ax.add_artist(leg_gen)

        ax.set_xscale("log")
        ax.set_title(titles[i], loc="left", pad=8, fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE", fontsize=10)
        ax.set_xlabel("Time(s)", fontsize=10)
        ax.set_ylim(-0.01, 0.21)


def plot_panel_f(ax_f: plt.Axes, df_merged: pd.DataFrame, col_map: Dict[str, str]) -> None:
    ax_f.set_title("f", loc="left", pad=8, fontsize=10, fontweight="bold")

    x_idx = np.arange(len(df_merged))
    y_true = df_merged["Label"]
    y_obs = df_merged["Label_observ"]
    y_tcr = df_merged["TCR-GIN"]
    y_tcr_std = df_merged["TCR-GIN (Std)"].fillna(0.0)

    plotted_base = False
    for col_detail in col_map.keys():
        if col_detail in df_merged.columns:
            lbl = "Baselines" if not plotted_base else None
            ax_f.plot(x_idx, df_merged[col_detail], color="gray", alpha=0.3, lw=0.4, label=lbl)
            plotted_base = True

    ax_f.plot(x_idx, y_true, color="black", lw=0.8, label="True Labels", zorder=100)
    ax_f.plot(x_idx, y_obs, color="#1f77b4", ls="--", lw=0.8, label="Observed Labels", zorder=101)
    ax_f.plot(x_idx, y_tcr, color="#FF0000", ls="-", lw=0.8, alpha=0.8, label="TCR-GIN", zorder=102)
    ax_f.fill_between(x_idx, y_tcr - y_tcr_std, y_tcr + y_tcr_std, color="#FF0000", alpha=0.2, zorder=101, linewidth=0)

    ax_f.set_xlabel("Test Sample ID", fontsize=10)
    ax_f.set_ylabel("CD Value", fontsize=10)
    ax_f.set_xlim(0, 400)
    ax_f.set_ylim(0, 1.0)

    leg_f = ax_f.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.7, edgecolor="lightgray", borderpad=0.3)
    leg_f.get_frame().set_linewidth(0.4)

    for line in leg_f.get_lines():
        if line.get_label() == "Baselines":
            line.set_alpha(0.9)
            line.set_linewidth(1.5)
            line.set_color("#666666")

    # Inset
    axins = ax_f.inset_axes([0.55, 0.08, 0.41, 0.28])
    n_points = len(df_merged)
    start_zoom, end_zoom = int(n_points * 0.7), int(n_points * 0.8)
    if end_zoom - start_zoom < 5:
        start_zoom, end_zoom = 0, max(5, n_points)

    for col_detail in col_map.keys():
        if col_detail in df_merged.columns:
            axins.plot(x_idx, df_merged[col_detail], color="gray", alpha=0.3, lw=0.4)
    axins.plot(x_idx, y_true, color="black", lw=0.8)
    axins.plot(x_idx, y_obs, color="#1f77b4", ls="--", lw=0.8)
    axins.plot(x_idx, y_tcr, color="#FF0000", ls="-", lw=0.8)
    axins.fill_between(x_idx, y_tcr - y_tcr_std, y_tcr + y_tcr_std, color="#FF0000", alpha=0.2, linewidth=0)

    axins.set_xlim(start_zoom, end_zoom)
    y_zoom_true = y_true.iloc[start_zoom:end_zoom]
    y_zoom_tcr = y_tcr.iloc[start_zoom:end_zoom]
    y_min_zoom = pd.concat([y_zoom_true, y_zoom_tcr]).min()
    y_max_zoom = pd.concat([y_zoom_true, y_zoom_tcr]).max()
    margin = (y_max_zoom - y_min_zoom) * 0.2 if y_max_zoom > y_min_zoom else 0.02
    axins.set_ylim(y_min_zoom - margin, y_max_zoom + margin)
    axins.tick_params(axis="both", which="major", labelsize=9)

    bbox_roi = TransformedBbox(
        Bbox.from_bounds(
            start_zoom,
            y_min_zoom - margin,
            end_zoom - start_zoom,
            (y_max_zoom + margin) - (y_min_zoom - margin),
        ),
        ax_f.transData
    )
    ax_f.add_patch(BboxPatch(bbox_roi, fill=False, ec="0.5", lw=0.5, zorder=200))
    ax_f.add_patch(BboxConnector(axins.bbox, bbox_roi, loc1=2, loc2=3, fc="none", ec="0.5", lw=0.5, zorder=200))
    ax_f.add_patch(BboxConnector(axins.bbox, bbox_roi, loc1=1, loc2=4, fc="none", ec="0.5", lw=0.5, zorder=200))


def plot_panel_g(ax_g: plt.Axes, df_mae_plot: pd.DataFrame, style_map: Dict[str, Dict[str, Any]]) -> None:
    ax_g.set_title("g", loc="left", pad=8, fontsize=10, fontweight="bold")

    raw_max = max(df_mae_plot["True_MAE"].max(), df_mae_plot["Observed_MAE"].max())
    max_val = np.ceil(raw_max / 0.05) * 0.05
    ax_g.plot([0, max_val], [0, max_val], color="black", ls="--", lw=1.0)

    for _, row in df_mae_plot.iterrows():
        algo = row["Algorithm"]
        style = style_map.get(algo, {"color": "#B0B0B0", "marker": "o"})
        if algo == "TCR-GIN":
            ax_g.scatter(
                row["True_MAE"], row["Observed_MAE"], color=style["color"], marker="*",
                s=80, zorder=200, edgecolors="black", linewidth=0.5
            )
        else:
            ms = 5 if style["marker"] == "s" else 6
            z_ord = 100 if algo in KEEP_ALGOS else 10
            alpha_val = 0.85 if algo in KEEP_ALGOS else 0.5
            ax_g.scatter(
                row["True_MAE"], row["Observed_MAE"], color=style["color"], marker=style["marker"],
                alpha=alpha_val, s=ms ** 2 * 1.2, edgecolors="white", linewidth=0.3, zorder=z_ord
            )

    ax_g.set_xlabel("Pred. vs true label", fontsize=10)
    ax_g.set_ylabel("Pred. vs observed label", fontsize=10)
    ax_g.set_xlim(0, 0.1)
    ax_g.set_ylim(0, 0.1)
    ax_g.set_aspect("equal")

    locator = MultipleLocator(0.05)
    ax_g.xaxis.set_major_locator(locator)
    ax_g.yaxis.set_major_locator(locator)


def plot_legend(ax_leg: plt.Axes, style_map: Dict[str, Dict[str, Any]]) -> None:
    ax_leg.axis("off")

    handles_all = []
    labels_all = []

    for algo in KEEP_ALGOS:
        s = style_map[algo]
        edge_c = "black" if algo == "TCR-GIN" else s["color"]
        m_size = 9 if algo == "TCR-GIN" else 6
        m_ew = 0.5 if algo == "TCR-GIN" else 0
        h = Line2D(
            [0], [0], color="w", marker=s["marker"], markerfacecolor=s["color"],
            markeredgecolor=edge_c, markersize=m_size, markeredgewidth=m_ew
        )
        handles_all.append(h)
        labels_all.append(algo)

    h_others = Line2D([0], [0], color="w", marker="o", markerfacecolor="#B0B0B0",
                      markeredgecolor="#B0B0B0", markersize=6)
    handles_all.append(h_others)
    labels_all.append("Others")

    ax_leg.legend(
        handles_all, labels_all,
        loc="center left", ncol=1, frameon=False,
        fontsize=10, labelspacing=1.3, handletextpad=0.5
    )


# =============================================================================
# Section 5. Main Figure Assembly / Save
# =============================================================================

def create_figure(df_merged: pd.DataFrame, df_mae_plot: pd.DataFrame, col_map: Dict[str, str]) -> plt.Figure:
    style_map = build_style_map()

    fig = plt.figure(figsize=(8.27, 10.5))
    outer_gs = gridspec.GridSpec(3, 1, hspace=0.25, left=0.08, right=0.98, bottom=0.06, top=0.96)

    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.3)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.3)
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2], wspace=0.3, width_ratios=[2.4, 2.8, 1.0])

    # a-e
    plot_panels_a_to_e(fig, gs_row1, gs_row2, style_map)

    # legend (moved to row 2, column 3)
    ax_leg = fig.add_subplot(gs_row2[2])
    plot_legend(ax_leg, style_map)

    # f
    ax_f = fig.add_subplot(gs_row3[0])
    plot_panel_f(ax_f, df_merged, col_map)

    # g
    ax_g = fig.add_subplot(gs_row3[1])
    plot_panel_g(ax_g, df_mae_plot, style_map)

    return fig


def save_figure(fig: plt.Figure, output_dir: Path, output_name: str, formats: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_name

    valid_formats = []
    for fmt in formats:
        f = fmt.lower().lstrip(".")
        if f in {"pdf", "svg", "png"}:
            valid_formats.append(f)

    if not valid_formats:
        valid_formats = ["pdf"]

    for fmt in valid_formats:
        save_path = base.with_suffix(f".{fmt}")
        kwargs = {"transparent": True, "bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 600
        fig.savefig(save_path, format=fmt, **kwargs)
        print(f"[OK] Saved: {save_path}")


def main() -> None:
    args = parse_args()

    exact_results_dir = resolve_exact_results_dir(args.exact_results_dir)
    output_dir = resolve_output_dir(args.output_dir)

    df_merged, df_mae_plot, col_map = load_exact_observed_data(exact_results_dir)

    fig = create_figure(df_merged, df_mae_plot, col_map)
    save_figure(fig, output_dir, args.output_name, args.formats)
    plt.close(fig)

    print(f"[DONE] Figure export completed in: {output_dir}")


if __name__ == "__main__":
    main()
