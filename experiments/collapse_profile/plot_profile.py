#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization script for collapse-profile results (Nature sub-journal style).

Reads:
    1. per_graph_predictions.csv          — from test_profile.py
    2. Baseline xlsx files                — from ERGM/baseline/{dataset}/results_final-{tau}/

Generates (in BOTH SVG and PDF):
    a–e) 5 networks selected by MAE percentiles (10,30,50,70,90)
    f)   (blank panel, reserved)
         — all 6 panels in one 2×3 combined figure

SVG → insert into Word | PDF → open in Adobe Illustrator

Usage:
    python plot_profile.py --config configs/plot_reddit_ergm.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for p in [str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

import yaml

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[!] pandas not installed; some features will fail. pip install pandas openpyxl")

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────

NPG_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
    "#7E6148", "#B09C85",
]

BASELINE_COLORS = [
    "#4DBBD5", "#00A087", "#3C5488", "#8491B4",
    "#91D1C2", "#7E6148", "#B09C85", "#F39B7F",
    "#1565C0", "#2E7D32", "#6A1B9A", "#00ACC1",
    "#558B2F", "#AD1457", "#FF8F00", "#283593",
    "#00838F", "#4E342E", "#546E7A", "#D84315",
    "#1B5E20", "#4527A0", "#BF360C", "#006064",
    "#FF6F00", "#880E4F", "#33691E", "#0D47A1",
    "#4A148C", "#827717",
]

BASELINE_LINE_STYLES      = ["-", "--", "-.", ":"]
BASELINE_COLORS_PER_STYLE = 7

TRUE_COLOR      = "#000000"
PRED_COLOR      = "#E64B35"
PRED_BAND_ALPHA = 0.18

# ──────────────────────────────────────────────────────────────
# Column / display-name map
# ──────────────────────────────────────────────────────────────

COL_MAP: Dict[str, str] = {
    "CollectiveInfluenceL1":             "CI \u2113-1",
    "CollectiveInfluenceL2":             "CI \u2113-2",
    "CollectiveInfluenceL3":             "CI \u2113-3",
    "CoreGDM":                           "CoreGDM",
    "CoreHD":                            "CoreHD",
    "Domirank":                          "DomiRank",
    "EGND":                              "EGND",
    "EI_s1":                             "EI_s1",
    "EI_s2":                             "EI_s2",
    "FINDER_CN":                         "FINDER",
    "GDM":                               "GDM",
    "GDMR":                              "GDMR",
    "GND":                               "GND",
    "GNDR":                              "GNDR",
    "MS":                                "MS",
    "MSR":                               "MSR",
    "betweenness_centrality_F":          "BCR",
    "betweenness_centrality_dynamic":    "BCR",
    "betweenness_centrality_T":          "BC",
    "degree_F":                          "DCR",
    "degree_centrality_dynamic":         "DCR",
    "degree_T":                          "DC",
    "degree_centrality":                 "DC",
    "network_entanglement_large":        "NEL",
    "network_entanglement_large_reinsertion":  "NELR",
    "network_entanglement_mid":          "NEM",
    "network_entanglement_mid_reinsertion":    "NEMR",
    "network_entanglement_small":        "NES",
    "network_entanglement_small_reinsertion":  "NESR",
    "vertex_entanglement":               "VE",
    "vertex_entanglement_reinsertion":   "VER",
}

REV_COL_MAP: Dict[str, str] = {}
for _k, _v in COL_MAP.items():
    REV_COL_MAP[_v] = _k
    REV_COL_MAP[_v.replace("\u2113", "l").replace(" ", "_")] = _k
REV_COL_MAP.update({
    "CI_l1":  "CollectiveInfluenceL1",
    "CI_l2":  "CollectiveInfluenceL2",
    "CI_l3":  "CollectiveInfluenceL3",
    "CI_l-1": "CollectiveInfluenceL1",
    "CI_l-2": "CollectiveInfluenceL2",
    "CI_l-3": "CollectiveInfluenceL3",
})

XLABEL_TAU    = "Collapse target \u03C4"
YLABEL_DIST   = "Collapse distance"


# ──────────────────────────────────────────────────────────────
# Nature sub-journal style
# ──────────────────────────────────────────────────────────────

def set_nature_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size":          7,
        "axes.titlesize":     7,
        "axes.labelsize":     7,
        "xtick.labelsize":    6,
        "ytick.labelsize":    6,
        "legend.fontsize":    5.5,
        "mathtext.fontset":   "custom",
        "mathtext.rm":        "Arial",
        "mathtext.it":        "Arial:italic",
        "mathtext.bf":        "Arial:bold",
        "mathtext.sf":        "Arial",
        "mathtext.default":   "it",
        "axes.linewidth":     0.5,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.spines.bottom": True,
        "axes.spines.left":   True,
        "axes.grid":          False,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.minor.size":   1.5,
        "ytick.minor.size":   1.5,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.pad":    3,
        "ytick.major.pad":    3,
        "xtick.top":          False,
        "ytick.right":        False,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#E0E0E0",
        "legend.borderpad":   0.3,
        "legend.handlelength":   1.5,
        "legend.handletextpad":  0.4,
        "legend.labelspacing":   0.3,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent":False,
        "svg.fonttype":       "none",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


# ──────────────────────────────────────────────────────────────
# Dual-format save
# ──────────────────────────────────────────────────────────────

def dual_save(fig: plt.Figure, save_path_no_ext: str) -> None:
    for fmt, tag in [("svg", "Word"), ("pdf", "AI")]:
        p = f"{save_path_no_ext}.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight")
        print(f"  [\u2713] {fmt.upper()} ({tag}):  {p}")


# ──────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────

def load_plot_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tau_cfg = cfg.pop("tau", None)
    if isinstance(tau_cfg, dict) and "values" in tau_cfg:
        cfg["tau_values"] = sorted(float(x) for x in tau_cfg["values"])
    return cfg


def resolve_plot_runs(cfg: Dict[str, Any]) -> List[int]:
    plot_runs = cfg.get("plot_runs", "all")
    num_runs  = int(cfg.get("num_runs", 5))
    if isinstance(plot_runs, str):
        if plot_runs.strip().lower() == "all":
            return list(range(1, num_runs + 1))
        return [int(x.strip()) for x in plot_runs.split(",") if x.strip()]
    elif isinstance(plot_runs, list):
        return [int(x) for x in plot_runs]
    return [int(plot_runs)]


def resolve_baseline_names(cfg: Dict[str, Any]) -> List[str]:
    raw = cfg.get("baseline_algorithms", [])
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return list(COL_MAP.keys())
    names: List[str] = []
    for item in raw:
        item = str(item).strip()
        if item in COL_MAP:
            names.append(item)
        elif item in REV_COL_MAP:
            names.append(REV_COL_MAP[item])
        else:
            print(f"[!] Unknown algorithm: '{item}'; skipping.")
    return names


# ──────────────────────────────────────────────────────────────
# CSV reading — predictions
# ──────────────────────────────────────────────────────────────

def read_per_graph_csv(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_predictions(
    csv_rows: List[Dict[str, str]],
    tau_values: List[float],
    run_ids: List[int],
) -> Dict[str, Dict]:
    tau_strs = [f"{t:.2f}" for t in tau_values]
    run_strs = {str(r) for r in run_ids}

    grouped: Dict[str, list] = defaultdict(list)
    for row in csv_rows:
        if row["run_id"] in run_strs:
            grouped[row["graph_id"]].append(row)

    result: Dict[str, Dict] = {}
    for gid, rows in grouped.items():
        true_profile = np.array([float(rows[0][f"true_{t}"]) for t in tau_strs])
        pred_per_run: Dict[int, np.ndarray] = {}
        for row in rows:
            pred_per_run[int(row["run_id"])] = np.array(
                [float(row[f"pred_{t}"]) for t in tau_strs]
            )
        preds_mat = np.stack(list(pred_per_run.values()), axis=0)
        result[gid] = {
            "true":         true_profile,
            "pred_per_run": pred_per_run,
            "pred_mean":    np.mean(preds_mat, axis=0),
            "pred_std":     np.std(preds_mat,  axis=0),
        }
    return result


# ──────────────────────────────────────────────────────────────
# MAE computation & percentile selection
# ──────────────────────────────────────────────────────────────

def compute_per_graph_mae(graph_data: Dict[str, Dict]) -> Dict[str, float]:
    return {
        gid: float(np.mean(np.abs(gd["true"] - gd["pred_mean"])))
        for gid, gd in graph_data.items()
    }


def select_by_mae_percentiles(
    mae_dict: Dict[str, float],
    percentiles: List[float] = (10, 30, 50, 70, 90),
    seed: int = 42,
) -> List[Tuple[str, float, float]]:
    rng  = random.Random(seed)
    gids = list(mae_dict.keys())
    maes = np.array([mae_dict[g] for g in gids])

    sorted_idx  = np.argsort(maes)
    sorted_gids = [gids[i] for i in sorted_idx]
    sorted_maes = maes[sorted_idx]
    n           = len(sorted_gids)

    thresholds = np.percentile(sorted_maes, percentiles)

    selected: List[Tuple[str, float, float]] = []
    used: set = set()

    for pct, thr in zip(percentiles, thresholds):
        dists  = np.abs(sorted_maes - thr)
        min_d  = min(dists[i] for i in range(n) if sorted_gids[i] not in used)
        candidates = [
            sorted_gids[i] for i in range(n)
            if sorted_gids[i] not in used
            and np.isclose(dists[i], min_d, atol=1e-12)
        ]
        chosen = rng.choice(candidates)
        used.add(chosen)
        actual_pct = float(np.sum(sorted_maes <= mae_dict[chosen]) / n * 100)
        selected.append((chosen, mae_dict[chosen], actual_pct))

    return selected


# ──────────────────────────────────────────────────────────────
# Baseline profile loading
# ──────────────────────────────────────────────────────────────

def _find_tau_folder(baseline_dir: str, tau: float) -> Optional[str]:
    for fmt in [f"{tau:g}", f"{tau:.1f}", f"{tau:.2f}", str(tau)]:
        p = os.path.join(baseline_dir, f"results_final-{fmt}")
        if os.path.isdir(p):
            return p
    return None


def _find_xlsx(tau_folder: str, alg_name: str) -> Optional[str]:
    for ext in ["xlsx", "csv"]:
        p = os.path.join(tau_folder, f"raw_data-{alg_name}.{ext}")
        if os.path.exists(p):
            return p
    return None


def _match_network_name(df_networks: List[str], graph_id: str) -> Optional[str]:
    if graph_id in df_networks:
        return graph_id
    for alt in [graph_id.replace("-", "_"), graph_id.replace("_", "-")]:
        if alt in df_networks:
            return alt
    if graph_id.startswith("net_"):
        s = graph_id[4:]
        for alt in [s, s.replace("-", "_")]:
            if alt in df_networks:
                return alt
    matches = [n for n in df_networks if graph_id in n or n in graph_id]
    return matches[0] if len(matches) == 1 else None


def load_baseline_profile(
    baseline_dir: str,
    graph_id: str,
    tau_values: List[float],
    alg_name: str,
) -> Optional[np.ndarray]:
    if not HAS_PANDAS:
        return None
    profile: List[float] = []
    for tau in tau_values:
        tau_folder = _find_tau_folder(baseline_dir, tau)
        if tau_folder is None:
            return None
        xlsx_path = _find_xlsx(tau_folder, alg_name)
        if xlsx_path is None:
            return None
        try:
            df = (pd.read_csv(xlsx_path) if xlsx_path.endswith(".csv")
                  else pd.read_excel(xlsx_path, engine="openpyxl"))
        except Exception as e:
            print(f"[!] Failed to read {xlsx_path}: {e}")
            return None
        if "network" not in df.columns or "critical_threshold" not in df.columns:
            return None
        matched = _match_network_name(df["network"].astype(str).tolist(), graph_id)
        if matched is None:
            return None
        row = df[df["network"].astype(str) == matched]
        if row.empty:
            return None
        profile.append(float(row.iloc[0]["critical_threshold"]))
    return np.array(profile)


def load_all_baselines(
    baseline_dir: str,
    graph_id: str,
    tau_values: List[float],
    algorithm_names: List[str],
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for alg in algorithm_names:
        p = load_baseline_profile(baseline_dir, graph_id, tau_values, alg)
        if p is not None:
            result[alg] = p
        else:
            print(f"    [!] Baseline '{alg}' not found for '{graph_id}'")
    return result


# ──────────────────────────────────────────────────────────────
# Axes helpers
# ──────────────────────────────────────────────────────────────

def style_axes_box(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#000000")
    ax.tick_params(axis="both", which="both", direction="in",
                   top=False, right=False, bottom=True, left=True,
                   width=0.5, length=3)
    ax.grid(False)


def style_axes_empty(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#000000")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ──────────────────────────────────────────────────────────────
# Shared y-limit (for profile panels only)
# ──────────────────────────────────────────────────────────────

def _compute_shared_ylim(
    graph_items: List[Tuple[str, Dict]],
    all_baselines: List[Dict[str, np.ndarray]],
    margin_frac: float = 0.06,
) -> Tuple[float, float]:
    all_y: List[float] = []
    for i, (_, gd) in enumerate(graph_items):
        all_y.extend(gd["true"].tolist())
        all_y.extend((gd["pred_mean"] + gd["pred_std"]).tolist())
        all_y.extend((gd["pred_mean"] - gd["pred_std"]).tolist())
        if i < len(all_baselines):
            for prof in all_baselines[i].values():
                all_y.extend(prof.tolist())
    if not all_y:
        return (0.0, 1.0)
    ymin, ymax = min(all_y), max(all_y)
    span = ymax - ymin if ymax > ymin else 1e-6
    return (ymin - margin_frac * span, ymax + margin_frac * span)


# ──────────────────────────────────────────────────────────────
# Single profile panel
# ──────────────────────────────────────────────────────────────

def _draw_one_panel(
    ax: plt.Axes,
    graph_id: str,
    tau_values: List[float],
    true_profile: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    baselines: Dict[str, np.ndarray],
    panel_label: str,
    subtitle: Optional[str] = None,
    shared_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    # Observed label
    ax.plot(tau_values, true_profile,
            color=TRUE_COLOR, linewidth=1.0, marker="o", markersize=2.5,
            markeredgewidth=0, markerfacecolor=TRUE_COLOR,
            label="Observed label", zorder=20)

    # TCR-GIN
    ax.plot(tau_values, pred_mean,
            color=PRED_COLOR, linewidth=1.0, marker="s", markersize=2,
            markeredgewidth=0, markerfacecolor=PRED_COLOR,
            label="TCR-GIN", zorder=15)
    ax.fill_between(tau_values, pred_mean - pred_std, pred_mean + pred_std,
                    color=PRED_COLOR, alpha=PRED_BAND_ALPHA, linewidth=0, zorder=10)

    # Baselines
    n_c = len(BASELINE_COLORS)
    n_s = len(BASELINE_LINE_STYLES)
    for idx, (alg_full, profile) in enumerate(baselines.items()):
        short = COL_MAP.get(alg_full, alg_full)
        color = BASELINE_COLORS[idx % n_c]
        ls    = BASELINE_LINE_STYLES[(idx // BASELINE_COLORS_PER_STYLE) % n_s]
        lw    = 0.7 if idx < BASELINE_COLORS_PER_STYLE else 0.6
        ax.plot(tau_values, profile,
                color=color, linewidth=lw, linestyle=ls,
                alpha=0.85, label=short, zorder=5)

    ax.set_xlabel(XLABEL_TAU, fontsize=6)
    ax.set_ylabel(YLABEL_DIST, fontsize=6)
    ax.set_xlim(min(tau_values) - 0.02, max(tau_values) + 0.02)
    if shared_ylim is not None:
        ax.set_ylim(shared_ylim)
    style_axes_box(ax)

    title_text = f"{graph_id}\n{subtitle}" if subtitle else graph_id
    ax.text(0.5, 0.93, title_text, transform=ax.transAxes,
            fontsize=6, fontweight="bold", ha="center", va="top",
            linespacing=1.3, zorder=30)
    ax.text(-0.01, 1.01, panel_label, transform=ax.transAxes,
            fontsize=7, fontweight="bold", ha="right", va="bottom", zorder=30)


# ──────────────────────────────────────────────────────────────
# ★ Combined figure: 5 profile panels + 1 blank panel
# ──────────────────────────────────────────────────────────────

def plot_selected_profiles(
    graph_items:      List[Tuple[str, Dict]],
    all_baselines:    List[Dict[str, np.ndarray]],
    tau_values:       List[float],
    save_path_no_ext: str,
    subtitles:        Optional[List[Optional[str]]] = None,
) -> None:
    """
    2-col × 3-row layout:
      panels a–e  → profile line charts (up to 5; extras left as empty frames)
      panel  f    → blank (reserved)
    Shared legend (Observed label / TCR-GIN / baselines) placed above the figure.
    """
    nrows, ncols = 3, 2
    total_panels = nrows * ncols      # 6
    n_nets       = min(len(graph_items), total_panels - 1)   # max 5

    shared_ylim = _compute_shared_ylim(
        graph_items[:n_nets], all_baselines[:n_nets]
    )
    print(f"    Shared y-axis: [{shared_ylim[0]:.4f}, {shared_ylim[1]:.4f}]")

    fig = plt.figure(figsize=(7.2, 6.8))
    gs  = GridSpec(nrows, ncols, figure=fig,
                   left=0.08, right=0.97,
                   bottom=0.06, top=0.88,
                   wspace=0.32, hspace=0.42)

    panel_labels = [chr(ord("a") + i) for i in range(total_panels)]
    axes: List[plt.Axes] = []

    for i in range(total_panels):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        if i < n_nets:
            # ── Profile panel ──
            gid, gd = graph_items[i]
            sub = subtitles[i] if subtitles and i < len(subtitles) else None
            _draw_one_panel(
                ax=ax,
                graph_id=gid,
                tau_values=tau_values,
                true_profile=gd["true"],
                pred_mean=gd["pred_mean"],
                pred_std=gd["pred_std"],
                baselines=all_baselines[i],
                panel_label=panel_labels[i],
                subtitle=sub,
                shared_ylim=shared_ylim,
            )
        else:
            # ── Blank panel (including panel f) ──
            style_axes_empty(ax)
            ax.set_xlim(min(tau_values) - 0.02, max(tau_values) + 0.02)
            ax.set_ylim(shared_ylim)
            ax.text(-0.01, 1.01, panel_labels[i], transform=ax.transAxes,
                    fontsize=7, fontweight="bold", ha="right", va="bottom")

    # ── Shared legend (from profile panels only) ──
    profile_ax = axes[0] if n_nets > 0 else None
    if profile_ax is not None:
        handles, labels = profile_ax.get_legend_handles_labels()
        seen: set = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                uniq_h.append(h)
                uniq_l.append(l)

        ncol_leg = max(1, (len(uniq_l) + 1) // 2)
        leg = fig.legend(
            uniq_h, uniq_l,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.99),
            ncol=ncol_leg,
            fontsize=5.5,
            handlelength=1.5, handletextpad=0.4,
            columnspacing=1.0, labelspacing=0.3,
            borderpad=0.4, frameon=True, framealpha=0.9,
            edgecolor="#E8E8E8", fancybox=False,
        )
        leg.get_frame().set_linewidth(0.3)

    dual_save(fig, save_path_no_ext)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot collapse-profile results (Nature sub-journal style)"
    )
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--plot_runs",   type=str, default=None,
                        help="Override: 'all' or '1,3,5'")
    parser.add_argument("--baselines",   type=str, default=None,
                        help="Override: comma-separated full or short names")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--percentiles", type=str, default="10,30,50,70,90",
                        help="Comma-separated MAE percentiles (default: 10,30,50,70,90)")
    cli = parser.parse_args()

    cfg = load_plot_config(cli.config)
    if cli.plot_runs is not None:
        cfg["plot_runs"] = cli.plot_runs
    if cli.baselines is not None:
        cfg["baseline_algorithms"] = [x.strip() for x in cli.baselines.split(",")]

    set_nature_style()

    dataset_name   = cfg["dataset_name"]
    result_dir     = cfg.get("result_dir", "experiments/collapse_profile/results")
    baseline_dir   = cfg["baseline_dir"]
    tau_values     = cfg["tau_values"]
    run_ids        = resolve_plot_runs(cfg)
    baseline_names = resolve_baseline_names(cfg)

    plot_dpi = int(cfg.get("plot_dpi", 300))
    plt.rcParams["savefig.dpi"] = plot_dpi
    plt.rcParams["figure.dpi"]  = plot_dpi

    target_pcts = [float(x.strip()) for x in cli.percentiles.split(",")]
    metrics_dir = os.path.join(result_dir, dataset_name, "metrics")

    print(f"Dataset:      {dataset_name}")
    print(f"Runs:         {run_ids}")
    print(f"Baselines:    {[COL_MAP.get(n, n) for n in baseline_names]}")
    print(f"Percentiles:  {target_pcts}")
    print(f"Metrics dir:  {metrics_dir}")
    print(f"Output:       SVG (Word) + PDF (AI)")
    print()

    # ── Read prediction CSV ──
    csv_path = os.path.join(metrics_dir, "per_graph_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"[\u2717] CSV not found: {csv_path}")
        print("    Run test_profile.py first.")
        return

    plot_dir = os.path.join(result_dir, dataset_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    csv_rows   = read_per_graph_csv(csv_path)
    graph_data = extract_predictions(csv_rows, tau_values, run_ids)

    if not graph_data:
        print(f"[\u2717] No data for runs {run_ids} in CSV.")
        return

    n_graphs = len(graph_data)
    print(f"Total graphs: {n_graphs} | \u03C4 points: {len(tau_values)}")
    print()

    # ── Compute MAE ──
    mae_dict = compute_per_graph_mae(graph_data)
    print("\u2500\u2500 Per-network MAE (sorted ascending) \u2500\u2500")
    for rank, gid in enumerate(sorted(mae_dict, key=mae_dict.get), 1):
        print(f"  #{rank:>3d}  {gid:>35s}  MAE = {mae_dict[gid]:.6f}")
    print()

    # ══════════════════════════════════════════════════════════
    # Combined figure — 5 profile panels (a–e) + blank panel (f)
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Selected profiles (a–e), panel f blank")
    print("=" * 60)

    selected = select_by_mae_percentiles(mae_dict, target_pcts, seed=cli.seed)

    print()
    print(f"  {'Target %':>9s}  {'Actual %':>9s}  {'MAE':>10s}  Network")
    print(f"  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*35}")
    for (gid, mae_val, actual_pct), tgt in zip(selected, target_pcts):
        print(f"  {tgt:>7.0f}th  {actual_pct:>7.1f}th  {mae_val:>10.6f}  {gid}")
    print()

    print("  Loading baselines for selected networks ...")
    sel_items:     List[Tuple[str, Dict]]      = []
    sel_baselines: List[Dict[str, np.ndarray]] = []
    for gid, _, _ in selected:
        print(f"    {gid}")
        sel_items.append((gid, graph_data[gid]))
        sel_baselines.append(
            load_all_baselines(baseline_dir, gid, tau_values, baseline_names)
        )
    print()

    sel_subtitles = [
        f"MAE pctl \u2248 {int(tgt)}th  (MAE = {mae_val:.4f})"
        for (_, mae_val, _), tgt in zip(selected, target_pcts)
    ]

    plot_selected_profiles(
        graph_items      = sel_items,
        all_baselines    = sel_baselines,
        tau_values       = tau_values,
        save_path_no_ext = os.path.join(plot_dir, "profiles_selected_by_mae"),
        subtitles        = sel_subtitles,
    )

    # ── Save selection record ──
    sel_csv = os.path.join(plot_dir, "selected_networks_by_mae.csv")
    with open(sel_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_percentile", "actual_percentile", "graph_id", "mae", "seed"])
        for (gid, mae_val, actual_pct), tgt in zip(selected, target_pcts):
            w.writerow([tgt, f"{actual_pct:.1f}", gid, f"{mae_val:.6f}", cli.seed])
    print(f"  [\u2713] Selection record: {sel_csv}")

    print()
    print(f"All plots saved to: {plot_dir}")
    print()
    print("Usage guide:")
    print("  \u2022 Word:  insert *.svg (font sizes preserved at 100%)")
    print("  \u2022 AI:    open  *.pdf (TrueType embedded, no glyph issues)")
    print("Done.")


if __name__ == "__main__":
    main()
