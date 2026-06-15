#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/trajectory_analysis/test_properties.py

Trajectory-property evaluation pipeline for TCR-GIN.

This script evaluates how well TCR-GIN preserves desirable trajectory-level
properties during graph dismantling, and compares its behavior with a set of
traditional baseline algorithms. It is designed for three closely related use
cases:
1) standard property evaluation,
2) ablation studies,
3) sensitivity studies.

Core objectives
---------------
The script focuses on trajectory analysis rather than only final prediction
accuracy. For each dismantling sequence, it analyzes whether the predicted
collapse-distance trajectory is:

1. Monotonic
   The predicted normalized collapse distance should not increase as attack
   steps progress, up to a configurable tolerance.

2. Smooth
   The predicted trajectory should not drop too sharply relative to the
   theoretical minimum change induced by node removals.

3. Additive / Consistent
   For disconnected remnants, the prediction made on the whole remnant graph
   should be consistent with the weighted aggregation of predictions made on
   its connected components.

In addition to evaluating TCR-GIN, the script also computes the same
trajectory-level statistics for traditional dismantling baselines, allowing
direct comparison under a unified metric definition.

What this script does
---------------------
Given one configuration file, the script will:

1. Resolve dataset paths and output paths.
2. Discover the appropriate TCR-GIN checkpoints from the configured model suite.
3. Support both:
   - single_param mode:
     evaluate a fixed model suite on one dataset;
   - multi_param mode:
     evaluate multiple experiment folders for ablation/sensitivity analysis.
4. For each run:
   - load remnant graphs,
   - choose the proper model according to graph size,
   - predict collapse distance for each remnant graph,
   - optionally apply component-aware aggregation for disconnected graphs.
5. Compute trajectory metrics for TCR-GIN:
   - monotonicity frequency/intensity,
   - smoothness frequency/intensity,
   - monotonicity accuracy MAE,
   - additivity consistency MAE.
6. Compute the same trajectory metrics for baseline algorithms from the
   dismantling result spreadsheets.
7. Aggregate results across runs and save a summary CSV.
8. Optionally generate publication-style figures.

Prediction modes
----------------
The script supports two prediction strategies for TCR-GIN:

1. holistic
   Treat each remnant graph as a whole and predict directly.

2. component_aware
   If a remnant graph is disconnected, split it into connected components,
   predict each component separately, and compute a node-count-weighted
   aggregate prediction.

If component-aware mode is requested but networkx is unavailable, the script
automatically falls back to holistic mode.

Supported figure outputs
------------------------
When plotting is enabled, the script can generate four types of figures:

1. Monochrome monotonicity panels
   Multi-panel black-and-white overview of trajectory comparisons between
   TCR-GIN and baselines, with inset zoom regions.

2. Colored monotonicity panels
   Focused color version for selected attack strategies, intended for compact
   publication-style presentation.

3. Large-scale monotonicity panels
   Monotonicity plots optimized for large-scale networks, using a symlog y-axis
   and inset zoom regions.

4. Additivity consistency scatter
   Scatter plot comparing whole-remnant predictions and aggregated
   component-level predictions.

The residual-style monotonicity figure is intentionally not included in this
version of the script.

Typical inputs
--------------
This script expects the dismantling dataset directory to already contain:
- remnant graph files,
- corresponding graph labels,
- optional baseline result spreadsheets.

It also expects the configuration file to specify:
- dismantling_data_dir,
- output_dir,
- dataset_type,
- prediction_strategy,
- monotonicity tolerance,
- test_type,
- model_suite,
- base_model_params,
- plotting options.

Typical outputs
---------------
The script writes:
1. A summary CSV containing:
   - dataset metadata,
   - algorithm name,
   - prediction strategy,
   - model-scale information,
   - model hyperparameters (for ablation/sensitivity),
   - aggregated trajectory metrics.
2. Optional figure files in PDF/SVG/PNG format under the configured output
   directory.

Usage
-----
Standard evaluation:
python experiments/trajectory_analysis/test_properties.py \
    --config experiments/trajectory_analysis/configs/test_properties_base_multisource-BA100.yaml

Ablation study:
python experiments/trajectory_analysis/test_properties.py \
    --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-BA100.yaml

Sensitivity study:
python experiments/trajectory_analysis/test_properties.py \
    --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-BA100.yaml

"""


from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from glob import glob
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# =============================================================================
# Section 0. Project Setup and Optional Dependencies
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from torch_geometric.data import Batch
    from torch_geometric.utils import to_networkx
    from data_loader import load_single_graph
    from model.tcr_gin import TCR_GIN
except ImportError as e:
    print(
        f"Failed to import project modules: {e}\n"
        "Please make sure the script is located under "
        "'TCR-GIN/experiments/trajectory_analysis/'."
    )
    sys.exit(1)

PLOTTING_ENABLED = False
NETWORKX_ENABLED = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: matplotlib is not available. Plotting is disabled.")

try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    print("Warning: networkx is not available. 'component_aware' will fall back to 'holistic'.")

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =============================================================================
# Section 1. Algorithm Naming and Styles
# =============================================================================

# Network name mapping: remnant directory name -> CSV network name
NETWORK_NAME_MAPPING = {
    "transport": "london_transport_multiplex_aggr",
    "power": "power-eris1176",
    "route": "route-views",
}


def _convert_seq_id_for_csv(seq_id: str) -> str:
    """Convert short seq_id (from remnant files) to full CSV network name format."""
    base_network = seq_id.split("-")[0]
    if base_network in NETWORK_NAME_MAPPING:
        csv_network_name = NETWORK_NAME_MAPPING[base_network]
        strategy_part = seq_id.split("-", 1)[1] if "-" in seq_id else ""
        return f"{csv_network_name}-{strategy_part}" if strategy_part else csv_network_name
    return seq_id

FILENAME_TO_SHORT_NAME = {
    "CollectiveInfluenceL1": r"CI $\ell$-1",
    "CollectiveInfluenceL2": r"CI $\ell$-2",
    "CollectiveInfluenceL3": r"CI $\ell$-3",
    "GDM": "GDM",
    "GDMR": "GDMR",
    "CoreGDM": "CoreGDM",
    "CoreHD": "CoreHD",
    "EGND": "EGND",
    "EI_s1": r"EI ${\sigma _1}$",
    "EI_s2": r"EI ${\sigma _2}$",
    "GND": "GND",
    "GNDR": "GNDR",
    "MS": "MS",
    "MSR": "MSR",
    "network_entanglement_small": "NES",
    "network_entanglement_small_reinsertion": "NESR",
    "network_entanglement_mid": "NEM",
    "network_entanglement_mid_reinsertion": "NEMR",
    "network_entanglement_large": "NEL",
    "network_entanglement_large_reinsertion": "NELR",
    "vertex_entanglement": "VE",
    "vertex_entanglement_reinsertion": "VER",
    "degree_T": "DC",
    "degree_F": "DCR",
    "betweenness_centrality_T": "BC",
    "betweenness_centrality_F": "BCR",
    "eigenvector_centrality_T": "EC",
    "eigenvector_centrality_F": "ECR",
    "FINDER_CN": "FINDER",
    "Domirank": "DomiRank",
}

_ALGO_DISPLAY_NAME_NORMALIZATION = {
    "CI1": r"CI $\ell$-1",
    "CI2": r"CI $\ell$-2",
    "CI3": r"CI $\ell$-3",
    "EIs1": r"EI ${\sigma _1}$",
    "EIs2": r"EI ${\sigma _2}$",
    "EI_s1": r"EI ${\sigma _1}$",
    "EI_s2": r"EI ${\sigma _2}$",
    "EI_sigma1": r"EI ${\sigma _1}$",
    "EI_sigma2": r"EI ${\sigma _2}$",
    "CollectiveInfluenceL1": r"CI $\ell$-1",
    "CollectiveInfluenceL2": r"CI $\ell$-2",
    "CollectiveInfluenceL3": r"CI $\ell$-3",
}

ORDERED_ALGOS = [
    "DC", "BC", "R1", "DCR", "BCR", "R2",
    "DomiRank", "FINDER", "CoreHD", "GDM", "GDMR", "CoreGDM",
    "MS", "MSR", r"CI $\ell$-1", r"CI $\ell$-2", r"CI $\ell$-3", "EGND",
    r"EI ${\sigma _1}$", r"EI ${\sigma _2}$", "GND", "GNDR", "VE", "VER",
    "NEL", "NELR", "NEM", "NEMR", "NES", "NESR",
]
ALGO_ORDER_MAP = {name: i for i, name in enumerate(ORDERED_ALGOS)}
ALGO_ORDER_MAP["TCR-GIN"] = -1

FIXED_ZOOM_BASELINES = ["CoreHD", "GDMR", "GDM", "FINDER", "CoreGDM", "MSR"]

if PLOTTING_ENABLED:
    colors = plt.get_cmap("Paired")(np.linspace(0, 1, 12))
else:
    colors = [(0.5, 0.5, 0.5, 1.0)] * 12

ALGO_STYLES: Dict[str, Dict[str, Any]] = {
    "TCR-GIN": {"color": colors[5], "marker": "o", "linestyle": "-", "zorder": 20, "markersize": 1.5, "linewidth": 1.0},
    "BCR": {"color": colors[1], "marker": "s", "linestyle": "--", "markersize": 2.0, "fillstyle": "none", "linewidth": 0.6},
    "DCR": {"color": colors[3], "marker": "^", "linestyle": ":", "markersize": 2.0, "fillstyle": "none", "linewidth": 0.6},
    "CoreGDM": {"color": colors[7], "marker": "D", "linestyle": "-.", "markersize": 2.0, "fillstyle": "none", "linewidth": 0.6},
    "CoreHD": {"color": colors[9], "marker": "p", "linestyle": (0, (3, 5, 1, 5)), "markersize": 2.5, "fillstyle": "none", "linewidth": 0.6},
    r"CI $\ell$-1": {"color": colors[11], "marker": "+", "linestyle": "--", "markersize": 2.5, "linewidth": 0.6},
    "GDM": {"color": "saddlebrown", "marker": "x", "linestyle": ":", "markersize": 2.5, "linewidth": 0.6},
    "default": {"color": "grey", "marker": "None", "linestyle": "--", "linewidth": 0.5},
}

other_algos = [name for name in FILENAME_TO_SHORT_NAME.values() if name not in ALGO_STYLES]
fallback_color_cycle = cycle(plt.get_cmap("tab20")(np.linspace(0, 1, 20))) if PLOTTING_ENABLED else cycle([(0.5, 0.5, 0.5, 1.0)])
fallback_linestyle_cycle = cycle([
    "--", ":", "-.", (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (1, 1)),
    (0, (5, 1)), (0, (1, 5)), (0, (3, 1, 1, 1)), (0, (5, 10)),
    (0, (1, 10)), (0, (3, 3, 1, 3, 1, 3)), (5, (10, 3)), (0, (10, 5, 3, 5)),
    (0, (8, 2, 2, 2, 2, 2)),
])

for algo in sorted(other_algos):
    ALGO_STYLES[algo] = {
        "color": next(fallback_color_cycle),
        "marker": "None",
        "linestyle": next(fallback_linestyle_cycle),
        "linewidth": 0.5,
    }


def normalize_algo_display_name(raw_name: str) -> str:
    return _ALGO_DISPLAY_NAME_NORMALIZATION.get(raw_name, raw_name)


def get_algo_sort_key(algo_name: str) -> int:
    return ALGO_ORDER_MAP.get(algo_name, 999)


def get_seq_sort_key(seq_id: str) -> int:
    raw_suffix = seq_id.rsplit("-", 1)[-1]
    normalized = normalize_algo_display_name(raw_suffix)
    return ALGO_ORDER_MAP.get(normalized, ALGO_ORDER_MAP.get(raw_suffix, 999))


def get_subplot_label(i: int) -> str:
    return chr(ord("a") + i) if i < 26 else f"a{chr(ord('a') + i - 26)}"


# =============================================================================
# Section 2. Environment and Config
# =============================================================================

def setup_environment(config_path_str: str) -> Dict[str, Any]:
    """
    Parse CLI config, resolve dynamic paths, and normalize options.
    """
    config_path = Path(config_path_str).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dismantling_path"] = (config_dir / config["dismantling_data_dir"]).resolve()
    config["output_dir"] = (config_dir / config["output_dir"]).resolve()
    config["dataset_name"] = config["dismantling_path"].name
    config["summary_csv_path"] = config["output_dir"] / f"summary-{config['dataset_name']}-{config_path.stem}.csv"

    config["prediction_strategy"] = config.get("prediction_strategy", "holistic")
    if config["prediction_strategy"] == "component_aware" and not NETWORKX_ENABLED:
        print("Warning: 'component_aware' requested but networkx is unavailable. Falling back to 'holistic'.")
        config["prediction_strategy"] = "holistic"

    return config


def parse_tolerance(value: Any) -> float:
    """
    Parse tolerance as float or fraction string, e.g. '1/100'.
    """
    if isinstance(value, str) and "/" in value:
        try:
            num, den = value.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.01
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.01


# =============================================================================
# Section 3. Model and Baseline Data Loading
# =============================================================================

def _get_short_algo_name(filename_stem: str) -> str:
    sorted_keys = sorted(FILENAME_TO_SHORT_NAME.keys(), key=len, reverse=True)
    for long_name in sorted_keys:
        if filename_stem.endswith(long_name):
            return FILENAME_TO_SHORT_NAME[long_name]
    return filename_stem


def parse_params_from_folder_name(dir_name: str) -> Dict[str, str]:
    """
    Parse hyperparameters from experiment folder name.
    Example:
      exp_0_dropout_0.2_feature_dim_3 -> {"dropout": "0.2", "feature": ...}
    """
    params = {}
    param_str = re.sub(r"^exp_\d+_", "", dir_name)
    parts = param_str.split("_")

    i = 0
    while i < len(parts) - 1:
        params[parts[i]] = parts[i + 1]
        i += 2
    return params


def normalize_model_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config/folder-derived params to model constructor arguments.
    """
    key_map = {
        "modelactivationfn": "activation_fn",
        "modeljktype": "jk_type",
        "modelusevirtualnode": "use_virtual_node",
        "pissconsistencylambda": "consistency_lambda",
        "pisspissk": "piss_k",
        "modelfeaturedim": "feature_dim",
    }

    final_params: Dict[str, Any] = {}
    for key, value in raw_params.items():
        short_key = key_map.get(key.lower(), key)

        if isinstance(value, str):
            lv = value.lower()
            if lv == "true":
                converted = True
            elif lv == "false":
                converted = False
            else:
                try:
                    converted = float(value) if "." in value else int(value)
                except ValueError:
                    converted = value
        else:
            converted = value

        final_params[short_key] = converted

    if "feature_dim" in final_params and final_params["feature_dim"] is not None:
        final_params["input_dim"] = final_params["feature_dim"]

    return final_params


def load_model_suite(
    suite_config: List[Dict[str, Any]],
    base_model_params: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Load model groups according to node ranges.
    """
    suite = []

    for model_info in suite_config:
        node_range = model_info["node_range"]
        model_paths = sorted(glob(model_info["path"]))
        if not model_paths:
            continue

        models = []
        for model_path in model_paths:
            try:
                current_model_params = base_model_params.copy()
                if "params" in model_info:
                    current_model_params.update(model_info["params"])

                final_params = normalize_model_params(current_model_params)
                model_args = argparse.Namespace(**final_params)

                model = TCR_GIN(model_args).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")

        if models:
            suite.append({"node_range": node_range, "models": models})

    return suite


def get_model_for_graph(graph: Any, model_suite: List[Dict[str, Any]], run_idx: int) -> Optional[TCR_GIN]:
    """
    Select the most suitable model for a graph by node count.
    If a group has fewer runs than requested, reuse the first model in that group.
    """
    n_nodes = int(graph.num_nodes)

    for model_group in model_suite:
        min_nodes, max_nodes = model_group["node_range"]
        in_range = min_nodes <= n_nodes < max_nodes
        if max_nodes == -1:
            in_range = min_nodes <= n_nodes

        if in_range:
            models_list = model_group["models"]
            if not models_list:
                return None
            model_index = run_idx if run_idx < len(models_list) else 0
            return models_list[model_index]

    if model_suite:
        models_list = model_suite[-1]["models"]
        if models_list:
            model_index = run_idx if run_idx < len(models_list) else 0
            return models_list[model_index]

    return None


def load_and_prepare_traditional_data(
    dismantling_path: Path,
    for_plotting: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load and clean baseline Excel results.
    """
    purpose = "plotting" if for_plotting else "metric computation"
    print(f"Loading traditional baseline data for {purpose}...")

    dataset_name = dismantling_path.name
    results_path = dismantling_path / f"{dataset_name}-Remnants" / "results_final"
    data_cache: Dict[str, pd.DataFrame] = {}

    if not results_path.exists():
        print(f"Warning: baseline result directory not found: {results_path}")
        return data_cache

    loaded_algos = []
    for xlsx_file in results_path.glob("*.xlsx"):
        try:
            short_algo_name = _get_short_algo_name(xlsx_file.stem)
            df = pd.read_excel(xlsx_file)

            if "network" not in df.columns:
                continue

            for col in ["critical_threshold", "network_size"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["critical_threshold", "network_size"], inplace=True)

            df["seq_id"] = df["network"].str.rsplit("_", n=1).str[0]
            df["step"] = pd.to_numeric(df["network"].str.rsplit("_", n=1).str[1], errors="coerce")
            df.dropna(subset=["step"], inplace=True)
            df["step"] = df["step"].astype(int)
            df.drop_duplicates(subset=["seq_id", "step"], keep="first", inplace=True)

            data_cache[short_algo_name] = df
            loaded_algos.append(short_algo_name)
        except Exception as e:
            print(f"Failed to load {xlsx_file.name}: {e}")

    if loaded_algos:
        print(f"  Loaded {len(set(loaded_algos))} baselines: {', '.join(sorted(set(loaded_algos)))}")
    else:
        print("  No baseline data loaded.")

    return data_cache


def find_remnants_dir(dismantling_path: Path, dataset_name: str) -> Optional[Path]:
    """
    Locate the remnants directory.
    """
    candidates = [
        dismantling_path / f"{dataset_name}-Remnants" / f"{dataset_name}-Remnants",
        dismantling_path / f"{dataset_name}-Remnants",
    ]
    for p in candidates:
        if p.exists() and list(p.glob("*.npz")):
            return p
    return None


# =============================================================================
# Section 4. Core Prediction and Metric Analysis
# =============================================================================

def prepare_and_predict_all_graphs(
    model_suite: List[Dict[str, Any]],
    device: torch.device,
    run_idx: int,
    config: Dict[str, Any],
    model_args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    """
    Stage 1:
    Iterate over all remnant graphs and compute both holistic and aggregated
    predictions (if component-aware mode is enabled).
    """
    dataset_name = config["dataset_name"]
    remnants_path = config["dismantling_path"] / f"{dataset_name}-Remnants" / f"{dataset_name}-Remnants"

    prediction_strategy = config.get("prediction_strategy", "holistic")
    force_component_aware = prediction_strategy == "component_aware"

    remnant_files = list(remnants_path.glob("*_edges.npz"))
    if not remnant_files:
        print("Warning: no remnant graphs found.")
        return {}

    prediction_cache: Dict[str, Dict[str, Any]] = {}
    print(f"  Stage 1: prepare graphs and compute predictions (strategy: {prediction_strategy})...")

    with torch.no_grad():
        for remnant_path in tqdm(remnant_files, desc="    Predicting", leave=False):
            graph_id = remnant_path.name.replace("_edges.npz", "")
            graph_prefix = str(remnant_path).replace("_edges.npz", "")

            g_remnant = load_single_graph(graph_prefix, feature_dim=model_args.input_dim)
            if g_remnant is None or g_remnant.num_nodes == 0:
                continue

            pred_holistic = 0.0
            model_remnant = get_model_for_graph(g_remnant, model_suite, run_idx)
            if model_remnant:
                batch_remnant = Batch.from_data_list([g_remnant]).to(device)
                pred_holistic = float(torch.clamp(model_remnant(batch_remnant), 0.0, 1.0).item())

            pred_aggregated = pred_holistic
            if force_component_aware and NETWORKX_ENABLED and g_remnant.num_edges > 0:
                g_nx = to_networkx(g_remnant, to_undirected=True)
                components = list(nx.connected_components(g_nx))

                if len(components) > 1:
                    total_weighted_pred = 0.0
                    total_nodes = int(g_remnant.num_nodes)

                    for node_indices in components:
                        node_idx_tensor = torch.tensor(list(node_indices), dtype=torch.long)
                        sub_g = g_remnant.subgraph(node_idx_tensor)
                        if sub_g.num_nodes == 0:
                            continue

                        model_comp = get_model_for_graph(sub_g, model_suite, run_idx)
                        if not model_comp:
                            continue

                        batch_comp = Batch.from_data_list([sub_g]).to(device)
                        pred_comp = float(torch.clamp(model_comp(batch_comp), 0.0, 1.0).item())
                        total_weighted_pred += pred_comp * sub_g.num_nodes

                    pred_aggregated = total_weighted_pred / total_nodes if total_nodes > 0 else 0.0

            prediction_cache[graph_id] = {
                "graph": g_remnant,
                "pred_holistic": pred_holistic,
                "pred_aggregated": pred_aggregated,
                "y": g_remnant.y,
            }

    return prediction_cache


def _compute_sequence_metrics(
    steps: Iterable[int],
    values: Iterable[float],
    n_initial: int,
    fixed_tolerance: float,
) -> Dict[str, float]:
    """
    Compute sequence-level monotonicity and smoothness metrics.
    """
    steps_arr = np.asarray(list(steps), dtype=float)
    vals_arr = np.asarray(list(values), dtype=float)

    if len(steps_arr) < 2 or n_initial <= 0:
        return {}

    sorted_idx = np.argsort(steps_arr)
    steps_arr = steps_arr[sorted_idx]
    vals_arr = vals_arr[sorted_idx]

    unit_cost = 1.0 / n_initial
    val_diffs = np.diff(vals_arr)
    step_spans = np.diff(steps_arr)

    total_steps = len(val_diffs)
    epsilon = 1e-9

    # Monotonicity
    threshold_mono = fixed_tolerance
    mono_violation_mask = val_diffs > (threshold_mono + epsilon)
    m_count = np.sum(mono_violation_mask)
    m_freq = m_count / total_steps if total_steps > 0 else 0.0
    m_excess = np.maximum(0, val_diffs - threshold_mono)
    m_int = float(np.mean(m_excess)) if len(m_excess) else 0.0

    # Smoothness
    theoretical_drop = step_spans * unit_cost
    threshold_smooth_drop = theoretical_drop + fixed_tolerance
    smooth_violation_mask = val_diffs < -(threshold_smooth_drop + epsilon)
    s_count = np.sum(smooth_violation_mask)
    s_freq = s_count / total_steps if total_steps > 0 else 0.0
    s_excess = np.maximum(0, -val_diffs - threshold_smooth_drop)
    s_int = float(np.mean(s_excess)) if len(s_excess) else 0.0

    return {
        "M_freq": float(m_freq),
        "M_int": float(m_int),
        "S_freq": float(s_freq),
        "S_int": float(s_int),
    }


def test_monotonicity(
    prediction_cache: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]]]:
    """
    Stage 2:
    Evaluate monotonicity and smoothness from predicted trajectories.
    """
    print("  Stage 2: analyze monotonicity and smoothness (fixed tolerance mode)...")
    prediction_strategy = config.get("prediction_strategy", "holistic")
    fixed_tolerance = parse_tolerance(config.get("monotonicity_test", {}).get("tolerance", 0.0))
    print(f"    -> fixed tolerance = {fixed_tolerance}")

    if not prediction_cache:
        return {}, {}

    sequences: Dict[str, List[int]] = defaultdict(list)
    for graph_id in prediction_cache.keys():
        match = re.match(r"(.+)_(\d+)$", graph_id)
        if match:
            full_seq_id, step = match.groups()

            # Convert seq_id to match CSV network names
            # e.g., transport-degree → london_transport_multiplex_aggr-degree
            base_network = full_seq_id.split("-")[0]
            if base_network in NETWORK_NAME_MAPPING:
                csv_network_name = NETWORK_NAME_MAPPING[base_network]
                strategy_part = full_seq_id.split("-", 1)[1] if "-" in full_seq_id else ""
                seq_id = f"{csv_network_name}-{strategy_part}" if strategy_part else csv_network_name
            else:
                seq_id = full_seq_id

            sequences[seq_id].append(int(step))

    all_metrics: Dict[str, List[float]] = defaultdict(list)
    plot_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"steps": [], "d_norm": []})
    all_abs_errors: List[float] = []

    is_route_dataset = "route" in config["dataset_name"].lower()

    for seq_id in sorted(sequences.keys()):
        steps = sorted(sequences[seq_id])
        if not steps:
            continue

        if is_route_dataset or "route" in seq_id.lower():
            n_initial = 6474
        else:
            initial_graph_id = f"{seq_id}_0"
            n_initial = prediction_cache.get(initial_graph_id, {}).get("graph", None)
            n_initial = int(n_initial.num_nodes) if n_initial is not None else 0

        if n_initial == 0:
            continue

        seq_steps = []
        seq_vals = []

        for step in steps:
            graph_id = f"{seq_id}_{step}"
            if graph_id not in prediction_cache:
                continue

            data = prediction_cache[graph_id]
            pred_val = data["pred_aggregated"] if prediction_strategy == "component_aware" else data["pred_holistic"]

            if data["y"] is not None:
                all_abs_errors.append(abs(pred_val - float(data["y"].item())))

            d_norm = pred_val * (data["graph"].num_nodes / n_initial)
            seq_steps.append(step)
            seq_vals.append(float(d_norm))

        plot_data[seq_id]["steps"] = seq_steps
        plot_data[seq_id]["d_norm"] = seq_vals

        metrics = _compute_sequence_metrics(seq_steps, seq_vals, n_initial, fixed_tolerance)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    final_metrics: Dict[str, float] = {}
    for k, v_list in all_metrics.items():
        if v_list:
            final_metrics[f"{k}_mean"] = float(np.mean(v_list))
            final_metrics[f"{k}_std"] = float(np.std(v_list))
        else:
            final_metrics[f"{k}_mean"] = -1.0
            final_metrics[f"{k}_std"] = 0.0

    if all_abs_errors:
        final_metrics["monotonicity_accuracy_mae_mean"] = float(np.mean(all_abs_errors))
        final_metrics["monotonicity_accuracy_mae_std"] = float(np.std(all_abs_errors))
    else:
        final_metrics["monotonicity_accuracy_mae_mean"] = -1.0
        final_metrics["monotonicity_accuracy_mae_std"] = 0.0

    return final_metrics, dict(plot_data)


def test_additivity(
    prediction_cache: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Stage 3:
    Evaluate additivity/consistency between holistic and aggregated predictions.
    """
    print("  Stage 3: analyze additivity (consistency)...")

    default_metrics = {
        "additivity_consistency_mae_mean": -1.0,
        "additivity_consistency_mae_std": 0.0,
    }
    if not prediction_cache:
        return default_metrics, {}

    all_abs_errors: List[float] = []
    plot_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"remnant_norm": [], "components_norm": [], "metric_val": 0.0}
    )

    for graph_id, data in prediction_cache.items():
        pred_holistic = data["pred_holistic"]
        pred_aggregated = data["pred_aggregated"]

        error = abs(pred_holistic - pred_aggregated)
        all_abs_errors.append(float(error))

        if error > 1e-6:
            base_network_id = graph_id.split("-")[0]
            plot_data[base_network_id]["remnant_norm"].append(pred_holistic)
            plot_data[base_network_id]["components_norm"].append(pred_aggregated)
            plot_data["global"]["remnant_norm"].append(pred_holistic)
            plot_data["global"]["components_norm"].append(pred_aggregated)

    mean_val = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0
    std_val = float(np.std(all_abs_errors)) if all_abs_errors else 0.0

    if "global" in plot_data:
        plot_data["global"]["metric_val"] = mean_val

    for k in list(plot_data.keys()):
        if k == "global":
            continue
        rem = np.asarray(plot_data[k]["remnant_norm"])
        comp = np.asarray(plot_data[k]["components_norm"])
        if len(rem) > 0:
            plot_data[k]["metric_val"] = float(np.mean(np.abs(rem - comp)))
        else:
            plot_data[k]["metric_val"] = 0.0

    metrics = {
        "additivity_consistency_mae_mean": mean_val,
        "additivity_consistency_mae_std": std_val,
    }
    return metrics, dict(plot_data)


# =============================================================================
# Section 5. Plotting Helpers
# =============================================================================

def _draw_equidistant_diagonal_grid(ax, slope: float, num_lines: int = 6, **line_kwargs) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    c_vals = [y - slope * x for x, y in corners]
    c_min, c_max = min(c_vals), max(c_vals)
    for c in np.linspace(c_min, c_max, num=num_lines):
        ax.axline(xy1=(0, c), slope=slope, **line_kwargs)


def _get_initial_size_for_sequence(
    seq_id: str,
    traditional_data: Dict[str, pd.DataFrame],
    dataset_name: Optional[str] = None,
) -> Optional[int]:
    if dataset_name and "route" in dataset_name.lower():
        return 6474
    if "route" in seq_id.lower():
        return 6474

    seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
    for df in traditional_data.values():
        initial_rows = df[(df["seq_id"] == seq_id_for_csv) & (df["step"] == 0)]
        if not initial_rows.empty:
            return int(initial_rows["network_size"].iloc[0])

    return None


def _collect_all_sequences(
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
) -> List[str]:
    all_seq_ids = set(plot_data.keys())
    for algo_df in traditional_data.values():
        all_seq_ids.update(algo_df["seq_id"].unique())
    return sorted(list(all_seq_ids), key=get_seq_sort_key)


def _build_monochrome_baseline_styles(traditional_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    gray_cycle = cycle(["#666666", "#888888", "#aaaaaa", "#cccccc"])
    styles = {}
    for name in traditional_data.keys():
        styles[name] = {
            **ALGO_STYLES.get(name, ALGO_STYLES["default"]),
            "color": next(gray_cycle),
            "zorder": 10,
        }
    return styles


def _build_colored_baseline_styles(traditional_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {**ALGO_STYLES.get(name, ALGO_STYLES["default"]), "zorder": 10}
        for name in traditional_data.keys()
    }


def _find_zoom_region(
    seq_id: str,
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    dataset_name: Optional[str] = None,
    num_neighbors: int = 4,
) -> Tuple[Optional[Tuple[int, int]], int]:
    """
    Find a zoom window that best highlights monotonicity differences.
    """
    if seq_id not in plot_data or not plot_data[seq_id]["steps"]:
        return None, 1

    tcr_gin_df = pd.DataFrame(plot_data[seq_id]).set_index("steps")["d_norm"]
    if tcr_gin_df.empty:
        return None, 1

    steps = sorted(plot_data[seq_id]["steps"])
    step_interval = max(1, int(np.median(np.diff(steps)))) if len(steps) > 1 else 1

    n_initial = _get_initial_size_for_sequence(seq_id, traditional_data, dataset_name)
    if not n_initial:
        return None, step_interval

    theoretical_slope_abs = 1.0 / n_initial
    search_start_idx = int(len(steps) * 0.1)
    search_end_idx = int(len(steps) * 0.9)

    best_strict = (None, -float("inf"))
    best_relaxed = (None, -float("inf"))

    for i in range(search_start_idx, search_end_idx):
        start_idx = max(0, i - num_neighbors)
        end_idx = min(len(steps) - 1, i + num_neighbors)
        win_steps = steps[start_idx:end_idx + 1]
        win_vals = tcr_gin_df.loc[win_steps]

        if len(win_vals) < 2:
            continue

        diffs = win_vals.diff().dropna()
        if (diffs > 1e-9).any():
            continue

        current_score = 0.0
        count_valid_baselines = 0

        seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
        for algo in FIXED_ZOOM_BASELINES:
            if algo not in traditional_data:
                continue

            df = traditional_data[algo]
            seq_df = df[df["seq_id"] == seq_id_for_csv].set_index("step")
            common_idx = seq_df.index.intersection(win_steps)
            if len(common_idx) < 2:
                continue

            base_vals = seq_df.loc[common_idx]
            base_d_norm = base_vals["critical_threshold"] * base_vals["network_size"] / n_initial
            gap = float((base_d_norm - tcr_gin_df.loc[common_idx]).mean())
            base_diffs = base_d_norm.diff().dropna()
            volatility = float(base_diffs[base_diffs > 0].sum())
            current_score += gap * 1.0 + volatility * 10.0
            count_valid_baselines += 1

        if count_valid_baselines == 0:
            continue

        total_drop = win_vals.iloc[0] - win_vals.iloc[-1]
        total_span = win_steps[-1] - win_steps[0]
        avg_slope_abs = total_drop / total_span if total_span > 0 else 0
        region = (win_steps[0], win_steps[-1])

        if avg_slope_abs < theoretical_slope_abs:
            if current_score > best_strict[1]:
                best_strict = (region, current_score)
        else:
            if current_score > best_relaxed[1]:
                best_relaxed = (region, current_score)

    if best_strict[0] is not None:
        return best_strict[0], step_interval
    if best_relaxed[0] is not None:
        return best_relaxed[0], step_interval
    return None, step_interval


def _collect_zoom_metadata(
    seq_chunk: List[str],
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    dataset_name: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    zoom_meta_data: Dict[str, Dict[str, Any]] = {}
    global_max_y_span = 0.05

    for seq_id in seq_chunk:
        zoom_region, _ = _find_zoom_region(seq_id, plot_data, traditional_data, dataset_name=dataset_name, num_neighbors=4)
        if not zoom_region:
            continue

        x_start, x_end = zoom_region
        local_y_min, local_y_max = 1.0, 0.0
        has_data = False

        if seq_id in plot_data:
            ti_df = pd.DataFrame(plot_data[seq_id])
            z_ti = ti_df[(ti_df["steps"] >= x_start) & (ti_df["steps"] <= x_end)]
            if not z_ti.empty:
                local_y_min = min(local_y_min, float(z_ti["d_norm"].min()))
                local_y_max = max(local_y_max, float(z_ti["d_norm"].max()))
                has_data = True

        n_initial = _get_initial_size_for_sequence(seq_id, traditional_data, dataset_name)
        if n_initial:
            seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
            for algo_name in FIXED_ZOOM_BASELINES:
                if algo_name not in traditional_data:
                    continue
                df = traditional_data[algo_name]
                seq_df = df[df["seq_id"] == seq_id_for_csv].sort_values("step")
                z_b = seq_df[(seq_df["step"] >= x_start) & (seq_df["step"] <= x_end)]
                if not z_b.empty:
                    d_norm = z_b["critical_threshold"] * z_b["network_size"] / n_initial
                    local_y_min = min(local_y_min, float(d_norm.min()))
                    local_y_max = max(local_y_max, float(d_norm.max()))
                    has_data = True

        if has_data:
            global_max_y_span = max(global_max_y_span, local_y_max - local_y_min)
            zoom_meta_data[seq_id] = {
                "region": zoom_region,
                "y_center": (local_y_min + local_y_max) / 2.0,
                "n_initial": n_initial,
            }

    return zoom_meta_data, global_max_y_span * 1.2


def _plot_monotonicity_inset(
    ax,
    seq_id: str,
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    baseline_styles: Dict[str, Dict[str, Any]],
    zoom_meta: Dict[str, Any],
    global_y_span: float,
    tcr_gin_style: Dict[str, Any],
    show_color: bool,
) -> None:
    x_start, x_end = zoom_meta["region"]
    y_center = zoom_meta["y_center"]
    n_init = zoom_meta["n_initial"]

    ax_inset = ax.inset_axes([0.45, 0.45, 0.52, 0.52], zorder=100)
    ax_inset._is_inset = True
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    ax_inset.patch.set_zorder(-1)

    if seq_id in plot_data:
        inset_style = tcr_gin_style.copy()
        inset_style["linewidth"] = max(1.0, tcr_gin_style.get("linewidth", 1.0))
        ax_inset.plot(plot_data[seq_id]["steps"], plot_data[seq_id]["d_norm"], **inset_style)

    seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
    for algo_name in FIXED_ZOOM_BASELINES:
        if algo_name not in traditional_data:
            continue
        df = traditional_data[algo_name]
        seq_df = df[df["seq_id"] == seq_id_for_csv].sort_values("step")
        z_b = seq_df[(seq_df["step"] >= x_start) & (seq_df["step"] <= x_end)]
        if not z_b.empty:
            d_norm = z_b["critical_threshold"] * z_b["network_size"] / n_init
            base_style = baseline_styles.get(algo_name, ALGO_STYLES["default"]).copy()
            base_style["linewidth"] = 0.6 if not show_color else 0.8
            ax_inset.plot(z_b["step"], d_norm, **base_style)

    target_bottom = y_center - global_y_span / 2
    target_top = y_center + global_y_span / 2
    if target_bottom < 0:
        offset = -target_bottom
        target_bottom += offset
        target_top += offset
    elif target_top > 1.05:
        offset = target_top - 1.05
        target_bottom -= offset
        target_top -= offset

    ax_inset.set_xlim(x_start, x_end)
    ax_inset.set_ylim(max(0, target_bottom), target_top)

    if n_init:
        _draw_equidistant_diagonal_grid(
            ax=ax_inset,
            slope=-1.0 / n_init,
            num_lines=6,
            color="gainsboro",
            linewidth=0.5,
            zorder=0,
            alpha=0.8,
        )

    ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ax_inset.tick_params(axis="both", which="major", labelsize=5, length=2)
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.5)

    ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=0.5)


# =============================================================================
# Section 6. Plotting Functions
# =============================================================================

def plot_additivity_consistency_scatter(
    plot_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    dataset_type: str,
    dataset_name: str,
) -> None:
    """
    Scatter plot for additivity consistency:
      x = aggregated prediction
      y = holistic prediction

    Output:
      save PDF / SVG / PNG with the same stem as output_path
    """
    if not PLOTTING_ENABLED or not plot_data:
        return

    groups = ["global"] if "global" in plot_data and len(plot_data["global"]["remnant_norm"]) > 0 else [
        k for k in sorted(plot_data.keys()) if k != "global" and len(plot_data[k].get("remnant_norm", [])) > 0
    ]
    if not groups:
        print("  Additivity plot: no valid points available.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_output_path = output_path.with_suffix("")

    print(f"Generating additivity plot -> {base_output_path}.[pdf/svg/png]")

    num_plots = len(groups)
    if num_plots == 1:
        nrows, ncols = 1, 1
        fig_size = (5.0, 5.0)
    else:
        ncols = 3
        nrows = math.ceil(num_plots / ncols)
        fig_size = (10, 3.8 * nrows)

    with plt.rc_context({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, group_name in enumerate(groups):
            ax = axes[i]
            data = plot_data[group_name]

            x_vals = np.asarray(data["components_norm"])
            y_vals = np.asarray(data["remnant_norm"])
            mae_val = float(data.get("metric_val", 0.0))

            ax.plot(
                [0, 1.2], [0, 1.2],
                color="black", linestyle="--", linewidth=1.5,
                zorder=1, label="Perfect Additivity"
            )
            ax.scatter(
                x_vals, y_vals,
                alpha=0.5, s=20,
                facecolors="#ff9999", edgecolors="none",
                zorder=2, label=r"Network $G_t$"
            )

            if len(x_vals) > 0 and len(y_vals) > 0:
                all_vals = np.concatenate([x_vals, y_vals])
                v_min, v_max = all_vals.min(), all_vals.max()
                span = max(v_max - v_min, 0.1)
                pad = span * 0.10
                lim_min = max(0, v_min - pad)
                lim_max = v_max + pad
                if lim_max > 1.0 and v_max <= 1.0:
                    lim_max = 1.02
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.text(
                0.05, 0.95, f"MAE = {mae_val:.4f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.0, alpha=0.9),
            )
            ax.tick_params(axis="both", which="major", labelsize=9)

        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)

        fig.supxlabel(r"$\hat{D}^{\mathrm{agg}}(G_t)$", fontsize=14, y=0.10)
        fig.supylabel(r"$\hat{D}(G_t)$", fontsize=14, x=0.08)

        if num_plots == 1:
            legend = axes[0].legend(loc="lower right", frameon=True, fontsize=10, framealpha=0.9)
            legend.get_frame().set_edgecolor("black")
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            legend = fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=2, frameon=True, fontsize=10
            )
            legend.get_frame().set_edgecolor("black")

        plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.96])

        fig.savefig(base_output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(base_output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(base_output_path.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight")

        plt.close(fig)


def plot_monotonicity_monochrome_panels(
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_type: str,
    dataset_name: str,
) -> None:
    """
    Monotonicity overview figure in grayscale with inset zoom windows.
    Formerly: visualize_monotonicity_scheme1_enhanced
    """
    if not PLOTTING_ENABLED or not traditional_data:
        return

    AX_WIDTH = 2.1
    AX_HEIGHT = 1.7
    GAP_WIDTH = 0.3
    GAP_HEIGHT = 0.4
    MARGIN_LEFT = 0.6
    MARGIN_RIGHT = 0.1
    MARGIN_BOTTOM = 0.5
    MARGIN_TOP_BASE = 0.25
    LEGEND_HEIGHT = 0.4
    TITLE_SPACE = 0.2
    LEGEND_GAP = 0.1

    sorted_sequences = _collect_all_sequences(plot_data, traditional_data)
    if not sorted_sequences:
        return

    base_network_id = sorted_sequences[0].rsplit("-", 1)[0]
    baseline_styles = _build_monochrome_baseline_styles(traditional_data)

    layouts = [(5, 3), (5, 3)]
    sequence_chunks = []
    start_idx = 0
    for rows, cols in layouts:
        count = rows * cols
        sequence_chunks.append(sorted_sequences[start_idx:start_idx + count])
        start_idx += count

    global_subplot_index = 0
    all_handles: Dict[str, Any] = {}

    for fig_idx, (layout, seq_chunk) in enumerate(zip(layouts, sequence_chunks)):
        if not seq_chunk:
            continue

        nrows, ncols = layout
        base_output_path = output_dir / f"monotonicity_monochrome_{base_network_id}_part{fig_idx + 1}"
        print(f"  Generating monochrome monotonicity plot (Part {fig_idx + 1}) -> {base_output_path}.[pdf/svg/png]")

        zoom_meta_data, global_max_y_span = _collect_zoom_metadata(seq_chunk, plot_data, traditional_data, dataset_name)

        content_width = ncols * AX_WIDTH + (ncols - 1) * GAP_WIDTH
        content_height = nrows * AX_HEIGHT + (nrows - 1) * GAP_HEIGHT
        current_top_margin_inches = TITLE_SPACE + LEGEND_GAP + LEGEND_HEIGHT + 0.1 if fig_idx == 0 else MARGIN_TOP_BASE

        fig_width = MARGIN_LEFT + content_width + MARGIN_RIGHT
        fig_height = MARGIN_BOTTOM + content_height + current_top_margin_inches

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        plot_area_top = 1.0 - (current_top_margin_inches / fig_height)
        plot_area_left = MARGIN_LEFT / fig_width
        plot_area_right = 1.0 - (MARGIN_RIGHT / fig_width)
        plot_area_center_x = (plot_area_left + plot_area_right) / 2.0

        plt.subplots_adjust(
            left=plot_area_left,
            right=plot_area_right,
            bottom=MARGIN_BOTTOM / fig_height,
            top=plot_area_top,
            wspace=GAP_WIDTH / AX_WIDTH,
            hspace=GAP_HEIGHT / AX_HEIGHT,
        )

        for i, seq_id in enumerate(seq_chunk):
            ax = axes[i]
            ax.set_facecolor("none")
            ax.patch.set_alpha(0.0)

            raw_strategy = seq_id.rsplit("-", 1)[-1]
            display_strategy = normalize_algo_display_name(raw_strategy)
            full_title = f"({get_subplot_label(global_subplot_index)}) {display_strategy} attack"
            ax.text(0.0, 1.05, full_title, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
            ax.set_title("")
            global_subplot_index += 1

            seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
            local_max_x = 0
            if seq_id in plot_data and plot_data[seq_id]["steps"]:
                local_max_x = max(local_max_x, max(plot_data[seq_id]["steps"]))
            for df in traditional_data.values():
                seq_df = df[df["seq_id"] == seq_id_for_csv]
                if not seq_df.empty:
                    local_max_x = max(local_max_x, int(seq_df["step"].max()))

            tcr_gin_style = {**ALGO_STYLES.get("TCR-GIN", {}), "linewidth": 1.0, "zorder": 20, "label": "TCR-GIN"}
            if seq_id in plot_data and plot_data[seq_id]["steps"]:
                h, = ax.plot(plot_data[seq_id]["steps"], plot_data[seq_id]["d_norm"], **tcr_gin_style)
                if "TCR-GIN" not in all_handles:
                    all_handles["TCR-GIN"] = h

            n_initial = _get_initial_size_for_sequence(seq_id, traditional_data, dataset_name)
            if not n_initial:
                continue

            for algo_name, df in traditional_data.items():
                seq_df = df[df["seq_id"] == seq_id_for_csv].sort_values("step")
                if not seq_df.empty:
                    d_norm = seq_df["critical_threshold"] * seq_df["network_size"] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES["default"])
                    display_name = normalize_algo_display_name(algo_name)
                    h, = ax.plot(seq_df["step"], d_norm, label=display_name, **{**style, "linewidth": 0.6})
                    if display_name not in all_handles:
                        all_handles[display_name] = h

            if seq_id in zoom_meta_data:
                _plot_monotonicity_inset(
                    ax=ax,
                    seq_id=seq_id,
                    plot_data=plot_data,
                    traditional_data=traditional_data,
                    baseline_styles=baseline_styles,
                    zoom_meta=zoom_meta_data[seq_id],
                    global_y_span=global_max_y_span,
                    tcr_gin_style=tcr_gin_style,
                    show_color=False,
                )

            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, max(10, local_max_x * 1.05))
            ax.tick_params(axis="both", which="major", labelsize=6)

        for j in range(len(seq_chunk), len(axes)):
            axes[j].set_visible(False)

        label_y_pos = (MARGIN_BOTTOM * 0.4) / fig_height
        label_x_pos = (MARGIN_LEFT * 0.4) / fig_width
        fig.supxlabel("Attack step", fontsize=8, y=label_y_pos)
        fig.supylabel("Collapse distance", fontsize=8, x=label_x_pos)

        if fig_idx == 0 and all_handles:
            sorted_keys = [k for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            sorted_handles = [all_handles[k] for k in sorted_keys]
            bbox_y_pos = plot_area_top + ((TITLE_SPACE + LEGEND_GAP) / fig_height) - 0.01
            fig.legend(
                sorted_handles, sorted_keys,
                loc="lower center",
                bbox_to_anchor=(plot_area_center_x, bbox_y_pos),
                ncol=10,
                frameon=False,
                fontsize=6,
                handlelength=1.5,
                columnspacing=1.0,
                bbox_transform=fig.transFigure,
            )

        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        for _ax in fig.axes:
            if hasattr(_ax, "_is_inset") and _ax._is_inset:
                continue
            _ax.set_facecolor("none")
            _ax.patch.set_alpha(0.0)

        fig.savefig(base_output_path.with_suffix(".pdf"), format="pdf", transparent=False, facecolor="none", edgecolor="none")
        fig.savefig(base_output_path.with_suffix(".svg"), format="svg", transparent=False, facecolor="none", edgecolor="none")
        fig.savefig(base_output_path.with_suffix(".png"), format="png", dpi=600, transparent=False, facecolor="none", edgecolor="none")
        plt.close(fig)


def plot_monotonicity_large_scale_panels(
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_type: str,
    dataset_name: str,
) -> None:
    """
    Large-scale monotonicity view with symlog y-axis.
    Formerly: visualize_monotonicity_large_scale
    """
    if not PLOTTING_ENABLED or not traditional_data:
        return

    AX_WIDTH = 2.1
    AX_HEIGHT = 1.7
    GAP_WIDTH = 0.4
    GAP_HEIGHT = 0.4
    MARGIN_LEFT = 0.7
    MARGIN_RIGHT = 0.1
    MARGIN_BOTTOM = 0.5
    MARGIN_TOP_BASE = 0.25
    LEGEND_HEIGHT = 0.4
    TITLE_SPACE = 0.2
    LEGEND_GAP = 0.1

    sorted_sequences = _collect_all_sequences(plot_data, traditional_data)
    if not sorted_sequences:
        return

    base_network_id = sorted_sequences[0].rsplit("-", 1)[0]
    baseline_styles = _build_monochrome_baseline_styles(traditional_data)

    layouts = [(5, 3), (5, 3)]
    sequence_chunks = []
    start_idx = 0
    for rows, cols in layouts:
        count = rows * cols
        sequence_chunks.append(sorted_sequences[start_idx:start_idx + count])
        start_idx += count

    global_subplot_index = 0
    all_handles: Dict[str, Any] = {}

    for fig_idx, (layout, seq_chunk) in enumerate(zip(layouts, sequence_chunks)):
        if not seq_chunk:
            continue

        nrows, ncols = layout
        base_output_path = output_dir / f"monotonicity_large_scale_{base_network_id}_part{fig_idx + 1}"
        print(f"  Generating large-scale monotonicity plot (Part {fig_idx + 1}) -> {base_output_path}.[pdf/svg/png]")

        zoom_meta_data, global_inset_y_span = _collect_zoom_metadata(seq_chunk, plot_data, traditional_data, dataset_name)

        content_width = ncols * AX_WIDTH + (ncols - 1) * GAP_WIDTH
        content_height = nrows * AX_HEIGHT + (nrows - 1) * GAP_HEIGHT
        current_top_margin_inches = TITLE_SPACE + LEGEND_GAP + LEGEND_HEIGHT + 0.1 if fig_idx == 0 else MARGIN_TOP_BASE

        fig_width = MARGIN_LEFT + content_width + MARGIN_RIGHT
        fig_height = MARGIN_BOTTOM + content_height + current_top_margin_inches

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        plot_area_top = 1.0 - (current_top_margin_inches / fig_height)
        plot_area_left = MARGIN_LEFT / fig_width
        plot_area_right = 1.0 - (MARGIN_RIGHT / fig_width)
        plot_area_center_x = (plot_area_left + plot_area_right) / 2.0

        plt.subplots_adjust(
            left=plot_area_left,
            right=plot_area_right,
            bottom=MARGIN_BOTTOM / fig_height,
            top=plot_area_top,
            wspace=GAP_WIDTH / AX_WIDTH,
            hspace=GAP_HEIGHT / AX_HEIGHT,
        )

        for i, seq_id in enumerate(seq_chunk):
            ax = axes[i]

            raw_strategy = seq_id.rsplit("-", 1)[-1]
            display_strategy = normalize_algo_display_name(raw_strategy)
            full_title = f"({get_subplot_label(global_subplot_index)}) {display_strategy} attack"
            ax.text(0.0, 1.05, full_title, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
            ax.set_title("")
            global_subplot_index += 1

            n_initial = _get_initial_size_for_sequence(seq_id, traditional_data, dataset_name)
            if not n_initial:
                continue

            seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
            current_max_x = 0
            if seq_id in plot_data and plot_data[seq_id]["steps"]:
                current_max_x = max(current_max_x, max(plot_data[seq_id]["steps"]))
            for df in traditional_data.values():
                seq_df = df[df["seq_id"] == seq_id_for_csv]
                if not seq_df.empty:
                    current_max_x = max(current_max_x, int(seq_df["step"].max()))

            local_target_xlim_right = max(10, current_max_x * 1.02)

            tcr_gin_style = {**ALGO_STYLES.get("TCR-GIN", {}), "linewidth": 1.2, "zorder": 20, "label": "TCR-GIN"}
            if seq_id in plot_data and plot_data[seq_id]["steps"]:
                h, = ax.plot(plot_data[seq_id]["steps"], plot_data[seq_id]["d_norm"], **tcr_gin_style)
                if "TCR-GIN" not in all_handles:
                    all_handles["TCR-GIN"] = h

            for algo_name, df in traditional_data.items():
                seq_df = df[df["seq_id"] == seq_id_for_csv].sort_values("step")
                if not seq_df.empty:
                    d_norm = seq_df["critical_threshold"] * seq_df["network_size"] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES["default"])
                    display_name = normalize_algo_display_name(algo_name)
                    h, = ax.plot(seq_df["step"], d_norm, label=display_name, **{**style, "linewidth": 0.6})
                    if display_name not in all_handles:
                        all_handles[display_name] = h

            ax.set_ylim(0, 1.05)
            linthresh_val = 0.07
            ax.set_yscale("symlog", linthresh=linthresh_val, linscale=1.0)
            ax.set_yticks([0, 0.035, 0.07, 1.0])
            ax.yaxis.set_major_formatter(ScalarFormatter())

            try:
                trans = ax.get_yaxis_transform()
                inv_trans = ax.transAxes.inverted()
                y_disp = trans.transform([0, linthresh_val])[1]
                y_axes_coord = inv_trans.transform([0, y_disp])[1]
                if 0 < y_axes_coord < 1:
                    d = 0.015
                    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.6)
                    ax.plot((-d, +d), (y_axes_coord - d, y_axes_coord + d), **kwargs)
                    ax.plot((-d, +d), (y_axes_coord - d - 0.015, y_axes_coord + d - 0.015), **kwargs)
            except Exception:
                pass

            start_xlim = 0
            if seq_id in plot_data and plot_data[seq_id]["steps"]:
                local_min = min(plot_data[seq_id]["steps"])
                if local_min > (local_target_xlim_right * 0.02):
                    start_xlim = max(0, local_min - local_target_xlim_right * 0.02)
                    d = 0.015
                    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.6)
                    ax.plot((-d, +d), (-d, +d), **kwargs)
                    ax.plot((-d, +d), (-d - 0.015, +d - 0.015), **kwargs)

            ax.set_xlim(start_xlim, local_target_xlim_right)
            ax.grid(False)
            ax.tick_params(axis="both", which="major", labelsize=6)

            if seq_id in zoom_meta_data:
                _plot_monotonicity_inset(
                    ax=ax,
                    seq_id=seq_id,
                    plot_data=plot_data,
                    traditional_data=traditional_data,
                    baseline_styles=baseline_styles,
                    zoom_meta=zoom_meta_data[seq_id],
                    global_y_span=global_inset_y_span,
                    tcr_gin_style=tcr_gin_style,
                    show_color=False,
                )

        for j in range(len(seq_chunk), len(axes)):
            axes[j].set_visible(False)

        label_y_pos = (MARGIN_BOTTOM * 0.4) / fig_height
        label_x_pos = (MARGIN_LEFT * 0.15) / fig_width + 0.02
        fig.supxlabel("Attack step", fontsize=8, y=label_y_pos)
        fig.supylabel("Collapse distance", fontsize=8, x=label_x_pos)

        if fig_idx == 0 and all_handles:
            sorted_keys = [k for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            sorted_handles = [all_handles[k] for k in sorted_keys]
            bbox_y_pos = plot_area_top + ((TITLE_SPACE + LEGEND_GAP) / fig_height) - 0.01
            fig.legend(
                sorted_handles, sorted_keys,
                loc="lower center",
                bbox_to_anchor=(plot_area_center_x, bbox_y_pos),
                ncol=10,
                frameon=False,
                fontsize=6,
                handlelength=1.5,
                columnspacing=1.0,
                bbox_transform=fig.transFigure,
            )

        plt.savefig(base_output_path.with_suffix(".pdf"), format="pdf", transparent=True)
        plt.savefig(base_output_path.with_suffix(".svg"), format="svg", transparent=True)
        plt.savefig(base_output_path.with_suffix(".png"), format="png", dpi=600, transparent=True)
        plt.close(fig)


def plot_monotonicity_colored_panels(
    plot_data: Dict[str, Dict[str, Any]],
    traditional_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_type: str,
    dataset_name: str,
) -> None:
    """
    Color monotonicity figure focused on selected attack strategies.
    Formerly: visualize_monotonicity_scheme3_enhanced
    """
    if not PLOTTING_ENABLED or not traditional_data:
        return

    AX_WIDTH = 2.8
    AX_HEIGHT = 2.0
    GAP_WIDTH = 0.5
    GAP_HEIGHT = 0.5
    MARGIN_LEFT = 0.8
    MARGIN_RIGHT = 0.2
    MARGIN_BOTTOM = 0.7
    LEGEND_HEIGHT = 0.6
    TITLE_SPACE = 0.2
    LEGEND_GAP = 0.1

    FONT_SIZE_MAIN = 12
    FONT_SIZE_TICKS = 11

    all_seq_ids = set(plot_data.keys())
    for algo_df in traditional_data.values():
        all_seq_ids.update(algo_df["seq_id"].unique())
    if not all_seq_ids:
        return

    base_network_id = list(all_seq_ids)[0].rsplit("-", 1)[0]
    target_suffixes = ["DC", "DCR", "BC", "BCR", "R1", "R2"]

    selected_seqs = []
    for strategy in target_suffixes:
        match = [s for s in all_seq_ids if s.endswith(f"-{strategy}")]
        selected_seqs.append(match[0] if match else None)

    if not any(selected_seqs):
        return

    base_output_path = output_dir / f"monotonicity_colored_{base_network_id}"
    print(f"  Generating colored monotonicity plot -> {base_output_path}.[pdf/svg/png]")

    baseline_styles = _build_colored_baseline_styles(traditional_data)
    seq_candidates = [s for s in selected_seqs if s]
    zoom_meta_data, global_max_y_span = _collect_zoom_metadata(seq_candidates, plot_data, traditional_data, dataset_name)

    nrows, ncols = 3, 2
    content_width = ncols * AX_WIDTH + (ncols - 1) * GAP_WIDTH
    content_height = nrows * AX_HEIGHT + (nrows - 1) * GAP_HEIGHT
    current_top_margin_inches = TITLE_SPACE + LEGEND_GAP + LEGEND_HEIGHT + 0.1

    fig_width = MARGIN_LEFT + content_width + MARGIN_RIGHT
    fig_height = MARGIN_BOTTOM + content_height + current_top_margin_inches

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    plot_area_top = 1.0 - (current_top_margin_inches / fig_height)
    plot_area_left = MARGIN_LEFT / fig_width
    plot_area_right = 1.0 - (MARGIN_RIGHT / fig_width)
    plot_area_center_x = (plot_area_left + plot_area_right) / 2.0

    plt.subplots_adjust(
        left=plot_area_left,
        right=plot_area_right,
        bottom=MARGIN_BOTTOM / fig_height,
        top=plot_area_top,
        wspace=GAP_WIDTH / AX_WIDTH,
        hspace=GAP_HEIGHT / AX_HEIGHT,
    )

    all_handles: Dict[str, Any] = {}

    for i, seq_id in enumerate(selected_seqs):
        ax = axes[i]
        if not seq_id:
            ax.set_visible(False)
            continue

        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)

        title_label = get_subplot_label(i)
        ax.text(0.0, 1.05, title_label, transform=ax.transAxes, fontsize=FONT_SIZE_MAIN, fontweight="bold", va="bottom", ha="left")
        ax.set_title("")

        seq_id_for_csv = _convert_seq_id_for_csv(seq_id)
        local_max_x = 0
        if seq_id in plot_data and plot_data[seq_id]["steps"]:
            local_max_x = max(local_max_x, max(plot_data[seq_id]["steps"]))
        for df in traditional_data.values():
            seq_df = df[df["seq_id"] == seq_id_for_csv]
            if not seq_df.empty:
                local_max_x = max(local_max_x, int(seq_df["step"].max()))

        tcr_gin_style = {**ALGO_STYLES.get("TCR-GIN", {}), "linewidth": 1.5, "zorder": 20, "label": "TCR-GIN"}
        if seq_id in plot_data and plot_data[seq_id]["steps"]:
            h, = ax.plot(plot_data[seq_id]["steps"], plot_data[seq_id]["d_norm"], **tcr_gin_style)
            if "TCR-GIN" not in all_handles:
                all_handles["TCR-GIN"] = h

        n_initial = _get_initial_size_for_sequence(seq_id, traditional_data, dataset_name)
        if n_initial:
            for algo_name, df in traditional_data.items():
                seq_df = df[df["seq_id"] == seq_id_for_csv].sort_values("step")
                if not seq_df.empty:
                    d_norm = seq_df["critical_threshold"] * seq_df["network_size"] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES["default"])
                    display_name = normalize_algo_display_name(algo_name)
                    h, = ax.plot(seq_df["step"], d_norm, label=display_name, **{**style, "linewidth": 0.8})
                    if display_name not in all_handles:
                        all_handles[display_name] = h

        if seq_id in zoom_meta_data:
            _plot_monotonicity_inset(
                ax=ax,
                seq_id=seq_id,
                plot_data=plot_data,
                traditional_data=traditional_data,
                baseline_styles=baseline_styles,
                zoom_meta=zoom_meta_data[seq_id],
                global_y_span=global_max_y_span,
                tcr_gin_style=tcr_gin_style,
                show_color=True,
            )

        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, max(10, local_max_x * 1.05))
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICKS)

    label_y_pos = (MARGIN_BOTTOM * 0.4) / fig_height
    label_x_pos = (MARGIN_LEFT * 0.4) / fig_width
    fig.supxlabel("Attack step", fontsize=FONT_SIZE_MAIN, y=label_y_pos)
    fig.supylabel("Collapse distance", fontsize=FONT_SIZE_MAIN, x=label_x_pos)

    if all_handles:
        sorted_keys = [k for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
        sorted_handles = [all_handles[k] for k in sorted_keys]
        bbox_y_pos = plot_area_top + ((TITLE_SPACE + LEGEND_GAP) / fig_height) - 0.01
        fig.legend(
            sorted_handles, sorted_keys,
            loc="lower center",
            bbox_to_anchor=(plot_area_center_x, bbox_y_pos),
            ncol=6,
            frameon=False,
            fontsize=FONT_SIZE_MAIN,
            handlelength=2.0,
            columnspacing=1.2,
            bbox_transform=fig.transFigure,
        )

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    for _ax in fig.axes:
        if hasattr(_ax, "_is_inset") and _ax._is_inset:
            continue
        _ax.set_facecolor("none")
        _ax.patch.set_alpha(0.0)

    fig.savefig(base_output_path.with_suffix(".pdf"), format="pdf", transparent=False, facecolor="none", edgecolor="none")
    fig.savefig(base_output_path.with_suffix(".svg"), format="svg", transparent=False, facecolor="none", edgecolor="none")
    fig.savefig(base_output_path.with_suffix(".png"), format="png", dpi=600, transparent=False, facecolor="none", edgecolor="none")
    plt.close(fig)


# =============================================================================
# Section 7. Job Discovery and Benchmarking
# =============================================================================

def discover_model_jobs(
    config: Dict[str, Any],
    base_model_params: Dict[str, Any],
    device: torch.device,
    config_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Discover evaluation jobs from config.
    Supports:
      - single_param
      - multi_param (ablation / sensitivity)
    """
    test_type = config["test_type"]
    jobs = []

    if test_type == "single_param":
        print("Discovering model jobs: single_param mode")
        suite_to_load = []

        for model_info in config["model_suite"]:
            if "base_dir" not in model_info:
                continue

            search_path = (config_dir / model_info["base_dir"]).resolve()
            exp_dirs = list(search_path.glob("exp_*"))
            if not exp_dirs:
                continue

            model_path_pattern = str(exp_dirs[0] / "model_run_*.pt")
            suite_to_load.append({
                "node_range": model_info["node_range"],
                "path": model_path_pattern,
            })

        if suite_to_load:
            model_suite = load_model_suite(suite_to_load, base_model_params, device)
            if model_suite:
                num_runs = max((len(m["models"]) for m in model_suite), default=0)
                jobs.append({
                    "name": "TCR-GIN",
                    "params": {},
                    "suite": model_suite,
                    "num_runs": num_runs,
                    "do_plotting": config.get("plotting", {}).get("enabled", True),
                })

    elif test_type == "multi_param":
        print("Discovering model jobs: multi_param mode")

        exp_dirs_by_range: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
        for model_info in config["model_suite"]:
            if "base_dir" not in model_info:
                continue
            node_range_tuple = tuple(model_info["node_range"])
            search_path = (config_dir / model_info["base_dir"]).resolve()
            exp_dirs_by_range[node_range_tuple].extend(search_path.glob("exp_*"))

        param_groups: Dict[str, Dict[Tuple[int, int], List[Path]]] = defaultdict(lambda: defaultdict(list))
        for node_range, exp_dirs in exp_dirs_by_range.items():
            for exp_dir in exp_dirs:
                param_str = re.sub(r"^exp_\d+_", "", exp_dir.name)
                param_groups[param_str][node_range].append(exp_dir)

        for param_str, range_dirs in param_groups.items():
            job_params = parse_params_from_folder_name(f"exp_0_{param_str}")
            suite_to_load = []
            num_runs_per_range = []

            for model_info in config["model_suite"]:
                node_range = tuple(model_info["node_range"])
                if node_range not in range_dirs:
                    continue

                model_path_pattern = str(range_dirs[node_range][0] / "model_run_*.pt")
                model_paths = glob(model_path_pattern)
                if not model_paths:
                    continue

                num_runs_per_range.append(len(model_paths))
                suite_to_load.append({
                    "node_range": list(node_range),
                    "path": model_path_pattern,
                    "params": job_params,
                })

            if not suite_to_load:
                continue

            model_suite = load_model_suite(suite_to_load, base_model_params, device)
            if model_suite:
                num_runs = max(num_runs_per_range) if num_runs_per_range else 0
                if num_runs == 0:
                    continue

                jobs.append({
                    "name": "TCR-GIN",
                    "params": job_params,
                    "suite": model_suite,
                    "num_runs": num_runs,
                    "do_plotting": False,
                })

    return jobs


def calculate_benchmarks(dismantling_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute baseline metrics using the same monotonicity/smoothness definitions.
    """
    data_cache = load_and_prepare_traditional_data(dismantling_path, for_plotting=False)
    if not data_cache:
        return pd.DataFrame()

    dataset_name = config["dataset_name"]
    is_route_dataset = "route" in dataset_name.lower()
    fixed_tolerance = parse_tolerance(config.get("monotonicity_test", {}).get("tolerance", 0.0))
    remnants_dir = find_remnants_dir(dismantling_path, dataset_name)
    input_dim = int(config.get("base_model_params", {}).get("input_dim", 7))

    all_results = []

    for short_algo_name, df in tqdm(data_cache.items(), desc="  Processing baselines"):
        algo_metrics: Dict[str, List[float]] = defaultdict(list)
        all_mae_errors: List[float] = []

        for seq_id, group in df.groupby("seq_id"):
            group = group.sort_values("step").reset_index(drop=True)
            if group.empty:
                continue

            if is_route_dataset or "route" in str(seq_id).lower():
                n_initial = 6474
            else:
                n_initial_row = group[group["step"] == 0]
                n_initial = int(n_initial_row["network_size"].iloc[0]) if not n_initial_row.empty else 0

            if n_initial == 0:
                continue

            vals = group["critical_threshold"].values * group["network_size"].values / n_initial
            steps = group["step"].values
            seq_metrics = _compute_sequence_metrics(steps, vals, n_initial, fixed_tolerance)

            for k, v in seq_metrics.items():
                algo_metrics[k].append(v)

            if remnants_dir:
                for _, row in group.iterrows():
                    graph_name = row.get("network")
                    recorded_val = row.get("critical_threshold")
                    if not graph_name or pd.isna(recorded_val):
                        continue

                    file_path = remnants_dir / f"{graph_name}_edges.npz"
                    if not file_path.exists():
                        continue

                    try:
                        g = load_single_graph(str(file_path).replace("_edges.npz", ""), feature_dim=input_dim)
                        if g is not None and g.y is not None:
                            all_mae_errors.append(abs(float(recorded_val) - float(g.y.item())))
                    except Exception:
                        pass

        row = {"algorithm": short_algo_name}
        for k, v_list in algo_metrics.items():
            if v_list:
                row[f"{k}_mean"] = float(np.mean(v_list))
                row[f"{k}_std"] = float(np.std(v_list))
            else:
                row[f"{k}_mean"] = -1.0
                row[f"{k}_std"] = 0.0

        row["additivity_consistency_mae_mean"] = 0.0
        row["additivity_consistency_mae_std"] = 0.0

        if all_mae_errors:
            row["monotonicity_accuracy_mae_mean"] = float(np.mean(all_mae_errors))
            row["monotonicity_accuracy_mae_std"] = float(np.std(all_mae_errors))
        else:
            row["monotonicity_accuracy_mae_mean"] = -1.0
            row["monotonicity_accuracy_mae_std"] = 0.0

        all_results.append(row)

    return pd.DataFrame(all_results)


# =============================================================================
# Section 8. Evaluation Pipeline
# =============================================================================

def run_evaluation_job(
    job: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """
    Run one TCR-GIN evaluation job, aggregate metrics, and optionally generate plots.
    """
    job_params = config.get("base_model_params", {}).copy()
    job_params.update(normalize_model_params(job["params"]))
    if "feature_dim" in job_params and job_params["feature_dim"] is not None:
        job_params["input_dim"] = job_params["feature_dim"]

    model_args = argparse.Namespace(**job_params)

    all_run_results = []
    monotonicity_plot_data_final = None
    additivity_plot_data_final = None

    job_desc = job["name"]
    if job["params"]:
        job_desc += f" ({', '.join(f'{k}={v}' for k, v in job['params'].items())})"

    for run_idx in tqdm(range(job["num_runs"]), desc=f"  Running {job_desc}", leave=False):
        prediction_cache = prepare_and_predict_all_graphs(job["suite"], device, run_idx, config, model_args)
        mono_metrics, mono_plot = test_monotonicity(prediction_cache, config)
        addi_metrics, addi_plot = test_additivity(prediction_cache, config)

        run_results = {**mono_metrics, **addi_metrics}
        all_run_results.append(run_results)

        if run_idx == 0:
            monotonicity_plot_data_final = mono_plot
            additivity_plot_data_final = addi_plot

    if not all_run_results:
        return None

    results_df = pd.DataFrame(all_run_results)

    if job["num_runs"] > 1:
        mean_series = results_df.mean()
        std_series = results_df.std()
        agg_metrics = {}
        for col in results_df.columns:
            agg_metrics[col] = float(mean_series[col])
            agg_metrics[f"{col}_std"] = float(std_series[col])
    else:
        agg_metrics = results_df.iloc[0].to_dict()

    agg_metrics["algorithm"] = job["name"]
    agg_metrics.update(job["params"])

    if job["do_plotting"]:
        print("Generating figures...")
        trad_dfs_for_plot = load_and_prepare_traditional_data(config["dismantling_path"], for_plotting=True)

        mono_bw_dir = config["output_dir"] / "monotonicity_plots_monochrome"
        mono_bw_dir.mkdir(parents=True, exist_ok=True)
        plot_monotonicity_monochrome_panels(
            monotonicity_plot_data_final,
            trad_dfs_for_plot,
            mono_bw_dir,
            config["dataset_type"],
            config["dataset_name"],
        )

        config_filename = Path(config.get("__file_path__", "")).name
        target_configs = {
            "test_properties_base_multisource-power.yaml",
            "test_properties_base_multisource-transport.yaml",
        }
        if config_filename in target_configs:
            mono_color_dir = config["output_dir"] / "monotonicity_plots_colored"
            mono_color_dir.mkdir(parents=True, exist_ok=True)
            plot_monotonicity_colored_panels(
                monotonicity_plot_data_final,
                trad_dfs_for_plot,
                mono_color_dir,
                config["dataset_type"],
                config["dataset_name"],
            )

        mono_large_dir = config["output_dir"] / "monotonicity_plots_large_scale"
        mono_large_dir.mkdir(parents=True, exist_ok=True)
        plot_monotonicity_large_scale_panels(
            monotonicity_plot_data_final,
            trad_dfs_for_plot,
            mono_large_dir,
            config["dataset_type"],
            config["dataset_name"],
        )

        addi_plot_dir = config["output_dir"] / "additivity_plots"
        addi_plot_dir.mkdir(parents=True, exist_ok=True)
        addi_plot_path = addi_plot_dir / f"{config['dataset_name']}_additivity.png"
        plot_additivity_consistency_scatter(
            additivity_plot_data_final,
            addi_plot_path,
            config["dataset_type"],
            config["dataset_name"],
        )

    return agg_metrics


def run_all_evaluations(config: Dict[str, Any], device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover, execute, and aggregate all TCR-GIN jobs and baseline evaluations.
    """
    base_model_params = config.get("base_model_params", {})
    config_dir = Path(config["__file_path__"]).parent

    jobs_to_run = discover_model_jobs(config, base_model_params, device, config_dir)
    all_job_summary_results = []

    if jobs_to_run:
        for job in tqdm(jobs_to_run, desc="Processing all model jobs"):
            summary_result = run_evaluation_job(job, config, device)
            if summary_result:
                all_job_summary_results.append(summary_result)
    else:
        print("No valid TCR-GIN jobs were discovered.")

    tgin_results_df = pd.DataFrame(all_job_summary_results)
    benchmark_df = calculate_benchmarks(config["dismantling_path"], config)

    return tgin_results_df, benchmark_df


# =============================================================================
# Section 9. Result Aggregation and Saving
# =============================================================================

def process_and_save_results(
    tgin_results_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """
    Merge TCR-GIN and baseline results, reorder columns, and save to CSV.
    """
    if not tgin_results_df.empty:
        model_scales = str(sorted([Path(m["base_dir"]).name for m in config.get("model_suite", [])]))
        tgin_results_df["test_dataset"] = config["dataset_name"]
        tgin_results_df["model_scales"] = model_scales
        tgin_results_df["prediction_strategy"] = config["prediction_strategy"]

    if not benchmark_df.empty:
        benchmark_df["test_dataset"] = config["dataset_name"]
        benchmark_df["model_scales"] = "N/A"
        benchmark_df["prediction_strategy"] = "N/A"

    final_df = pd.concat([tgin_results_df, benchmark_df], ignore_index=True)
    if final_df.empty:
        print("\nEvaluation finished, but no valid results were collected.")
        return

    meta_cols_def = ["test_dataset", "algorithm", "prediction_strategy", "model_scales"]
    metric_roots = [
        "M_freq", "M_int",
        "S_freq", "S_int",
        "additivity_consistency_mae",
        "monotonicity_accuracy_mae",
    ]

    all_cols = list(final_df.columns)
    found_meta, found_metrics, found_params = [], [], []

    for col in all_cols:
        if col in meta_cols_def:
            found_meta.append(col)
            continue

        is_metric = any(col.startswith(root) for root in metric_roots)
        if is_metric:
            found_metrics.append(col)
        else:
            found_params.append(col)

    found_meta.sort(key=lambda x: meta_cols_def.index(x))
    found_params.sort()
    found_metrics.sort()

    ordered_cols = found_meta + found_params + found_metrics
    final_df = final_df.reindex(columns=ordered_cols)

    config["summary_csv_path"].parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(config["summary_csv_path"], index=False, float_format="%.6f")

    print(f"\nEvaluation completed. Aggregated results saved to: {config['summary_csv_path']}")
    print("\n--- Final Result Preview ---")
    with pd.option_context("display.max_rows", 40, "display.max_columns", None, "display.width", 200):
        print(final_df)


# =============================================================================
# Section 10. Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TCR-GIN trajectory property evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config file")
    cli_args = parser.parse_args()

    config = setup_environment(cli_args.config)
    config["__file_path__"] = cli_args.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-" * 60)
    print(f"Dataset           : {config['dataset_name']} (type: {config.get('dataset_type', 'N/A')})")
    print(f"Test type         : {config['test_type']}")
    print(f"Prediction mode   : {config['prediction_strategy']}")
    print(f"Device            : {device}")
    print(f"Output directory  : {config['output_dir']}")
    print("-" * 60)

    tgin_results_df, benchmark_df = run_all_evaluations(config, device)
    process_and_save_results(tgin_results_df, benchmark_df, config)


if __name__ == "__main__":
    main()
