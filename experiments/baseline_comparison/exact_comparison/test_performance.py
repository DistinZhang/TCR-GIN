#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/baseline_comparison/exact_comparison/test_performance.py

Evaluate a trained GNN model on test datasets and compare against multiple
baseline dismantling heuristics.

This script reads an experiment YAML config, caches the required dataset "test"
folders into a temporary directory (to avoid touching large raw datasets),
loads one or multiple model checkpoints, runs inference, and exports:
1) A summary CSV containing MAE/RMSE/R2 and baseline comparisons.
2) An optional "detailed" CSV containing per-network label, model prediction
   (mean/std over multiple runs), and baseline thresholds.

usage
------------
python experiments/baseline_comparison/exact_comparison/test_performance.py \
  --config experiments/baseline_comparison/exact_comparison/configs/test_exact.yaml

python experiments/baseline_comparison/exact_comparison/test_performance.py \
  --config experiments/baseline_comparison/exact_comparison/configs/test_observ.yaml \
  --algo-name MyFinalAlgoName

Notes
-----
- Run from project root.
- The "algo-name" is only used for column naming in exported detailed results.
- This script assumes dataset files include:
    *_edges.npz, *_features.npy, *_label.json
"""

# =============================================================================
# Section 0. Imports and Path Configuration
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from tqdm.auto import tqdm

# Project root: .../TCR-GIN
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.collate")

# Local imports (project code)
from model.tcr_gin import TCR_GIN  # noqa: E402


# =============================================================================
# Section 1. Shared Utilities (Constants, Mappings, Small Helpers)
# =============================================================================

# Mapping used for output readability (file naming / display abbreviation).
ALGO_FILENAME_MAPPING: Dict[str, str] = {
    "CollectiveInfluenceL1": "CollectiveInfluenceL1",
    "CollectiveInfluenceL2": "CollectiveInfluenceL2",
    "CollectiveInfluenceL3": "CollectiveInfluenceL3",
    "GDM": "GDM",
    "GDMR": "GDMR",
    "CoreGDM": "CoreGDM",
    "CoreHD": "CoreHD",
    "EGND": "EGND",
    "EI_s1": "EI_s1",
    "EI_s2": "EI_s2",
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
    "FINDER_CN": "FINDER",
    "Domirank": "DomiRank",
}
ALGO_ABBREVIATION_MAPPING: Dict[str, str] = {v: k for k, v in ALGO_FILENAME_MAPPING.items()}

# When reinsertion variants exist, some pipelines store reinsertion-time separately.
REINSERTION_TIME_MAPPING: Dict[str, str] = {
    "network_entanglement_small_reinsertion": "network_entanglement_small",
    "network_entanglement_mid_reinsertion": "network_entanglement_mid",
    "network_entanglement_large_reinsertion": "network_entanglement_large",
    "GNDR": "GND",
    "vertex_entanglement_reinsertion": "vertex_entanglement",
}

DEFAULT_OUTPUT_ORDER: List[str] = [
    "CollectiveInfluenceL1",
    "CollectiveInfluenceL2",
    "CollectiveInfluenceL3",
    "GDM",
    "GDMR",
    "CoreGDM",
    "CoreHD",
    "EGND",
    "EI_s1",
    "EI_s2",
    "GND",
    "GNDR",
    "MS",
    "MSR",
    "NES",
    "NESR",
    "NEM",
    "NEMR",
    "NEL",
    "NELR",
    "VE",
    "VER",
    "DC",
    "DCR",
    "BC",
    "BCR",
]


def calculate_robust_mean(series: pd.Series) -> float:
    """
    Compute a robust mean by IQR filtering when enough samples exist.

    If the series is too small (<4) or IQR filtering removes everything,
    fall back to the plain mean.
    """
    if series.empty or len(series) < 4:
        return float(series.mean())

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = series[(series >= lower_bound) & (series <= upper_bound)]
    return float(filtered.mean()) if not filtered.empty else float(series.mean())


def _ensure_list(x: Union[str, int, List, Tuple, None]) -> List:
    """Ensure a value becomes a list (None -> [])."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


# =============================================================================
# Section 2. Phase 1 — Dataset Discovery and Temporary Cache
# =============================================================================

def collect_all_dataset_paths(config: dict) -> List[str]:
    """
    Parse all dataset '.../test' relative paths required by the YAML config.

    The config supports multiple experiment formats:
    - experiments[].instances[] + templates
    - experiments[].dataset_instances[]
    """
    all_paths: Set[str] = set()
    experiments = config.get("experiments", [])

    for exp_config in experiments:
        templates = exp_config.get("templates", {})

        # Template-based instances
        if "instances" in exp_config:
            for instance in exp_config.get("instances", []):
                if "source_scales" in instance:
                    scales = instance.get("source_scales", [])
                    generators = instance.get("source_generators", [])
                    for scale in scales:
                        for generator in generators:
                            d_path = templates["d_path"].format(scale=scale, generator=generator)
                            all_paths.add(d_path)
                else:
                    scales = instance.get("scales", [])
                    generators = instance.get("generators", [])
                    for scale in scales:
                        for generator in generators:
                            d_path = templates["d_path"].format(
                                scale=scale,
                                generator=generator,
                                model_type=instance.get("model_type", generator),
                            )
                            all_paths.add(d_path)

        # Direct dataset instances
        if "dataset_instances" in exp_config:
            for instance in exp_config.get("dataset_instances", []):
                paths = instance["d_path"] if isinstance(instance["d_path"], list) else [instance["d_path"]]
                for p in paths:
                    all_paths.add(p)

    return sorted(all_paths)


@contextlib.contextmanager
def manage_test_data(config: dict):
    """
    Context manager: copy required dataset 'test' folders into a temp cache.

    This helps:
    - avoid modifying original datasets,
    - reduce repeated I/O when running multiple tasks,
    - simplify packaging for GitHub release (you can point datasets_root_dir to a smaller folder).

    Yields
    ------
    temp_dir : str
        Absolute path to the cached dataset root.
    """
    source_root = config["base_config"]["global_settings"]["datasets_root_dir"]
    source_root = str(source_root)
    if not os.path.isabs(source_root):
        source_root = str(PROJECT_ROOT / source_root)

    dataset_rel_paths = collect_all_dataset_paths(config)

    temp_dir = PROJECT_ROOT / f"temp_data_cache_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Caching data to temporary directory: {temp_dir}")
    try:
        if not dataset_rel_paths:
            print("[WARNING] No dataset paths found to cache.")
        else:
            print(f"[INFO] Found {len(dataset_rel_paths)} unique 'test' directories to copy.")
            for rel_path in tqdm(dataset_rel_paths, desc="  Copying data"):
                src = Path(source_root) / rel_path
                dest = temp_dir / rel_path
                if src.is_dir():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    print(f"  [Warning] Source directory not found, skipping: {src}")

        yield str(temp_dir)

    finally:
        print(f"\n[INFO] Cleaning up temporary data directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_graph_ids(dataset_dirs: Union[str, List[str]], limit: int = -1) -> List[str]:
    """
    Collect unique graph ids by scanning '*_edges.npz' files.

    Parameters
    ----------
    dataset_dirs:
        One or multiple dataset directories.
    limit:
        If >0, returns only the first N graph ids (after sorting).
    """
    dataset_dirs = dataset_dirs if isinstance(dataset_dirs, list) else [dataset_dirs]

    all_graph_ids: Set[str] = set()
    for directory in dataset_dirs:
        if not os.path.isdir(directory):
            print(f"  [Warning] Dataset directory does not exist: {directory}")
            continue

        for f in glob.glob(os.path.join(directory, "*_edges.npz")):
            graph_id = os.path.basename(f).replace("_edges.npz", "")
            all_graph_ids.add(graph_id)

    sorted_ids = sorted(all_graph_ids)
    if 0 < limit < len(sorted_ids):
        print(f"  [INFO] Sampling first {limit} graphs out of {len(sorted_ids)} available.")
        return sorted_ids[:limit]
    return sorted_ids


# =============================================================================
# Section 3. Feature Utilities (Node Features / Feature-Set Helpers)
# =============================================================================

def get_feature_set_from_dim(dim: int) -> str:
    """Infer feature set name by feature dimension."""
    if dim <= 3:
        return "basic"
    if dim <= 5:
        return "extended"
    return "full"


def calculate_node_features(G: nx.Graph, feature_set: str = "full") -> Tuple[np.ndarray, List[str]]:
    """
    Calculate node features from a NetworkX graph.

    Supported feature sets
    ----------------------
    basic    : degree, clustering, k-core
    extended : basic + avg_neighbor_deg, pagerank
    full     : extended + betweenness, eigenvector

    Returns
    -------
    feature_matrix : np.ndarray, shape (n_nodes, n_features)
    feature_names  : list of str
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        dim_map = {"basic": 3, "extended": 5, "full": 7}
        return np.empty((0, dim_map.get(feature_set, 7))), []

    features: Dict[str, np.ndarray] = {}

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    features["degree"] = degrees
    features["clustering"] = np.array([nx.clustering(G, u) for u in nodes], dtype=float)

    core_numbers = nx.core_number(G)
    features["kcore"] = np.array([core_numbers.get(u, 0) for u in nodes], dtype=float)

    if feature_set in ("extended", "full"):
        avg_neighbor_deg = np.zeros(n_nodes, dtype=float)
        for i, u in enumerate(nodes):
            if G.degree(u) > 0:
                avg_neighbor_deg[i] = sum(G.degree(v) for v in G.neighbors(u)) / G.degree(u)
        features["avg_neighbor_deg"] = avg_neighbor_deg

        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
        features["pagerank"] = np.array([pr.get(u, 0.0) for u in nodes], dtype=float)

    if feature_set == "full":
        # Betweenness
        try:
            k = min(50, n_nodes - 1) if n_nodes > 50 else None
            bc = nx.betweenness_centrality(G, k=k, seed=42) if k is not None else nx.betweenness_centrality(G)
            features["betweenness"] = np.array([bc.get(u, 0.0) for u in nodes], dtype=float)
        except Exception:
            features["betweenness"] = np.zeros(n_nodes, dtype=float)

        # Eigenvector
        try:
            ec = nx.eigenvector_centrality_numpy(G, max_iter=100, tol=1e-4)
            features["eigenvector"] = np.array([ec.get(u, 0.0) for u in nodes], dtype=float)
        except Exception:
            max_deg = float(max(degrees)) if len(degrees) > 0 else 1.0
            features["eigenvector"] = degrees / max(1.0, max_deg)

    feature_names = list(features.keys())
    feature_matrix = np.column_stack([features[name] for name in feature_names])
    return feature_matrix, feature_names


# =============================================================================
# Section 4. Phase 2 — Baseline Loading and Label Parsing
# =============================================================================
def load_baselines(baselines_dir: str, scale: Union[str, int, List, None]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load baseline results and build pivot tables.

    Returns
    -------
    threshold_pivot : pd.DataFrame
        index=network, columns=algorithm, values=critical_threshold
    time_pivot : pd.DataFrame
        index=network, columns=algorithm, values=dismantle_time
    """
    all_dfs: List[pd.DataFrame] = []

    # ====== Modifiedsection START ======
    # Find results_final directly under baselines_dir,no longer append scale subdirectory
    results_path = os.path.join(baselines_dir, "results_final")

    # Fallback: if no results_final subdirectory, find files directly under baselines_dir
    if not os.path.isdir(results_path):
        has_files = bool(
            glob.glob(os.path.join(baselines_dir, "*.csv"))
            or glob.glob(os.path.join(baselines_dir, "*.xlsx"))
        )
        if os.path.isdir(baselines_dir) and has_files:
            results_path = baselines_dir
        else:
            print(f"  [Warning] No results_final found under {baselines_dir}")
            return pd.DataFrame(), pd.DataFrame()
    # ====== Modifiedsection END ======

    baseline_files = glob.glob(os.path.join(results_path, "*.xlsx")) + glob.glob(os.path.join(results_path, "*.csv"))
    if not baseline_files:
        print(f"  [Warning] No baseline files (*.xlsx/*.csv) found in {results_path}")
        return pd.DataFrame(), pd.DataFrame()

    for f in baseline_files:
        try:
            if f.endswith(".xlsx"):
                df = pd.read_excel(f, engine="openpyxl")
            else:
                df = pd.read_csv(f, sep=None, engine="python")


            # Ensure a 'network' column exists
            if "network" not in df.columns:
                candidates = [c for c in df.columns if "network" in c.lower() or "graph" in c.lower()]
                if candidates:
                    df.rename(columns={candidates[0]: "network"}, inplace=True)
                else:
                    continue

            # Normalize algorithm naming for degree/betweenness static vs dynamic
            if "heuristic" in df.columns:
                def _algo_name(row: pd.Series) -> str:
                    h = str(row["heuristic"])
                    if h in ("degree", "betweenness_centrality") and "static" in row and pd.notna(row["static"]):
                        static_flag = "T" if str(row.get("static", "")).upper() == "TRUE" else "F"
                        return f"{h}_{static_flag}"
                    return h

                df["algorithm"] = df.apply(_algo_name, axis=1)
            else:
                # Fallback: infer from filename
                df["algorithm"] = (
                    os.path.basename(f)
                    .replace(".xlsx", "")
                    .replace(".csv", "")
                )

            df["network"] = df["network"].astype(str).str.strip()
            all_dfs.append(df)

        except Exception as e:
            print(f"    [Warning] Failed to read or parse baseline file {f}: {e}")

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    threshold_pivot = combined_df.pivot_table(index="network", columns="algorithm", values="critical_threshold", aggfunc="first")
    time_pivot = combined_df.pivot_table(index="network", columns="algorithm", values="dismantle_time", aggfunc="first")

    # Merge base + reinsertion times when both exist
    for reinsertion_algo, base_algo in REINSERTION_TIME_MAPPING.items():
        if reinsertion_algo in time_pivot.columns and base_algo in time_pivot.columns:
            time_pivot[reinsertion_algo] = time_pivot[reinsertion_algo].add(time_pivot[base_algo], fill_value=0.0)

    return threshold_pivot, time_pivot


def load_labels(dataset_dirs: Union[str, List[str]]) -> Dict[str, float]:
    """
    Load label json files from dataset directories.

    Expected JSON keys
    ------------------
    critical_threshold : float
    """
    dataset_dirs = dataset_dirs if isinstance(dataset_dirs, list) else [dataset_dirs]

    labels: Dict[str, float] = {}
    for directory in dataset_dirs:
        if not os.path.isdir(directory):
            continue

        for f in glob.glob(os.path.join(directory, "*_label.json")):
            try:
                network_name = os.path.basename(f).replace("_label.json", "")
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                labels[network_name] = float(data["critical_threshold"])
            except Exception as e:
                print(f"  [Warning] Failed to read label file {f}: {e}")

    return labels


# =============================================================================
# Section 5. Phase 3 — Experiment Runner (Inference + Metrics + Exports)
# =============================================================================

def run_single_experiment(
    task_config: dict,
    global_settings: dict,
    algo_display_name: str,
) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    """
    Run a single task:
    - Load model checkpoints under exp_dir/model_run_*.pt
    - Enumerate test graphs from dataset dirs
    - Predict y_pred for each graph and compute metrics against label y_true
    - Optionally merge baseline thresholds into a detailed per-network table

    Returns
    -------
    final_metrics : dict | None
        A single-row summary (to be concatenated for all tasks).
    detailed_df : pd.DataFrame | None
        Per-network detailed predictions (mean/std across runs) and baselines.
    """
    exp_dir = task_config["model_source"]["path"]
    full_job_name = task_config["job_name"]
    print(f"\n--- Running Task: {full_job_name} ---")

    # Baselines directory
    raw_baselines_path = global_settings["baselines_root_dir"]
    expanded_path = os.path.expanduser(raw_baselines_path)
    baselines_dir = expanded_path if os.path.isabs(expanded_path) else str(PROJECT_ROOT / expanded_path)

    # Dataset directory (cached root)
    datasets_root = global_settings["datasets_root_dir"]
    datasets_root = str(datasets_root)
    if not os.path.isabs(datasets_root):
        datasets_root = str(PROJECT_ROOT / datasets_root)

    test_path_config = task_config["test_dataset"]["path"]
    test_path_config = test_path_config if isinstance(test_path_config, list) else [test_path_config]
    dataset_dirs = [str(Path(datasets_root) / p) for p in test_path_config]

    scale = task_config["test_dataset"]["scale"]
    feature_dim = int(task_config["feature_dim"])

    device = torch.device(
        "cuda" if global_settings.get("device", "auto") == "auto" and torch.cuda.is_available() else "cpu"
    )

    sample_limit = int(task_config.get("sampling", {}).get("test_limit", -1))

    print(f"  - Model Dir: {exp_dir}")
    print(f"  - Dataset Dir(s) (from cache): {dataset_dirs}")
    print(f"  - Feature Dim: {feature_dim}")
    print(f"  - Device: {device}")

    # Load model runs
    model_paths = sorted(glob.glob(os.path.join(exp_dir, "model_run_*.pt")))
    models: List[TCR_GIN] = []
    if model_paths:
        model_params = task_config["model_params"]
        for path in model_paths:
            try:
                args = argparse.Namespace(**model_params)
                args.input_dim = feature_dim
                model = TCR_GIN(args).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"  [Warning] Failed to load model {path}: {e}")

    # Load graph ids and labels
    graph_ids = get_graph_ids(dataset_dirs, limit=sample_limit)
    labels = load_labels(dataset_dirs)

    threshold_baselines_df, time_baselines_df = load_baselines(baselines_dir, scale)

    # Keep only labels that exist in graph ids
    labels = {gid: val for gid, val in labels.items() if gid in graph_ids}

    if not models or not graph_ids or not labels:
        print("  [ERROR] Missing models, graphs, or labels for this sample. Skipping task.")
        return None, None

    job_run_metrics: List[dict] = []

    # [NEW] Collect raw per-run predictions for detailed output
    all_raw_predictions: List[dict] = []

    pbar_models = tqdm(enumerate(models), total=len(models), desc="  Testing model runs", leave=False)
    for run_idx, model in pbar_models:
        run_results: List[dict] = []

        for graph_id in tqdm(graph_ids, desc=f"    Run {run_idx + 1}/{len(models)}", leave=False):
            if graph_id not in labels:
                continue

            graph_path_prefix = None
            for d_dir in dataset_dirs:
                potential_prefix = os.path.join(d_dir, graph_id)
                if os.path.exists(f"{potential_prefix}_edges.npz"):
                    graph_path_prefix = potential_prefix
                    break
            if graph_path_prefix is None:
                continue

            # Feature timing (calculated features) — kept for accounting, even if final uses saved features
            try:
                t_feature_start = time.time()
                edges = np.load(f"{graph_path_prefix}_edges.npz", allow_pickle=True)["edges"]
                G = nx.Graph()
                G.add_edges_from(edges)
                feature_set = get_feature_set_from_dim(feature_dim)
                calculated_features, _ = calculate_node_features(G, feature_set)
                calculated_features = calculated_features[:, :feature_dim]
                t_feature_end = time.time()
                _feature_time = t_feature_end - t_feature_start
            except Exception:
                continue

            # Load precomputed features (actual features used for model input)
            try:
                features_to_use = np.load(f"{graph_path_prefix}_features.npy")[:, :feature_dim]
            except Exception:
                continue

            # Inference
            t_inference_start = time.time()
            x = torch.tensor(features_to_use, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            graph_data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0]).to(device)
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

            with torch.no_grad():
                y_pred = float(torch.clamp(model(graph_data), 0.0, 1.0).item())
            t_inference_end = time.time()

            total_time = (t_feature_end - t_feature_start) + (t_inference_end - t_inference_start)
            run_results.append(
                {
                    "network": graph_id,
                    "y_true": float(labels[graph_id]),
                    "y_pred": y_pred,
                    "time_total": float(total_time),
                }
            )

        if run_results:
            # [NEW] Store raw predictions for later aggregation (mean/std)
            all_raw_predictions.extend(run_results)

            df_run = pd.DataFrame(run_results)
            job_run_metrics.append(
                {
                    "mae": float(mean_absolute_error(df_run["y_true"], df_run["y_pred"])),
                    "rmse": float(np.sqrt(mean_squared_error(df_run["y_true"], df_run["y_pred"]))),
                    "r2": float(r2_score(df_run["y_true"], df_run["y_pred"])),
                    "time": float(df_run["time_total"].mean()),
                }
            )

    if not job_run_metrics:
        print("  [ERROR] No valid results were produced. Skipping.")
        return None, None

    # [NEW] Build detailed per-network prediction table (mean/std across runs)
    detailed_df: Optional[pd.DataFrame] = None
    try:
        if all_raw_predictions:
            df_all_runs = pd.DataFrame(all_raw_predictions)

            grouped = df_all_runs.groupby("network")
            labels_series = grouped["y_true"].first()
            preds_mean = grouped["y_pred"].mean()
            preds_std = grouped["y_pred"].std()

            algo_col = algo_display_name
            algo_std_col = f"{algo_display_name} (Std)"

            detailed_df = pd.DataFrame(
                {
                    "Label": labels_series,
                    algo_col: preds_mean,
                    algo_std_col: preds_std.fillna(0.0),  # std is NaN if only one run
                }
            )

            # Merge baseline thresholds (left join on network index)
            if not threshold_baselines_df.empty:
                threshold_baselines_df.index = threshold_baselines_df.index.astype(str).str.strip()
                detailed_df.index = detailed_df.index.astype(str).str.strip()
                detailed_df = detailed_df.join(threshold_baselines_df, how="left")

    except Exception as e:
        print(f"  [Warning] Error creating detailed dataframe: {e}")

    # Summary metrics (aggregated over model runs)
    df_job = pd.DataFrame(job_run_metrics)

    final_metrics = {
        "Job Name": full_job_name,
        "Model Path": os.path.relpath(exp_dir, str(PROJECT_ROOT)),
        "Test Dataset": str(task_config["test_dataset"]["path"]),
        "MAE": f"{df_job['mae'].mean():.4f} ± {df_job['mae'].std():.4f}",
        "RMSE": f"{df_job['rmse'].mean():.4f} ± {df_job['rmse'].std():.4f}",
        "R2": f"{df_job['r2'].mean():.4f} ± {df_job['r2'].std():.4f}",
    }

    # Per-run MAE columns for quick inspection
    for idx, run in enumerate(job_run_metrics, 1):
        final_metrics[f"{algo_display_name} Run{idx} (MAE)"] = f"{run['mae']:.4f}"

    # Baseline comparisons: MAE against labels and robust mean time
    temp_mae_metrics: Dict[str, str] = {}
    temp_time_metrics: Dict[str, str] = {}

    if not threshold_baselines_df.empty:
        threshold_baselines_df.index = threshold_baselines_df.index.astype(str).str.strip()
        time_baselines_df.index = time_baselines_df.index.astype(str).str.strip()

        labels_series = pd.Series(labels)
        labels_series.index = labels_series.index.astype(str).str.strip()
        test_networks = list(labels_series.index)

        relevant_thresholds = threshold_baselines_df[threshold_baselines_df.index.isin(test_networks)]
        relevant_times = time_baselines_df[time_baselines_df.index.isin(test_networks)]

        if not relevant_thresholds.empty:
            baseline_labels = labels_series.loc[relevant_thresholds.index]

            for algo_fullname in relevant_thresholds.columns:
                abbr = ALGO_FILENAME_MAPPING.get(algo_fullname, algo_fullname)
                preds = relevant_thresholds[algo_fullname].dropna()

                common_index = baseline_labels.index.intersection(preds.index)
                if not common_index.empty:
                    errors = (baseline_labels.loc[common_index] - preds.loc[common_index]).abs()
                    temp_mae_metrics[abbr] = f"{errors.mean():.4f}"

                if algo_fullname in relevant_times.columns:
                    relevant_time_values = relevant_times.loc[common_index, algo_fullname].dropna()
                    if not relevant_time_values.empty:
                        temp_time_metrics[abbr] = f"{calculate_robust_mean(relevant_time_values):.4f}"

    # Write baseline MAE columns in a stable order first
    for abbr in DEFAULT_OUTPUT_ORDER:
        if abbr in temp_mae_metrics:
            final_metrics[f"{abbr} (MAE)"] = temp_mae_metrics[abbr]
    for abbr, value in sorted(temp_mae_metrics.items()):
        if abbr not in DEFAULT_OUTPUT_ORDER:
            final_metrics[f"{abbr} (MAE)"] = value

    # Time summary for model
    final_metrics["Time (s)"] = f"{df_job['time'].mean():.4f}"

    # Baseline time columns
    for abbr in DEFAULT_OUTPUT_ORDER:
        if abbr in temp_time_metrics:
            final_metrics[f"{abbr} (Time)"] = temp_time_metrics[abbr]
    for abbr, value in sorted(temp_time_metrics.items()):
        if abbr not in DEFAULT_OUTPUT_ORDER:
            final_metrics[f"{abbr} (Time)"] = value

    return final_metrics, detailed_df


# =============================================================================
# Section 6. CLI and Entry Point (Config Merge + Task Generation + Main)
# =============================================================================

def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two config dicts (override wins)."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def generate_tasks_from_experiment(exp_config: dict, base_config: dict) -> List[dict]:
    """
    Expand a high-level experiment group into runnable tasks.

    Supports:
    - dataset_instances (direct list)
    - instances + templates (grid expansion)
    """
    tasks: List[dict] = []

    base_config = deepcopy(base_config)
    base_config.setdefault("model_source", {})
    base_config.setdefault("test_dataset", {})

    base_task_template = merge_configs(
        base_config,
        {k: v for k, v in exp_config.items() if k not in ["instances", "templates"]},
    )

    # Direct dataset instances
    if "dataset_instances" in exp_config:
        for instance in exp_config.get("dataset_instances", []):
            task = merge_configs(deepcopy(base_task_template), instance)
            task["model_source"] = {"path": instance["m_path"]}
            task["test_dataset"] = {"path": instance["d_path"], "scale": instance["scale"]}
            task["job_name"] = instance["name"]
            tasks.append(task)
        return tasks

    # Template-based instances
    if "instances" not in exp_config:
        return []

    templates = exp_config.get("templates", {})

    for instance in exp_config.get("instances", []):
        task_base_for_instance = merge_configs(deepcopy(base_task_template), instance)

        # Mix/source mode
        if "source_scales" in instance:
            task = deepcopy(task_base_for_instance)
            placeholders = {"mix_id": instance["mix_id"]}

            task["model_source"]["path"] = templates["m_path"].format_map(placeholders)
            d_paths = [
                templates["d_path"].format(scale=s, generator=g)
                for s in instance["source_scales"]
                for g in instance["source_generators"]
            ]
            task["test_dataset"]["path"] = d_paths
            task["test_dataset"]["scale"] = instance["source_scales"]
            task["job_name"] = templates["job_name"].format_map(placeholders)
            tasks.append(task)
            continue

        # Grid expansion: scales x generators
        for scale in instance.get("scales", []):
            for generator in instance.get("generators", []):
                task = deepcopy(task_base_for_instance)
                placeholders = {
                    "scale": scale,
                    "generator": generator,
                    "model_type": instance.get("model_type", generator),
                }

                task["model_source"]["path"] = templates["m_path"].format_map(placeholders)
                task["test_dataset"]["path"] = templates["d_path"].format_map(placeholders)
                task["test_dataset"]["scale"] = str(scale)
                task["job_name"] = templates["job_name"].format_map(placeholders)
                tasks.append(task)

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN Performance Testing Framework (Exact Comparison)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="TCR-GIN",
        help="Display name used in exported columns (default: TCR-GIN).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GNN Performance Testing Framework")
    print("=" * 60)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_results: List[dict] = []
    all_detailed_dfs: List[pd.DataFrame] = []

    # Cache dataset test folders to a temporary root
    with manage_test_data(config) as temp_data_root:
        base_config = config["base_config"]
        base_config["global_settings"]["datasets_root_dir"] = temp_data_root

        experiments = config.get("experiments", [])
        for exp_config in experiments:
            print(f"\nParsing Experiment Group: {exp_config.get('name', 'Unnamed')}")
            tasks_to_run = generate_tasks_from_experiment(exp_config, base_config)

            for task_config in tasks_to_run:
                # Resolve models root
                models_root = task_config["global_settings"]["models_root_dir"]
                models_root = str(models_root)
                if not os.path.isabs(models_root):
                    models_root = str(PROJECT_ROOT / models_root)

                base_model_path = os.path.join(models_root, task_config["model_source"]["path"])
                exp_dirs = sorted(glob.glob(os.path.join(base_model_path, "exp_*")))

                # Fallback: model files directly under base_model_path
                if not exp_dirs and glob.glob(os.path.join(base_model_path, "model_run_*.pt")):
                    exp_dirs = [base_model_path]

                if not exp_dirs:
                    print(
                        f"[Warning] No 'exp_*' subdirectories or models found in {base_model_path}. "
                        f"Skipping job '{task_config.get('job_name', 'Unnamed')}'."
                    )
                    continue

                for exp_dir in exp_dirs:
                    final_task_config = deepcopy(task_config)
                    final_task_config["model_source"]["path"] = exp_dir

                    result, detailed_df = run_single_experiment(
                        final_task_config,
                        task_config["global_settings"],
                        algo_display_name=args.algo_name,
                    )
                    if result:
                        all_results.append(result)
                    if detailed_df is not None:
                        all_detailed_dfs.append(detailed_df)

    output_dir = os.path.join(str(PROJECT_ROOT), config["base_config"]["global_settings"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    output_filename = config["base_config"]["global_settings"]["output_filename"]

    # 1) Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Summary results saved to: {output_path}")
    else:
        print("\n[INFO] No summary results were generated.")

    # 2) Save detailed per-network predictions
    if all_detailed_dfs:
        full_detailed_df = pd.concat(all_detailed_dfs, ignore_index=False)

        filename_root, ext = os.path.splitext(output_filename)
        detailed_filename = f"{filename_root}_detailed{ext}" if ext else f"{filename_root}_detailed"
        detailed_output_path = os.path.join(output_dir, detailed_filename)

        full_detailed_df.to_csv(detailed_output_path)
        print(f"[SUCCESS] Detailed predictions saved to: {detailed_output_path}")
    else:
        print("[INFO] No detailed results were generated.")

    print("\n" + "=" * 60)
    print("Test Run Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()
