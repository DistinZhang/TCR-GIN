#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/baseline_comparison/test_performance.py

Evaluate TCR-GIN checkpoints on configured test datasets and compare with
classical baselines.

This script:
1) parses experiment tasks from a YAML config,
2) caches required test folders into a temporary local directory,
3) evaluates all model runs found in each experiment folder,
4) reports TCR-GIN MAE and average test time,
5) appends baseline MAE/time columns when baseline files are available,
6) optionally exports edge-time analysis CSVs for time-scaling tests.

Important timing behavior
-------------------------
For each graph, the recorded model test time now includes:
- data loading from disk,
- online node feature computation,
- tensor/data object preparation,
- model forward inference.

Usage examples
-------------------------
python experiments/baseline_comparison/test_performance.py
  --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU.yaml

python experiments/baseline_comparison/test_performance.py
  --config experiments/baseline_comparison/configs/test_multisource-LBWE-GPU.yaml

python experiments/baseline_comparison/test_performance.py
  --config experiments/baseline_comparison/configs/test_multisource-REDDIT-CPU.yaml

python experiments/baseline_comparison/test_performance.py
  --config experiments/baseline_comparison/configs/test_multisource-REDDIT-GPU.yaml

python experiments/baseline_comparison/test_performance.py
  --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU-time.yaml
"""


from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import shutil
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from tqdm.auto import tqdm


# =============================================================================
# Section 0. Project Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.collate")

from model.tcr_gin import TCR_GIN  # noqa: E402


# =============================================================================
# Section 1. Constants and Mappings
# =============================================================================

ALGO_FILENAME_MAPPING = {
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


# =============================================================================
# Section 2. Config/Data Cache Utilities
# =============================================================================

def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def collect_all_dataset_paths(config: Dict[str, Any]) -> List[str]:
    """
    Parse all relative test dataset paths required by the config.
    """
    all_paths = set()
    experiments = config.get("experiments", [])

    for exp_config in experiments:
        templates = exp_config.get("templates", {})

        if "instances" in exp_config:
            for instance in exp_config.get("instances", []):
                d_template = templates.get("d_path")
                if not d_template:
                    continue

                if "source_scales" in instance:
                    scales = instance.get("source_scales", [])
                    generators = instance.get("source_generators", [])
                    for scale in scales:
                        for generator in generators:
                            try:
                                d_path = d_template.format(scale=scale, generator=generator)
                                all_paths.add(d_path)
                            except Exception:
                                continue
                else:
                    scales = instance.get("scales", [])
                    generators = instance.get("generators", [])
                    for scale in scales:
                        for generator in generators:
                            model_type = instance.get("model_type", generator)
                            try:
                                d_path = d_template.format(
                                    scale=scale,
                                    generator=generator,
                                    model_type=model_type
                                )
                                all_paths.add(d_path)
                            except Exception:
                                continue

        if "dataset_instances" in exp_config:
            for instance in exp_config.get("dataset_instances", []):
                for p in _as_list(instance.get("d_path")):
                    all_paths.add(p)

    return sorted(all_paths)


@contextlib.contextmanager
def manage_test_data(config: Dict[str, Any]):
    """
    Context manager:
    copy only required test folders into a temporary cache and clean up after run.
    """
    source_root = Path(config["base_config"]["global_settings"]["datasets_root_dir"]).expanduser()
    if not source_root.is_absolute():
        source_root = PROJECT_ROOT / source_root

    dataset_rel_paths = collect_all_dataset_paths(config)
    temp_dir = PROJECT_ROOT / f"temp_data_cache_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Caching data to temporary directory: {temp_dir}")

    try:
        for rel_path in tqdm(dataset_rel_paths, desc="  Copying data"):
            src = source_root / rel_path
            dst = temp_dir / rel_path
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
        yield str(temp_dir)
    finally:
        print(f"\n[INFO] Cleaning up temporary data directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Section 3. Feature Utilities / Data Loading / Baselines
# =============================================================================

def get_graph_ids(dataset_dirs: Union[str, List[str]], limit: int = -1) -> List[str]:
    """
    Discover graph IDs by scanning *_edges.npz files in one or multiple directories.
    """
    directories = _as_list(dataset_dirs)
    all_graph_ids = set()

    for directory in directories:
        d = Path(directory)
        if not d.is_dir():
            continue
        for f in d.glob("*_edges.npz"):
            all_graph_ids.add(f.stem.replace("_edges", ""))

    sorted_ids = sorted(all_graph_ids)
    if limit > 0 and limit < len(sorted_ids):
        return sorted_ids[:limit]
    return sorted_ids


def calculate_robust_mean(series: pd.Series) -> float:
    """
    Robust mean using 1.5*IQR filtering; fallback to simple mean.
    """
    if series.empty:
        return float("nan")
    if len(series) < 4:
        return float(series.mean())

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = series[(series >= lb) & (series <= ub)]
    return float(filtered.mean()) if not filtered.empty else float(series.mean())


def get_feature_set_from_dim(dim: int) -> str:
    """
    Infer feature set from requested feature dimension.
    """
    if dim <= 3:
        return "basic"
    if dim <= 5:
        return "extended"
    return "full"


def calculate_node_features(G, feature_set='full'):
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        dim_map = {'basic': 3, 'extended': 5, 'full': 7}
        return np.empty((0, dim_map.get(feature_set, 7))), []

    features_dict = {
        'degree': np.array([d for _, d in G.degree()], dtype=float),
        'clustering': np.array([nx.clustering(G, u) for u in nodes], dtype=float),
    }

    core_numbers = nx.core_number(G)
    features_dict['kcore'] = np.array([core_numbers.get(u, 0) for u in nodes], dtype=float)

    if feature_set in ['extended', 'full']:
        avg_neighbor_deg = np.zeros(n_nodes, dtype=float)
        for i, u in enumerate(nodes):
            if G.degree(u) > 0:
                avg_neighbor_deg[i] = sum(G.degree(v) for v in G.neighbors(u)) / G.degree(u)
        features_dict['avg_neighbor_deg'] = avg_neighbor_deg

        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
        features_dict['pagerank'] = np.array([pr.get(u, 0) for u in nodes], dtype=float)

    if feature_set == 'full':
        try:
            k = min(50, n_nodes - 1) if n_nodes > 50 else None
            bc = nx.betweenness_centrality(G, k=k, seed=42) if k is not None else nx.betweenness_centrality(G)
            features_dict['betweenness'] = np.array([bc.get(u, 0) for u in nodes], dtype=float)
        except Exception:
            features_dict['betweenness'] = np.zeros(n_nodes, dtype=float)

        try:
            ec = nx.eigenvector_centrality_numpy(G, max_iter=100, tol=1e-4)
            features_dict['eigenvector'] = np.array([ec.get(u, 0) for u in nodes], dtype=float)
        except Exception:
            max_deg = max(1.0, float(np.max(features_dict['degree']))) if len(features_dict['degree']) > 0 else 1.0
            features_dict['eigenvector'] = features_dict['degree'] / max_deg

    feature_matrix = np.column_stack([features_dict[f] for f in features_dict.keys()])
    return feature_matrix, list(features_dict.keys())


def _normalize_edges_array(edges: np.ndarray) -> np.ndarray:
    """
    Normalize loaded edges into int64 shape (E, 2).
    """
    arr = np.asarray(edges)

    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid edge shape: {arr.shape}, expected (E, 2).")
    return arr


def load_baselines(
    baselines_dir: Union[str, Path],
    scale: Union[int, str, List[Union[int, str]], None]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load baseline threshold/time tables and return pivoted DataFrames:
      - threshold_df: index=network, columns=algorithm, values=critical_threshold
      - time_df:      index=network, columns=algorithm, values=dismantle_time
    """
    if scale in (None, 0, "0"):
        return pd.DataFrame(), pd.DataFrame()

    scales_to_load = scale if isinstance(scale, list) else [scale]
    root = Path(baselines_dir).expanduser()
    all_dfs = []

    for s in scales_to_load:
        results_path = root / str(s) / "results_final"
        if not results_path.is_dir():
            continue

        files = sorted(results_path.glob("*.xlsx")) + sorted(results_path.glob("*.csv"))
        for f in files:
            try:
                if f.suffix.lower() == ".xlsx":
                    df = pd.read_excel(f, engine="openpyxl")
                else:
                    df = pd.read_csv(f)

                def _algo_name(row):
                    heuristic = str(row.get("heuristic", ""))
                    if heuristic in {"degree", "betweenness_centrality"}:
                        is_static = str(row.get("static", "")).upper() == "TRUE"
                        return f"{heuristic}_{'T' if is_static else 'F'}"
                    return heuristic

                df["algorithm"] = df.apply(_algo_name, axis=1)

                if "network" in df.columns:
                    df["network"] = df["network"].astype(str)
                    all_dfs.append(df)

            except Exception:
                continue

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    threshold_df = combined_df.pivot_table(
        index="network",
        columns="algorithm",
        values="critical_threshold",
        aggfunc="first",
    )
    time_df = combined_df.pivot_table(
        index="network",
        columns="algorithm",
        values="dismantle_time",
        aggfunc="first",
    )
    return threshold_df, time_df


def load_labels(dataset_dirs: Union[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Load labels from *_label.json.

    Returns
    -------
    dict:
      {
        network_name: {
            "threshold": <critical_threshold>,
            "num_edges": <num_edges>
        }
      }
    """
    directories = _as_list(dataset_dirs)
    labels_info: Dict[str, Dict[str, float]] = {}

    for directory in directories:
        d = Path(directory)
        if not d.is_dir():
            continue

        for f in d.glob("*_label.json"):
            try:
                network_name = f.name.replace("_label.json", "")
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)

                if "critical_threshold" not in data:
                    continue

                labels_info[network_name] = {
                    "threshold": float(data["critical_threshold"]),
                    "num_edges": int(data.get("num_edges", 0)),
                }
            except Exception:
                continue

    return labels_info


# =============================================================================
# Section 4. Experiment Execution
# =============================================================================

def _resolve_dataset_dirs(datasets_root: Union[str, Path], path_cfg: Union[str, List[str]]) -> List[str]:
    root = Path(datasets_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return [str(root / p) for p in _as_list(path_cfg)]


def _find_graph_prefix(graph_id: str, dataset_dirs: List[str]) -> Optional[str]:
    for d in dataset_dirs:
        prefix = Path(d) / graph_id
        if (prefix.parent / f"{prefix.name}_edges.npz").exists():
            return str(prefix)
    return None


def _build_undirected_edge_index(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.as_tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _build_graph_for_online_feature_calc(edges: np.ndarray, num_nodes: int) -> nx.Graph:
    """
    Build a NetworkX graph for online feature calculation.

    Important:
    - we explicitly add nodes [0, ..., num_nodes-1] first, so isolated nodes
      present in the saved feature matrix are also included in the online graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if edges.size > 0:
        G.add_edges_from(edges.tolist())
    return G


def _load_models(exp_dir: str, task_config: Dict[str, Any], device: torch.device) -> List[TCR_GIN]:
    model_paths = sorted(glob.glob(os.path.join(exp_dir, "model_run_*.pt")))
    if not model_paths:
        model_paths = sorted(glob.glob(os.path.join(exp_dir, "*.pt")))

    models: List[TCR_GIN] = []
    for path in model_paths:
        try:
            m_args = argparse.Namespace(**task_config["model_params"])
            m_args.input_dim = int(task_config["feature_dim"])
            model = TCR_GIN(m_args).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        except Exception:
            continue
    return models


def run_single_experiment(
    task_config: Dict[str, Any],
    global_settings: Dict[str, Any]
) -> Tuple[Optional[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Execute one task (possibly with multiple model runs in one exp folder).

    Timing definition for each graph:
      total_time = data_loading + online_feature_calc + tensor_prep + model_forward

    Note:
      Online-calculated features are NOT used as model inputs. They are computed
      only for timing/accounting. Model inference still uses the loaded
      `*_features.npy` file.
    """
    exp_dir = task_config["model_source"]["path"]
    full_job_name = task_config["job_name"]
    print(f"\n--- Running Task: {full_job_name} ---")

    datasets_root = global_settings["datasets_root_dir"]
    dataset_dirs = _resolve_dataset_dirs(datasets_root, task_config["test_dataset"]["path"])

    scale = task_config["test_dataset"]["scale"]
    feature_dim = int(task_config["feature_dim"])
    sample_limit = int(task_config.get("sampling", {}).get("test_limit", -1))

    use_cuda = (global_settings.get("device", "auto") == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    models = _load_models(exp_dir, task_config, device)
    graph_ids = get_graph_ids(dataset_dirs, limit=sample_limit)

    labels_full_info = load_labels(dataset_dirs)
    labels_full_info = {gid: info for gid, info in labels_full_info.items() if gid in graph_ids}

    threshold_baselines_df, time_baselines_df = load_baselines(
        os.path.expanduser(global_settings["baselines_root_dir"]),
        scale
    )

    if not models or not graph_ids or not labels_full_info:
        return None, []

    graph_prefix_cache = {gid: _find_graph_prefix(gid, dataset_dirs) for gid in graph_ids}
    job_run_metrics = []
    all_individual_results: List[Dict[str, Any]] = []

    for run_idx, model in tqdm(
        enumerate(models),
        total=len(models),
        desc="  Testing model runs",
        leave=False
    ):
        run_results = []

        for graph_id in tqdm(graph_ids, desc=f"    Run {run_idx + 1}", leave=False):
            info = labels_full_info.get(graph_id)
            if info is None:
                continue

            graph_path_prefix = graph_prefix_cache.get(graph_id)
            if not graph_path_prefix:
                continue

            try:
                # -------------------------------------------------------------
                # 1) Total timing starts BEFORE any per-graph data loading
                # -------------------------------------------------------------
                t_total_start = time.perf_counter()

                # -------------------------------------------------------------
                # 2) Data loading from disk
                # -------------------------------------------------------------
                t_load_start = t_total_start

                raw_edges = np.load(f"{graph_path_prefix}_edges.npz", allow_pickle=True)["edges"]
                loaded_features = np.load(f"{graph_path_prefix}_features.npy")

                t_load_end = time.perf_counter()

                edges = _normalize_edges_array(raw_edges)
                if loaded_features.ndim != 2:
                    continue

                features_for_model = loaded_features[:, :feature_dim]
                num_nodes = int(features_for_model.shape[0])

                if num_nodes == 0:
                    continue

                # -------------------------------------------------------------
                # 3) Online feature calculation (timed but NOT used for model)
                # -------------------------------------------------------------
                t_feat_start = time.perf_counter()

                feature_set = get_feature_set_from_dim(feature_dim)
                G_online = _build_graph_for_online_feature_calc(edges, num_nodes)

                # Compute and discard; only used for timing/accounting
                calculated_features, _ = calculate_node_features(G_online, feature_set=feature_set)
                _ = calculated_features[:, :feature_dim] if calculated_features.ndim == 2 else calculated_features

                t_feat_end = time.perf_counter()

                # -------------------------------------------------------------
                # 4) Tensor/Data preparation for model input
                #    IMPORTANT: use LOADED features, not online-computed ones
                # -------------------------------------------------------------
                t_prep_start = time.perf_counter()

                x = torch.as_tensor(features_for_model, dtype=torch.float32, device=device)
                edge_index = _build_undirected_edge_index(edges, device)

                graph_data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0]).to(device)
                graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

                t_prep_end = time.perf_counter()

                # -------------------------------------------------------------
                # 5) Model forward inference
                # -------------------------------------------------------------
                t_infer_start = time.perf_counter()

                with torch.no_grad():
                    y_pred = float(torch.clamp(model(graph_data), 0.0, 1.0).item())

                t_infer_end = time.perf_counter()

                # -------------------------------------------------------------
                # 6) Time breakdown
                # -------------------------------------------------------------
                data_load_time = t_load_end - t_load_start
                feature_calc_time = t_feat_end - t_feat_start
                prep_time = t_prep_end - t_prep_start
                inference_time = t_infer_end - t_infer_start
                total_time = t_infer_end - t_total_start

                n_edges = int(info["num_edges"])

                run_results.append({
                    "network": graph_id,
                    "y_true": float(info["threshold"]),
                    "y_pred": y_pred,
                    "time_total": total_time,
                    "time_data_load": data_load_time,
                    "time_feature_calc": feature_calc_time,
                    "time_prep": prep_time,
                    "time_inference": inference_time,
                })

                all_individual_results.append({
                    "network": graph_id,
                    "num_edges": n_edges,
                    "job_name": full_job_name,
                    "run_idx": run_idx + 1,
                    "time_total": total_time,
                    "time_data_load": data_load_time,
                    "time_feature_calc": feature_calc_time,
                    "time_prep": prep_time,
                    "time_inference": inference_time,
                })

            except Exception:
                continue

        if run_results:
            df_run = pd.DataFrame(run_results)
            job_run_metrics.append({
                "mae": float(mean_absolute_error(df_run["y_true"], df_run["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(df_run["y_true"], df_run["y_pred"]))),
                "r2": float(r2_score(df_run["y_true"], df_run["y_pred"])),
                "time_total": float(df_run["time_total"].mean()),
                "time_data_load": float(df_run["time_data_load"].mean()),
                "time_feature_calc": float(df_run["time_feature_calc"].mean()),
                "time_prep": float(df_run["time_prep"].mean()),
                "time_inference": float(df_run["time_inference"].mean()),
            })

    if not job_run_metrics:
        return None, []

    df_job = pd.DataFrame(job_run_metrics)

    final_metrics: Dict[str, str] = {
        "Job Name": full_job_name,
        "MAE": f"{df_job['mae'].mean():.4f} ± {df_job['mae'].std(ddof=0):.4f}",
        "RMSE": f"{df_job['rmse'].mean():.4f} ± {df_job['rmse'].std(ddof=0):.4f}",
        "R2": f"{df_job['r2'].mean():.4f} ± {df_job['r2'].std(ddof=0):.4f}",
        "Time (s)": f"{df_job['time_total'].mean():.4f}",
        "Data Load Time (s)": f"{df_job['time_data_load'].mean():.4f}",
        "Feature Calc Time (s)": f"{df_job['time_feature_calc'].mean():.4f}",
        "Prep Time (s)": f"{df_job['time_prep'].mean():.4f}",
        "Inference Time (s)": f"{df_job['time_inference'].mean():.4f}",
    }

    # Baseline MAE/time on common networks
    if not threshold_baselines_df.empty:
        label_series = pd.Series({k: v["threshold"] for k, v in labels_full_info.items()})
        common_nets = threshold_baselines_df.index.intersection(label_series.index)

        if not common_nets.empty:
            for algo in threshold_baselines_df.columns:
                abbr = ALGO_FILENAME_MAPPING.get(algo, algo)

                errs = (
                    label_series.loc[common_nets]
                    - threshold_baselines_df.loc[common_nets, algo]
                ).abs().dropna()

                if not errs.empty:
                    final_metrics[f"{abbr} (MAE)"] = f"{errs.mean():.4f}"

                if algo in time_baselines_df.columns:
                    t_vals = time_baselines_df.loc[common_nets, algo].dropna()
                    if not t_vals.empty:
                        final_metrics[f"{abbr} (Time)"] = f"{calculate_robust_mean(t_vals):.4f}"

    return final_metrics, all_individual_results


# =============================================================================
# Section 5. Config Merging and Task Generation
# =============================================================================

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def generate_tasks_from_experiment(exp_config: Dict[str, Any], base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand one experiment block into concrete task configs.
    """
    tasks = []

    base_cfg = deepcopy(base_config)
    base_cfg.setdefault("model_source", {})
    base_cfg.setdefault("test_dataset", {})

    base_task_template = merge_configs(
        base_cfg,
        {k: v for k, v in exp_config.items() if k not in ["instances", "templates", "dataset_instances"]}
    )

    # Direct dataset instances mode
    if "dataset_instances" in exp_config:
        for instance in exp_config.get("dataset_instances", []):
            task = merge_configs(deepcopy(base_task_template), instance)
            task["model_source"] = {"path": instance["m_path"]}
            task["test_dataset"] = {"path": instance["d_path"], "scale": instance["scale"]}
            task["job_name"] = instance["name"]
            tasks.append(task)
        return tasks

    # Template+instances mode
    if "instances" not in exp_config:
        return []

    templates = exp_config.get("templates", {})

    for instance in exp_config.get("instances", []):
        task_base = merge_configs(base_task_template, instance)

        if "source_scales" in instance:
            task = deepcopy(task_base)
            task["model_source"]["path"] = templates["m_path"].format(mix_id=instance["mix_id"])
            task["test_dataset"]["path"] = [
                templates["d_path"].format(scale=s, generator=g)
                for s in instance["source_scales"]
                for g in instance["source_generators"]
            ]
            task["test_dataset"]["scale"] = instance["source_scales"]
            task["job_name"] = templates["job_name"].format(mix_id=instance["mix_id"])
            tasks.append(task)
        else:
            for scale in instance.get("scales", []):
                for generator in instance.get("generators", []):
                    task = deepcopy(task_base)
                    p = {
                        "scale": scale,
                        "generator": generator,
                        "model_type": instance.get("model_type", generator),
                    }
                    task["model_source"]["path"] = templates["m_path"].format(**p)
                    task["test_dataset"]["path"] = templates["d_path"].format(**p)
                    task["test_dataset"]["scale"] = str(scale)
                    task["job_name"] = templates["job_name"].format(**p)
                    tasks.append(task)

    return tasks


# =============================================================================
# Section 6. Output Utilities
# =============================================================================

def _resolve_output_dir(global_settings: Dict[str, Any]) -> Path:
    out_dir = Path(global_settings["output_dir"]).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def export_time_scaling_outputs(raw_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Export raw and binned edge-time CSV files for time-scaling analysis.

    Note:
      The exported total time is full pipeline time:
      data load + online feature calc + prep + inference.
    """
    if not raw_rows:
        return

    raw_df = pd.DataFrame(raw_rows)

    # Keep original filename for compatibility, but content is no longer inference-only
    raw_path = out_dir / "raw_edge_time_details-only-inf.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"[INFO] Individual network details saved to: {raw_path}")
    print("[INFO] The exported time columns now include full pipeline timing and per-stage breakdown.")

    if len(raw_df) > 5:
        raw_df["edge_bin"] = pd.cut(raw_df["num_edges"], bins=20)

        time_cols = [
            c for c in [
                "time_total",
                "time_data_load",
                "time_feature_calc",
                "time_prep",
                "time_inference",
            ]
            if c in raw_df.columns
        ]

        agg_spec = {"num_edges": "mean"}
        for col in time_cols:
            agg_spec[col] = "mean"

        binned_df = (
            raw_df.groupby("edge_bin", observed=True)
            .agg(agg_spec)
            .dropna()
            .reset_index(drop=True)
        )

        bin_path = out_dir / "binned_edge_time_analysis.csv"
        binned_df.to_csv(bin_path, index=False)
        print(f"[SUCCESS] Binned edge-time analysis saved to: {bin_path}")


# =============================================================================
# Section 7. Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TCR-GIN against baselines from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_global_settings = config["base_config"]["global_settings"]
    out_dir = _resolve_output_dir(base_global_settings)
    report_path = out_dir / base_global_settings["output_filename"]

    is_time_scaling_test = bool(
        base_global_settings.get("export_edge_time_analysis", False)
        or config_path.name == "test_multisource-LBWE-CPU-time.yaml"
    )

    all_results = []
    all_raw_time_data = []

    with manage_test_data(config) as temp_data_root:
        base_config = deepcopy(config["base_config"])
        base_config["global_settings"]["datasets_root_dir"] = temp_data_root

        for exp_config in config.get("experiments", []):
            tasks = generate_tasks_from_experiment(exp_config, base_config)

            for task_config in tasks:
                models_root = Path(task_config["global_settings"]["models_root_dir"]).expanduser()
                if not models_root.is_absolute():
                    models_root = PROJECT_ROOT / models_root

                source_path = Path(task_config["model_source"]["path"])
                base_path = source_path if source_path.is_absolute() else models_root / source_path

                exp_dirs = sorted([str(p) for p in base_path.glob("exp_*") if p.is_dir()])
                if not exp_dirs and list(base_path.glob("*.pt")):
                    exp_dirs = [str(base_path)]

                if not exp_dirs:
                    print(f"[WARN] No model checkpoints found under: {base_path}")
                    continue

                for exp_dir in exp_dirs:
                    final_task = deepcopy(task_config)
                    final_task["model_source"]["path"] = exp_dir

                    res, raw_data = run_single_experiment(final_task, task_config["global_settings"])
                    if res:
                        all_results.append(res)
                        if is_time_scaling_test:
                            all_raw_time_data.extend(raw_data)

    if all_results:
        pd.DataFrame(all_results).to_csv(report_path, index=False)
        print(f"\n[SUCCESS] Main report saved to: {report_path}")
    else:
        print("\n[INFO] No valid results generated.")

    if is_time_scaling_test and all_raw_time_data:
        export_time_scaling_outputs(all_raw_time_data, out_dir)


if __name__ == "__main__":
    main()
