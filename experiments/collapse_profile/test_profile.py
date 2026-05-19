#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference, evaluation, and baseline comparison for trained collapse-profile models.

Generates FIVE CSV files per dataset:
    1. aggregate_summary.csv              — per-run aggregate metrics + mean/std (+ timing)
    2. per_tau_metrics.csv                — per-run per-τ metrics + mean/std
    3. per_graph_predictions.csv          — per-run per-graph predictions
    4. baseline_comparison_per_tau.csv    — per-τ MAE: TCR-GIN vs all baselines
    5. baseline_comparison_aggregate.csv  — overall MAE: TCR-GIN vs all baselines + ranking + timing

Usage:
    python test_profile.py --config configs/test_reddit_ergm.yaml
    python test_profile.py --config configs/test_power_ergm.yaml --test_runs 1,3
    python test_profile.py --config configs/test_transport_ergm.yaml --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

for p in [str(THIS_DIR), str(THIS_DIR.parent), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

import yaml

from data_loader_profile import (
    CachedProfileGraphDataset,
    PISSGraphDataset,
    get_graph_ids,
    piss_collate,
)
from model.tcr_gin_profile import TCR_GIN_Profile
from utils_profile import (
    compute_profile_metrics,
    convert_numpy_types,
    set_seed,
    setup_logger,
)

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ──────────────────────────────────────────────────────────────
# Algorithm short names (same as plot script)
# ──────────────────────────────────────────────────────────────

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
    "betweenness_centrality_F": "BCR",
    "betweenness_centrality_dynamic": "BCR",
    "betweenness_centrality_T": "BC",
    "degree_F": "DCR",
    "degree_centrality_dynamic": "DCR",
    "degree_T": "DC",
    "degree_centrality": "DC",
    "network_entanglement_large": "NEL",
    "network_entanglement_large_reinsertion": "NELR",
    "network_entanglement_mid": "NEM",
    "network_entanglement_mid_reinsertion": "NEMR",
    "network_entanglement_small": "NES",
    "network_entanglement_small_reinsertion": "NESR",
    "vertex_entanglement": "VE",
    "vertex_entanglement_reinsertion": "VER",
}

REV_COL_MAP = {v: k for k, v in COL_MAP.items()}

# ──────────────────────────────────────────────────────────────
# Reinsertion algorithms → their base algorithms (full names)
# Time = own time + base algorithm time
# ──────────────────────────────────────────────────────────────

REINSERTION_BASE_MAP = {
    "GNDR": "GND",
    "vertex_entanglement_reinsertion": "vertex_entanglement",
    "network_entanglement_large_reinsertion": "network_entanglement_large",
    "network_entanglement_mid_reinsertion": "network_entanglement_mid",
    "network_entanglement_small_reinsertion": "network_entanglement_small",
}

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tau_cfg = cfg.pop("tau", None)
    if isinstance(tau_cfg, dict) and "values" in tau_cfg:
        cfg["tau_values"] = sorted(float(x) for x in tau_cfg["values"])
    return cfg


def build_model_args(cfg: Dict[str, Any]) -> argparse.Namespace:
    args = argparse.Namespace(
        input_dim=cfg.get("input_dim", 3),
        feature_dim=cfg.get("feature_dim", None),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 5),
        dropout=cfg.get("dropout", 0.2),
        jk_type=cfg.get("jk_type", "cat"),
        use_virtual_node=cfg.get("use_virtual_node", True),
        use_residual=cfg.get("use_residual", True),
        activation_fn=cfg.get("activation_fn", "gelu"),
        tau_values=cfg["tau_values"],
    )
    if args.feature_dim is not None and args.feature_dim > 0:
        args.input_dim = args.feature_dim
    return args


def resolve_run_ids(cfg: Dict[str, Any]) -> List[int]:
    test_runs = cfg.get("test_runs", "all")
    num_runs = int(cfg.get("num_runs", 5))
    if isinstance(test_runs, str):
        if test_runs.strip().lower() == "all":
            return list(range(1, num_runs + 1))
        return [int(x.strip()) for x in test_runs.split(",") if x.strip()]
    elif isinstance(test_runs, list):
        return [int(x) for x in test_runs]
    return [int(test_runs)]


def resolve_baseline_names(cfg: Dict[str, Any]) -> List[str]:
    raw = cfg.get("baseline_algorithms", [])
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return list(COL_MAP.keys())
    names = []
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
# Feature computation (for timing only)
# ──────────────────────────────────────────────────────────────

def _feature_dim_to_set(feature_dim: Optional[int]) -> str:
    if feature_dim is None or feature_dim <= 3:
        return "basic"
    elif feature_dim <= 5:
        return "extended"
    else:
        return "full"


def calculate_node_features(G, feature_set="basic"):
    nodes = list(G.nodes())
    features_dict = {}
    features_dict["degree"] = np.array([G.degree(u) for u in nodes])
    features_dict["clustering"] = np.array([nx.clustering(G, u) for u in nodes])
    features_dict["kcore"] = np.array([nx.core_number(G)[u] for u in nodes])

    if feature_set in ["extended", "full"]:
        features_dict["avg_neighbor_deg"] = np.array(
            [sum(G.degree(v) for v in G.neighbors(u)) / max(1, G.degree(u))
             for u in nodes]
        )
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        features_dict["pagerank"] = np.array([pr[u] for u in nodes])

    if feature_set == "full":
        try:
            bc = nx.betweenness_centrality(
                G, k=min(30, len(G)), seed=random.randint(0, 10000)
            )
            features_dict["betweenness"] = np.array([bc[u] for u in nodes])
        except Exception:
            features_dict["betweenness"] = np.zeros(len(nodes))
        try:
            ec = nx.eigenvector_centrality_numpy(G)
            features_dict["eigenvector"] = np.array([ec[u] for u in nodes])
        except Exception:
            max_deg = max(features_dict["degree"]) if len(features_dict["degree"]) > 0 else 1
            features_dict["eigenvector"] = features_dict["degree"] / max(1, max_deg)

    feature_matrix = np.column_stack([features_dict[f] for f in features_dict])
    return feature_matrix, list(features_dict.keys())


def pyg_to_networkx(data) -> nx.Graph:
    G = nx.Graph()
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else int(data.x.size(0))
    G.add_nodes_from(range(num_nodes))
    ei = data.edge_index.cpu().numpy()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u <= v:
            G.add_edge(u, v)
    return G


def time_feature_computation(
    data_list: List,
    feature_set: str,
    logger=None,
) -> Tuple[float, float, List[float]]:
    """Compute node features for every graph to measure wall-clock time.

    Returns:
        (total_seconds, avg_per_graph_seconds, [per_graph_seconds_list])
    """
    per_graph_times: List[float] = []
    total_start = time.time()

    for idx, data in enumerate(data_list):
        G = pyg_to_networkx(data)
        t0 = time.time()
        _ = calculate_node_features(G, feature_set=feature_set)
        elapsed = time.time() - t0
        per_graph_times.append(elapsed)
        if logger and (idx + 1) % 50 == 0:
            logger.info(
                f"    Feature timing: {idx + 1}/{len(data_list)} graphs done "
                f"({elapsed:.4f}s for this graph)"
            )

    total_elapsed = time.time() - total_start
    avg_elapsed = total_elapsed / len(data_list) if data_list else 0.0
    return total_elapsed, avg_elapsed, per_graph_times


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_dataset(
    data_path: str,
    tau_values: List[float],
    cache_path: str,
    feature_dim: Optional[int],
    label_suffix: str,
    label_tau_key: str,
    label_profile_key: str,
) -> Tuple[List, List[str]]:
    graph_ids = get_graph_ids(data_path, label_suffix=label_suffix)
    if not graph_ids:
        return [], []

    dataset = CachedProfileGraphDataset(
        dataset_path=data_path,
        graph_ids=graph_ids,
        tau_values=tau_values,
        cache_path=cache_path,
        rebuild_cache=True,
        feature_dim=feature_dim,
        label_suffix=label_suffix,
        label_tau_key=label_tau_key,
        label_profile_key=label_profile_key,
    )

    data_list, valid_ids = [], []
    for i, gid in enumerate(graph_ids):
        d = dataset[i]
        if d is not None:
            data_list.append(d)
            valid_ids.append(gid)
    return data_list, valid_ids


def make_dataloader(data_list, batch_size, num_workers, use_gpu):
    wrapped = PISSGraphDataset(data_list, k=0)
    kw = {"num_workers": num_workers, "pin_memory": use_gpu}
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(
        wrapped, batch_size=batch_size, shuffle=False,
        collate_fn=piss_collate, **kw,
    )


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def run_inference(model, dataloader, device, label_scale):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_data in dataloader:
            batch_anchor = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            if batch_anchor is None:
                continue
            batch_anchor = batch_anchor.to(device)
            pred_scaled = model(batch_anchor)
            pred = pred_scaled / label_scale
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_anchor.y.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


# ──────────────────────────────────────────────────────────────
# Metrics (TCR-GIN)
# ──────────────────────────────────────────────────────────────

def compute_per_graph(y_true, y_pred, tau_values, graph_ids):
    records = []
    for i, gid in enumerate(graph_ids):
        yt, yp = y_true[i], y_pred[i]
        diff = yt - yp
        rec = {
            "graph_id": gid,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_abs_error": float(np.max(np.abs(diff))),
            "mono_violation": float(np.sum(np.maximum(yp[1:] - yp[:-1], 0.0))),
        }
        if np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
            rec["corr"] = float(np.corrcoef(yt, yp)[0, 1])
        else:
            rec["corr"] = None
        for j, tau in enumerate(tau_values):
            k = f"{tau:.2f}"
            rec[f"true_{k}"] = float(yt[j])
            rec[f"pred_{k}"] = float(yp[j])
            rec[f"abs_err_{k}"] = float(abs(diff[j]))
        records.append(rec)
    return records


def compute_per_tau(y_true, y_pred, tau_values):
    diff = y_true - y_pred
    records = []
    for j, tau in enumerate(tau_values):
        records.append({
            "tau": float(tau),
            "mae": float(np.mean(np.abs(diff[:, j]))),
            "rmse": float(np.sqrt(np.mean(diff[:, j] ** 2))),
        })
    return records


# ──────────────────────────────────────────────────────────────
# Baseline loading from xlsx
# ──────────────────────────────────────────────────────────────

def _find_tau_folder(baseline_dir: str, tau: float) -> Optional[str]:
    candidates = [
        f"results_final-{tau:g}",
        f"results_final-{tau:.1f}",
        f"results_final-{tau:.2f}",
        f"results_final-{tau}",
    ]
    for c in candidates:
        p = os.path.join(baseline_dir, c)
        if os.path.isdir(p):
            return p
    return None


def _find_xlsx(tau_folder: str, alg_name: str) -> Optional[str]:
    candidates = [
        os.path.join(tau_folder, f"raw_data-{alg_name}.xlsx"),
        os.path.join(tau_folder, f"raw_data-{alg_name}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _match_network_name(df_networks: List[str], graph_id: str) -> Optional[str]:
    if graph_id in df_networks:
        return graph_id
    alt1 = graph_id.replace("-", "_")
    if alt1 in df_networks:
        return alt1
    alt2 = graph_id.replace("_", "-")
    if alt2 in df_networks:
        return alt2
    if graph_id.startswith("net_"):
        stripped = graph_id[4:]
        if stripped in df_networks:
            return stripped
        if stripped.replace("-", "_") in df_networks:
            return stripped.replace("-", "_")
    matches = [n for n in df_networks if graph_id in n or n in graph_id]
    if len(matches) == 1:
        return matches[0]
    return None


def load_baseline_predictions(
    baseline_dir: str,
    graph_ids: List[str],
    tau_values: List[float],
    alg_name: str,
) -> Optional[np.ndarray]:
    if not HAS_PANDAS:
        return None

    N = len(graph_ids)
    T = len(tau_values)
    predictions = np.full((N, T), np.nan)

    for j, tau in enumerate(tau_values):
        tau_folder = _find_tau_folder(baseline_dir, tau)
        if tau_folder is None:
            continue

        xlsx_path = _find_xlsx(tau_folder, alg_name)
        if xlsx_path is None:
            continue

        try:
            if xlsx_path.endswith(".csv"):
                df = pd.read_csv(xlsx_path)
            else:
                df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception:
            continue

        if "network" not in df.columns or "critical_threshold" not in df.columns:
            continue

        df_networks = df["network"].astype(str).tolist()

        for i, gid in enumerate(graph_ids):
            matched = _match_network_name(df_networks, gid)
            if matched is None:
                continue
            row = df[df["network"].astype(str) == matched]
            if row.empty:
                continue
            predictions[i, j] = float(row.iloc[0]["critical_threshold"])

    if np.isnan(predictions).sum() > 0.5 * N * T:
        return None

    return predictions


# ──────────────────────────────────────────────────────────────
# Baseline timing loading
# ──────────────────────────────────────────────────────────────

def load_baseline_times(
    baseline_dir: str,
    graph_ids: List[str],
    tau_values: List[float],
    alg_name: str,
) -> Optional[Dict[str, Any]]:
    """Load dismantle_time for one algorithm from ONE tau folder.

    Reads the 'dismantle_time' column from source files.
    Only reads one tau folder since timing is identical across tau values.
    """
    if not HAS_PANDAS:
        return None

    for tau in tau_values:
        tau_folder = _find_tau_folder(baseline_dir, tau)
        if tau_folder is None:
            continue

        xlsx_path = _find_xlsx(tau_folder, alg_name)
        if xlsx_path is None:
            continue

        try:
            if xlsx_path.endswith(".csv"):
                df = pd.read_csv(xlsx_path)
            else:
                df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception:
            continue

        # Read dismantle_time from the source file.
        if "network" not in df.columns or "dismantle_time" not in df.columns:
            continue

        df_networks = df["network"].astype(str).tolist()
        per_graph: Dict[str, float] = {}

        for gid in graph_ids:
            matched = _match_network_name(df_networks, gid)
            if matched is None:
                continue
            row = df[df["network"].astype(str) == matched]
            if row.empty:
                continue
            try:
                per_graph[gid] = float(row.iloc[0]["dismantle_time"])  # ★ dismantle_time
            except (ValueError, KeyError):
                continue

        if per_graph:
            total = sum(per_graph.values())
            return {
                "per_graph": per_graph,
                "total": total,
                "mean": total / len(per_graph),
                "num_found": len(per_graph),
            }

    return None




def load_all_baseline_times(
    baseline_dir: str,
    graph_ids: List[str],
    tau_values: List[float],
    algorithm_names: List[str],
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Load time for all requested baseline algorithms.

    For reinsertion algorithms (GNDR, VER, NELR, NEMR, NESR),
    their per-graph time = own time + base algorithm time.
    """
    # Step 1: load raw times for all algorithms (including bases that might not
    #         be in algorithm_names but are needed for reinsertion calculation)
    needed_bases = set()
    for alg in algorithm_names:
        if alg in REINSERTION_BASE_MAP:
            needed_bases.add(REINSERTION_BASE_MAP[alg])

    all_to_load = set(algorithm_names) | needed_bases
    raw_times: Dict[str, Dict[str, Any]] = {}

    for alg in all_to_load:
        short = COL_MAP.get(alg, alg)
        tinfo = load_baseline_times(baseline_dir, graph_ids, tau_values, alg)
        if tinfo is not None:
            raw_times[alg] = tinfo
            if logger:
                logger.info(
                    f"    Baseline raw time '{short}': "
                    f"avg={tinfo['mean']:.4f}s/graph, "
                    f"total={tinfo['total']:.4f}s, "
                    f"graphs={tinfo['num_found']}"
                )
        else:
            if logger:
                logger.warning(f"    Baseline raw time '{short}': no time data found")

    # Step 2: for reinsertion algorithms, add base algorithm's per-graph time
    result: Dict[str, Dict[str, Any]] = {}
    for alg in algorithm_names:
        short = COL_MAP.get(alg, alg)

        if alg in REINSERTION_BASE_MAP:
            base_alg = REINSERTION_BASE_MAP[alg]
            base_short = COL_MAP.get(base_alg, base_alg)

            if alg not in raw_times:
                if logger:
                    logger.warning(f"    Time '{short}': own time not available, skipped")
                continue
            if base_alg not in raw_times:
                if logger:
                    logger.warning(
                        f"    Time '{short}': base '{base_short}' time not available, "
                        f"using own time only"
                    )
                result[alg] = raw_times[alg]
                continue

            # Merge: reinsertion_time + base_time per graph
            own_pg = raw_times[alg]["per_graph"]
            base_pg = raw_times[base_alg]["per_graph"]
            merged_pg: Dict[str, float] = {}

            for gid in own_pg:
                own_t = own_pg[gid]
                base_t = base_pg.get(gid, 0.0)
                merged_pg[gid] = own_t + base_t

            if merged_pg:
                total = sum(merged_pg.values())
                merged_info = {
                    "per_graph": merged_pg,
                    "total": total,
                    "mean": total / len(merged_pg),
                    "num_found": len(merged_pg),
                }
                result[alg] = merged_info
                if logger:
                    logger.info(
                        f"    Time '{short}' = own + '{base_short}': "
                        f"avg={merged_info['mean']:.4f}s/graph, "
                        f"total={merged_info['total']:.4f}s"
                    )
        else:
            # Non-reinsertion algorithm: use raw time directly
            if alg in raw_times:
                result[alg] = raw_times[alg]
                if logger:
                    logger.info(
                        f"    Time '{short}': "
                        f"avg={raw_times[alg]['mean']:.4f}s/graph, "
                        f"graphs={raw_times[alg]['num_found']}"
                    )
            else:
                if logger:
                    logger.warning(f"    Time '{short}': no time data found")

    return result


def load_all_baseline_predictions(
    baseline_dir: str,
    graph_ids: List[str],
    tau_values: List[float],
    algorithm_names: List[str],
    logger=None,
) -> Dict[str, np.ndarray]:
    baselines = {}
    for alg in algorithm_names:
        short = COL_MAP.get(alg, alg)
        preds = load_baseline_predictions(baseline_dir, graph_ids, tau_values, alg)
        if preds is not None:
            baselines[alg] = preds
            if logger:
                n_valid = np.count_nonzero(~np.isnan(preds))
                n_total = preds.size
                logger.info(f"    Baseline '{short}' ({alg}): {n_valid}/{n_total} values loaded")
        else:
            if logger:
                logger.warning(f"    Baseline '{short}' ({alg}): insufficient data, skipped")
    return baselines


# ──────────────────────────────────────────────────────────────
# Baseline comparison metrics
# ──────────────────────────────────────────────────────────────

def compute_baseline_per_tau_metrics(
    y_true: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
    tau_values: List[float],
) -> List[Dict]:
    records = []
    for alg_name, preds in baseline_preds.items():
        short = COL_MAP.get(alg_name, alg_name)
        for j, tau in enumerate(tau_values):
            valid_mask = ~np.isnan(preds[:, j])
            if valid_mask.sum() == 0:
                continue
            diff = y_true[valid_mask, j] - preds[valid_mask, j]
            records.append({
                "algorithm": alg_name,
                "algorithm_short": short,
                "tau": float(tau),
                "mae": float(np.mean(np.abs(diff))),
                "num_valid": int(valid_mask.sum()),
            })
    return records


def compute_baseline_aggregate_metrics(
    y_true: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
) -> List[Dict]:
    records = []
    for alg_name, preds in baseline_preds.items():
        short = COL_MAP.get(alg_name, alg_name)
        valid_mask = ~np.isnan(preds)
        if valid_mask.sum() == 0:
            continue
        diff = y_true[valid_mask] - preds[valid_mask]
        records.append({
            "algorithm": alg_name,
            "algorithm_short": short,
            "mae": float(np.mean(np.abs(diff))),
            "num_valid": int(valid_mask.sum()),
        })
    return records


# ──────────────────────────────────────────────────────────────
# CSV writers (original)
# ──────────────────────────────────────────────────────────────

def write_aggregate_csv(all_run_metrics, save_path, run_times=None,
                        feature_avg_time=None, num_graphs=None):
    if not all_run_metrics:
        return
    metric_keys = [
        "profile_mae", "profile_rmse", "profile_mse",
        "profile_corr", "monotonicity_violation_mean",
    ]
    has_timing = run_times is not None
    fieldnames = ["run_id"] + metric_keys
    if has_timing:
        # Use a compact output column name.
        fieldnames += [
            "feature_time",
            "inference_time",
            "time",
        ]

    n = num_graphs if num_graphs and num_graphs > 0 else 1

    rows = []
    for run_id in sorted(all_run_metrics.keys()):
        m = all_run_metrics[run_id]
        row = {"run_id": str(run_id)}
        for k in metric_keys:
            row[k] = f"{m.get(k, 0.0):.8f}"
        if has_timing:
            inf_per_graph = run_times.get(run_id, 0.0) / n
            feat_per_graph = feature_avg_time if feature_avg_time is not None else 0.0
            row["feature_time"] = f"{feat_per_graph:.6f}"
            row["inference_time"] = f"{inf_per_graph:.6f}"
            row["time"] = f"{feat_per_graph + inf_per_graph:.6f}"
        rows.append(row)

    for stat_name, stat_fn in [("mean", np.mean), ("std", np.std)]:
        row = {"run_id": stat_name}
        for k in metric_keys:
            vals = [all_run_metrics[rid].get(k, 0.0) for rid in sorted(all_run_metrics.keys())]
            row[k] = f"{float(stat_fn(vals)):.8f}"
        if has_timing:
            feat_per_graph = feature_avg_time if feature_avg_time is not None else 0.0
            inf_per_graph_vals = [run_times.get(rid, 0.0) / n
                                  for rid in sorted(all_run_metrics.keys())]
            row["inference_time"] = f"{float(stat_fn(inf_per_graph_vals)):.6f}"
            if stat_name == "mean":
                row["feature_time"] = f"{feat_per_graph:.6f}"
                row["time"] = f"{feat_per_graph + float(np.mean(inf_per_graph_vals)):.6f}"
            else:
                row["feature_time"] = "0.000000"
                row["time"] = f"{float(np.std(inf_per_graph_vals)):.6f}"
        rows.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_tau_csv(all_run_tau, tau_values, save_path):
    fieldnames = ["run_id", "tau", "mae", "rmse"]
    rows = []
    for run_id in sorted(all_run_tau.keys()):
        for rec in all_run_tau[run_id]:
            rows.append({
                "run_id": str(run_id),
                "tau": f"{rec['tau']:.2f}",
                "mae": f"{rec['mae']:.8f}",
                "rmse": f"{rec['rmse']:.8f}",
            })
    for stat_name, stat_fn in [("mean", np.mean), ("std", np.std)]:
        for j, tau in enumerate(tau_values):
            mae_vals = [all_run_tau[rid][j]["mae"] for rid in sorted(all_run_tau.keys())]
            rmse_vals = [all_run_tau[rid][j]["rmse"] for rid in sorted(all_run_tau.keys())]
            rows.append({
                "run_id": stat_name,
                "tau": f"{tau:.2f}",
                "mae": f"{float(stat_fn(mae_vals)):.8f}",
                "rmse": f"{float(stat_fn(rmse_vals)):.8f}",
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_graph_csv(all_run_graphs, tau_values, save_path):
    tau_strs = [f"{t:.2f}" for t in tau_values]
    fieldnames = ["run_id", "graph_id", "mae", "rmse", "max_abs_error", "corr", "mono_violation"]
    for t in tau_strs:
        fieldnames.append(f"true_{t}")
    for t in tau_strs:
        fieldnames.append(f"pred_{t}")
    for t in tau_strs:
        fieldnames.append(f"abs_err_{t}")

    rows = []
    for run_id in sorted(all_run_graphs.keys()):
        for rec in all_run_graphs[run_id]:
            row = {
                "run_id": str(run_id),
                "graph_id": rec["graph_id"],
                "mae": f"{rec['mae']:.8f}",
                "rmse": f"{rec['rmse']:.8f}",
                "max_abs_error": f"{rec['max_abs_error']:.8f}",
                "corr": f"{rec['corr']:.8f}" if rec["corr"] is not None else "",
                "mono_violation": f"{rec['mono_violation']:.8f}",
            }
            for t in tau_strs:
                row[f"true_{t}"] = f"{rec[f'true_{t}']:.8f}"
                row[f"pred_{t}"] = f"{rec[f'pred_{t}']:.8f}"
                row[f"abs_err_{t}"] = f"{rec[f'abs_err_{t}']:.8f}"
            rows.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────
# CSV writers (baseline comparison)
# ──────────────────────────────────────────────────────────────

def write_baseline_per_tau_csv(
    tcr_gin_per_tau_mean: List[Dict],
    baseline_per_tau: List[Dict],
    tau_values: List[float],
    save_path: str,
):
    """Write per-tau comparison CSV — MAE only (no RMSE)."""
    tau_data = {}
    for tau in tau_values:
        tau_key = f"{tau:.2f}"
        tau_data[tau_key] = {}

    for rec in tcr_gin_per_tau_mean:
        tau_key = f"{rec['tau']:.2f}"
        tau_data[tau_key]["TCR-GIN"] = {"mae": rec["mae"]}

    alg_shorts_seen = set()
    for rec in baseline_per_tau:
        tau_key = f"{rec['tau']:.2f}"
        short = rec["algorithm_short"]
        tau_data[tau_key][short] = {"mae": rec["mae"]}
        alg_shorts_seen.add(short)

    alg_order = ["TCR-GIN"] + sorted(alg_shorts_seen)

    fieldnames = ["tau"]
    for alg in alg_order:
        fieldnames.append(f"{alg}_mae")
    fieldnames.extend(["best_algorithm", "best_mae", "tcr_gin_rank"])

    rows = []
    for tau in tau_values:
        tau_key = f"{tau:.2f}"
        row = {"tau": tau_key}
        mae_list = []

        for alg in alg_order:
            if alg in tau_data[tau_key]:
                m = tau_data[tau_key][alg]
                row[f"{alg}_mae"] = f"{m['mae']:.8f}"
                mae_list.append((alg, m["mae"]))
            else:
                row[f"{alg}_mae"] = ""

        if mae_list:
            mae_list.sort(key=lambda x: x[1])
            row["best_algorithm"] = mae_list[0][0]
            row["best_mae"] = f"{mae_list[0][1]:.8f}"
            tcr_rank = next((i + 1 for i, (a, _) in enumerate(mae_list) if a == "TCR-GIN"), "")
            row["tcr_gin_rank"] = str(tcr_rank)
        else:
            row["best_algorithm"] = ""
            row["best_mae"] = ""
            row["tcr_gin_rank"] = ""

        rows.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_aggregate_csv(
    tcr_gin_agg_mean: Dict,
    baseline_agg: List[Dict],
    save_path: str,
    tcr_gin_avg_time_per_graph: Optional[float] = None,
    baseline_times: Optional[Dict[str, Dict]] = None,
):
    all_entries = []

    all_entries.append({
        "algorithm": "TCR-GIN",
        "algorithm_short": "TCR-GIN",
        "mae": tcr_gin_agg_mean.get("profile_mae", 0.0),
        "num_valid": "all",
        "time": tcr_gin_avg_time_per_graph,
    })

    for rec in baseline_agg:
        alg = rec["algorithm"]
        avg_t = None
        if baseline_times and alg in baseline_times:
            avg_t = baseline_times[alg]["mean"]
        all_entries.append({
            "algorithm": rec["algorithm"],
            "algorithm_short": rec["algorithm_short"],
            "mae": rec["mae"],
            "num_valid": rec["num_valid"],
            "time": avg_t,
        })

    all_entries.sort(key=lambda x: x["mae"])

    # Use a compact output column name.
    fieldnames = [
        "rank", "algorithm", "algorithm_short", "mae", "num_valid",
        "time", "speedup_vs_tcr_gin",
    ]

    tcr_avg_t = tcr_gin_avg_time_per_graph

    rows = []
    for rank, entry in enumerate(all_entries, 1):
        t = entry["time"]

        if t is not None and tcr_avg_t is not None and tcr_avg_t > 1e-12:
            speedup = t / tcr_avg_t
            speedup_str = f"{speedup:.2f}x"
        elif entry["algorithm_short"] == "TCR-GIN":
            speedup_str = "1.00x"
        else:
            speedup_str = ""

        rows.append({
            "rank": str(rank),
            "algorithm": entry["algorithm"],
            "algorithm_short": entry["algorithm_short"],
            "mae": f"{entry['mae']:.8f}",
            "num_valid": str(entry["num_valid"]),
            "time": f"{t:.6f}" if t is not None else "",
            "speedup_vs_tcr_gin": speedup_str,
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test collapse-profile model on a single dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_runs", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    if cli.test_runs is not None:
        cfg["test_runs"] = cli.test_runs
    if cli.device is not None:
        cfg["device"] = cli.device
    if cli.batch_size is not None:
        cfg["batch_size"] = cli.batch_size

    experiment_name = cfg["experiment_name"]
    dataset_name = cfg["dataset_name"]
    model_dir = os.path.join(cfg["model_dir"], experiment_name)
    result_dir = cfg.get("result_dir", "experiments/collapse_profile/results")
    ds_metric_dir = os.path.join(result_dir, dataset_name, "metrics")
    tau_values = cfg["tau_values"]
    label_scale = float(cfg.get("label_scale", 100.0))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))
    feature_dim = cfg.get("feature_dim", None)
    label_suffix = cfg.get("label_suffix", "_profile_label.json")
    label_tau_key = cfg.get("label_tau_key", "tau_grid_full")
    label_profile_key = cfg.get("label_profile_key", "collapse_profile_full")
    cache_base = cfg.get("cache_path", "experiments/collapse_profile/cache/test")

    baseline_dir = cfg.get("baseline_dir", None)
    baseline_alg_names = resolve_baseline_names(cfg) if baseline_dir else []

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    run_ids = resolve_run_ids(cfg)
    model_args = build_model_args(cfg)

    set_seed(seed)

    log_dir = os.path.join(result_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Test path: {cfg['test_path']}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Runs: {run_ids}")
    logger.info(f"Tau: {tau_values}")
    logger.info(f"Device: {device}")
    if baseline_dir:
        logger.info(f"Baseline dir: {baseline_dir}")
        logger.info(f"Baselines: {[COL_MAP.get(a, a) for a in baseline_alg_names]}")

    # ── Load models ──
    models: Dict[int, TCR_GIN_Profile] = {}
    for run_id in run_ids:
        mp = os.path.join(model_dir, f"model_run_{run_id}.pt")
        if not os.path.exists(mp):
            logger.warning(f"Model not found: {mp}; skipping run {run_id}")
            continue
        model = TCR_GIN_Profile(model_args).to(device)
        model.load_state_dict(torch.load(mp, map_location=device))
        model.eval()
        models[run_id] = model
        logger.info(f"Loaded run {run_id}: {mp}")

    if not models:
        logger.error("No models loaded. Exiting.")
        return

    # ── Load data ──
    ds_cache = os.path.join(cache_base, dataset_name)
    data_list, valid_ids = load_dataset(
        data_path=cfg["test_path"],
        tau_values=tau_values,
        cache_path=ds_cache,
        feature_dim=feature_dim,
        label_suffix=label_suffix,
        label_tau_key=label_tau_key,
        label_profile_key=label_profile_key,
    )
    if not data_list:
        logger.error(f"No valid graphs in {cfg['test_path']}; exiting.")
        return

    num_graphs = len(valid_ids)
    logger.info(f"Loaded {num_graphs} graphs: {valid_ids}")
    dataloader = make_dataloader(data_list, batch_size, num_workers, device.type == "cuda")

    # ── Feature computation timing ──
    feature_set = _feature_dim_to_set(feature_dim)
    logger.info(f"  ── Timing feature computation (feature_set='{feature_set}', "
                f"feature_dim={feature_dim}) ──")
    feature_total_time, feature_avg_time, feature_per_graph_times = time_feature_computation(
        data_list, feature_set=feature_set, logger=logger
    )
    logger.info(
        f"  Feature computation: total={feature_total_time:.4f}s, "
        f"avg={feature_avg_time:.6f}s/graph (×{num_graphs} graphs)"
    )

    # ── Per-run inference ──
    all_run_agg: Dict[int, Dict] = {}
    all_run_tau: Dict[int, List] = {}
    all_run_graphs: Dict[int, List] = {}
    all_run_preds: Dict[int, np.ndarray] = {}
    all_run_times: Dict[int, float] = {}  # total inference time per run (all graphs)

    y_true_global = None

    for run_id, model in models.items():
        logger.info(f"  ── Run {run_id} ──")
        t0 = time.time()
        y_pred, y_true = run_inference(model, dataloader, device, label_scale)
        elapsed = time.time() - t0
        all_run_times[run_id] = elapsed
        inf_per_graph = elapsed / num_graphs
        total_per_graph = feature_avg_time + inf_per_graph
        logger.info(
            f"    Inference: {y_pred.shape[0]} graphs × {y_pred.shape[1]} τ, "
            f"total={elapsed:.4f}s, avg={inf_per_graph:.6f}s/graph"
        )
        logger.info(
            f"    TCR-GIN avg per graph: feat={feature_avg_time:.6f}s + "
            f"inf={inf_per_graph:.6f}s = {total_per_graph:.6f}s"
        )

        if y_true_global is None:
            y_true_global = y_true

        all_run_preds[run_id] = y_pred

        metrics = compute_profile_metrics(
            torch.tensor(y_true), torch.tensor(y_pred), tau_values
        )
        all_run_agg[run_id] = metrics
        logger.info(
            f"    MAE={metrics['profile_mae']:.6f}  "
            f"RMSE={metrics['profile_rmse']:.6f}  "
            f"Corr={metrics['profile_corr']:.6f}  "
            f"MonoViol={metrics['monotonicity_violation_mean']:.6f}"
        )
        all_run_tau[run_id] = compute_per_tau(y_true, y_pred, tau_values)
        all_run_graphs[run_id] = compute_per_graph(y_true, y_pred, tau_values, valid_ids)

    # ── Write original CSVs ──
    write_aggregate_csv(
        all_run_agg,
        os.path.join(ds_metric_dir, "aggregate_summary.csv"),
        run_times=all_run_times,
        feature_avg_time=feature_avg_time,
        num_graphs=num_graphs,
    )
    logger.info(f"  [✓] aggregate_summary.csv")

    write_per_tau_csv(all_run_tau, tau_values, os.path.join(ds_metric_dir, "per_tau_metrics.csv"))
    logger.info(f"  [✓] per_tau_metrics.csv")

    write_per_graph_csv(all_run_graphs, tau_values, os.path.join(ds_metric_dir, "per_graph_predictions.csv"))
    logger.info(f"  [✓] per_graph_predictions.csv")

    # ── Cross-run summary ──
    if len(models) > 1:
        logger.info(f"  ── Cross-run summary ({dataset_name}) ──")
        for k in ["profile_mae", "profile_rmse", "profile_corr"]:
            vals = [all_run_agg[rid][k] for rid in sorted(all_run_agg.keys())]
            logger.info(f"    {k}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")
        inf_per_graph_vals = [all_run_times[rid] / num_graphs for rid in sorted(all_run_times.keys())]
        mean_inf_pg = float(np.mean(inf_per_graph_vals))
        logger.info(f"    avg_feature_time/graph:   {feature_avg_time:.6f}s")
        logger.info(f"    avg_inference_time/graph:  {mean_inf_pg:.6f} ± {float(np.std(inf_per_graph_vals)):.6f}s")
        logger.info(f"    avg_total_time/graph:      {feature_avg_time + mean_inf_pg:.6f}s")

    # ══════════════════════════════════════════════════════════
    # Baseline comparison
    # ══════════════════════════════════════════════════════════

    if baseline_dir and baseline_alg_names and y_true_global is not None:
        logger.info(f"  ── Loading baselines ({dataset_name}) ──")

        if not HAS_PANDAS:
            logger.error("  pandas not installed! pip install pandas openpyxl")
        else:
            baseline_preds = load_all_baseline_predictions(
                baseline_dir=baseline_dir,
                graph_ids=valid_ids,
                tau_values=tau_values,
                algorithm_names=baseline_alg_names,
                logger=logger,
            )

            logger.info(f"  ── Loading baseline timing ──")
            baseline_times = load_all_baseline_times(
                baseline_dir=baseline_dir,
                graph_ids=valid_ids,
                tau_values=tau_values,
                algorithm_names=baseline_alg_names,
                logger=logger,
            )

            if baseline_preds:
                logger.info(f"  ── Computing baseline comparison ──")

                # TCR-GIN mean predictions across runs
                pred_stack = np.stack([all_run_preds[rid] for rid in sorted(all_run_preds.keys())], axis=0)
                y_pred_mean = np.mean(pred_stack, axis=0)

                tcr_gin_per_tau_mean = compute_per_tau(y_true_global, y_pred_mean, tau_values)

                tcr_gin_agg_mean = {
                    "profile_mae": float(np.mean(np.abs(y_true_global - y_pred_mean))),
                }

                # ★ TCR-GIN avg time per graph = feature_avg + inference_avg
                mean_inf_total = float(np.mean([all_run_times[rid] for rid in sorted(all_run_times.keys())]))
                mean_inf_per_graph = mean_inf_total / num_graphs
                tcr_gin_avg_time_per_graph = feature_avg_time + mean_inf_per_graph

                logger.info(
                    f"    TCR-GIN avg time/graph: "
                    f"feat={feature_avg_time:.6f}s + inf={mean_inf_per_graph:.6f}s "
                    f"= {tcr_gin_avg_time_per_graph:.6f}s"
                )

                baseline_per_tau = compute_baseline_per_tau_metrics(
                    y_true_global, baseline_preds, tau_values
                )

                baseline_agg = compute_baseline_aggregate_metrics(
                    y_true_global, baseline_preds
                )

                # --- Write comparison CSVs ---
                write_baseline_per_tau_csv(
                    tcr_gin_per_tau_mean,
                    baseline_per_tau,
                    tau_values,
                    os.path.join(ds_metric_dir, "baseline_comparison_per_tau.csv"),
                )
                logger.info(f"  [✓] baseline_comparison_per_tau.csv")

                write_baseline_aggregate_csv(
                    tcr_gin_agg_mean,
                    baseline_agg,
                    os.path.join(ds_metric_dir, "baseline_comparison_aggregate.csv"),
                    tcr_gin_avg_time_per_graph=tcr_gin_avg_time_per_graph,
                    baseline_times=baseline_times,
                )
                logger.info(f"  [✓] baseline_comparison_aggregate.csv")

                # --- Log summary ---
                logger.info(f"  ── Baseline comparison summary ──")
                logger.info(
                    f"    TCR-GIN: MAE={tcr_gin_agg_mean['profile_mae']:.6f}, "
                    f"avg_time/graph={tcr_gin_avg_time_per_graph:.6f}s"
                )

                sorted_baselines = sorted(baseline_agg, key=lambda x: x["mae"])
                for rec in sorted_baselines:
                    short = rec["algorithm_short"]
                    alg_full = rec["algorithm"]
                    diff = rec["mae"] - tcr_gin_agg_mean["profile_mae"]
                    marker = "✓ TCR-GIN better" if diff > 0 else "✗ TCR-GIN worse"
                    time_str = ""
                    if alg_full in baseline_times:
                        bt_avg = baseline_times[alg_full]["mean"]
                        speedup = bt_avg / tcr_gin_avg_time_per_graph if tcr_gin_avg_time_per_graph > 1e-12 else 0
                        if speedup >= 1:
                            time_str = f", avg_time/graph={bt_avg:.6f}s ({speedup:.1f}x slower than TCR-GIN)"
                        else:
                            time_str = f", avg_time/graph={bt_avg:.6f}s ({1/speedup:.1f}x faster than TCR-GIN)"
                    logger.info(
                        f"    {short:12s}: MAE={rec['mae']:.6f}  "
                        f"(Δ={diff:+.6f}, {marker}){time_str}"
                    )

                # TCR-GIN rank per tau
                logger.info(f"  ── TCR-GIN rank per τ ──")
                for j, tau in enumerate(tau_values):
                    tcr_mae = tcr_gin_per_tau_mean[j]["mae"]
                    all_maes = [(rec["algorithm_short"], rec["mae"])
                               for rec in baseline_per_tau if abs(rec["tau"] - tau) < 0.001]
                    all_maes.append(("TCR-GIN", tcr_mae))
                    all_maes.sort(key=lambda x: x[1])
                    rank = next(i + 1 for i, (a, _) in enumerate(all_maes) if a == "TCR-GIN")
                    total = len(all_maes)
                    best = all_maes[0][0]
                    logger.info(f"    τ={tau:.2f}: TCR-GIN rank {rank}/{total}, best={best}")

                # Timing ranking
                logger.info(f"  ── Timing ranking (avg per graph) ──")
                time_ranking = [("TCR-GIN", tcr_gin_avg_time_per_graph)]
                for alg_full, tinfo in baseline_times.items():
                    short = COL_MAP.get(alg_full, alg_full)
                    time_ranking.append((short, tinfo["mean"]))
                time_ranking.sort(key=lambda x: x[1])
                for i, (name, t) in enumerate(time_ranking, 1):
                    logger.info(f"    #{i} {name:12s}: {t:.6f}s/graph")

            else:
                logger.warning(f"  No baselines loaded successfully; skipping comparison.")
    elif baseline_dir and not HAS_PANDAS:
        logger.warning(f"  Baseline comparison skipped: pip install pandas openpyxl")

    logger.info(f"Results saved to: {ds_metric_dir}")
    logger.info("Done.\n")


if __name__ == "__main__":
    main()
