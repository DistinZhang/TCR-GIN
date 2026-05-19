#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/baseline_comparison/real_generalization/test_generalization_real.py

Evaluate cross-dataset generalization of TCR-GIN on real datasets.

Usage
-----
python experiments/baseline_comparison/real_generalization/test_generalization_real.py \
    --config experiments/baseline_comparison/real_generalization/configs/test_generalization_real.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
from tqdm.auto import tqdm

# =============================================================================
# Section 0. Project Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.collate")

from model.tcr_gin import TCR_GIN  

# =============================================================================
# Section 1. Constants
# =============================================================================

ALGO_FILENAME_MAPPING = {
    "CollectiveInfluenceL1": "CI-L1",
    "CollectiveInfluenceL2": "CI-L2",
    "CollectiveInfluenceL3": "CI-L3",
    "GDM": "GDM",
    "GDMR": "GDMR",
    "CoreGDM": "CoreGDM",
    "CoreHD": "CoreHD",
    "EGND": "EGND",
    "EI_s1": "EI-s1",
    "EI_s2": "EI-s2",
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

REINSERTION_TIME_MAPPING = {
    "network_entanglement_small_reinsertion": "network_entanglement_small",
    "network_entanglement_mid_reinsertion": "network_entanglement_mid",
    "network_entanglement_large_reinsertion": "network_entanglement_large",
    "GNDR": "GND",
    "vertex_entanglement_reinsertion": "vertex_entanglement",
    "GDMR": "GDM",
    "MSR": "MS",
}

# =============================================================================
# Section 2. Utility Functions
# =============================================================================


def resolve_path(path_str: str, base: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (base / p)


def parse_config_path(config_arg: str) -> Path:
    p = Path(config_arg).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def get_graph_ids(dataset_dir: Path) -> List[str]:
    if not dataset_dir.is_dir():
        return []
    graph_ids = {f.name.replace("_edges.npz", "") for f in dataset_dir.glob("*_edges.npz")}
    return sorted(graph_ids)


def load_labels(dataset_dir: Path) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    if not dataset_dir.is_dir():
        return labels

    for f in dataset_dir.glob("*_label.json"):
        try:
            network_name = f.name.replace("_label.json", "")
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            labels[network_name] = float(data["critical_threshold"])
        except Exception as e:
            print(f"  [WARN] Failed to read label file {f}: {e}")
    return labels


def normalize_algorithm_name(row: pd.Series) -> str:
    heuristic = str(row.get("heuristic", ""))
    if heuristic in {"degree", "betweenness_centrality"}:
        static_flag = str(row.get("static", "")).upper() == "TRUE"
        return f"{heuristic}_{'T' if static_flag else 'F'}"
    return heuristic


def load_baselines(baselines_dir: Path, baseline_source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load baseline threshold/time tables from:
      <baselines_dir>/<baseline_source>/results_final/*
    """
    results_path = baselines_dir / baseline_source / "results_final"
    if not results_path.is_dir():
        print(f"    [WARN] Baseline directory does not exist: {results_path}")
        return pd.DataFrame(), pd.DataFrame()

    files = sorted(results_path.glob("*.xlsx")) + sorted(results_path.glob("*.csv"))
    if not files:
        print(f"    [WARN] No baseline files found in: {results_path}")
        return pd.DataFrame(), pd.DataFrame()

    all_dfs: List[pd.DataFrame] = []

    for f in files:
        try:
            if f.suffix.lower() == ".xlsx":
                df = pd.read_excel(f, engine="openpyxl")
            else:
                df = pd.read_csv(f)

            if "heuristic" in df.columns:
                df["algorithm"] = df.apply(normalize_algorithm_name, axis=1)
            elif "algorithm" not in df.columns:
                # fallback from filename
                algo_name = f.name.replace(f"{baseline_source}-", "").replace(f.suffix, "")
                df["algorithm"] = algo_name

            if "network" not in df.columns:
                continue

            df["network"] = df["network"].astype(str)
            all_dfs.append(df)
        except Exception as e:
            print(f"    [WARN] Failed to parse baseline file {f}: {e}")

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    threshold_pivot = pd.DataFrame()
    time_pivot = pd.DataFrame()

    if "critical_threshold" in combined_df.columns:
        threshold_pivot = combined_df.pivot_table(
            index="network",
            columns="algorithm",
            values="critical_threshold",
            aggfunc="first",
        )

    if "dismantle_time" in combined_df.columns:
        time_pivot = combined_df.pivot_table(
            index="network",
            columns="algorithm",
            values="dismantle_time",
            aggfunc="first",
        )

        # reinsertion runtime = reinsertion phase + base phase (if both exist)
        for reinsertion_algo, base_algo in REINSERTION_TIME_MAPPING.items():
            if reinsertion_algo in time_pivot.columns and base_algo in time_pivot.columns:
                time_pivot[reinsertion_algo] = time_pivot[reinsertion_algo].add(
                    time_pivot[base_algo], fill_value=0
                )

    return threshold_pivot, time_pivot


def adjust_feature_dim(features: np.ndarray, feature_dim: int) -> np.ndarray:
    feat = np.asarray(features, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(f"Invalid feature shape: {feat.shape}")

    if feat.shape[1] == feature_dim:
        return feat
    if feat.shape[1] > feature_dim:
        return feat[:, :feature_dim]

    # pad with zeros if dim is smaller than expected
    pad = np.zeros((feat.shape[0], feature_dim - feat.shape[1]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=1)


def build_undirected_edge_index(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(edges)
    if arr.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = np.asarray(arr, dtype=np.int64).reshape(-1, 2)

    edge_index = torch.as_tensor(arr, dtype=torch.long, device=device).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


# =============================================================================
# Section 3. GNN Evaluation
# =============================================================================


def load_models_for_task(task_config: Dict[str, Any], model_dir: Path, device: torch.device) -> List[TCR_GIN]:
    # Search order:
    # 1) model_dir/exp_*/model_run_*.pt
    # 2) model_dir/model_run_*.pt
    # 3) model_dir/*.pt
    model_files = sorted(glob.glob(str(model_dir / "exp_*" / "model_run_*.pt")))
    if not model_files:
        model_files = sorted(glob.glob(str(model_dir / "model_run_*.pt")))
    if not model_files:
        model_files = sorted(glob.glob(str(model_dir / "*.pt")))

    models: List[TCR_GIN] = []
    for path in model_files:
        try:
            m_args = argparse.Namespace(**task_config["model_params"])
            m_args.input_dim = int(task_config["feature_dim"])

            model = TCR_GIN(m_args).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"  [WARN] Failed to load model {path}: {e}")

    return models


def run_gnn_exp(task_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run one generalization task and return:
      - Run_1 ... Run_n
      - Mean
      - Std
    """
    global_settings = task_config["global_settings"]
    model_source = task_config["model_source"]
    test_dataset = task_config["test_dataset"]

    models_root = resolve_path(global_settings["models_root_dir"], PROJECT_ROOT)
    datasets_root = resolve_path(global_settings["datasets_root_dir"], PROJECT_ROOT)

    model_dir = resolve_path(model_source["path"], models_root)
    dataset_dir = resolve_path(test_dataset["path"], datasets_root)

    print(
        f"\n- Testing model trained on '{model_source['name']}' "
        f"-> real dataset '{test_dataset['name']}'"
    )
    print(f"  model_dir  : {model_dir}")
    print(f"  dataset_dir: {dataset_dir}")

    use_cuda = global_settings.get("device", "auto") == "auto" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    models = load_models_for_task(task_config, model_dir, device)
    if not models:
        print(f"  [WARN] No valid model checkpoints found in {model_dir}.")
        return []

    labels = load_labels(dataset_dir)
    if not labels:
        print(f"  [WARN] No labels found in {dataset_dir}.")
        return []

    feature_dim = int(task_config["feature_dim"])
    sorted_items = sorted(labels.items(), key=lambda x: x[0])

    run_maes: List[float] = []

    for run_idx, model in enumerate(models, start=1):
        preds, truths = [], []

        for gid, truth in sorted_items:
            prefix = dataset_dir / gid
            edge_file = Path(f"{prefix}_edges.npz")
            feat_file = Path(f"{prefix}_features.npy")

            if not edge_file.exists() or not feat_file.exists():
                continue

            try:
                edges = np.load(edge_file, allow_pickle=True)["edges"]
                features = np.load(feat_file)

                features = adjust_feature_dim(features, feature_dim)
                if features.shape[0] == 0:
                    continue

                x = torch.as_tensor(features, dtype=torch.float32, device=device)
                edge_index = build_undirected_edge_index(edges, device)

                data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0]).to(device)
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

                with torch.no_grad():
                    y_pred = float(model(data).item())

                preds.append(y_pred)
                truths.append(float(truth))

            except Exception as e:
                print(f"  [WARN] Failed graph {gid}: {e}")

        if truths:
            mae_val = float(mean_absolute_error(truths, preds))
            run_maes.append(mae_val)

    if not run_maes:
        print("  [WARN] No valid predictions were produced.")
        return []

    results = []
    for i, mae_val in enumerate(run_maes, start=1):
        results.append({
            "Dataset": test_dataset["name"],
            "Model": model_source["name"],
            "Type": f"Run_{i}",
            "Value": mae_val,
        })

    results.append({
        "Dataset": test_dataset["name"],
        "Model": model_source["name"],
        "Type": "Mean",
        "Value": float(np.mean(run_maes)),
    })
    results.append({
        "Dataset": test_dataset["name"],
        "Model": model_source["name"],
        "Type": "Std",
        "Value": float(np.std(run_maes, ddof=0)),
    })

    return results


# =============================================================================
# Section 4. Task Generation
# =============================================================================


def generate_real_world_tasks(exp_config: Dict[str, Any], base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    print(f"\nParsing group: '{exp_config.get('name', 'Unnamed Group')}'")

    training_sources = exp_config.get("training_sources", [])
    testing_targets = exp_config.get("testing_targets", [])

    single_tpl = exp_config.get("single_source_model_path_template", "{dataset_name}")
    multi_tpl = exp_config.get("multi_source_model_path_template", "{dataset_name}")
    dataset_tpl = exp_config.get("real_dataset_path_template", "data_real/{dataset_name}")

    for model_info in training_sources:
        model_name = model_info["name"]
        constituents = model_info.get("constituents", [model_name])

        model_tpl = single_tpl if len(constituents) == 1 else multi_tpl
        model_rel_path = model_tpl.format(dataset_name=model_name)

        for target_info in testing_targets:
            target_name = target_info["name"]
            target_path = dataset_tpl.format(dataset_name=target_name)

            task = deepcopy(base_config)
            task["model_source"] = {
                "name": model_name,
                "path": model_rel_path,
            }
            task["test_dataset"] = {
                "name": target_name,
                "path": target_path,
                "baseline_source": target_info.get("baseline_source", target_name),
            }
            tasks.append(task)

    print(f"  - Generated {len(tasks)} tasks.")
    return tasks


def get_experiment_groups(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "experiment_groups" in config and isinstance(config["experiment_groups"], list):
        return config["experiment_groups"]
    if "experiment_settings" in config and isinstance(config["experiment_settings"], dict):
        return [config["experiment_settings"]]
    return []


def extract_model_dataset_order(experiment_groups: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    model_order: List[str] = []
    dataset_order: List[str] = []

    for group in experiment_groups:
        for src in group.get("training_sources", []):
            name = src.get("name")
            if name and name not in model_order:
                model_order.append(name)

        for tgt in group.get("testing_targets", []):
            name = tgt.get("name")
            if name and name not in dataset_order:
                dataset_order.append(name)

    return model_order, dataset_order


# =============================================================================
# Section 5. Baseline Evaluation
# =============================================================================


def collect_unique_targets(experiment_groups: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Collect unique testing targets (preserve insertion order), and keep:
      - dataset name
      - baseline source
      - dataset path template (from the first group where it appears)
    """
    targets: Dict[str, Dict[str, str]] = {}

    for group in experiment_groups:
        dataset_tpl = group.get("real_dataset_path_template", "data_real/{dataset_name}")
        for t in group.get("testing_targets", []):
            t_name = t.get("name")
            if not t_name:
                continue
            if t_name not in targets:
                targets[t_name] = {
                    "name": t_name,
                    "baseline_source": t.get("baseline_source", t_name),
                    "dataset_template": dataset_tpl,
                }

    return list(targets.values())


def calculate_baseline_performance(config: Dict[str, Any], experiment_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n[INFO] Calculating baseline performance...")

    base_cfg = config["base_config"]["global_settings"]
    baselines_dir = resolve_path(base_cfg["baselines_root_dir"], PROJECT_ROOT)
    datasets_root = resolve_path(base_cfg["datasets_root_dir"], PROJECT_ROOT)

    all_results: List[Dict[str, Any]] = []
    targets = collect_unique_targets(experiment_groups)

    for t in targets:
        target_name = t["name"]
        baseline_source = t["baseline_source"]
        dataset_rel = t["dataset_template"].format(dataset_name=target_name)

        print(f"\n--- Baselines for dataset: {target_name} (source={baseline_source}) ---")

        dataset_dir = resolve_path(dataset_rel, datasets_root)
        labels = load_labels(dataset_dir)
        if not labels:
            print(f"  [WARN] No labels found for {target_name}.")
            continue

        threshold_df, time_df = load_baselines(baselines_dir, baseline_source)
        if threshold_df.empty:
            print("  [WARN] No baseline threshold data.")
            continue

        labels_series = pd.Series(labels, dtype=float)
        common_networks = labels_series.index.intersection(threshold_df.index)
        if common_networks.empty:
            print("  [WARN] No overlapping network IDs between labels and baseline files.")
            continue

        for algo in threshold_df.columns:
            if algo not in ALGO_FILENAME_MAPPING:
                continue

            preds = threshold_df.loc[common_networks, algo].dropna()
            common_final = labels_series.index.intersection(preds.index)
            if common_final.empty:
                continue

            mae = float(mean_absolute_error(labels_series.loc[common_final], preds.loc[common_final]))

            avg_time = np.nan
            if not time_df.empty and algo in time_df.columns:
                times = time_df.loc[common_final, algo].dropna()
                if not times.empty:
                    avg_time = float(times.mean())

            all_results.append({
                "Algorithm": ALGO_FILENAME_MAPPING[algo],
                "Dataset": target_name,
                "MAE": mae,
                "Time": avg_time,
            })

    return all_results


# =============================================================================
# Section 6. Reporting
# =============================================================================


def build_gnn_pivot_table(
    all_gnn_results: List[Dict[str, Any]],
    model_order: List[str],
    dataset_order: List[str],
) -> Optional[pd.DataFrame]:
    if not all_gnn_results:
        return None

    df = pd.DataFrame(all_gnn_results)

    # keep stable order and append unseen values
    for m in df["Model"].dropna().unique():
        if m not in model_order:
            model_order.append(m)
    for d in df["Dataset"].dropna().unique():
        if d not in dataset_order:
            dataset_order.append(d)

    run_types = sorted(
        [x for x in df["Type"].unique() if isinstance(x, str) and x.startswith("Run_")],
        key=lambda x: int(x.split("_")[1]),
    )
    type_order = run_types + [x for x in ["Mean", "Std"] if x in df["Type"].unique()]

    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df["Type"] = pd.Categorical(df["Type"], categories=type_order, ordered=True)

    pivot_df = df.pivot_table(
        index="Dataset",
        columns=["Model", "Type"],
        values="Value",
        aggfunc="first",
    )

    # Reorder columns manually
    ordered_cols = [(m, t) for m in model_order for t in type_order if (m, t) in pivot_df.columns]
    if ordered_cols:
        pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))

    pivot_df = pivot_df.reindex(index=dataset_order)
    return pivot_df


def write_output(
    output_path: Path,
    pivot_df: Optional[pd.DataFrame],
    baseline_results: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if pivot_df is not None and not pivot_df.empty:
            pivot_df.to_csv(f)
            f.write("\n\n")
        else:
            f.write("No GNN results were generated.\n\n")

    print(f"\n[INFO] Found {len(baseline_results)} baseline rows.")

    if baseline_results:
        df_baselines = pd.DataFrame(baseline_results)
        agg_baselines = (
            df_baselines
            .groupby("Algorithm", as_index=False)
            .agg(MAE=("MAE", "mean"), Time=("Time", "mean"))
        )

        agg_baselines["MAE"] = agg_baselines["MAE"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        agg_baselines["Time"] = agg_baselines["Time"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        with open(output_path, "a", encoding="utf-8") as f:
            f.write("Baseline Algorithm Performance (Averaged across all tested real datasets)\n")
            agg_baselines.to_csv(f, index=False)


# =============================================================================
# Section 7. Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCR-GIN Real-World Generalization Testing")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("TCR-GIN Real-World Generalization Testing")
    print("=" * 70)

    config_path = parse_config_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_groups = get_experiment_groups(config)
    if not experiment_groups:
        print("[ERROR] No experiment groups found in config.")
        return

    print(f"Found {len(experiment_groups)} experiment group(s).")

    all_tasks: List[Dict[str, Any]] = []
    base_config = config["base_config"]

    for group in experiment_groups:
        all_tasks.extend(generate_real_world_tasks(group, base_config))

    if not all_tasks:
        print("[ERROR] No tasks generated from config.")
        return

    # Run GNN tasks
    all_gnn_results: List[Dict[str, Any]] = []
    for task in tqdm(all_tasks, desc="Executing GNN generalization tasks"):
        rows = run_gnn_exp(task)
        if rows:
            all_gnn_results.extend(rows)

    model_order, dataset_order = extract_model_dataset_order(experiment_groups)
    pivot_df = build_gnn_pivot_table(all_gnn_results, model_order, dataset_order)

    # Baseline comparison
    baseline_results = calculate_baseline_performance(config, experiment_groups)

    # Save report
    gs = config["base_config"]["global_settings"]
    output_dir = resolve_path(gs["output_dir"], PROJECT_ROOT)
    output_path = output_dir / gs["output_filename"]

    write_output(output_path, pivot_df, baseline_results)

    print(f"\n[SUCCESS] All jobs completed. Results saved to:\n{output_path}")
    print("\n" + "=" * 70)
    print("Test Run Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
