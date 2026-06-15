#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/utils/aggregate_remnants.py

Aggregate remnant-level results and labels from component-level precomputed
results.

What this script does
---------------------
1. Generate remnant-level label JSON files by aggregating component labels.
2. Aggregate remnant-level dismantling results from component-level XLSX files.
3. Use BCR results as a fallback when the target algorithm is missing some
   component-level entries.
4. Process different algorithms in parallel.
5. Sort the output `network` column in natural order.
6. Save final aggregated tables in `.xlsx` format.

Expected directory structure
----------------------------
Components directory:
- <components_dir>/
    - results_final/
        - *.xlsx
    - <components_dir.name>/
        - *_edges.npz
        - *_label.json

Remnants directory:
- <remnants_dir>/
    - *_edges.npz
    - (output) *_label.json
    - results_final/
        - *.xlsx

Usage
-----
From the repository root:

1) Basic usage:
    python utils/aggregate_remnants.py \
        --components-dir experiments/trajectory_analysis/data/power/power-Components \
        --remnants-dir experiments/trajectory_analysis/data/power/power-Remnants

2) Use a fixed number of CPU cores:
    python utils/aggregate_remnants.py \
        --components-dir experiments/trajectory_analysis/data/transport/transport-Components \
        --remnants-dir experiments/trajectory_analysis/data/transport/transport-Remnants \
        --cores 8

3) Another example:
    python utils/aggregate_remnants.py \
        --components-dir experiments/trajectory_analysis/data/synth-20-100/synth-20-100-Components \
        --remnants-dir experiments/trajectory_analysis/data/synth-20-100/synth-20-100-Remnants

Notes
-----
- The script reads component-level result files from:
    <components_dir>/results_final/*.xlsx
- It writes remnant-level result files to:
    <remnants_dir>/results_final/*.xlsx
- It writes remnant-level labels next to each remnant graph file.
- If BCR results are available, they are used only as fallback for missing
  component-level rows of other algorithms.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import natsort
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Algorithm mapping
# =============================================================================

ALGO_MAPPING: Dict[str, object] = {
    "CI1": "CollectiveInfluenceL1",
    "CI2": "CollectiveInfluenceL2",
    "CI3": "CollectiveInfluenceL3",
    "GDM": "GDM",
    "GDMR": "GDMR",
    "CoreGDM": "CoreGDM",
    "CoreHD": "CoreHD",
    "EGND": "EGND",
    "EIs1": "EI_s1",
    "EIs2": "EI_s2",
    "GND": "GND",
    "GNDR": "GNDR",
    "MS": "MS",
    "MSR": "MSR",
    "NES": "network_entanglement_small",
    "NESR": "network_entanglement_small_reinsertion",
    "NEM": "network_entanglement_mid",
    "NEMR": "network_entanglement_mid_reinsertion",
    "NEL": "network_entanglement_large",
    "NELR": "network_entanglement_large_reinsertion",
    "VE": "vertex_entanglement",
    "VER": "vertex_entanglement_reinsertion",
    "FINDER": "FINDER_CN",
    "DomiRank": "Domirank",
    "DC": ("degree", True),
    "DCR": ("degree", False),
    "BC": ("betweenness_centrality", True),
    "BCR": ("betweenness_centrality", False),
    "EC": ("eigenvector_centrality", True),
    "ECR": ("eigenvector_centrality", False),
}

REVERSE_ALGO_MAPPING: Dict[str, str] = {}
for short_name, mapping_val in ALGO_MAPPING.items():
    if isinstance(mapping_val, str):
        signature = mapping_val
    else:
        heuristic, static_bool = mapping_val
        signature = f"{heuristic}_{'T' if static_bool else 'F'}"
    REVERSE_ALGO_MAPPING[signature] = short_name


# =============================================================================
# Helpers
# =============================================================================

def load_graph_from_npz(file_path: Path) -> Optional[nx.Graph]:
    """Load a graph from a compressed npz file containing an edge list."""
    try:
        data = np.load(file_path, allow_pickle=False)
        edge_list = data["edges"] if "edges" in data else data[data.files[0]]
        return nx.from_edgelist(edge_list)
    except Exception as exc:
        print(f"ERROR: failed to load graph from {file_path}. Error: {exc}")
        return None


def get_algo_details(short_name: str) -> Tuple[Optional[str], Optional[bool]]:
    """Return (heuristic_name, static_flag) for an algorithm short name."""
    target = ALGO_MAPPING.get(short_name)
    if target is None:
        return None, None
    if isinstance(target, tuple):
        return target[0], target[1]
    return target, None


def read_result_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Read an XLSX result table and normalize the `network` column."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    if "network" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "network"})
        else:
            raise ValueError(f"Missing 'network' column in {xlsx_path}")
    return df


# =============================================================================
# Label aggregation
# =============================================================================

def generate_all_labels(remnant_files: List[Path], components_files_path: Path) -> None:
    """
    Generate remnant label JSON files once before parallel result aggregation.
    This avoids concurrent writes to the same label files.
    """
    print("\nGenerating remnant label files...")
    for remnant_npz_path in tqdm(remnant_files, desc="  Labels"):
        remnant_name = remnant_npz_path.name.replace("_edges.npz", "")
        component_pattern = f"{remnant_name}_*_edges.npz"
        component_npz_paths = list(components_files_path.glob(f"**/{component_pattern}"))
        if not component_npz_paths:
            continue

        weighted_ct_sum = 0.0
        total_component_nodes = 0
        first_label_data = None

        for comp_path in component_npz_paths:
            base_name = comp_path.name.replace("_edges.npz", "")
            comp_json_path = comp_path.parent / f"{base_name}_label.json"
            if not comp_json_path.exists():
                continue

            with open(comp_json_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            if first_label_data is None:
                first_label_data = label_data

            num_nodes = int(label_data.get("num_nodes", 0))
            weighted_ct_sum += float(label_data.get("critical_threshold", 0.0)) * num_nodes
            total_component_nodes += num_nodes

        if first_label_data is None or total_component_nodes <= 0:
            continue

        remnant_graph = load_graph_from_npz(remnant_npz_path)
        if remnant_graph is None:
            continue

        num_nodes = remnant_graph.number_of_nodes()
        num_edges = remnant_graph.number_of_edges()
        final_ct = weighted_ct_sum / total_component_nodes

        new_label = {
            "critical_threshold": final_ct,
            "removed_nodes": first_label_data.get("removed_nodes", []),
            "feature_names": first_label_data.get("feature_names", []),
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0,
        }

        output_base_name = remnant_npz_path.name.replace("_edges.npz", "")
        output_json_path = remnant_npz_path.parent / f"{output_base_name}_label.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(new_label, f, indent=4)


# =============================================================================
# Per-algorithm aggregation
# =============================================================================

def process_single_algorithm(
    algo_info: Tuple[str, Path],
    remnant_files: List[Path],
    components_files_path: Path,
    output_results_path: Path,
    remnants_path_name: str,
    bcr_fallback_df: Optional[pd.DataFrame],
) -> Tuple[str, bool]:
    """
    Aggregate remnant-level results for one algorithm.

    If some component-level rows are missing for the target algorithm, BCR rows
    are used as fallback when available.
    """
    algo_short, target_xlsx = algo_info

    try:
        current_algo_df = read_result_xlsx(target_xlsx)
    except Exception as exc:
        print(f"Warning: failed to read {target_xlsx.name}. Error: {exc}")
        return algo_short, False

    new_results_rows = []

    for remnant_npz_path in remnant_files:
        remnant_name = remnant_npz_path.name.replace("_edges.npz", "")
        component_pattern = f"{remnant_name}_*_edges.npz"
        component_npz_paths = list(components_files_path.glob(f"**/{component_pattern}"))
        if not component_npz_paths:
            continue

        component_names = [p.name.replace("_edges.npz", "") for p in component_npz_paths]

        found_data = current_algo_df[current_algo_df["network"].isin(component_names)].copy()
        component_data_parts = [found_data]

        if bcr_fallback_df is not None and algo_short != "BCR":
            found_networks = set(found_data["network"])
            missing_networks = [name for name in component_names if name not in found_networks]

            if missing_networks:
                fallback_data = bcr_fallback_df[bcr_fallback_df["network"].isin(missing_networks)].copy()
                if not fallback_data.empty:
                    fallback_data["heuristic"] = "fallback_from_BCR"
                    fallback_data["static"] = "N/A"
                    component_data_parts.append(fallback_data)

        component_data = pd.concat(component_data_parts, ignore_index=True)

        if component_data.empty:
            continue

        remnant_graph = load_graph_from_npz(remnant_npz_path)
        if remnant_graph is None:
            continue

        rem_num = component_data["rem_num"].sum()
        dismantle_time = component_data["dismantle_time"].sum()
        network_size = remnant_graph.number_of_nodes()
        critical_threshold = rem_num / network_size if network_size > 0 else 0.0

        heuristic_name, static_val = get_algo_details(algo_short)

        new_row = {
            "network": remnant_name,
            "heuristic": heuristic_name,
            "static": static_val,
            "rem_num": rem_num,
            "network_size": network_size,
            "critical_threshold": critical_threshold,
            "dismantle_time": dismantle_time,
        }
        new_results_rows.append(new_row)

    if not new_results_rows:
        return algo_short, False

    group_df = pd.DataFrame(new_results_rows)
    group_df = group_df.sort_values(
        by="network",
        key=natsort.natsort_keygen(),
    ).reset_index(drop=True)

    heuristic_name, static_val = get_algo_details(algo_short)
    static_suffix = f"_{'T' if static_val else 'F'}" if static_val is not None else ""
    output_xlsx_filename = f"{remnants_path_name}-{heuristic_name}{static_suffix}.xlsx"
    output_xlsx_path = output_results_path / output_xlsx_filename

    group_df.to_excel(output_xlsx_path, index=False, engine="openpyxl")
    return algo_short, True


# =============================================================================
# Main workflow
# =============================================================================

def process_remnants_dataset(components_dir: str, remnants_dir: str, num_cores: int) -> None:
    """Coordinate label generation and parallel result aggregation."""
    components_path = Path(components_dir).expanduser().resolve()
    remnants_path = Path(remnants_dir).expanduser().resolve()

    if not components_path.exists():
        raise FileNotFoundError(f"Components directory not found: {components_path}")
    if not remnants_path.exists():
        raise FileNotFoundError(f"Remnants directory not found: {remnants_path}")

    components_results_path = components_path / "results_final"
    component_dataset_name = components_path.name
    components_files_path = components_path / component_dataset_name

    output_results_path = remnants_path / "results_final"
    output_results_path.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print(f"Components Dir : {components_path}")
    print(f"Remnants Dir   : {remnants_path}")
    print(f"Output Dir     : {output_results_path}")
    print(f"Parallel Cores : {num_cores}")
    print("-" * 60)

    xlsx_files = sorted(components_results_path.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(
            f"No result files (.xlsx) found in '{components_results_path}'."
        )

    prefix_to_remove = f"{components_path.name}-"
    algo_to_file_map: Dict[str, Path] = {}

    for xlsx_file in xlsx_files:
        algo_signature = xlsx_file.stem.replace(prefix_to_remove, "")
        algo_short = REVERSE_ALGO_MAPPING.get(algo_signature)
        if algo_short:
            algo_to_file_map[algo_short] = xlsx_file
        else:
            print(
                f"Warning: could not map filename signature '{algo_signature}'. "
                f"Skipping {xlsx_file.name}"
            )

    print(f"Found {len(algo_to_file_map)} available algorithms from filenames.")

    bcr_fallback_df = None
    if "BCR" in algo_to_file_map:
        try:
            print("Loading BCR results for fallback...")
            bcr_fallback_df = read_result_xlsx(algo_to_file_map["BCR"])
        except Exception as exc:
            print(f"Warning: failed to load BCR fallback data. Fallback disabled. Error: {exc}")
            bcr_fallback_df = None
    else:
        print("Warning: BCR results not found. Fallback mechanism disabled.")

    remnant_files = sorted(
        remnants_path.rglob("*_edges.npz"),
        key=lambda p: natsort.natsort_key(str(p)),
    )
    print(f"Found {len(remnant_files)} remnant graphs.")

    generate_all_labels(remnant_files, components_files_path)

    print("\nAggregating remnant-level results in parallel...")

    worker_func = partial(
        process_single_algorithm,
        remnant_files=remnant_files,
        components_files_path=components_files_path,
        output_results_path=output_results_path,
        remnants_path_name=remnants_path.name,
        bcr_fallback_df=bcr_fallback_df,
    )

    tasks = sorted(algo_to_file_map.items(), key=lambda x: x[0])

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_func, tasks),
                total=len(tasks),
                desc="  Algorithms",
            )
        )

    success_count = sum(1 for _, ok in results if ok)
    print(
        f"\nDone. Successfully generated aggregated results for "
        f"{success_count}/{len(tasks)} algorithms."
    )


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate remnant-level results and labels from component-level "
            "precomputed outputs."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--components-dir",
        type=str,
        required=True,
        help="Path to the Components dataset directory.",
    )
    parser.add_argument(
        "--remnants-dir",
        type=str,
        required=True,
        help="Path to the Remnants dataset directory.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU cores to use. Default: all available cores.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    max_cores = os.cpu_count() or 1
    cores_to_use = min(args.cores, max_cores)
    if cores_to_use <= 0:
        cores_to_use = 1

    process_remnants_dataset(
        components_dir=args.components_dir,
        remnants_dir=args.remnants_dir,
        num_cores=cores_to_use,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
