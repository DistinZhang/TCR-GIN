#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/utils/process_network.py

Utility script for generating remnant-network and connected-component snapshots
from dismantling results.

This script reads a dataset directory containing:
1. graph files stored as `*_edges.npz`, and
2. one or more CSV/TSV result files containing node-removal sequences produced
   by dismantling algorithms.

For each graph and each selected algorithm, the script simulates the
dismantling process step by step and periodically saves:
- the remnant graph after filtering out small connected components,
- each retained connected component as an individual subgraph,
- a node-feature matrix for every saved graph.

Main features
-------------
1. Parallel processing across algorithm tasks.
2. Dynamic node-feature computation during dismantling.
3. Interval-based snapshot sampling via `--interval`.
4. Optional LCC-triggered saving via `--lcc-threshold`.
5. Automatic self-loop removal before simulation.
6. Reindexed edge lists and aligned feature matrices for downstream GNN use.
7. Robust CSV loading and removal-sequence parsing.

Expected inputs
---------------
Inside the target directory:
- `*_edges.npz` graph files
- one or more `.csv` files containing at least:
  - `heuristic`
  - `removals`
  - optionally `static`

Outputs
-------
Two subdirectories will be created under the dataset directory:
- `<dataset>-Remnants`
- `<dataset>-Components`

Saved files follow the pattern:
- remnant graph:
  `<net_name>-<algo>_<step>_edges.npz`
  `<net_name>-<algo>_<step>_features.npy`
- connected component:
  `<net_name>-<algo>_<step>_<component_id>_edges.npz`
  `<net_name>-<algo>_<step>_<component_id>_features.npy`

Usage
-----
Example:
    python utils/process_network.py \
        --dir experiments/trajectory_analysis/data/power \
        --algos CI1 CI2 CI3 GDM CoreHD FINDER DomiRank BC BCR DC DCR \
        --random-runs 2 \
        --workers 32 \
        --size 4 \
        --interval 10 \
        --lcc-threshold 2000

Another example:
    python utils/process_network.py \
        --dir experiments/trajectory_analysis/data/transport \
        --algos CI1 CI2 CI3 GDM GDMR CoreGDM CoreHD EGND FINDER DomiRank \
                DC DCR BC BCR NES NESR NEM NEMR NEL NELR VE VER \
        --random-runs 2 \
        --workers 32 \
        --size 4 \
        --interval 5 \
        --lcc-threshold 2000

Notes
-----
- The script preserves the original behavior of selecting one representative
  row for each `(heuristic, static)` group:
  - use the second row if available,
  - otherwise use the first row.
- Small connected components are filtered using `number_of_nodes() > min_size`,
  matching the original implementation.
"""

from __future__ import annotations

import argparse
import ast
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

# =============================================================================
# Algorithm name mapping
# =============================================================================

ALGO_MAPPING: Dict[tuple[str, bool], str] = {
    ("degree", True): "DC",
    ("degree", False): "DCR",
    ("betweenness_centrality", True): "BC",
    ("betweenness_centrality", False): "BCR",
    ("eigenvector_centrality", True): "EC",
    ("eigenvector_centrality", False): "ECR",
    ("CollectiveInfluenceL1", True): "CI1",
    ("CollectiveInfluenceL2", True): "CI2",
    ("CollectiveInfluenceL3", True): "CI3",
    ("GDM", True): "GDM",
    ("CoreHD", True): "CoreHD",
    ("Domirank", True): "DomiRank",
    ("CollectiveInfluenceL1", False): "CI1",
    ("CollectiveInfluenceL2", False): "CI2",
    ("CollectiveInfluenceL3", False): "CI3",
}

SIMPLE_NAME_MAP: Dict[str, str] = {
    "network_entanglement_small": "NES",
    "network_entanglement_small_reinsertion": "NESR",
    "network_entanglement_mid": "NEM",
    "network_entanglement_mid_reinsertion": "NEMR",
    "network_entanglement_large": "NEL",
    "network_entanglement_large_reinsertion": "NELR",
    "vertex_entanglement": "VE",
    "vertex_entanglement_reinsertion": "VER",
    "FINDER_CN": "FINDER",
    "GDM": "GDM",
    "GDMR": "GDMR",
}

FEATURE_NAMES: List[str] = [
    "degree",
    "clustering",
    "core_number",
    "average_neighbor_degree",
    "pagerank",
    "betweenness_centrality",
    "eigenvector_centrality",
]


def normalize_static_col(value: Any) -> bool:
    """Convert a `static` column value to boolean."""
    s = str(value).strip().lower()
    return s in {"true", "1", "t", "yes"}


def load_all_results(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV/TSV result files in a directory."""
    all_dfs: List[pd.DataFrame] = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, sep="\t")
            if df.shape[1] < 2:
                df = pd.read_csv(csv_file, sep=",")
            all_dfs.append(df)
        except Exception as exc:
            print(f"Warning: could not read {csv_file}: {exc}")

    if not all_dfs:
        return pd.DataFrame()

    results_df = pd.concat(all_dfs, ignore_index=True)
    results_df.columns = [str(c).strip().lower() for c in results_df.columns]
    return results_df


def load_graph_from_npz(file_path: Path) -> nx.Graph:
    """
    Load a graph from:
    1. a scipy sparse adjacency matrix saved as .npz, or
    2. a compressed npz containing an `edges` array.
    """
    try:
        adj = sp.load_npz(file_path)
        return nx.from_scipy_sparse_array(adj)
    except Exception:
        data = np.load(file_path, allow_pickle=False)
        if "edges" not in data:
            raise ValueError(f"Missing 'edges' array in {file_path}")
        return nx.from_edgelist(data["edges"])


def parse_removals(removals_value: Any) -> Optional[List[int]]:
    """
    Parse a removal sequence from a CSV field.

    Supported examples:
    - [1, 2, 3]
    - [(0, 12), (1, 8), ...]
    - strings that contain tuple-like '(step, node)' patterns
    """
    if pd.isna(removals_value):
        return None

    s = str(removals_value).strip().replace("nan", "None").replace("inf", "'inf'")

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and parsed:
            result: List[int] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    result.append(int(item[1]))
                else:
                    result.append(int(item))
            return result
    except Exception:
        pass

    matches = re.findall(r"\(\s*\d+\s*,\s*(\d+)", s)
    if matches:
        return [int(x) for x in matches]

    return None


def choose_short_algorithm_name(algo_name: str, is_static: bool) -> str:
    """Map raw heuristic names to exported short names."""
    short_name = ALGO_MAPPING.get((algo_name, is_static))
    if short_name:
        return short_name

    short_name = SIMPLE_NAME_MAP.get(algo_name)
    if short_name:
        return short_name

    return algo_name if is_static else f"{algo_name}_R"


def calculate_features(graph: nx.Graph) -> np.ndarray:
    """Compute node-level structural features for a graph."""
    if graph.number_of_nodes() == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    nodes = sorted(graph.nodes())

    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    core_number = nx.core_number(graph)
    avg_neighbor_degree = nx.average_neighbor_degree(graph)
    pagerank = nx.pagerank(graph, alpha=0.85)
    betweenness = nx.betweenness_centrality(graph)

    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-5)
    except Exception:
        eigenvector = {n: 0.0 for n in nodes}

    return np.array(
        [
            [
                degree.get(n, 0.0),
                clustering.get(n, 0.0),
                core_number.get(n, 0.0),
                avg_neighbor_degree.get(n, 0.0),
                pagerank.get(n, 0.0),
                betweenness.get(n, 0.0),
                eigenvector.get(n, 0.0),
            ]
            for n in nodes
        ],
        dtype=np.float32,
    )


def save_subgraph(subgraph: nx.Graph, path_prefix: Path, features: Optional[np.ndarray]) -> None:
    """Save a graph as a reindexed edge list and optional feature matrix."""
    if subgraph.number_of_nodes() == 0:
        return

    nodes = sorted(subgraph.nodes())
    mapping = {old: new for new, old in enumerate(nodes)}
    edges = np.array([(mapping[u], mapping[v]) for u, v in subgraph.edges()], dtype=np.int64)

    if edges.size == 0:
        edges = edges.reshape(0, 2)

    np.savez_compressed(f"{path_prefix}_edges.npz", edges=edges)

    if features is not None:
        np.save(f"{path_prefix}_features.npy", features)


def save_feature_metadata(output_dir: Path) -> None:
    """Save feature-name metadata once per output directory."""
    metadata_path = output_dir / "feature_names.txt"
    if not metadata_path.exists():
        metadata_path.write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")


def should_start_saving(graph: nx.Graph, lcc_threshold: Optional[int]) -> bool:
    """Decide whether snapshot saving should already be active."""
    if lcc_threshold is None:
        return True
    if graph.number_of_nodes() == 0:
        return False
    largest_cc = len(max(nx.connected_components(graph), key=len))
    return largest_cc < lcc_threshold


def build_random_sequence(graph: nx.Graph) -> np.ndarray:
    """Generate a random node-removal order."""
    return np.random.default_rng().permutation(list(graph.nodes()))


def worker_task(
    task: Dict[str, Any],
    graph_init: nx.Graph,
    remnants_dir: Path,
    components_dir: Path,
    min_size: int,
    interval: int,
    lcc_threshold: Optional[int],
) -> str:
    """Process one algorithm on one network."""
    algo = task["algo"]
    net_name = task["net_name"]

    if algo.startswith("R") and algo[1:].isdigit():
        sequence = build_random_sequence(graph_init)
    else:
        sequence = parse_removals(task["removals"])
        if sequence is None:
            return f"Skipped {algo} on {net_name}: could not parse removals"

    graph = graph_init.copy()
    saving_enabled = should_start_saving(graph, lcc_threshold)

    if saving_enabled:
        initial_features = calculate_features(graph)
        save_subgraph(graph, remnants_dir / f"{net_name}-{algo}_0", initial_features)
        save_subgraph(graph, components_dir / f"{net_name}-{algo}_0_1", initial_features)

    total_steps = len(sequence)

    for step, node in enumerate(tqdm(sequence, desc=f"{algo} on {net_name}", leave=False), start=1):
        if graph.has_node(node):
            graph.remove_node(node)

        if not saving_enabled and graph.number_of_nodes() > 0 and lcc_threshold is not None:
            if should_start_saving(graph, lcc_threshold):
                saving_enabled = True

        if saving_enabled and (step % interval == 0 or step == total_steps):
            if graph.number_of_nodes() == 0:
                break

            components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
            valid_components = [c for c in components if c.number_of_nodes() > min_size]

            if valid_components:
                remnant = nx.disjoint_union_all(valid_components)
                save_subgraph(
                    remnant,
                    remnants_dir / f"{net_name}-{algo}_{step}",
                    calculate_features(remnant),
                )

                for comp_idx, comp in enumerate(valid_components, start=1):
                    save_subgraph(
                        comp,
                        components_dir / f"{net_name}-{algo}_{step}_{comp_idx}",
                        calculate_features(comp),
                    )

        if graph.number_of_nodes() == 0:
            break

    return f"Done {algo} on {net_name}"


def build_task_templates(
    results_df: pd.DataFrame,
    specified_algos: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build per-algorithm task templates from CSV results."""
    task_templates: List[Dict[str, Any]] = []

    if results_df.empty:
        return task_templates

    if "heuristic" not in results_df.columns or "removals" not in results_df.columns:
        print("Warning: result CSVs must contain at least 'heuristic' and 'removals' columns.")
        return task_templates

    if "static" not in results_df.columns:
        results_df["static"] = True

    results_df["static_bool"] = results_df["static"].apply(normalize_static_col)
    grouped = results_df.groupby(["heuristic", "static_bool"], dropna=False)

    print(f"[CSV Analysis] Found {len(grouped)} unique algorithm types (heuristic + static).")

    for (algo_name, is_static), group in grouped:
        target_row = group.iloc[1] if len(group) >= 2 else group.iloc[0]
        short_name = choose_short_algorithm_name(str(algo_name), bool(is_static))

        if specified_algos:
            if short_name not in specified_algos and str(algo_name) not in specified_algos:
                continue

        task_templates.append(
            {
                "algo": short_name,
                "removals": target_row["removals"],
            }
        )

    return task_templates


def process_dataset(
    data_dir: str,
    min_size: int,
    workers: Optional[int],
    specified_algos: Optional[List[str]],
    random_runs: int,
    interval: int,
    lcc_threshold: Optional[int],
) -> None:
    data_path = Path(data_dir).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_path}")
    if interval <= 0:
        raise ValueError("--interval must be a positive integer")

    results_df = load_all_results(data_path)
    graph_files = sorted(data_path.glob("*_edges.npz"))

    if not graph_files:
        print("No '*_edges.npz' files found.")
        return

    remnants_dir = data_path / f"{data_path.name}-Remnants"
    components_dir = data_path / f"{data_path.name}-Components"
    remnants_dir.mkdir(exist_ok=True)
    components_dir.mkdir(exist_ok=True)

    save_feature_metadata(remnants_dir)
    save_feature_metadata(components_dir)

    task_templates = build_task_templates(results_df, specified_algos)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for graph_file in graph_files:
            net_name = graph_file.name.replace("_edges.npz", "")
            print(f"\n[Processing File] {net_name}")

            try:
                graph = load_graph_from_npz(graph_file)
                graph.remove_edges_from(nx.selfloop_edges(graph))
            except Exception as exc:
                print(f"Error loading {graph_file}: {exc}")
                continue

            final_tasks: List[Dict[str, Any]] = []

            for template in task_templates:
                task = template.copy()
                task["net_name"] = net_name
                final_tasks.append(task)

            for run_id in range(1, random_runs + 1):
                final_tasks.append(
                    {
                        "algo": f"R{run_id}",
                        "net_name": net_name,
                        "removals": None,
                    }
                )

            if not final_tasks:
                print("  [Warning] No tasks generated.")
                continue

            print(f"  -> Submitting {len(final_tasks)} tasks: {[t['algo'] for t in final_tasks]}")

            futures = [
                executor.submit(
                    worker_task,
                    task,
                    graph,
                    remnants_dir,
                    components_dir,
                    min_size,
                    interval,
                    lcc_threshold,
                )
                for task in final_tasks
            ]

            for future in as_completed(futures):
                try:
                    message = future.result()
                    if message:
                        print(f"  {message}")
                except Exception as exc:
                    print(f"  [Worker Error] {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate remnant and component snapshots from dismantling results."
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="Dataset directory containing '*_edges.npz' graph files and result CSVs.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=2,
        help="Minimum connected-component size threshold. Components with size <= this value are discarded.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: Python executor default.",
    )
    parser.add_argument(
        "-a",
        "--algos",
        nargs="+",
        default=None,
        help="Optional whitelist of algorithms to process (short names or raw heuristic names).",
    )
    parser.add_argument(
        "-r",
        "--random-runs",
        type=int,
        default=0,
        help="Number of random attack runs to add.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=1,
        help="Snapshot interval in removal steps.",
    )
    parser.add_argument(
        "--lcc-threshold",
        type=int,
        default=None,
        help="Start saving only after the largest connected component becomes smaller than this threshold.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    process_dataset(
        data_dir=args.dir,
        min_size=args.size,
        workers=args.workers,
        specified_algos=args.algos,
        random_runs=args.random_runs,
        interval=args.interval,
        lcc_threshold=args.lcc_threshold,
    )


if __name__ == "__main__":
    main()
