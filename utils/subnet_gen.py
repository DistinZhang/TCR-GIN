"""
TCR-GIN/utils/subnet_gen.py - Subgraph Generation, Feature Computation, and Label Assignment

Description:
    A modular preprocessing tool for graph machine learning datasets.  It
    automates the full pipeline from raw networks to labelled subgraph
    datasets, organised around the concept of network *dismantling* —
    progressively removing nodes until the largest connected component
    collapses below a size threshold.

    The script exposes three operating modes (--mode):

    1.  subgraph
        Starting point of the data-generation pipeline.  For every source
        network it runs many parallel random-removal simulations.  At each
        removal step the resulting connected components are extracted as
        new subgraph samples.  A graph-fingerprint deduplication step
        ensures that structurally identical subgraphs are stored only once.
        Subgraphs are binned into directories by node-count range
        (e.g. 20-40, 40-100, 100, 200, …).

    2.  features_only
        Computes node-level topological features (degree, clustering
        coefficient, k-core number, PageRank, betweenness / eigenvector
        centrality, …) for every subgraph that does not yet have a
        companion *_features.npy file.  Feature computation is
        parallelised across workers.

    3.  labels
        Assigns a graph-level label — the *critical threshold*, i.e. the
        minimum fraction of nodes whose removal disconnects the network —
        to every subgraph.  Two independent strategies are provided:

        a)  External-file strategy (triggered by --xlsx_file + --edges_dir):
            Reads pre-computed thresholds from an XLSX / CSV table and
            matches them to the corresponding edge-list files.

        b)  Brute-force strategy (triggered by --input):
            For each subgraph, performs an exact combinatorial search to
            find the smallest node set whose removal disrupts the network.
            The search starts from an upper bound (an existing JSON label
            if available, otherwise a degree-heuristic estimate) and
            iterates downward, pruning via high-degree-first ordering.

Input (command-line arguments):
    --mode              One of {subgraph, features_only, labels}.
    --input             Directory or single .npz file (subgraph /
                        features_only / brute-force labels).
    --processes         Number of parallel workers (default: 4).

    subgraph-specific:
        --num_sequences             Random removal sequences per graph (default: 128).
        --min_remaining_fraction    Stop when this fraction of nodes remains (default: 0.01).
        --min_remaining_nodes       Absolute node floor (overrides fraction).

    features_only-specific:
        --feature_set               One of {basic, extended, full} (default: full).

    labels-specific:
        --xlsx_file     Path to XLSX file with pre-computed thresholds.
        --edges_dir     Comma-separated directories containing edge-list files.

Output:
    subgraph mode:
        <source_dir>/<base>-<range>/<base>-<range>_<id>_edges.npz

    features_only mode:
        <same_dir>/<prefix>_features.npy   (one per input .npz)

    labels mode:
        <same_dir>/<prefix>_label.json     (one per input .npz)

Usage:
    # Mode 1 — generate subgraphs via random dismantling
    python subnet_gen.py --mode subgraph --input real_networks --processes 32 \\
                         --min_remaining_fraction 0.05

    # Mode 2 — compute node features for existing subgraphs
    python subnet_gen.py --mode features_only --input real_networks --processes 32 \\
                         --feature_set full

    # Mode 3a — assign labels from an external XLSX file
    python subnet_gen.py --mode labels \\
                         --xlsx_file "data_synth/results/results.xlsx" \\
                         --edges_dir "data_synth/BA-800-1000,data_synth/ER-800-1000" \\
                         --processes 32

    # Mode 3b — assign labels via brute-force exact search
    python subnet_gen.py --mode labels --input real_networks --processes 32
"""

import os
import sys
import argparse
import numpy as np
import networkx as nx
import json
import random
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
import time
import glob
import math
from itertools import combinations, islice
from math import comb
import gc

# Allow imports from the parent package (e.g. gen_data.calculate_node_features)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from gen_data import calculate_node_features


# ===========================================================================
#  Core helpers
# ===========================================================================

def load_graph_from_npz(npz_file):
    """
    Load a NetworkX graph from a compressed file.

    Supports two storage conventions:
      - 'edges' key  → Nx2 array of (u, v) pairs.
      - 'adjacency'  → dense or sparse adjacency matrix.

    Self-loop edges are removed automatically.  Returns None on failure.
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        edges = None

        if 'edges' in data:
            edges = data['edges']
        elif 'adjacency' in data:
            adj = data['adjacency'].item() if data['adjacency'].ndim == 0 else data['adjacency']
            edges = np.argwhere(adj > 0)
        else:
            raise ValueError(
                f"NPZ file '{os.path.basename(npz_file)}' contains neither 'edges' nor 'adjacency' key."
            )

        G = nx.Graph()
        G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except Exception as e:
        print(f"Failed to load graph '{os.path.basename(npz_file)}': {e}")
        return None


def find_npz_files_in_dir(input_path):
    """
    Recursively find all .npz files under *input_path*.

    If *input_path* is itself a .npz file, returns a single-element list.
    """
    all_files = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            all_files.extend(
                os.path.join(root, f) for f in files if f.endswith('.npz')
            )
    elif input_path.endswith('.npz'):
        all_files.append(input_path)

    if not all_files:
        print(f"Warning: no .npz files found in '{input_path}'.")
    return all_files


# ===========================================================================
#  Mode 1 — Subgraph generation
# ===========================================================================

def graph_fingerprint(G):
    """
    Compute a hash-based fingerprint for *G*.

    Two graphs with identical adjacency matrices (after sorting node
    labels) will produce the same fingerprint, enabling O(1)
    deduplication via a shared set.
    """
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    return hash(A.tobytes())


def save_graph_to_npz(G, output_file):
    """Save an edge list to a compressed .npz file, creating parent dirs as needed."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(output_file, edges=np.array(list(G.edges())))
        return True
    except Exception as e:
        print(f"Failed to save graph to '{output_file}': {e}")
        return False


def renumber_graph(G):
    """Relabel nodes to consecutive integers starting from 0."""
    return nx.convert_node_labels_to_integers(G, first_label=0)


def get_size_range_folder_name(n_nodes, base_name):
    """
    Map a node count to the appropriate output sub-directory name.

    Binning rules:
      -  20 <= n < 40   → <base>-20-40
      -  40 <= n < 100  → <base>-40-100
      - 100 <= n        → <base>-<floor to nearest 100>   (e.g. 356 → <base>-300)
      - n < 20          → None  (too small; discard)
    """
    if 20 <= n_nodes < 40:
        return f"{base_name}-20-40"
    if 40 <= n_nodes < 100:
        return f"{base_name}-40-100"
    if n_nodes >= 100:
        return f"{base_name}-{(n_nodes // 100) * 100}"
    return None


def process_single_sequence(args):
    """
    Worker function: execute one random dismantling sequence.

    For each removal step, every resulting connected component with a
    valid size is fingerprinted.  If the fingerprint has not been seen
    before, the component is saved as a new subgraph sample.

    Returns the number of unique subgraphs produced by this sequence.
    """
    G, base_name, output_root, max_removals, shared_counter, lock, seen_fps = args
    graph_count = 0

    # Random permutation defines the removal order
    removal_order = random.sample(list(G.nodes()), G.number_of_nodes())

    for step in range(1, max_removals + 1):
        subgraph = G.copy()
        subgraph.remove_nodes_from(removal_order[:step])
        if subgraph.number_of_nodes() == 0:
            continue

        for component_nodes in nx.connected_components(subgraph):
            comp_graph = subgraph.subgraph(component_nodes).copy()
            folder_name = get_size_range_folder_name(comp_graph.number_of_nodes(), base_name)
            if not folder_name:
                continue

            comp_graph = renumber_graph(comp_graph)
            fp = graph_fingerprint(comp_graph)

            # Atomically check-and-register the fingerprint
            with lock:
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                file_idx = shared_counter.value
                shared_counter.value += 1

            save_dir = os.path.join(output_root, folder_name)
            out_file = os.path.join(save_dir, f"{folder_name}_{file_idx}_edges.npz")
            if save_graph_to_npz(comp_graph, out_file):
                graph_count += 1

    return graph_count


def generate_subnet_series(
    input_path, num_processes, num_sequences,
    min_remaining_fraction, min_remaining_nodes
):
    """
    Orchestrator: generate subgraphs from all source networks in
    *input_path* via parallel random dismantling.

    For each source graph, *num_sequences* independent random-removal
    orderings are executed in parallel.  The maximum number of nodes
    removed per sequence is governed by *min_remaining_nodes* (absolute)
    or *min_remaining_fraction* (relative).
    """
    print("Mode 'subgraph': starting subgraph generation...")
    npz_files = find_npz_files_in_dir(input_path)
    if not npz_files:
        return

    manager = Manager()
    shared_counter = manager.Value('i', 0)
    seen_fingerprints = manager.Set()
    lock = manager.Lock()

    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        print(f"\n--- Processing graph: {filename} ---")
        G = load_graph_from_npz(npz_file)
        if G is None or G.number_of_nodes() < 20:
            print(f"Skipping {filename} (too small or failed to load).")
            continue

        total_nodes = G.number_of_nodes()

        # Determine how many nodes may be removed
        if min_remaining_nodes:
            if total_nodes <= min_remaining_nodes:
                continue
            max_removals = total_nodes - min_remaining_nodes
        else:
            min_nodes_to_keep = math.ceil(total_nodes * min_remaining_fraction)
            if total_nodes <= min_nodes_to_keep:
                continue
            max_removals = total_nodes - int(min_nodes_to_keep)

        if max_removals <= 0:
            continue

        base_name = filename.removesuffix('_edges.npz').removesuffix('.npz')
        output_root = os.path.dirname(npz_file)
        task_args = (
            G, base_name, output_root, max_removals,
            shared_counter, lock, seen_fingerprints,
        )

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_single_sequence, [task_args] * num_sequences),
                total=num_sequences,
                desc=f"Dismantling {base_name}",
            ))

    print(f"\nSubgraph generation complete. "
          f"{shared_counter.value} unique subgraph files produced.")


# ===========================================================================
#  Mode 2 — Feature computation only
# ===========================================================================

def process_single_feature_completion(npz_file, feature_set):
    """
    Worker function: compute and save node features for a single graph.

    Loads the graph from *npz_file*, computes the requested feature set,
    and writes the resulting matrix to <prefix>_features.npy.  Returns
    True on success.
    """
    try:
        G = load_graph_from_npz(npz_file)
        if G is None or G.number_of_nodes() == 0:
            return False

        features, _ = calculate_node_features(G, feature_set)

        base_name = os.path.splitext(npz_file)[0].removesuffix('_edges')
        feature_file = f"{base_name}_features.npy"
        np.save(feature_file, features)
        return True
    except Exception as e:
        print(f"Feature computation failed for '{os.path.basename(npz_file)}': {e}")
        return False


def complete_features_only(input_path, num_processes, feature_set):
    """
    Orchestrator: compute node features for every subgraph in
    *input_path* that does not already have a companion _features.npy
    file.
    """
    print(f"Mode 'features_only': computing features for graphs in '{input_path}'...")
    npz_files = find_npz_files_in_dir(input_path)

    files_to_process = [
        f for f in npz_files
        if not os.path.exists(
            f.replace('.npz', '_features.npy').replace('_edges.npz', '_features.npy')
        )
    ]

    if not files_to_process:
        print("All graphs already have feature files. Nothing to do.")
        return

    print(f"Found {len(files_to_process)} graphs requiring feature computation.")
    with Pool(processes=num_processes) as pool:
        worker_func = partial(process_single_feature_completion, feature_set=feature_set)
        results = list(tqdm(
            pool.imap_unordered(worker_func, files_to_process),
            total=len(files_to_process),
            desc="Computing features",
        ))

    success_count = sum(results)
    print(f"Feature computation complete. "
          f"{success_count}/{len(files_to_process)} files processed successfully.")


# ===========================================================================
#  Mode 3 — Label assignment
# ===========================================================================

# ---------------------------------------------------------------------------
#  3a: Labels from an external file
# ---------------------------------------------------------------------------

def process_single_network_label(args):
    """
    Worker function: write a JSON label for one network using a
    pre-computed threshold from a DataFrame.

    Looks up the minimum critical_threshold for the given network name
    in *df*, loads the graph to obtain structural metadata, and writes
    the label file.
    """
    network_name, df, edge_file = args
    try:
        min_threshold = df.loc[
            df['network'] == network_name, 'critical_threshold'
        ].min()

        G = load_graph_from_npz(edge_file)
        if G is None:
            return False

        label_data = {
            "critical_threshold": round(float(min_threshold), 4),
            "removed_nodes": [],
            "feature_names": ["degree", "clustering", "kcore", "pagerank"],
            "network_params": {},
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / max(1, G.number_of_nodes()),
        }

        base_name = os.path.splitext(edge_file)[0].removesuffix('_edges')
        with open(f"{base_name}_label.json", 'w') as f:
            json.dump(label_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Label generation failed for network '{network_name}': {e}")
        return False


def generate_labels_from_xlsx(xlsx_file, edges_dirs, num_processes):
    """
    Orchestrator: read thresholds from an XLSX file and write JSON
    labels for every matching edge-list file found in *edges_dirs*.
    """
    print(f"Mode 'labels': generating labels from external file '{xlsx_file}'...")
    try:
        df = pd.read_excel(xlsx_file, usecols=['network', 'critical_threshold'])
        network_min_threshold = (
            df.groupby('network')['critical_threshold'].min().reset_index()
        )

        edge_files = {
            os.path.basename(f).removesuffix('_edges.npz').removesuffix('.npz'): f
            for d in edges_dirs
            for f in find_npz_files_in_dir(d)
        }

        tasks = [
            (net, network_min_threshold, edge_files[net])
            for net in network_min_threshold['network'].unique()
            if net in edge_files
        ]

        if not tasks:
            print("No matching network files found.")
            return

        print(f"Will generate labels for {len(tasks)} matched networks.")
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_network_label, tasks),
                total=len(tasks),
                desc="Generating labels",
            ))

        print(f"Label generation complete. "
              f"{sum(results)}/{len(tasks)} networks processed successfully.")
    except Exception as e:
        print(f"Failed to process XLSX file: {e}")


# ---------------------------------------------------------------------------
#  3b: Labels via brute-force exact search
# ---------------------------------------------------------------------------

def _is_network_disrupted(G_temp, min_cc_size):
    """
    Return True if the network is considered disrupted, i.e. its largest
    connected component has fewer than *min_cc_size* nodes.
    """
    if G_temp.number_of_nodes() < min_cc_size:
        return True
    components = list(nx.connected_components(G_temp))
    return not components or len(max(components, key=len)) < min_cc_size


def _degree_heuristic_upper_bound(G, min_cc_size=5):
    """
    Greedily remove the highest-degree node at each step and return the
    number of removals needed to disrupt the network.

    This provides a fast (but not necessarily tight) upper bound on the
    true critical removal count.
    """
    G_temp = G.copy()
    for removed_count in range(G.number_of_nodes()):
        if _is_network_disrupted(G_temp, min_cc_size):
            return removed_count
        if G_temp.number_of_nodes() == 0:
            break
        max_degree_node = max(G_temp.nodes(), key=G_temp.degree)
        G_temp.remove_node(max_degree_node)
    return G.number_of_nodes() - 1


def _get_search_upper_bound(G, base_name, filename):
    """
    Determine the starting upper bound for the brute-force search.

    Strategy:
      1. If a JSON label file already exists alongside the edge file,
         use its critical_threshold (converted to an absolute count) as
         the upper bound — this is the tightest available estimate.
      2. Otherwise, fall back to the degree-greedy heuristic.
    """
    potential_label_file = f"{base_name}_label.json"
    if os.path.exists(potential_label_file):
        try:
            with open(potential_label_file, 'r') as f:
                ratio = json.load(f).get('critical_threshold')
                if ratio and ratio > 0:
                    count = int(round(ratio * G.number_of_nodes()))
                    return count
        except Exception as e:
            print(f"Warning: failed to read JSON for '{filename}': {e}")

    count = _degree_heuristic_upper_bound(G)
    return count


def compute_critical_threshold(G, start_remove_count, min_cc_size=5):
    """
    Find the minimum number of node removals that disrupts the network,
    using a top-down exact combinatorial search.

    Algorithm outline
    -----------------
    1.  Accept *start_remove_count* (a known feasible solution from a
        heuristic or a previous label) as the initial best.
    2.  Iterate k from start_remove_count − 1 down to 1.
    3.  At each level k, enumerate combinations of k nodes (ordered by
        decreasing degree for early success) and test whether their
        removal disrupts the graph.
        -  If a disrupting set of size k is found → update best to k
           and continue searching k − 1 (a tighter solution may exist).
        -  If no disrupting set of size k exists (after exhaustive or
           budget-limited enumeration) → k + 1 is optimal.  The current
           best already equals k + 1, so stop immediately.
    4.  Return the critical threshold as a fraction of total nodes,
        together with the specific node set of the best solution found.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    start_remove_count : int
        Upper bound on the number of removals (exclusive starting point
        for the downward search).
    min_cc_size : int
        A graph is "disrupted" when its largest connected component has
        fewer than this many nodes.

    Returns
    -------
    threshold : float
        Fraction of nodes in the minimum disrupting set (0.0 – 1.0).
    removed_nodes : list[int]
        The specific nodes in the best disrupting set found (may be
        empty if only the heuristic upper bound was used).
    """
    total_nodes = G.number_of_nodes()

    if total_nodes < min_cc_size:
        return 0.0, []

    start_count = max(1, min(start_remove_count, total_nodes - 1))
    best_threshold_count = start_count
    best_removed_nodes = []

    # Order candidate nodes by degree (descending) so that the most
    # impactful removals are tried first within each combination level.
    nodes_by_degree = sorted(G.nodes(), key=G.degree, reverse=True)

    for k in range(start_count - 1, 0, -1):
        total_combs = comb(total_nodes, k)
        combs_to_check = total_combs

        found_solution_at_k = False
        combo_generator = combinations(nodes_by_degree, k)

        for nodes_to_remove in islice(combo_generator, combs_to_check):
            G_temp = G.copy()
            G_temp.remove_nodes_from(nodes_to_remove)
            if _is_network_disrupted(G_temp, min_cc_size):
                best_threshold_count = k
                best_removed_nodes = list(nodes_to_remove)
                found_solution_at_k = True
                break  # Found a solution at level k; try k − 1 next

        if not found_solution_at_k:
            # No k-sized set can disrupt the network → k + 1 is optimal.
            # best_threshold_count already holds that value, so stop.
            break

    final_threshold = best_threshold_count / total_nodes
    return final_threshold, best_removed_nodes


def process_single_brute_force_label(npz_file):
    """
    Worker function: compute (or update) the exact critical-threshold
    label for a single graph via brute-force search.

    Workflow:
      1. Read any existing label (if present) to obtain the old threshold.
      2. Load the graph and determine the search upper bound.
      3. Run the exact combinatorial search.
      4. If the new threshold differs from the old one (at 4-decimal
         precision), write an updated JSON label; otherwise skip the write.

    Returns True on success (whether or not a write occurred).
    """
    filename = os.path.basename(npz_file)
    try:
        base_name = os.path.splitext(npz_file)[0].removesuffix('_edges')
        label_file = f"{base_name}_label.json"

        # Step 1: read existing threshold (if any)
        old_threshold = None
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    old_data = json.load(f)
                    if 'critical_threshold' in old_data:
                        old_threshold = old_data['critical_threshold']
            except Exception as e:
                print(f"Warning: failed to read existing label for '{filename}': {e}")

        # Step 2: load graph and compute new threshold
        G = load_graph_from_npz(npz_file)
        if G is None or G.number_of_nodes() < 3:
            return False

        start_count = _get_search_upper_bound(G, base_name, filename)
        new_threshold_raw, removed_nodes = compute_critical_threshold(
            G, start_remove_count=start_count
        )
        rounded_new_threshold = round(
            float(new_threshold_raw) if new_threshold_raw is not None else 0.0, 4
        )

        # Step 3: skip write if the threshold is unchanged
        if old_threshold is not None and round(old_threshold, 4) == rounded_new_threshold:
            return True

        # Step 4: write updated label
        _, feature_names = calculate_node_features(G, 'full')

        label_data = {
            "critical_threshold": rounded_new_threshold,
            "removed_nodes": [int(n) for n in removed_nodes],
            "feature_names": feature_names,
            "network_params": {},
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / max(1, G.number_of_nodes()),
        }

        with open(label_file, 'w') as f:
            json.dump(label_data, f, indent=4)

        if old_threshold is None:
            print(f"INFO: {filename}: new label created, "
                  f"critical_threshold = {rounded_new_threshold}.")
        else:
            print(f"INFO: {filename}: critical_threshold updated "
                  f"from {old_threshold} to {rounded_new_threshold}.")

        return True
    except Exception as e:
        print(f"Label computation failed for '{filename}': {e}")
        return False


def brute_force_labels(input_path, num_processes):
    """
    Orchestrator: compute exact critical-threshold labels for every
    subgraph in *input_path* via parallel brute-force search.
    """
    print(f"Mode 'labels': computing brute-force labels for graphs in '{input_path}'...")
    npz_files = find_npz_files_in_dir(input_path)
    if not npz_files:
        return

    print(f"Found {len(npz_files)} graphs requiring label computation.")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_brute_force_label, npz_files),
            total=len(npz_files),
            desc="Computing labels",
        ))

    success_count = sum(results)
    print(f"Label computation complete. "
          f"{success_count}/{len(npz_files)} files processed successfully.")


# ===========================================================================
#  CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph ML data preprocessing: subgraph generation, feature computation, and label assignment.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['subgraph', 'features_only', 'labels'],
        help=(
            "Operating mode:\n"
            "  subgraph       Generate subgraphs by random dismantling.\n"
            "  features_only  Compute node-level topological features.\n"
            "  labels         Assign graph-level labels (external file or brute-force)."
        ),
    )
    parser.add_argument(
        '--input', type=str,
        help="Input directory or .npz file (subgraph / features_only / brute-force labels).",
    )
    parser.add_argument(
        '--processes', type=int, default=4,
        help="Number of parallel worker processes (default: 4).",
    )

    # --- subgraph mode ---
    subgraph_group = parser.add_argument_group('subgraph mode options')
    subgraph_group.add_argument(
        '--num_sequences', type=int, default=128,
        help="Number of random dismantling sequences per source graph (default: 128).",
    )
    subgraph_group.add_argument(
        '--min_remaining_fraction', type=float, default=0.01,
        help="Stop removing nodes when this fraction remains (default: 0.01).",
    )
    subgraph_group.add_argument(
        '--min_remaining_nodes', type=int,
        help="Absolute minimum remaining nodes (overrides --min_remaining_fraction).",
    )

    # --- features_only mode ---
    feature_group = parser.add_argument_group('features_only mode options')
    feature_group.add_argument(
        '--feature_set', type=str, default='full',
        choices=['basic', 'extended', 'full'],
        help="Feature computation level (default: full).",
    )

    # --- labels mode ---
    label_group = parser.add_argument_group('labels mode options')
    label_group.add_argument(
        '--xlsx_file', type=str,
        help="[External-file strategy] Path to XLSX file containing pre-computed thresholds.",
    )
    label_group.add_argument(
        '--edges_dir', type=str,
        help="[External-file strategy] Comma-separated directories with edge-list files.",
    )

    args = parser.parse_args()

    if args.processes <= 0:
        args.processes = mp.cpu_count()

    # --- Dispatch by mode ---
    if args.mode == 'subgraph':
        if not args.input:
            parser.error("'subgraph' mode requires --input.")
        generate_subnet_series(
            args.input,
            num_processes=args.processes,
            num_sequences=args.num_sequences,
            min_remaining_fraction=args.min_remaining_fraction,
            min_remaining_nodes=args.min_remaining_nodes,
        )

    elif args.mode == 'features_only':
        if not args.input:
            parser.error("'features_only' mode requires --input.")
        complete_features_only(
            args.input,
            num_processes=args.processes,
            feature_set=args.feature_set,
        )

    elif args.mode == 'labels':
        if args.xlsx_file and args.edges_dir:
            edges_dirs = [d.strip() for d in args.edges_dir.split(',')]
            generate_labels_from_xlsx(
                args.xlsx_file, edges_dirs, num_processes=args.processes
            )
        elif args.input:
            brute_force_labels(args.input, num_processes=args.processes)
        else:
            parser.error(
                "'labels' mode requires either (--xlsx_file and --edges_dir) or (--input)."
            )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
