"""
TCR-GIN/utils/gen_data.py - Synthetic Network Graph Dataset Generator

Description:
    Generates synthetic network graph datasets in parallel using multiple
    classical graph models (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz,
    and LFR benchmark). For each generated graph, node-level structural
    features are computed and saved alongside the edge list.

Input (command-line arguments):
    - Simple mode: --types, --ranges, --num_samples to generate all
      combinations of network types and node ranges.
    - Advanced job mode: --jobs with format TYPE:RANGE:COUNT[:START_ID]
      for fine-grained control over each generation task.
    - Global options: --workers, --feature_set, --base_dir.

Output:
    For each generated network with ID <id> and prefix <prefix>:
        <base_dir>/<scale>/<prefix>/<prefix>_<id>_edges.npz   - compressed edge list
        <base_dir>/<scale>/<prefix>/<prefix>_<id>_features.npy - node feature matrix

Usage Examples:
    # Simple mode: generate 500 BA and WS graphs for two node ranges
    python gen_data.py --types BA WS --ranges 100-200 300-400 --num_samples 500

    # Advanced job mode: specify each task independently
    python gen_data.py --jobs BA:100-200:1000 LFR:300-400:500:100

    # With options
    python gen_data.py --jobs ER:50-100:200 --feature_set full --workers 8 --base_dir ./my_data
"""

import os
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support
import json
import networkx as nx
from itertools import repeat
import random
import time
import argparse
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Network generation
# ---------------------------------------------------------------------------

def generate_network(network_type, params):
    """Generate a single graph of the specified type."""
    n = params.get('n', 100)

    if network_type == 'ER':
        p = params['avg_degree'] / (n - 1)
        return nx.erdos_renyi_graph(n, p)

    elif network_type == 'BA':
        m = params.get('m', 3)
        return nx.barabasi_albert_graph(n, m)

    elif network_type == 'WS':
        k = params.get('k', 4)
        p = params.get('p', 0.1)
        return nx.watts_strogatz_graph(n, k, p)

    elif network_type == 'LFR':
        max_iters = 100
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=params['n'],
                tau1=params['tau1'],
                tau2=params['tau2'],
                mu=params['mu'],
                average_degree=params['avg_degree'],
                max_degree=params['max_degree'],
                min_community=params['min_community'],
                max_community=params['max_community'],
                max_iters=max_iters,
                seed=random.randint(0, 10000)
            )
            return G
        except nx.ExceededMaxIterations:
            return nx.erdos_renyi_graph(n, params.get('avg_degree', 5) / (n - 1))

    else:
        raise ValueError(f"Unsupported network type: {network_type}")


def compute_critical_threshold(G):
    return


# ---------------------------------------------------------------------------
#  Random parameter sampling (scale-aware for LFR)
# ---------------------------------------------------------------------------

def generate_random_params(network_type, node_range=(20, 100)):
    """Sample random graph parameters, with scale-aware tuning for LFR."""
    n = random.randint(*node_range)
    params = {'n': n}

    if network_type == 'LFR':
        scale = 'small'
        if 300 <= node_range[0] < 700:
            scale = 'medium'
        elif 700 <= node_range[0] < 1000:
            scale = 'large'
        elif node_range[0] >= 1000:
            scale = 'extra_large'

        if scale == 'small':
            avg_degree_range = (3, 8)
            mu_range = (0.2, 0.5)
            max_community_ratio = 0.4
            min_community_base = 5
        elif scale == 'medium':
            avg_degree_range = (4, 12)
            mu_range = (0.25, 0.6)
            max_community_ratio = 0.2
            min_community_base = 10
        elif scale == 'large':
            avg_degree_range = (5, 15)
            mu_range = (0.3, 0.6)
            max_community_ratio = 0.15
            min_community_base = 15
        else:
            avg_degree_range = (6, 18)
            mu_range = (0.3, 0.65)
            max_community_ratio = 0.1
            min_community_base = 20

        params['avg_degree'] = random.uniform(*avg_degree_range)
        params['mu'] = random.uniform(*mu_range)
        params['tau1'] = random.uniform(2.0, 3.0)
        params['tau2'] = random.uniform(1.5, 2.5)

        params['max_degree'] = int(n * 0.95)
        if params['max_degree'] <= params['avg_degree']:
            params['max_degree'] = int(params['avg_degree']) + 5

        params['min_community'] = max(min_community_base, int(params['avg_degree']))
        params['max_community'] = int(n * max_community_ratio)

        if params['min_community'] > n / 2:
            params['min_community'] = int(n / 2)
        if params['min_community'] >= params['max_community']:
            params['max_community'] = params['min_community'] + 10
        if params['max_community'] > n:
            params['max_community'] = n

        return params

    elif network_type == 'ER':
        params['avg_degree'] = random.uniform(2, min(8, n / 2))
    elif network_type == 'BA':
        max_m = min(8, n // 4)
        params['m'] = random.randint(1, max(1, max_m))
    elif network_type == 'WS':
        max_k = min(16, n - 1)
        k = random.randint(4, max_k)
        params['k'] = k if k % 2 == 0 else k - 1
        params['p'] = random.uniform(0.01, 0.9)

    return params


# ---------------------------------------------------------------------------
#  Node feature computation
# ---------------------------------------------------------------------------

def calculate_node_features(G, feature_set='basic'):
    """Compute structural node features for every node in G."""
    nodes = list(G.nodes())
    features_dict = {}

    features_dict['degree'] = np.array([G.degree(u) for u in nodes])
    features_dict['clustering'] = np.array([nx.clustering(G, u) for u in nodes])
    features_dict['kcore'] = np.array([nx.core_number(G)[u] for u in nodes])

    if feature_set in ['extended', 'full']:
        features_dict['avg_neighbor_deg'] = np.array(
            [sum(G.degree(v) for v in G.neighbors(u)) / max(1, G.degree(u)) for u in nodes])
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        features_dict['pagerank'] = np.array([pr[u] for u in nodes])

    if feature_set == 'full':
        try:
            bc = nx.betweenness_centrality(G, k=min(30, len(G)), seed=random.randint(0, 10000))
            features_dict['betweenness'] = np.array([bc[u] for u in nodes])
        except Exception:
            features_dict['betweenness'] = np.zeros(len(nodes))
        try:
            ec = nx.eigenvector_centrality_numpy(G)
            features_dict['eigenvector'] = np.array([ec[u] for u in nodes])
        except Exception:
            max_deg = max(features_dict['degree']) if len(features_dict['degree']) > 0 else 1
            features_dict['eigenvector'] = features_dict['degree'] / max(1, max_deg)

    feature_matrix = np.column_stack([features_dict[f] for f in features_dict])
    return feature_matrix, list(features_dict.keys())


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

def save_network_data(folder, G, features, feature_names, net_id, params, filename_prefix):
    """Save edge list, features, and metadata to disk."""
    os.makedirs(folder, exist_ok=True)
    base_filename = f"{filename_prefix}_{net_id}"
    np.savez_compressed(os.path.join(folder, f"{base_filename}_edges.npz"),
                        edges=np.array(list(G.edges())))
    np.save(os.path.join(folder, f"{base_filename}_features.npy"), features)
    label_data = {
        "feature_names": feature_names,
        "network_params": params,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes()
    }


# ---------------------------------------------------------------------------
#  Worker process
# ---------------------------------------------------------------------------

def worker_process(task_args):
    """Generate a single network sample (retries on failure)."""
    net_id, network_type, node_range, output_dir, filename_prefix, feature_set = task_args
    while True:
        try:
            params = generate_random_params(network_type, node_range)
            G_raw = generate_network(network_type, params)

            if G_raw is None or G_raw.number_of_nodes() < 2:
                continue

            if not nx.is_connected(G_raw):
                components = list(nx.connected_components(G_raw))
                if len(components) > 1:
                    main_component_nodes = max(components, key=len)
                    G_main_component = G_raw.subgraph(main_component_nodes).copy()
                    for comp_nodes in components:
                        if comp_nodes != main_component_nodes:
                            node_from_main = random.choice(list(G_main_component.nodes()))
                            node_from_small = random.choice(list(comp_nodes))
                            G_main_component.add_edge(node_from_main, node_from_small)
                    G_raw = G_main_component

            G_final = nx.convert_node_labels_to_integers(G_raw, first_label=0, ordering='default')
            features, feature_names = calculate_node_features(G_final, feature_set)
            save_network_data(output_dir, G_final, features, feature_names,
                              net_id, params, filename_prefix)
            return True
        except Exception:
            continue


# ---------------------------------------------------------------------------
#  Parallel dataset generation
# ---------------------------------------------------------------------------

def parallel_generate_dataset(
        network_type, node_range, num_samples, output_dir, filename_prefix,
        start_id=0, feature_set='extended', num_workers=None
):
    """Dispatch parallel workers to generate a batch of network samples."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Task Start] ==> Model: {network_type}, Range: {node_range}, "
          f"Count: {num_samples}, Start ID: {start_id}")
    print(f"             Output dir: '{output_dir}'")

    ids = range(start_id, start_id + num_samples)
    tasks = zip(
        ids, repeat(network_type), repeat(node_range),
        repeat(output_dir), repeat(filename_prefix), repeat(feature_set)
    )

    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(worker_process, tasks)
        for _ in tqdm(results_iterator, total=num_samples,
                      desc=f"Generating {network_type} ({node_range[0]}-{node_range[1]})"):
            pass

    print(f"[Task Done] Successfully generated {num_samples} networks.")
    return start_id + num_samples


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel synthetic network graph dataset generator.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    simple_group = parser.add_argument_group('Simple mode')
    simple_group.add_argument('--types', nargs='+',
                              help='List of network types (e.g. BA WS)')
    simple_group.add_argument('--ranges', nargs='+',
                              help='List of node ranges (e.g. 100-200 300-400)')
    simple_group.add_argument('--num_samples', type=int,
                              help='Number of samples per combination')
    simple_group.add_argument('--start_id', type=int, default=0,
                              help='Starting ID (default: 0)')

    job_group = parser.add_argument_group('Advanced job mode')
    job_group.add_argument('--jobs', nargs='+',
                           help='Define independent jobs. Format: TYPE:RANGE:COUNT[:START_ID]')

    parser.add_argument('--workers', type=int, default=None,
                        help='Number of CPU workers (default: cpu_count - 1)')
    parser.add_argument('--feature_set', type=str, default='full',
                        choices=['basic', 'extended', 'full'],
                        help='Feature set to compute (default: full)')
    parser.add_argument('--base_dir', type=str, default='./data_synth',
                        help='Root output directory (default: ./data_synth)')

    args = parser.parse_args()
    tasks = []

    print("==================== Batch Network Generation ====================")

    if args.jobs:
        print("[INFO] Advanced job mode detected.")
        for job_str in args.jobs:
            parts = job_str.split(':')
            if len(parts) < 3 or len(parts) > 4:
                print(f"[ERROR] Job '{job_str}' has invalid format.")
                continue
            task = {'type': parts[0], 'range_str': parts[1]}
            try:
                task['count'] = int(parts[2])
                task['start_id'] = int(parts[3]) if len(parts) == 4 else 0
                tasks.append(task)
            except ValueError:
                print(f"[ERROR] Job '{job_str}' has invalid count or ID.")
                continue
    else:
        print("[INFO] Using simple combination mode.")
        if not all([args.types, args.ranges, args.num_samples is not None]):
            print("[ERROR] Simple mode requires --types, --ranges, and --num_samples.")
            parser.print_help()
            return
        for range_str in args.ranges:
            for net_type in args.types:
                tasks.append({
                    'type': net_type, 'range_str': range_str,
                    'count': args.num_samples, 'start_id': args.start_id
                })

    if not tasks:
        print("[WARNING] No valid tasks found. Exiting.")
        return

    for task in tasks:
        net_type = task['type']
        range_str = task['range_str']
        num_samples = task['count']
        start_id = task['start_id']
        min_n, max_n = map(int, range_str.split('-'))
        node_range = (min_n, max_n)

        if min_n >= 100 and (max_n - min_n) == 100:
            scale_folder_name = str(min_n)
        else:
            scale_folder_name = f"{min_n}-{max_n}"

        model_folder_and_prefix = f"{net_type}-{scale_folder_name}"
        final_output_dir = os.path.join(args.base_dir, scale_folder_name, model_folder_and_prefix)

        parallel_generate_dataset(
            network_type=net_type, node_range=node_range, num_samples=num_samples,
            output_dir=final_output_dir, filename_prefix=model_folder_and_prefix,
            start_id=start_id, feature_set=args.feature_set, num_workers=args.workers
        )

    print("\n==================== All Generation Tasks Completed ====================")


if __name__ == "__main__":
    freeze_support()
    main()
