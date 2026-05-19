"""
TCR-GIN/utils/network_data_consolidator.py - Network Result Consolidation and Label Generation

Description:
    Consolidates multi-algorithm experiment results for network datasets.
    For each network, it:
      1. Scans a network directory for edge-list files (*_edges.npz).
      2. Loads all result files (.xlsx / .csv) from a results directory.
      3. Maps raw algorithm identifiers to standardized names.
      4. Generates per-algorithm XLSX files (filtered, deduplicated, sorted).
      5. Produces a win-count report showing how often each algorithm
         achieved the minimum critical threshold.
      6. Creates / updates JSON label files alongside each edge-list,
         recording the best-known critical threshold and graph metadata.

Input:
    -n / --network_dir   Directory containing *_edges.npz network files
                         (searched recursively).
    -r / --results_dir   Directory containing .xlsx / .csv result files
                         (searched recursively).
    -o / --output_dir    Directory where output XLSX reports are saved.

Output:
    <output_dir>/<dataset>-<algorithm>.xlsx          Per-algorithm result sheets
    <output_dir>/<dataset>-algorithm_win_counts.xlsx  Win-count summary report
    <network_dir>/.../<network>_label.json            JSON label per network
                                                      (written next to the .npz)

Usage:
    python network_data_consolidator.py \
        -n ./datasets/data_synth/100 \
        -r ./datasets/data_synth/100/results \
        -o ./datasets/data_synth/100/results_final
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import json
import glob
import argparse
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

NETWORK_COL = 'network'
THRESHOLD_COL = 'critical_threshold'
HEURISTIC_COL = 'heuristic'
STATIC_COL = 'static'


# ---------------------------------------------------------------------------
#  Network discovery
# ---------------------------------------------------------------------------

def find_target_networks(network_dir):
    """Recursively scan *network_dir* and return {network_name: npz_path}."""
    print(f"[*] Recursively scanning '{network_dir}' for network files...")
    if not os.path.isdir(network_dir):
        print(f"[!] Error: network directory '{network_dir}' does not exist.")
        return {}

    network_paths = {}
    npz_files = glob.glob(os.path.join(network_dir, '**', '*_edges.npz'), recursive=True)

    for file_path in npz_files:
        base_name = os.path.basename(file_path).replace('_edges.npz', '')
        network_paths[base_name] = file_path

    if not network_paths:
        print(f"[!] Warning: no '*_edges.npz' files found under '{network_dir}'.")
    else:
        print(f"[+] Found {len(network_paths)} target networks.")
    return network_paths


# ---------------------------------------------------------------------------
#  Result loading
# ---------------------------------------------------------------------------

def load_all_results_data(results_dir):
    """Recursively load all .xlsx / .csv result files into a single DataFrame."""
    if not os.path.isdir(results_dir):
        print(f"[!] Error: results directory '{results_dir}' does not exist.")
        return pd.DataFrame()

    print(f"[*] Recursively scanning '{results_dir}' for result files...")
    all_files = glob.glob(os.path.join(results_dir, '**/*.*'), recursive=True)
    valid_files = [f for f in all_files if f.endswith('.xlsx') or f.endswith('.csv')]

    if not valid_files:
        print(f"[!] Warning: no .xlsx or .csv files found under '{results_dir}'.")
        return pd.DataFrame()

    all_dfs = []
    for f in tqdm(valid_files, desc="    Reading files"):
        try:
            df = pd.read_excel(f) if f.endswith('.xlsx') else pd.read_csv(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"\n[!] Warning: failed to read '{f}': {e}")

    if not all_dfs:
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[*] All result files loaded: {len(full_df)} records in total.")
    return full_df


# ---------------------------------------------------------------------------
#  Algorithm name mapping
# ---------------------------------------------------------------------------

def map_algorithm_names(df):
    """Create a unified 'algorithm_name' column from heuristic/static columns."""
    if HEURISTIC_COL not in df.columns:
        print(f"[!] Warning: column '{HEURISTIC_COL}' is missing; cannot create algorithm names.")
        return df

    df['algorithm_name'] = df[HEURISTIC_COL]
    df.loc[df[HEURISTIC_COL] == 'FINDER', 'algorithm_name'] = 'FINDER_CN'
    df.loc[df[HEURISTIC_COL] == 'DomiRank', 'algorithm_name'] = 'Domirank'

    if STATIC_COL in df.columns:
        df[STATIC_COL] = df[STATIC_COL].astype(str).str.upper()
        is_degree = df[HEURISTIC_COL] == 'degree'
        is_betweenness = df[HEURISTIC_COL] == 'betweenness_centrality'
        is_static_true = df[STATIC_COL] == 'TRUE'

        df.loc[is_degree & is_static_true, 'algorithm_name'] = 'degree_T'
        df.loc[is_degree & ~is_static_true, 'algorithm_name'] = 'degree_F'
        df.loc[is_betweenness & is_static_true, 'algorithm_name'] = 'betweenness_centrality_T'
        df.loc[is_betweenness & ~is_static_true, 'algorithm_name'] = 'betweenness_centrality_F'

    return df


# ---------------------------------------------------------------------------
#  Per-algorithm XLSX generation
# ---------------------------------------------------------------------------

def generate_algorithm_xlsx_files(df, output_dir, dataset_name):
    """Write one deduplicated, sorted XLSX file per algorithm."""
    print("\n[*] Generating per-algorithm XLSX files...")

    if 'algorithm_name' not in df.columns:
        print("[!] Error: 'algorithm_name' column is missing.")
        return

    grouped = df.groupby('algorithm_name')

    for algo_name, group_df in tqdm(grouped, desc="    Saving algorithm files"):
        df_deduplicated = group_df.drop_duplicates(subset=[NETWORK_COL], keep='last').copy()
        sort_keys = df_deduplicated[NETWORK_COL].str.extract(r'(.*?_)([0-9]+)$')
        sort_keys[0] = sort_keys[0].fillna(df_deduplicated[NETWORK_COL])
        sort_keys[1] = sort_keys[1].fillna(0)
        df_deduplicated['sort_key_text'] = sort_keys[0]
        df_deduplicated['sort_key_num'] = pd.to_numeric(sort_keys[1])
        df_sorted = df_deduplicated.sort_values(by=['sort_key_text', 'sort_key_num'])

        output_filename = os.path.join(output_dir, f"{dataset_name}-{algo_name}.xlsx")
        columns_to_drop = ['algorithm_name', 'sort_key_text', 'sort_key_num']
        df_to_save = df_sorted.drop(columns=columns_to_drop, errors='ignore')
        df_to_save.to_excel(output_filename, index=False, engine='openpyxl')

    print(f"[+] Generated XLSX files for {len(grouped)} algorithms in '{output_dir}'.")


# ---------------------------------------------------------------------------
#  Graph I/O helper
# ---------------------------------------------------------------------------

def load_graph_from_npz(npz_file):
    """Load a NetworkX graph from a compressed edge-list file."""
    try:
        edges = np.load(npz_file, allow_pickle=True)['edges']
        G = nx.Graph()
        G.add_edges_from(edges)
        return G
    except Exception as e:
        print(f"\n[!] Warning: failed to load graph '{npz_file}': {e}")
        return None


# ---------------------------------------------------------------------------
#  JSON label generation
# ---------------------------------------------------------------------------

def generate_network_json_labels(df, network_paths_map):
    """
    Generate / update a JSON label file for each network.

    Threshold selection strategy:
      - Prefer the smallest *positive* threshold across all algorithms.
      - Fall back to 0 only when every algorithm reports 0.
      - Write only when the content has actually changed (incremental update).
    """
    print("\n" + "=" * 50)
    print("[*] Generating / updating JSON label files (smart zero handling & incremental update)...")

    if THRESHOLD_COL not in df.columns:
        print(f"[!] Error: column '{THRESHOLD_COL}' is missing.")
        return

    df[THRESHOLD_COL] = pd.to_numeric(df[THRESHOLD_COL], errors='coerce')

    positive_df = df[df[THRESHOLD_COL] > 0]
    min_positive = positive_df.groupby(NETWORK_COL)[THRESHOLD_COL].min()
    min_all = df.groupby(NETWORK_COL)[THRESHOLD_COL].min()
    final_thresholds = min_positive.combine_first(min_all)

    corrected_networks = [
        net for net in final_thresholds.index
        if net in min_all.index and min_all[net] == 0 and final_thresholds[net] > 0
    ]
    if corrected_networks:
        print(f"    -> [Smart correction] {len(corrected_networks)} networks had mixed zero/positive values.")
        print(f"       Ignored zeros; chose smallest positive "
              f"(e.g. {corrected_networks[0]} -> {final_thresholds[corrected_networks[0]]}).")

    still_zero = final_thresholds[final_thresholds == 0]
    if not still_zero.empty:
        print(f"    -> [Note] {len(still_zero)} networks remain at 0 (all algorithms reported 0).")
        print(f"       Examples: {still_zero.index[:5].tolist()}")

    updated_count = 0
    skipped_count = 0

    for network_name, npz_file_path in tqdm(network_paths_map.items(), desc="    Processing JSON files"):
        if network_name not in final_thresholds.index:
            continue

        G = load_graph_from_npz(npz_file_path)
        if G is None:
            continue

        val = float(final_thresholds[network_name])

        label_data = {
            "critical_threshold": val,
            "removed_nodes": [],
            "feature_names": [
                "degree", "clustering", "kcore",
                "avg_neighbor_deg", "pagerank",
                "betweenness", "eigenvector"
            ],
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / max(1, G.number_of_nodes())
        }

        output_path = npz_file_path.replace('_edges.npz', '_label.json')

        should_write = True
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                if existing_data == label_data:
                    should_write = False
            except Exception:
                should_write = True

        if should_write:
            with open(output_path, 'w') as f:
                json.dump(label_data, f, indent=4)
            updated_count += 1
        else:
            skipped_count += 1

    print(f"[+] Done: {updated_count} files updated/created, {skipped_count} files unchanged (skipped).")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
#  Win-count report
# ---------------------------------------------------------------------------

def generate_win_count_report(df, output_dir, dataset_name):
    """
    Count how many times each algorithm achieves the minimum critical_threshold.

    Ties are broken by a predefined priority list; algorithms not on the list
    receive the highest priority (lowest rank number).
    """
    print("\n[*] Computing per-algorithm win counts (with tie-breaking priority)...")
    if THRESHOLD_COL not in df.columns or 'algorithm_name' not in df.columns:
        print(f"[!] Error: columns '{THRESHOLD_COL}' or 'algorithm_name' are missing.")
        return

    specified_priority_group = [
        'network_entanglement_large_reinsertion',
        'network_entanglement_mid_reinsertion',
        'network_entanglement_small_reinsertion',
        'vertex_entanglement_reinsertion',
        'vertex_entanglement',
        'network_entanglement_mid',
        'network_entanglement_small',
        'network_entanglement_large',
        'GDM',
        'GDMR',
        'CoreGDM',
        'EGND'
    ]

    priority_map = {algo: i + 1 for i, algo in enumerate(specified_priority_group)}

    df_copy = df.copy()
    df_copy['priority_level'] = df_copy['algorithm_name'].map(priority_map).fillna(0)

    df_sorted = df_copy.sort_values(
        by=[NETWORK_COL, THRESHOLD_COL, 'priority_level'],
        ascending=[True, True, True]
    )

    df_winners = df_sorted.drop_duplicates(subset=[NETWORK_COL], keep='first')

    all_algorithms = df['algorithm_name'].unique()
    df_all_algos = pd.DataFrame(all_algorithms, columns=['algorithm'])

    win_counts = df_winners['algorithm_name'].value_counts().reset_index()
    win_counts.columns = ['algorithm', 'win_count']

    final_report = pd.merge(df_all_algos, win_counts, on='algorithm', how='left')
    final_report['win_count'] = final_report['win_count'].fillna(0).astype(int)
    final_report = final_report.sort_values(by=['win_count', 'algorithm'], ascending=[False, True])

    output_filename = os.path.join(output_dir, f"{dataset_name}-algorithm_win_counts.xlsx")
    final_report.to_excel(output_filename, index=False, engine='openpyxl')

    print(f"[+] Win-count report saved: '{output_filename}'")
    print(f"    Covers {len(final_report)} algorithms with tie-breaking applied.")


# ---------------------------------------------------------------------------
#  Main processing pipeline
# ---------------------------------------------------------------------------

def process_data(results_dir, network_dir, output_dir):
    """Orchestrate the full consolidation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] XLSX output directory: '{output_dir}'")

    network_paths_map = find_target_networks(network_dir)
    if not network_paths_map:
        return

    target_networks = list(network_paths_map.keys())

    df_results = load_all_results_data(results_dir)
    if df_results.empty:
        return

    df_filtered = df_results[df_results[NETWORK_COL].isin(target_networks)].copy()
    if df_filtered.empty:
        print(f"[!] Warning: no records in '{results_dir}' match networks in '{network_dir}'.")
        return
    print(f"[+] Filtered to {len(df_filtered)} records matching target networks.")

    dataset_name = os.path.basename(os.path.normpath(network_dir))
    if not dataset_name:
        dataset_name = "dataset"

    print("[*] Mapping standardized algorithm names...")
    df_filtered = map_algorithm_names(df_filtered)

    generate_win_count_report(df_filtered, output_dir, dataset_name)
    generate_algorithm_xlsx_files(df_filtered, output_dir, dataset_name)
    generate_network_json_labels(df_filtered, network_paths_map)


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate multi-algorithm results and generate analysis reports / JSON labels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-r', '--results_dir', type=str, required=True,
        help="Directory containing source result files (.xlsx, .csv).\n"
             "Searched recursively."
    )
    parser.add_argument(
        '-n', '--network_dir', type=str, required=True,
        help="Directory containing source network files (*_edges.npz).\n"
             "Searched recursively."
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True,
        help="Directory for generated XLSX report files."
    )
    args = parser.parse_args()

    print("--- Starting consolidation ---")
    process_data(args.results_dir, args.network_dir, args.output_dir)
    print("\n--- All tasks completed ---")


if __name__ == "__main__":
    main()
