#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/early_warning/evaluate_classic_robustness.py

Evaluate classical robustness metrics on remnant graphs and optionally compare
them with TCR-GIN predictions.

Function
--------
This script:
1. Scans remnant-graph files and deduplicates them by algorithm and step
2. Computes classical robustness metrics, including:
   - LCC
   - natural connectivity
   - R(rand)
   - R(DCR)
3. Optionally runs TCR-GIN inference to obtain predicted collapse distance
4. Parses external results files to recover or merge LCC curves
5. Supplements collapse-step records for random-attack baselines (R1/R2)
6. Saves a unified metrics CSV
7. Generates multi-panel robustness plots unless `--skip_plot` is set

Inputs
------
- `--input_root`: root directory of the dataset / remnants
- `--output_dir`: directory for CSV and figures
- `--initial_size`: initial graph size for normalization
- `--collapse_target`: target LCC threshold for collapse
- `--model_config`: optional TCR-GIN config for prediction
- `--n_workers`: number of worker processes for CPU metric computation
- `--skip_plot`: skip plotting and only export CSV

Outputs
-------
- `<dataset_name>_metrics.csv`
- robustness plot files in PDF / SVG / PNG format

Usage
-----
Example:
    python experiments/early_warning/evaluate_classic_robustness.py \
        --input_root path/to/dataset \
        --output_dir path/to/output \
        --initial_size 6474 \
        --collapse_target 0.30 \
        --model_config experiments/early_warning/configs/example.yaml
"""

import os
import sys
import re
import argparse
import math
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import json
import torch
import glob
import ast
from scipy import linalg
from pathlib import Path
from tqdm import tqdm
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator


# ==============================================================================
# Project imports and path setup
# ==============================================================================
try:
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from torch_geometric.data import Batch
    from data_loader import load_single_graph
    from model.tcr_gin import TCR_GIN
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False

warnings.filterwarnings('ignore')


# ==============================================================================
# Algorithm-name mappings and constants
# ==============================================================================
ORDERED_ALGOS = [
    'DC', 'DCR', 'BC', 'BCR', 'R1', 'R2', 'MS', 'MSR',
    'CI1', 'CI2', 'CI3', 'EIs1', 'EIs2',
    'CoreHD', 'DomiRank', 'FINDER',
    'GND', 'GNDR', 'VE', 'VER',
    'GDM', 'GDMR', 'CoreGDM',
    'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR',
    'EGND'
]

FILENAME_TO_SHORT_NAME = {
    'CollectiveInfluenceL1': 'CI1',
    'CollectiveInfluenceL2': 'CI2',
    'CollectiveInfluenceL3': 'CI3',
    'GDM': 'GDM',
    'GDMR': 'GDMR',
    'CoreGDM': 'CoreGDM',
    'CoreHD': 'CoreHD',
    'EGND': 'EGND',
    'EI_s1': 'EIs1',
    'EI_s2': 'EIs2',
    'GND': 'GND',
    'GNDR': 'GNDR',
    'MS': 'MS',
    'MSR': 'MSR',
    'network_entanglement_small': 'NES',
    'network_entanglement_small_reinsertion': 'NESR',
    'network_entanglement_mid': 'NEM',
    'network_entanglement_mid_reinsertion': 'NEMR',
    'network_entanglement_large': 'NEL',
    'network_entanglement_large_reinsertion': 'NELR',
    'vertex_entanglement': 'VE',
    'vertex_entanglement_reinsertion': 'VER',
    'degree_T': 'DC',
    'degree_F': 'DCR',
    'betweenness_centrality_T': 'BC',
    'betweenness_centrality_F': 'BCR',
    'eigenvector_centrality_T': 'EC',
    'eigenvector_centrality_F': 'ECR',
    'FINDER_CN': 'FINDER',
    'Domirank': 'DomiRank',
    'degree': 'DC',
    'betweenness_centrality': 'BC',
    'betweenness': 'BC',
    'eigenvector_centrality': 'EC',
    'eigenvector': 'EC',
}

REQUIRED_COLUMNS = [
    'filename', 'algorithm', 'step', 'network_size',
    'LCC', 'natural_connectivity',
    'R(rand)', 'R(DCR)', 'Predicted DC'
]

DEFAULT_SCALE_FACTOR = 1.0
MODEL_INDEX_TO_USE = 0


# ==============================================================================
# General helpers
# ==============================================================================
def to_short_algo(name: str) -> str:
    if name is None:
        return "Unknown"
    s = str(name).strip()
    if s.startswith('_'):
        s = s[1:]
    return FILENAME_TO_SHORT_NAME.get(s, s)


def _parse_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().upper()
    return True if s in ['TRUE', '1', 'T', 'YES'] else False


def infer_algo_from_results_row(heuristic_raw, static_raw):
    h = to_short_algo(heuristic_raw)
    st = _parse_bool(static_raw)
    raw = str(heuristic_raw).strip().lower() if heuristic_raw is not None else ""
    if raw in ['degree', 'deg']:
        return 'DC' if st else 'DCR'
    if raw in ['betweenness_centrality', 'betweenness', 'bc']:
        return 'BC' if st else 'BCR'
    return h


def extract_and_normalize_algo(raw_name):
    s = str(raw_name).strip().replace('.npz', '').replace('_edges', '')
    s = re.sub(r'[-_]\d+$', '', s)
    s = re.sub(r'[-_]\d+$', '', s)

    candidates = list(FILENAME_TO_SHORT_NAME.keys()) + ORDERED_ALGOS
    candidates = sorted(list(set(candidates)), key=len, reverse=True)

    for cand in candidates:
        if s == cand:
            return to_short_algo(cand)
        if s.endswith(cand):
            idx = len(s) - len(cand)
            if idx > 0 and s[idx - 1] in ['_', '-']:
                return to_short_algo(cand)
    return "Unknown"


def parse_filename_info(filename: str):
    stem = filename.replace('.npz', '')
    match = re.search(r'[_-](?P<step>\d+)(?:_\d+)?_edges$', stem)
    if match:
        step = int(match.group('step'))
        raw_algo = stem[:match.start()]
        return raw_algo, step

    parts = stem.split('_')
    nums = []
    while parts and re.fullmatch(r'-?\d+', parts[-1]):
        nums.append(int(parts.pop()))

    if not nums:
        return "Unknown", -1
    if len(nums) >= 2:
        return '_'.join(parts), nums[1]
    return ('_'.join(parts) if parts else "Unknown", nums[0])


def load_graph_robust(file_path: Path):
    try:
        with np.load(file_path) as loader:
            if 'edges' in loader:
                edges = loader['edges']
            elif 'data' in loader:
                edges = loader['data']
            elif 'edge_index' in loader:
                edges = loader['edge_index']
            else:
                edges = loader[loader.files[0]]
            if edges.shape[0] == 2 and edges.shape[1] != 2:
                edges = edges.T

        G = nx.Graph()
        G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except Exception:
        return None


# ==============================================================================
# CSV / XLSX parsing
# ==============================================================================
def get_results_file_path(input_root_path: Path):
    path = Path(input_root_path)
    pool = list(path.glob("*results.csv")) + list(path.glob("*results.xlsx"))
    if not pool:
        pool = list(path.parent.glob("*results.csv")) + list(path.parent.glob("*results.xlsx"))

    valid_pool = [
        p for p in pool
        if "_metrics" not in p.name and not p.name.startswith("._") and p.exists()
    ]
    if valid_pool:
        valid_pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return valid_pool[0]
    return None


def _clean_removals_string(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = re.sub(r'\bnan\b', 'None', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTRUE\b', 'True', s)
    s = re.sub(r'\bFALSE\b', 'False', s)
    return s


def replay_order_to_lcc(initial_graph, order_list, initial_size):
    if initial_graph is None:
        return pd.DataFrame()

    G = initial_graph.copy()
    records = []
    lcc_0 = len(max(nx.connected_components(G), key=len)) / initial_size if G.number_of_nodes() > 0 else 0.0
    records.append({'step': 0, 'LCC': lcc_0})

    step = 0
    for node_id in order_list:
        step += 1
        try:
            nid = int(node_id)
            if G.has_node(nid):
                G.remove_node(nid)
        except Exception:
            continue

        if G.number_of_nodes() > 0:
            lcc = len(max(nx.connected_components(G), key=len)) / initial_size
        else:
            lcc = 0.0
        records.append({'step': step, 'LCC': float(lcc)})

    return pd.DataFrame(records)


def extract_lcc_from_removal_item(item):
    if isinstance(item, dict):
        for k in ['lcc', 'LCC', 'giant', 'gcc']:
            if k in item:
                try:
                    v = float(item[k])
                    return v if not math.isnan(v) else None
                except Exception:
                    pass
        return None

    if not isinstance(item, (list, tuple)):
        return None

    seq = list(item)

    if len(seq) >= 4:
        try:
            v = float(seq[-2])
            if not math.isnan(v) and 0.0 <= v <= 1.1:
                return v
        except Exception:
            pass

    start_idx = len(seq) - 2 if len(seq) >= 2 else len(seq) - 1
    for j in range(start_idx, -1, -1):
        try:
            v = float(seq[j])
            if not math.isnan(v) and 0.0 <= v <= 1.1:
                return v
        except Exception:
            continue
    return None


def parse_csv_result_file(file_path: Path, initial_graph, initial_size: int, default_threshold: float):
    print(f" -> [Parse] Parsing results file: {file_path.name}")
    try:
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path)

        df.columns = [str(c).strip().lower() for c in df.columns]

        rem_col = None
        for c in df.columns:
            if 'removal' in c:
                rem_col = c
                break

        if 'heuristic' not in df.columns or not rem_col:
            print(" [Warning] Results file is missing 'heuristic' or 'removals'.")
            return {}, []

        results = {}
        for _, row in df.iterrows():
            algo_short = infer_algo_from_results_row(row.get('heuristic'), row.get('static'))
            if algo_short == "Unknown":
                continue

            raw = row.get(rem_col)
            if pd.isna(raw):
                continue

            s = _clean_removals_string(raw)
            if not s:
                continue

            try:
                data_list = ast.literal_eval(s)
            except Exception:
                continue

            if not isinstance(data_list, (list, tuple)) or not data_list:
                continue

            is_simple_list = all(
                isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.isdigit())
                for x in data_list
            )

            if is_simple_list:
                if initial_graph:
                    results[algo_short] = replay_order_to_lcc(initial_graph, list(data_list), initial_size)
                continue

            steps, lccs = [], []
            for item in data_list:
                step_val, lcc = None, None
                if isinstance(item, dict):
                    step_val = item.get('step', item.get('k', item.get('t')))
                    lcc = item.get('lcc', item.get('LCC', item.get('giant')))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    step_val = int(item[0])
                    lcc = extract_lcc_from_removal_item(item)

                if step_val is not None and lcc is not None:
                    steps.append(int(step_val))
                    lccs.append(float(lcc))

            if steps:
                df_curve = pd.DataFrame({'step': steps, 'LCC': lccs}).sort_values('step')
                if df_curve['step'].min() > 0:
                    df_curve = pd.concat([pd.DataFrame([{'step': 0, 'LCC': 1.0}]), df_curve])

                # Mark collapse step (first step where LCC rounded to 2 decimals <= threshold)
                df_curve['is_collapse_step'] = False
                df_curve['LCC_rounded'] = df_curve['LCC'].round(2)
                below_or_equal_threshold = df_curve[df_curve['LCC_rounded'] <= round(default_threshold, 2)]
                if not below_or_equal_threshold.empty:
                    first_collapse_idx = below_or_equal_threshold.index[0]
                    df_curve.loc[first_collapse_idx, 'is_collapse_step'] = True
                else:
                    df_curve.loc[df_curve.index[-1], 'is_collapse_step'] = True

                df_curve = df_curve.drop(columns=['LCC_rounded'])
                results[algo_short] = df_curve

        return results, []
    except Exception as e:
        print(f" [Error] Failed to parse results file: {e}")
        return {}, []


def find_original_graph(input_root: Path, dataset_name: str):
    """
    Find the original network graph file.

    Tries multiple common patterns:
    - <dataset_name>/<network_name>_edges.npz
    - <dataset_name>/<network_name>.gt
    - <dataset_name>/<network_name>.npz
    """
    # Common patterns for original graph files
    patterns = [
        f"*_aggr_edges.npz",
        f"*_aggr.npz",
        f"*_edges.npz",
        f"*.gt",
        f"*_multiplex_aggr_edges.npz",
        f"*_multiplex_aggr.npz",
    ]

    for pattern in patterns:
        candidates = list(input_root.glob(pattern))
        if candidates:
            # Return the first match that doesn't contain "Remnants" in path
            for c in candidates:
                if "Remnants" not in str(c):
                    return c

    return None


def extract_removal_sequence_and_last_lcc(file_path: Path):
    """
    Extract removal sequences and last LCC from CSV results file.

    Returns:
        dict with structure:
        {
            'simple_lists': {algo: [node_ids]},  # DC/BC/DCR/BCR
            'tuple_data': {algo: last_lcc_value}  # Other 24 algorithms
        }
    """
    file_path = Path(file_path)
    simple_lists = {}
    tuple_data = {}

    try:
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path)

        df.columns = [str(c).strip().lower() for c in df.columns]

        rem_col = None
        for c in df.columns:
            if 'removal' in c:
                rem_col = c
                break

        if 'heuristic' not in df.columns or not rem_col:
            return {'simple_lists': {}, 'tuple_data': {}}

        for _, row in df.iterrows():
            heuristic = row.get('heuristic')
            static = row.get('static')
            raw = row.get(rem_col)

            if pd.isna(raw):
                continue

            # Determine algorithm name
            algo_short = infer_algo_from_results_row(heuristic, static)
            if algo_short == "Unknown":
                continue

            s = _clean_removals_string(raw)
            if not s:
                continue

            try:
                data_list = ast.literal_eval(s)
            except Exception:
                continue

            if not isinstance(data_list, (list, tuple)) or not data_list:
                continue

            # Check if it's a simple list (DC/BC/DCR/BCR)
            is_simple_list = all(
                isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.isdigit())
                for x in data_list
            )

            if is_simple_list:
                # Simple node ID list
                simple_lists[algo_short] = [int(x) for x in data_list]
            else:
                # Tuple-based data - extract last LCC
                last_item = data_list[-1]
                lcc = extract_lcc_from_removal_item(last_item)
                if lcc is not None:
                    tuple_data[algo_short] = float(lcc)

        return {'simple_lists': simple_lists, 'tuple_data': tuple_data}
    except Exception as e:
        print(f" [Error] Failed to extract removal sequences: {e}")
        return {'simple_lists': {}, 'tuple_data': {}}


# ==============================================================================
# Core robustness metrics
# ==============================================================================
def calc_lcc_global(G, initial_network_size):
    if G.number_of_nodes() == 0 or initial_network_size <= 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len) if not nx.is_empty(G) else []
    return len(largest_cc) / initial_network_size


def calc_natural_connectivity(G, initial_network_size):
    if G.number_of_nodes() == 0:
        return -10.0
    try:
        adj = nx.to_numpy_array(G)
        evals = linalg.eigvalsh(adj)
        return np.logaddexp.reduce(evals) - np.log(initial_network_size)
    except Exception:
        return 0.0


def simulate_attack_metrics(G, _ignored_initial_size, mode='random'):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {'R': 0.0}

    g_sim = G.copy()
    acc_sum = 0.0
    seq = []

    if mode == 'random':
        seq = np.random.permutation(nodes)

    for i in range(N):
        if g_sim.number_of_nodes() > 0:
            if nx.is_connected(g_sim):
                current_lcc_size = g_sim.number_of_nodes()
            else:
                current_lcc_size = len(max(nx.connected_components(g_sim), key=len))
        else:
            current_lcc_size = 0

        acc_sum += current_lcc_size
        node_to_remove = None

        if mode == 'random':
            node_to_remove = seq[i]
        elif mode == 'degree':
            deg = dict(g_sim.degree())
            if not deg:
                break
            node_to_remove = max(deg, key=lambda k: (deg[k], k))
        else:
            break

        g_sim.remove_node(node_to_remove)

    return {'R': acc_sum / (N * N)}


# ==============================================================================
# Multiprocessing worker utilities
# ==============================================================================
def _init_worker():
    """Limit BLAS threads in child processes to avoid oversubscription."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    import warnings
    warnings.filterwarnings('ignore')


def compute_metrics_for_file(task):
    """
    Run CPU-only metric computation in a worker process.

    Args:
        task: (algo, step, fpath_str, initial_size)

    Returns:
        (algo, step, metrics_dict) or (algo, step, None)
    """
    algo, step, fpath_str, initial_size = task
    fpath = Path(fpath_str)
    G = load_graph_robust(fpath)
    if G is None:
        return (algo, step, None)

    metrics = {
        'network_size': G.number_of_nodes(),
        'LCC': calc_lcc_global(G, initial_size),
        'natural_connectivity': calc_natural_connectivity(G, initial_size),
        'R(rand)': simulate_attack_metrics(G, initial_size, 'random')['R'],
        'R(DCR)': simulate_attack_metrics(G, initial_size, 'degree')['R'],
        # 'natural_connectivity': 0,
        # 'R(rand)': 0,
        # 'R(DCR)': 0,
    }
    return (algo, step, metrics)


def compute_collapse_step_from_original_graph(original_graph, removal_sequence, initial_size, collapse_target):
    """
    Remove nodes from the original graph according to the removal sequence,
    and return the LCC after the last removal.

    Args:
        original_graph: The original graph
        removal_sequence: List of node IDs to remove in order
        initial_size: Original network size
        collapse_target: Collapse threshold

    Returns:
        final_lcc: LCC ratio after removing all nodes in sequence
    """
    if original_graph is None or not removal_sequence:
        return None

    G = original_graph.copy()

    # Remove nodes in sequence
    for node_id in removal_sequence:
        if G.has_node(node_id):
            G.remove_node(node_id)

    # Calculate final LCC
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        final_lcc = len(largest_cc) / initial_size
    else:
        final_lcc = 0.0

    return final_lcc


def compute_collapse_step_random_ordered(last_graph, initial_size, collapse_target):
    """
    For R1/R2: Try removing nodes from the last remaining graph in order of node ID
    (smallest to largest) until LCC drops below threshold.

    Args:
        last_graph: The last remaining graph
        initial_size: Original network size
        collapse_target: Collapse threshold

    Returns:
        final_lcc: LCC ratio after successful removal, or None if no node works
    """
    if last_graph is None or last_graph.number_of_nodes() == 0:
        return None

    threshold_val = collapse_target * initial_size
    connected_components = sorted(nx.connected_components(last_graph), key=len, reverse=True)

    if not connected_components:
        return 0.0

    curr_lcc = len(connected_components[0])
    curr_lcc_ratio = curr_lcc / initial_size

    # If already below threshold, return current LCC
    if curr_lcc_ratio < collapse_target:
        return curr_lcc_ratio

    # Get nodes from largest component and sort by ID
    largest_cc_nodes = sorted(list(connected_components[0]))

    # Try removing nodes in order of ID
    for node in largest_cc_nodes:
        g_temp = last_graph.copy()
        g_temp.remove_node(node)

        if g_temp.number_of_nodes() > 0:
            temp_lcc = len(max(nx.connected_components(g_temp), key=len))
        else:
            temp_lcc = 0

        if temp_lcc < threshold_val:
            return temp_lcc / initial_size

    return None


def supplement_collapse_with_csv_data(clean_file_map, initial_size, collapse_target, dataset_name,
                                     original_graph, csv_data):
    """
    Compute collapse steps for algorithms based on CSV data and original graph.

    Three types of algorithms:
    1. DC/BC/DCR/BCR: Use original graph + CSV removal sequence
    2. R1/R2: Random ordered removal from last remaining graph
    3. Others (24 algos): Use LCC value directly from CSV tuple data

    Args:
        clean_file_map: dict mapping (algo, step) to filepath
        initial_size: initial network size
        collapse_target: collapse threshold
        dataset_name: dataset name
        original_graph: original network graph
        csv_data: dict with 'simple_lists' and 'tuple_data'

    Returns:
        list of extra records for step=-1
    """
    extra_records = []
    algo_max_files = {}

    simple_lists = csv_data.get('simple_lists', {})
    tuple_data = csv_data.get('tuple_data', {})

    # Find the last step file for each algorithm
    for (algo, step), fpath in clean_file_map.items():
        if algo not in algo_max_files or step > algo_max_files[algo][0]:
            algo_max_files[algo] = (step, fpath)

    for algo, (last_step, fpath) in algo_max_files.items():
        print(f"    -> {algo}: checking step {last_step}...")

        # Type 1: DC/BC/DCR/BCR - use original graph + CSV removal sequence
        if algo in simple_lists:
            if original_graph is None:
                print(f"       ERROR: No original graph for {algo}")
                continue

            removal_seq = simple_lists[algo]
            print(f"       Using CSV removal sequence ({len(removal_seq)} nodes)")

            final_lcc = compute_collapse_step_from_original_graph(
                original_graph, removal_seq, initial_size, collapse_target
            )

            if final_lcc is not None:
                virt_fname = f"{dataset_name}-{algo}_-1_edges.npz"
                rec = {
                    'filename': virt_fname,
                    'algorithm': algo,
                    'step': -1,
                    'network_size': max(0, initial_size - len(removal_seq)),
                    'LCC': final_lcc,
                    'natural_connectivity': np.nan,
                    'R(rand)': np.nan,
                    'R(DCR)': np.nan,
                    'Predicted DC': np.nan
                }
                extra_records.append(rec)
                print(f"       Collapse found: LCC={final_lcc:.4f}")
            else:
                print(f"       Failed to compute collapse")

        # Type 2: R1/R2 - random ordered removal from last remaining graph
        elif algo in ['R1', 'R2']:
            G = load_graph_robust(fpath)
            if G is None or G.number_of_nodes() == 0:
                print(f"       ERROR: Cannot load graph")
                continue

            print(f"       Using ordered node removal (by node ID)")
            final_lcc = compute_collapse_step_random_ordered(G, initial_size, collapse_target)

            if final_lcc is not None:
                virt_fname = f"{dataset_name}-{algo}_-1_edges.npz"
                rec = {
                    'filename': virt_fname,
                    'algorithm': algo,
                    'step': -1,
                    'network_size': G.number_of_nodes() - 1,
                    'LCC': final_lcc,
                    'natural_connectivity': np.nan,
                    'R(rand)': np.nan,
                    'R(DCR)': np.nan,
                    'Predicted DC': np.nan
                }
                extra_records.append(rec)
                print(f"       Collapse found: LCC={final_lcc:.4f}")
            else:
                print(f"       Failed to find collapse node")

        # Type 3: Other 24 algorithms - use LCC from CSV tuple data
        elif algo in tuple_data:
            final_lcc = tuple_data[algo]
            print(f"       Using LCC from CSV tuple data: {final_lcc:.4f}")

            G = load_graph_robust(fpath)
            network_size = G.number_of_nodes() if G else initial_size

            virt_fname = f"{dataset_name}-{algo}_-1_edges.npz"
            rec = {
                'filename': virt_fname,
                'algorithm': algo,
                'step': -1,
                'network_size': network_size,
                'LCC': final_lcc,
                'natural_connectivity': np.nan,
                'R(rand)': np.nan,
                'R(DCR)': np.nan,
                'Predicted DC': np.nan
            }
            extra_records.append(rec)
            print(f"       Collapse record created")

        else:
            print(f"       WARNING: No CSV data for {algo}, skipping")

    return extra_records


# ==============================================================================
# Sequential supplement for all algorithms that need collapse steps (OLD VERSION - REPLACED)
# ==============================================================================
def supplement_random_collapse_OLD(clean_file_map, initial_size, collapse_target, dataset_name, random_seed=42, csv_last_nodes=None):
    """
    Add collapse points for all algorithms whose removal lists end while LCC remains above the threshold.

    Prefer the final node specified in csv_last_nodes when available; otherwise remove a random node.

    Args:
        clean_file_map: File mapping {(algo, step): filepath}.
        initial_size: Initial network size.
        collapse_target: Collapse threshold.
        dataset_name: Dataset name.
        random_seed: Random seed for reproducibility.
        csv_last_nodes: {algo: last_node_id} read from CSV files.
    """
    extra_records = []
    algo_max_files = {}
    csv_last_nodes = csv_last_nodes or {}

    # Find the last step for each algorithm
    for (algo, step), fpath in clean_file_map.items():
        if algo not in algo_max_files or step > algo_max_files[algo][0]:
            algo_max_files[algo] = (step, fpath)

    threshold_val = collapse_target * initial_size

    for algo, (last_step, fpath) in algo_max_files.items():
        print(f" -> [Supp] Checking collapse point for {algo} from step {last_step}...")
        G = load_graph_robust(fpath)
        if G is None or G.number_of_nodes() == 0:
            continue

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        if not connected_components:
            curr_lcc = 0
        else:
            curr_lcc = len(connected_components[0])

        # Check if already below threshold
        curr_lcc_ratio = curr_lcc / initial_size
        if curr_lcc_ratio < collapse_target:
            print(f"    -> {algo} step {last_step} already below threshold (LCC={curr_lcc_ratio:.4f})")
            continue

        # Need to find collapse point
        print(f"    -> {algo} step {last_step} above threshold (LCC={curr_lcc_ratio:.4f}), estimating collapse...")
        found_collapse = False
        final_lcc = 0.0
        removed_node = None

        # First try the last node from CSV if available
        if algo in csv_last_nodes:
            candidate_node = csv_last_nodes[algo]
            print(f"    -> Trying CSV last node {candidate_node}...")
            if G.has_node(candidate_node):
                g_temp = G.copy()
                g_temp.remove_node(candidate_node)
                if g_temp.number_of_nodes() > 0:
                    temp_lcc = len(max(nx.connected_components(g_temp), key=len))
                else:
                    temp_lcc = 0

                if temp_lcc < threshold_val:
                    found_collapse = True
                    final_lcc = temp_lcc / initial_size
                    removed_node = candidate_node
                    print(f"    -> CSV node {candidate_node} works: LCC={final_lcc:.4f} < {collapse_target}")
                else:
                    print(f"    -> CSV node {candidate_node} does not bring LCC below threshold (LCC={temp_lcc/initial_size:.4f}), falling back to random")
            else:
                print(f"    -> CSV node {candidate_node} not in graph, falling back to random")

        # If CSV node doesn't work, fall back to random removal
        if not found_collapse:
            largest_cc_nodes = list(connected_components[0])

            # Set random seed for reproducibility
            seed = random_seed + hash(algo) % 10000 + last_step
            np.random.seed(seed)

            np.random.shuffle(largest_cc_nodes)

            for node in largest_cc_nodes:
                g_temp = G.copy()
                g_temp.remove_node(node)
                if g_temp.number_of_nodes() > 0:
                    temp_lcc = len(max(nx.connected_components(g_temp), key=len))
                else:
                    temp_lcc = 0

                if temp_lcc < threshold_val:
                    found_collapse = True
                    final_lcc = temp_lcc / initial_size
                    removed_node = node
                    break

        if found_collapse:
            virt_fname = f"{dataset_name}-{algo}_-1_edges.npz"
            rec = {
                'filename': virt_fname,
                'algorithm': algo,
                'step': -1,  # Explicitly mark as -1
                'network_size': G.number_of_nodes() - 1,
                'LCC': final_lcc,
                'natural_connectivity': np.nan,
                'R(rand)': np.nan,
                'R(DCR)': np.nan,
                'Predicted DC': np.nan
            }
            extra_records.append(rec)
            print(f"    -> Collapse found: removed node {removed_node}, LCC={final_lcc:.4f} < {collapse_target}")
        else:
            print("    -> Collapse threshold was not reached by removing one node from the largest component.")

    return extra_records


# ==============================================================================
# TCR-GIN predictor
# ==============================================================================
class TCRGINPredictor:
    def __init__(self, config_path_str, device):
        self.device = device
        self.models_map = []
        self.enabled = False
        self.input_dim_global = 3
        self.config = {}
        self.config_path = None

        if not config_path_str:
            return

        if not IMPORTS_OK:
            print("[TCR-GIN] Project imports failed. Predictor disabled.")
            return

        try:
            self.config_path = Path(config_path_str).resolve()
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}

            base_params = self.config.get('base_model_params', {})
            self.input_dim_global = base_params.get(
                'feature_dim',
                base_params.get('input_dim', 7)
            )

            self._load_suite()

            if self.models_map:
                self.enabled = True
                print(f"[TCR-GIN] Loaded {len(self.models_map)} model(s).")
        except Exception as e:
            print(f"[TCR-GIN] Init error: {e}")
            self.models_map = []
            self.enabled = False

    @staticmethod
    def _norm_key(k):
        """
        Normalize keys like:
          label_scale_factor
          labelScaleFactor
          Label-Scale-Factor
        into:
          labelscalefactor
        """
        return re.sub(r'[^a-z0-9]', '', str(k).lower())

    @staticmethod
    def _to_positive_float(v):
        try:
            x = float(v)
            if x <= 0:
                return None
            return x
        except Exception:
            return None

    def _lookup_label_scale_factor(self, d):
        """
        Read label_scale_factor from one dictionary only.
        This intentionally does not read label_stats.json or any outputs file.
        """
        if not isinstance(d, dict):
            return None

        for k, v in d.items():
            if self._norm_key(k) == "labelscalefactor":
                return self._to_positive_float(v)

        return None

    def _get_label_scale_factor_from_config(self, model_info, merged_params):
        """
        Strictly read label_scale_factor from the YAML model config.

        Search priority:
          1. model_suite item params
          2. model_suite item itself
          3. merged base_params + model_info params
          4. base_model_params
          5. fixed_params
          6. training_params
          7. train_params
          8. top-level config

        This function never reads label_stats.json.
        """
        candidate_dicts = [
            model_info.get("params", {}) if isinstance(model_info, dict) else {},
            model_info if isinstance(model_info, dict) else {},
            merged_params if isinstance(merged_params, dict) else {},
            self.config.get("base_model_params", {}),
            self.config.get("fixed_params", {}),
            self.config.get("training_params", {}),
            self.config.get("train_params", {}),
            self.config,
        ]

        for d in candidate_dicts:
            scale = self._lookup_label_scale_factor(d)
            if scale is not None:
                return scale

        raise KeyError(
            "label_scale_factor was not found in model_config YAML. "
            "Add, for example, `label_scale_factor: 100.0` under "
            "`base_model_params`, `training_params`, top-level config, "
            "or each `model_suite[].params`."
        )

    def _resolve_paths(self, model_info, config_dir, project_root_guess):
        if 'path' in model_info:
            raw = model_info['path']
            candidates = [
                raw,
                str(config_dir / raw),
                str(project_root_guess / raw)
            ]
            for cand in candidates:
                if glob.glob(cand):
                    return cand

        if 'base_dir' in model_info:
            raw_base = model_info['base_dir']
            candidates = [
                config_dir / raw_base,
                project_root_guess / raw_base
            ]

            for base_search in candidates:
                if base_search.exists():
                    exp_dirs = sorted(list(base_search.glob('exp_*')))
                    if exp_dirs:
                        return str(exp_dirs[0] / 'model_run_*.pt')

        return None

    def _load_suite(self):
        base_params = self.config.get('base_model_params', {})
        config_dir = self.config_path.parent

        # Prefer the real project_root computed near the import block.
        try:
            project_root_guess = project_root
        except NameError:
            project_root_guess = (
                config_dir.parents[1]
                if len(config_dir.parents) > 1
                else config_dir.parent
            )

        key_map = {
            'modelactivationfn': 'activation_fn',
            'modeljktype': 'jk_type',
            'modelusevirtualnode': 'use_virtual_node',
            'pissconsistencylambda': 'consistency_lambda',
            'pisspissk': 'piss_k',
            'modelfeaturedim': 'feature_dim',
            'labelscalefactor': 'label_scale_factor',
            'modellabelscalefactor': 'label_scale_factor',
        }

        model_suite = self.config.get('model_suite', [])

        for model_info in model_suite:
            node_range = model_info.get('node_range', [0, -1])
            path_pattern = self._resolve_paths(model_info, config_dir, project_root_guess)

            if not path_pattern:
                print(f"[TCR-GIN] No model path resolved for model_info: {model_info}")
                continue

            model_files = sorted(glob.glob(path_pattern))
            if not model_files:
                print(f"[TCR-GIN] No model files matched: {path_pattern}")
                continue

            if len(model_files) <= MODEL_INDEX_TO_USE:
                target_model_path = Path(model_files[-1])
            else:
                target_model_path = Path(model_files[MODEL_INDEX_TO_USE])

            params = {
                **base_params,
                **(model_info.get('params', {}) if isinstance(model_info, dict) else {})
            }


            scale_factor = self._get_label_scale_factor_from_config(model_info, params)

            final_params = {}

            for k, v in params.items():
                short_k = key_map.get(self._norm_key(k), k)

                if isinstance(v, str):
                    vv = v.strip()
                    if vv.lower() == 'true':
                        v = True
                    elif vv.lower() == 'false':
                        v = False
                    else:
                        try:
                            if any(ch in vv for ch in ['.', 'e', 'E']):
                                v = float(vv)
                            else:
                                v = int(vv)
                        except Exception:
                            v = vv

                final_params[short_k] = v

            if 'feature_dim' in final_params:
                final_params['input_dim'] = final_params['feature_dim']

            try:
                model_args = argparse.Namespace(**final_params)
                model = TCR_GIN(model_args).to(self.device)
                model.load_state_dict(torch.load(target_model_path, map_location=self.device))
                model.eval()

                self.models_map.append({
                    'range': node_range,
                    'model': model,
                    'scale': scale_factor,
                    'input_dim': final_params.get('input_dim', 7),
                })

            except Exception as e:
                print(f"[TCR-GIN] Failed to load {target_model_path}: {e}")

    def predict(self, file_path):
        if not self.enabled:
            return 0.001

        try:
            input_dim = (
                self.models_map[0]['input_dim']
                if self.models_map
                else self.input_dim_global
            )

            stem = str(file_path).replace('_edges.npz', '')
            data = load_single_graph(stem, feature_dim=input_dim)

            if data is None:
                return 0.001

            N = data.num_nodes
            selected = None

            for item in self.models_map:
                lo, hi = item['range']
                if hi == -1:
                    if N >= lo:
                        selected = item
                        break
                else:
                    if lo <= N < hi:
                        selected = item
                        break

            if selected is None and self.models_map:
                selected = self.models_map[-1]

            if selected is None:
                return 0.001

            batch = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                pred_norm = selected['model'](batch)

            # Same as calculate_decision_window.py:
            #   pred_real = model_output / label_scale_factor
            #   pred_holistic = clamp(pred_real, 0, 1)
            pred_real = pred_norm / selected['scale']
            pred_holistic = float(
                torch.clamp(pred_real, 1/N, 1.0).view(-1)[0].item()
            )

            return pred_holistic

        except Exception as e:
            print(f"[TCR-GIN] Predict failed for {file_path}: {e}")
            return 0.001



# ==============================================================================
# Plotting configuration
# ==============================================================================
SHOW_LCC = True
SHOW_TGT = True
SHOW_NAT = True
SHOW_R_RAND = True
SHOW_R_DCR = True
SHOW_R_BCR = False
SHOW_PRED_DC = True
SHOW_EST_DC = False

PLOT_ORDERED_ALGOS = [
    'DC', 'BC', 'R1', 'DCR', 'BCR', 'R2',
    'DomiRank', 'FINDER', 'CoreHD', 'GDM', 'GDMR', 'CoreGDM',
    'MS', 'MSR', 'CI1', 'CI2', 'CI3', 'EGND',
    'EIs1', 'EIs2', 'GND', 'GNDR', 'VE', 'VER',
    'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR'
]

ALGO_DISPLAY_NAME = {
    'CI1': r'CI $\ell$-1',
    'CI2': r'CI $\ell$-2',
    'CI3': r'CI $\ell$-3',
    'EIs1': r'EI ${\sigma_1}$',
    'EI_s1': r'EI ${\sigma_1}$',
    'EIs2': r'EI ${\sigma_2}$',
    'EI_s2': r'EI ${\sigma_2}$',
}

try:
    COLORS = sns.color_palette("Paired", 12)
except Exception:
    COLORS = plt.get_cmap('tab20').colors

C_LCC = '#1f77b4'
C_NAT = '#2ca02c'
C_RRAND = '#9467bd'
C_RDEG = '#ff7f0e'
C_RBCR = '#e377c2'
C_TIGIN = COLORS[5]
C_EST = '#d62728'
C_TGT = '0.55'

STYLE_LCC_LINE = dict(color=C_LCC, lw=1.5, ls='-', alpha=0.95, label='LCC')
STYLE_LCC_PT = dict(color=C_LCC, lw=0, ls='', marker=None, ms=0, zorder=10)
STYLE_NAT = dict(color=C_NAT, lw=1.3, ls='-', marker=None, alpha=0.9, label='Natural connectivity')
STYLE_RRAND = dict(color=C_RRAND, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(rand)')
STYLE_RDCR = dict(color=C_RDEG, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(DCR)')
STYLE_RBCR = dict(color=C_RBCR, lw=1.3, ls='-', marker=None, alpha=0.9, label='R(BCR)')
STYLE_PRED = dict(color=C_TIGIN, lw=1.1, ls='-', marker=None, label='Collapse distance')
STYLE_EST = dict(color=C_EST, lw=1.3, ls='-', marker=None, label='Est DC (BCR)')


def get_algo_sort_key(name):
    if name in PLOT_ORDERED_ALGOS:
        return PLOT_ORDERED_ALGOS.index(name)
    return 999


def get_subplot_label(idx):
    if idx < 26:
        return chr(ord('a') + idx)
    first = idx // 26 - 1
    second = idx % 26
    return f"{chr(ord('a') + first)}{chr(ord('a') + second)}"


def beautify_axis_basic(ax):
    ax.grid(False)
    ax.tick_params(direction='in', which='both', labelsize=6, length=2.0, width=0.6, pad=1.5)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)


def set_clean_yticks(ax, policy='mid'):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span < 1e-9:
        span = 1.0

    is_integer_mode = False
    if ymax > 10 and span > 1.0:
        fmt_str = "{:.0f}"
        steps = [1, 2, 5, 10]
        is_integer_mode = True
    elif span < 0.05:
        fmt_str = "{:.3f}"
        steps = [1, 2, 5, 10]
    else:
        fmt_str = "{:.2f}"
        steps = [1, 2, 5, 10]

    loc = MaxNLocator(nbins=7, steps=steps, integer=is_integer_mode)
    candidates = loc.tick_values(ymin, ymax)

    margin = 0.15 * span
    exclude_bottom_limit = ymin + margin
    exclude_top_limit = ymax - margin

    valid_ticks = []
    seen_labels = set()
    for t in candidates:
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue

        is_bad = False
        if policy in ['top', 'mid'] and t <= exclude_bottom_limit:
            is_bad = True
        if policy in ['bottom', 'mid'] and t >= exclude_top_limit:
            is_bad = True

        label_str = fmt_str.format(t)
        if label_str in seen_labels:
            is_bad = True

        if not is_bad:
            valid_ticks.append(t)
            seen_labels.add(label_str)

    if len(valid_ticks) < 2:
        center = (ymin + ymax) / 2
        if is_integer_mode:
            t1 = np.floor(center)
            t2 = np.ceil(center)
            if t1 == t2:
                t2 += 1
            valid_ticks = [t1, t2]
        elif span < 0.05:
            step_val = 0.002
            valid_ticks = [center - step_val, center + step_val]
        else:
            step_val = 0.01
            valid_ticks = [center - step_val, center + step_val]

    final_ticks = [t for t in valid_ticks if ymin - 0.01 * span <= t <= ymax + 0.01 * span]
    if not final_ticks:
        final_ticks = valid_ticks

    ax.set_yticks(final_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt_str.replace("{:", "%").replace("}", "")))


def get_local_ylim(series, padding=0.1, min_val=-0.001):
    series = series.dropna()
    if series.empty:
        return (0, 1)

    ymin, ymax = series.min(), series.max()
    span = ymax - ymin
    target_min_span = 0.005 if span < 0.05 else 0.04

    if span < target_min_span:
        center = (ymin + ymax) / 2.0
        if center == 0:
            lower, upper = 0, target_min_span
        else:
            lower = center - target_min_span / 2.0
            upper = center + target_min_span / 2.0
    else:
        lower = ymin - span * padding
        upper = ymax + span * padding

    if min_val is not None:
        if lower < min_val:
            lower = min_val
        if upper - lower < target_min_span:
            upper = lower + target_min_span

    return (lower, upper)


def plot_grid(df, algos_to_plot, n_rows, n_cols, start_idx,
              out_path, filename_prefix, collapse_target,
              is_first_part_of_sequence=False):

    AX_WIDTH = 1.70
    AX_HEIGHT = 2.70
    GAP_W = 0.35
    GAP_H = 0.32
    MARGIN_LEFT = 0.55
    MARGIN_RIGHT = 0.15
    MARGIN_BOTTOM = 0.60
    MARGIN_TOP = 0.30
    LEGEND_SPACE = 0.45

    top_margin_actual = MARGIN_TOP
    if is_first_part_of_sequence:
        top_margin_actual += LEGEND_SPACE

    fig_width = MARGIN_LEFT + (n_cols * AX_WIDTH) + ((n_cols - 1) * GAP_W) + MARGIN_RIGHT
    fig_height = MARGIN_BOTTOM + (n_rows * AX_HEIGHT) + ((n_rows - 1) * GAP_H) + top_margin_actual
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs_left = MARGIN_LEFT / fig_width
    gs_right = 1.0 - (MARGIN_RIGHT / fig_width)
    gs_bottom = MARGIN_BOTTOM / fig_height
    gs_top = 1.0 - (top_margin_actual / fig_height)

    outer_gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        left=gs_left,
        right=gs_right,
        bottom=gs_bottom,
        top=gs_top,
        wspace=GAP_W / AX_WIDTH,
        hspace=GAP_H / AX_HEIGHT
    )

    legend_handles = []
    legend_labels = []

    for i, algo in enumerate(algos_to_plot):
        r, c = i // n_cols, i % n_cols
        sub_gs = outer_gs[r, c].subgridspec(5, 1, hspace=0.00)

        df_algo = df[df['algorithm'] == algo].sort_values('step')
        df_files = df_algo[~df_algo['filename'].astype(str).str.contains('_-1_edges', na=False)]
        df_files = df_files[df_files['filename'] != 'step=-1']

        label_str = get_subplot_label(start_idx + i)
        display_name = ALGO_DISPLAY_NAME.get(algo, algo)

        ax1 = fig.add_subplot(sub_gs[0])
        h_tgt = None
        if SHOW_TGT:
            h_tgt = ax1.axhline(collapse_target, color=C_TGT, ls='--', lw=1.0)
        if SHOW_LCC:
            l1, = ax1.plot(df_algo['step'], df_algo['LCC'], **STYLE_LCC_LINE)
            if i == 0:
                if h_tgt is not None:
                    legend_handles.append(h_tgt)
                    legend_labels.append('Collapse target')
                legend_handles.append(l1)
                legend_labels.append('LCC')

        ax1.set_title(f"({label_str}) {display_name} attack", fontsize=8, fontweight='bold', loc='left', pad=3)
        ax1.set_ylim(get_local_ylim(df_algo['LCC'], min_val=0))
        beautify_axis_basic(ax1)
        set_clean_yticks(ax1, policy='top')
        ax1.tick_params(labelbottom=False)

        ax2 = fig.add_subplot(sub_gs[1], sharex=ax1)
        if SHOW_NAT and not df_files.empty and 'natural_connectivity' in df_files:
            tmp = df_files.dropna(subset=['natural_connectivity'])
            if not tmp.empty:
                l, = ax2.plot(tmp['step'], tmp['natural_connectivity'], **STYLE_NAT)
                if i == 0:
                    legend_handles.append(l)
                    legend_labels.append(STYLE_NAT['label'])
                ax2.set_ylim(get_local_ylim(tmp['natural_connectivity']))
        beautify_axis_basic(ax2)
        set_clean_yticks(ax2, policy='mid')
        ax2.tick_params(labelbottom=False)

        ax3 = fig.add_subplot(sub_gs[2], sharex=ax1)
        if SHOW_R_RAND and not df_files.empty and 'R(rand)' in df_files:
            tmp = df_files.dropna(subset=['R(rand)'])
            if not tmp.empty:
                l, = ax3.plot(tmp['step'], tmp['R(rand)'], **STYLE_RRAND)
                if i == 0:
                    legend_handles.append(l)
                    legend_labels.append(STYLE_RRAND['label'])
                ax3.set_ylim(get_local_ylim(tmp['R(rand)']))
        beautify_axis_basic(ax3)
        set_clean_yticks(ax3, policy='mid')
        ax3.tick_params(labelbottom=False)

        ax4 = fig.add_subplot(sub_gs[3], sharex=ax1)
        vals_layer4 = []
        if not df_files.empty:
            if SHOW_R_DCR and 'R(DCR)' in df_files:
                tmp = df_files.dropna(subset=['R(DCR)'])
                if not tmp.empty:
                    l, = ax4.plot(tmp['step'], tmp['R(DCR)'], **STYLE_RDCR)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_RDCR['label'])
                    vals_layer4.append(tmp['R(DCR)'])
            if SHOW_R_BCR and 'R(BCR)' in df_files:
                tmp = df_files.dropna(subset=['R(BCR)'])
                if not tmp.empty:
                    l, = ax4.plot(tmp['step'], tmp['R(BCR)'], **STYLE_RBCR)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_RBCR['label'])
                    vals_layer4.append(tmp['R(BCR)'])
        if vals_layer4:
            ax4.set_ylim(get_local_ylim(pd.concat(vals_layer4)))
        beautify_axis_basic(ax4)
        set_clean_yticks(ax4, policy='mid')
        ax4.tick_params(labelbottom=False)

        ax5 = fig.add_subplot(sub_gs[4], sharex=ax1)
        vals_layer5 = []
        if not df_files.empty:
            if SHOW_PRED_DC and 'Predicted DC' in df_files:
                tmp = df_files.dropna(subset=['Predicted DC'])
                if not tmp.empty:
                    l, = ax5.plot(tmp['step'], tmp['Predicted DC'], **STYLE_PRED)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_PRED['label'])
                    vals_layer5.append(tmp['Predicted DC'])
            if SHOW_EST_DC and 'Estimated DC (BCR)' in df_files:
                tmp = df_files.dropna(subset=['Estimated DC (BCR)'])
                if not tmp.empty:
                    l, = ax5.plot(tmp['step'], tmp['Estimated DC (BCR)'], **STYLE_EST)
                    if i == 0:
                        legend_handles.append(l)
                        legend_labels.append(STYLE_EST['label'])
                    vals_layer5.append(tmp['Estimated DC (BCR)'])
        if vals_layer5:
            ax5.set_ylim(get_local_ylim(pd.concat(vals_layer5)))
        beautify_axis_basic(ax5)
        set_clean_yticks(ax5, policy='bottom')

        max_step = int(df_algo['step'].max()) if not df_algo.empty else 0
        if max_step <= 20:
            step_interval = 5
        elif max_step <= 50:
            step_interval = 10
        elif max_step <= 100:
            step_interval = 20
        elif max_step <= 200:
            step_interval = 40
        else:
            step_interval = 50

        ax5.xaxis.set_major_locator(MultipleLocator(step_interval))
        ax5.set_xlim(left=0, right=max_step * 1.05 if max_step > 0 else 1.0)
        ax5.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    fig.supxlabel('Attack step', fontsize=9, y=0.02)

    if is_first_part_of_sequence and legend_handles:
        center_x = (gs_left + gs_right) / 2.0
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(center_x, gs_top + 0.015),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=7,
            columnspacing=1.2,
            bbox_transform=fig.transFigure
        )

    for ext in ['pdf', 'svg', 'png']:
        save_to = out_path / f"{filename_prefix}-new.{ext}"
        plt.savefig(save_to, dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"  Saved: {save_to}")
    plt.close()


def run_plotting(csv_path: Path, out_path: Path, collapse_target: float):
    """Load the metrics CSV and generate all plots."""
    print(f"\n{'=' * 50}")
    print(" [Plot] Starting plotting...")
    print(f"{'=' * 50}")

    df = pd.read_csv(csv_path)
    dataset_name = csv_path.name.replace('_metrics.csv', '')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'legend.fontsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    baseline_algos = ['DC', 'BC']
    other_algos = [x for x in df['algorithm'].unique() if x not in baseline_algos and x in PLOT_ORDERED_ALGOS]
    other_algos = sorted(other_algos, key=get_algo_sort_key)

    print("  Plotting baseline algorithms (DC, BC)...")
    plot_grid(
        df, baseline_algos, n_rows=1, n_cols=2, start_idx=0,
        out_path=out_path, filename_prefix=f"{dataset_name}_baseline",
        collapse_target=collapse_target, is_first_part_of_sequence=True
    )

    chunk1 = other_algos[:12]
    if chunk1:
        print(f"  Plotting other algorithms, part 1 ({len(chunk1)} algos)...")
        n_cols1 = 4
        n_rows1 = math.ceil(len(chunk1) / n_cols1)
        plot_grid(
            df, chunk1, n_rows=n_rows1, n_cols=n_cols1, start_idx=0,
            out_path=out_path, filename_prefix=f"{dataset_name}_others_part1",
            collapse_target=collapse_target, is_first_part_of_sequence=True
        )

    chunk2 = other_algos[12:]
    if chunk2:
        print(f"  Plotting other algorithms, part 2 ({len(chunk2)} algos)...")
        n_cols2 = 4
        n_rows2 = math.ceil(len(chunk2) / n_cols2)
        plot_grid(
            df, chunk2, n_rows=n_rows2, n_cols=n_cols2, start_idx=12,
            out_path=out_path, filename_prefix=f"{dataset_name}_others_part2",
            collapse_target=collapse_target, is_first_part_of_sequence=False
        )

    print(" [Plot] Plotting finished.")


# ==============================================================================
# Main workflow
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--initial_size', type=int, required=True)
    parser.add_argument('--collapse_target', type=float, default=0.30)
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument(
        '--n_workers',
        type=int,
        default=25,
        help="Number of worker processes. Use 0 for automatic cpu_count()."
    )
    parser.add_argument(
        '--skip_plot',
        action='store_true',
        help="Skip plotting and only export the CSV."
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    # Set global random seeds for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    input_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    dataset_name = input_root.name

    nested_dir = input_root / f"{dataset_name}-Remnants" / f"{dataset_name}-Remnants"
    single_dir = input_root / f"{dataset_name}-Remnants"
    if nested_dir.exists():
        target_dir = nested_dir
    elif single_dir.exists():
        target_dir = single_dir
    else:
        target_dir = input_root
    print(f" -> [Target Dir] Using directory: {target_dir}")

    initial_graph = None
    step0_files = list(target_dir.glob("*_0_edges.npz"))
    if not step0_files:
        step0_files = list(target_dir.glob("*_0.npz"))
    if step0_files:
        step0_files.sort(key=lambda p: len(str(p)))
        print(f" -> [Init] Found initial graph file: {step0_files[0].name}")
        initial_graph = load_graph_robust(step0_files[0])
        if initial_graph:
            args.initial_size = initial_graph.number_of_nodes()
    else:
        print(" [Error] No step-0 graph file was found in the target directory.")

    csv_path = out_dir / f"{input_root.name}_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    print(" -> [Scan] Scanning .npz files...")
    raw_files = list(target_dir.glob("*.npz"))
    raw_files = [f for f in raw_files if 'edges' in f.name or 'remnant' in str(f).lower()]

    clean_file_map = {}
    for f in raw_files:
        algo_raw, step = parse_filename_info(f.name)
        algo = extract_and_normalize_algo(algo_raw)
        if algo == "Unknown":
            continue
        key = (algo, step)
        if key not in clean_file_map:
            clean_file_map[key] = f
        else:
            if len(f.name) < len(clean_file_map[key].name):
                clean_file_map[key] = f

    print(f" -> [Filter] Retained {len(clean_file_map)} unique files (from {len(raw_files)} raw files).")

    data_records = df.to_dict('records') if not df.empty else []
    existing_keys = set((row['algorithm'], row['step']) for row in data_records if 'algorithm' in row and 'step' in row)
    for (algo, step), fpath in clean_file_map.items():
        if (algo, step) not in existing_keys:
            data_records.append({
                'filename': fpath.name,
                'algorithm': algo,
                'step': step,
                'network_size': -1
            })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = TCRGINPredictor(args.model_config, device)

    tasks = []
    task_indices = []
    pred_tasks = []

    for idx, row in enumerate(data_records):
        fname = row.get('filename')
        if not fname or str(fname) == '-1' or 'step=-1' in str(fname):
            continue

        algo = row['algorithm']
        step = row['step']
        fpath = clean_file_map.get((algo, step))
        if not fpath:
            continue

        if 'predicted_cd' in row:
            row['Predicted DC'] = row.pop('predicted_cd')
        if 'R(deg)' in row:
            row['R(DCR)'] = row.pop('R(deg)')

        cpu_missing = False
        for c in ['network_size', 'LCC', 'natural_connectivity', 'R(rand)', 'R(DCR)']:
            if c not in row or pd.isna(row.get(c)) or row.get(c) == -1:
                cpu_missing = True
                break

        if cpu_missing:
            tasks.append((algo, step, str(fpath), args.initial_size))
            task_indices.append(idx)

        need_pred = predictor.enabled and ('Predicted DC' not in row or pd.isna(row.get('Predicted DC')))
        if need_pred:
            pred_tasks.append((idx, fpath))

    n_workers = args.n_workers if args.n_workers > 0 else mp.cpu_count()
    n_workers = min(n_workers, len(tasks)) if tasks else 0
    print(f" -> Need to compute CPU metrics for {len(tasks)} files using {n_workers} workers.")

    if tasks:
        results_map = {}

        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
            future_to_task = {executor.submit(compute_metrics_for_file, t): t for t in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Computing metrics (parallel)"):
                try:
                    algo, step, metrics = future.result()
                    if metrics is not None:
                        results_map[(algo, step)] = metrics
                except Exception as e:
                    t = future_to_task[future]
                    print(f" [Error] Worker failed for {t[0]} step={t[1]}: {e}")

        for task_idx, (algo, step, fpath_str, _) in zip(task_indices, tasks):
            m = results_map.get((algo, step))
            if m is None:
                continue
            row = data_records[task_idx]
            for key, val in m.items():
                if key not in row or pd.isna(row.get(key)) or row.get(key) == -1:
                    row[key] = val

    if pred_tasks:
        print(f" -> Need to compute Predicted DC for {len(pred_tasks)} files on GPU/accelerator.")
        for idx, fpath in tqdm(pred_tasks, desc="GPU inference"):
            row = data_records[idx]
            pred = predictor.predict(fpath)
            ns = row.get('network_size', args.initial_size)
            if ns == -1:
                ns = args.initial_size
            row['Predicted DC'] = pred * (ns / args.initial_size)

    # Don't call supplement here - will do it after merge
    df_updated = pd.DataFrame(data_records)

    results_file = get_results_file_path(input_root)
    if results_file:
        curves, _ = parse_csv_result_file(results_file, initial_graph, args.initial_size, args.collapse_target)

        dataset_long_prefix = input_root.name
        if not df_updated.empty:
            first_fname = df_updated.iloc[0]['filename']
            if '-' in first_fname:
                parts = first_fname.split('-')
                if len(parts) >= 2:
                    dataset_long_prefix = "-".join(parts[:-1])

        # Only keep collapse steps from CSV curves
        curve_rows = []
        for algo, curve_df in curves.items():
            collapse_rows = curve_df[curve_df.get('is_collapse_step', pd.Series([False]*len(curve_df))) == True]
            if not collapse_rows.empty:
                for _, r in collapse_rows.iterrows():
                    curr_step = int(r['step'])
                    estimated_size = max(0, args.initial_size - curr_step)
                    curve_rows.append({
                        'filename': f"{dataset_long_prefix}-{algo}_-1_edges.npz",
                        'algorithm': algo,
                        'step': curr_step,
                        'network_size': estimated_size,
                        'LCC_curve': float(r['LCC']),
                        'is_collapse_step': True
                    })

        df_curve = pd.DataFrame(curve_rows)
        if not df_curve.empty:
            if df_updated.empty:
                df_final = df_curve.rename(columns={'LCC_curve': 'LCC'})
                df_final = df_final.drop(columns=['is_collapse_step'])
            else:
                df_updated['step'] = df_updated['step'].fillna(0).astype(int)
                df_curve['step'] = df_curve['step'].astype(int)

                # Merge logic: for each algorithm, add collapse row
                all_rows = []
                for algo in df_updated['algorithm'].unique():
                    algo_disk_rows = df_updated[df_updated['algorithm'] == algo].copy()
                    algo_curve_rows = df_curve[df_curve['algorithm'] == algo]

                    # Remove any existing -1 rows first
                    algo_disk_rows = algo_disk_rows[~algo_disk_rows['filename'].astype(str).str.contains('_-1_', na=False)]

                    if not algo_curve_rows.empty:
                        collapse_row = algo_curve_rows.iloc[0]
                        collapse_step = int(collapse_row['step'])
                        collapse_lcc = float(collapse_row['LCC_curve'])

                        max_disk_step = algo_disk_rows['step'].max() if not algo_disk_rows.empty else 0

                        if collapse_step <= max_disk_step:
                            # Collapse step exists in disk files, replace it
                            if collapse_step in algo_disk_rows['step'].values:
                                idx = algo_disk_rows[algo_disk_rows['step'] == collapse_step].index[0]
                                algo_disk_rows.loc[idx, 'LCC'] = collapse_lcc
                                algo_disk_rows.loc[idx, 'filename'] = f"{dataset_long_prefix}-{algo}_-1_edges.npz"
                        else:
                            # Collapse step is beyond disk files, add as new row
                            new_row = pd.DataFrame([{
                                'filename': f"{dataset_long_prefix}-{algo}_-1_edges.npz",
                                'algorithm': algo,
                                'step': collapse_step,
                                'network_size': max(0, args.initial_size - collapse_step),
                                'LCC': collapse_lcc,
                                'natural_connectivity': np.nan,
                                'R(rand)': np.nan,
                                'R(DCR)': np.nan,
                                'Predicted DC': np.nan
                            }])
                            algo_disk_rows = pd.concat([algo_disk_rows, new_row], ignore_index=True)

                    all_rows.append(algo_disk_rows)

                curve_only_algos = set(df_curve['algorithm'].unique()) - set(df_updated['algorithm'].unique())
                for algo in curve_only_algos:
                    algo_curve_rows = df_curve[df_curve['algorithm'] == algo]
                    for _, r in algo_curve_rows.iterrows():
                        new_row = pd.DataFrame([{
                            'filename': f"{dataset_long_prefix}-{algo}_-1_edges.npz",
                            'algorithm': algo,
                            'step': int(r['step']),
                            'network_size': max(0, args.initial_size - int(r['step'])),
                            'LCC': float(r['LCC_curve']),
                            'natural_connectivity': np.nan,
                            'R(rand)': np.nan,
                            'R(DCR)': np.nan,
                            'Predicted DC': np.nan
                        }])
                        all_rows.append(new_row)

                df_final = pd.concat(all_rows, ignore_index=True)
        else:
            df_final = df_updated
    else:
        df_final = df_updated

    # For algorithms without collapse steps, compute them using CSV data and original graph
    if not df_final.empty:
        print(" -> [Supp] Checking algorithms that need collapse steps...")
        algos_with_minus1 = set(
            df_final[df_final['filename'].astype(str).str.contains('_-1_', na=False)]['algorithm'].unique()
        )
        all_algos = set(df_final['algorithm'].unique())
        algos_needing_supp = all_algos - algos_with_minus1

        if algos_needing_supp:
            print(f" -> [Supp] Need to compute collapse steps for: {algos_needing_supp}")

            # Find original graph
            original_graph_file = find_original_graph(input_root, dataset_name)
            if original_graph_file:
                print(f" -> [Info] Found original graph: {original_graph_file.name}")
                original_graph_for_supp = load_graph_robust(original_graph_file)
            else:
                print(f" -> [Warning] Original graph not found")
                original_graph_for_supp = None

            # Extract CSV data (removal sequences and last LCC values)
            csv_data = extract_removal_sequence_and_last_lcc(results_file) if results_file else {}

            # Build clean_file_map ONLY for algorithms that need supplement
            supp_clean_file_map = {}
            for _, row in df_final.iterrows():
                algo = row['algorithm']
                if algo not in algos_needing_supp:
                    continue
                step = int(row['step'])
                fname = str(row['filename'])
                if fname and fname != 'nan' and '_-1_' not in fname:
                    # Use the file path from clean_file_map if available
                    if (algo, step) in clean_file_map:
                        supp_clean_file_map[(algo, step)] = clean_file_map[(algo, step)]

            extra_records = supplement_collapse_with_csv_data(
                supp_clean_file_map,
                args.initial_size,
                args.collapse_target,
                dataset_name,
                original_graph_for_supp,
                csv_data
            )

            if extra_records:
                df_extra = pd.DataFrame(extra_records)
                # Ensure all required columns exist
                for col in REQUIRED_COLUMNS:
                    if col not in df_extra.columns:
                        df_extra[col] = np.nan
                df_final = pd.concat([df_final, df_extra], ignore_index=True)

    # Remove rows without filename
    if not df_final.empty:
        df_final = df_final[df_final['filename'].notna() & (df_final['filename'] != '')]

    for col in REQUIRED_COLUMNS:
        if col not in df_final.columns:
            df_final[col] = np.nan

    # Fix step=-1 ordering: move -1 records to end by setting step = max_step + 1
    for algo in df_final['algorithm'].unique():
        algo_mask = df_final['algorithm'] == algo
        minus_one_mask = algo_mask & (df_final['step'] == -1)
        if minus_one_mask.any():
            max_step = df_final.loc[algo_mask & (df_final['step'] != -1), 'step'].max()
            df_final.loc[minus_one_mask, 'step'] = max_step + 1

    df_final = df_final.sort_values(by=['algorithm', 'step'])
    df_final = df_final[REQUIRED_COLUMNS]
    df_final.to_csv(csv_path, index=False)
    print(f" -> [Done] CSV saved to: {csv_path}")

    if not args.skip_plot:
        run_plotting(csv_path, out_dir, args.collapse_target)

    print(f"\n {'=' * 50}")
    print(f" Finished. Output directory: {out_dir}")
    print(f" {'=' * 50}")


if __name__ == "__main__":
    main()
