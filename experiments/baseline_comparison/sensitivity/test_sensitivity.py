#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/baseline_comparison/sensitivity/test_sensitivity.py

Sensitivity-study evaluation script.

Example:
    python experiments/baseline_comparison/sensitivity/test_sensitivity.py \
        --config experiments/baseline_comparison/sensitivity/configs/test_sensitivity_synth.yaml

    python experiments/baseline_comparison/sensitivity/test_sensitivity.py \
        --config experiments/baseline_comparison/sensitivity/configs/test_sensitivity_REDDIT.yaml
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import glob
from tqdm.auto import tqdm
import warnings
import json
import time
import contextlib
import shutil
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
from copy import deepcopy
import networkx as nx

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.data.collate")
from model.tcr_gin import TCR_GIN


def merge_configs(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# --- Feature Calculation Helper Functions ---
def get_feature_set_from_dim(dim):
    if dim <= 3:
        return 'basic'
    if dim <= 5:
        return 'extended'
    return 'full'


def calculate_node_features(G, feature_set='full'):
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        dim_map = {'basic': 3, 'extended': 5, 'full': 7}
        return np.empty((0, dim_map.get(feature_set, 7))), []

    features_dict = {}
    degrees = np.array([d for _, d in G.degree()])
    features_dict['degree'] = degrees
    features_dict['clustering'] = np.array([nx.clustering(G, u) for u in nodes])
    core_numbers = nx.core_number(G)
    features_dict['kcore'] = np.array([core_numbers.get(u, 0) for u in nodes])

    if feature_set in ['extended', 'full']:
        avg_neighbor_deg = np.zeros(n_nodes)
        for i, u in enumerate(nodes):
            if G.degree(u) > 0:
                avg_neighbor_deg[i] = sum(G.degree(v) for v in G.neighbors(u)) / G.degree(u)
        features_dict['avg_neighbor_deg'] = avg_neighbor_deg
        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
            features_dict['pagerank'] = np.array([pr.get(u, 0) for u in nodes])
        except Exception:
            features_dict['pagerank'] = np.zeros(n_nodes)

    if feature_set == 'full':
        try:
            k = min(50, n_nodes - 1) if n_nodes > 50 else None
            bc = nx.betweenness_centrality(G, k=k, seed=42) if k is not None else nx.betweenness_centrality(G)
            features_dict['betweenness'] = np.array([bc.get(u, 0) for u in nodes])
        except Exception:
            features_dict['betweenness'] = np.zeros(n_nodes)
        try:
            ec = nx.eigenvector_centrality_numpy(G, max_iter=100, tol=1e-4)
            features_dict['eigenvector'] = np.array([ec.get(u, 0) for u in nodes])
        except Exception:
            max_deg = max(degrees) if len(degrees) > 0 else 1
            features_dict['eigenvector'] = degrees / max(1, max_deg)

    feature_names = list(features_dict.keys())
    feature_matrix = np.column_stack([features_dict[f] for f in feature_names])
    return feature_matrix, feature_names
# ----------------------------------------------------------------------


def collect_sensitivity_dataset_paths(config):
    all_paths = set()
    experiments = config.get('experiments', [])
    for exp_config in experiments:
        templates = exp_config.get('templates', {})
        d_path_template = templates.get('d_path', '')

        for instance in exp_config.get('instances', []):
            if 'source_scales' in instance and 'source_generators' in instance:
                scales = instance['source_scales']
                generators = instance['source_generators']
                for s in scales:
                    for g in generators:
                        if d_path_template:
                            all_paths.add(d_path_template.format(scale=s, generator=g))
            elif 'constituents' in instance and '{constituent_name}' in d_path_template:
                for c in instance['constituents']:
                    all_paths.add(d_path_template.format(constituent_name=c))
            elif 'path' in instance:
                all_paths.add(instance['path'])

    return sorted(list(all_paths))


@contextlib.contextmanager
def manage_test_data(config):
    source_root = config['base_config']['global_settings']['datasets_root_dir']
    if not os.path.isabs(source_root):
        source_root = os.path.join(PROJECT_ROOT, source_root)

    dataset_rel_paths = collect_sensitivity_dataset_paths(config)

    temp_dir = os.path.join(PROJECT_ROOT, f"temp_data_cache_sensitivity_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\n[INFO] Caching data to temporary directory: {temp_dir}")
    try:
        if not dataset_rel_paths:
            print("[WARNING] No dataset paths found to cache.")
        else:
            print(f"[INFO] Found {len(dataset_rel_paths)} unique 'test' directories to copy.")
            for rel_path in tqdm(dataset_rel_paths, desc="  Copying data"):
                src = os.path.join(source_root, rel_path)
                dest = os.path.join(temp_dir, rel_path)
                if os.path.isdir(src):
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    print(f"  [Warning] Source directory not found, skipping: {src}")
        yield temp_dir
    finally:
        print(f"\n[INFO] Cleaning up temporary data directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_graph_ids(dataset_dir, limit=-1):
    if not os.path.isdir(dataset_dir):
        print(f"  [Warning] Dataset directory does not exist: {dataset_dir}")
        return []
    edge_files = glob.glob(os.path.join(dataset_dir, '*_edges.npz'))
    graph_ids = [os.path.basename(f).replace('_edges.npz', '') for f in edge_files]
    sorted_ids = sorted(graph_ids)
    if limit > 0 and limit < len(sorted_ids):
        print(f"  [INFO] Sampling first {limit} graphs out of {len(sorted_ids)} available.")
        return sorted_ids[:limit]
    return sorted_ids


def parse_exp_name_to_params(exp_name):
    params = {}
    # Expanded key mapping for sensitivity analysis variables
    key_mapping = {
        # PISS Parameters
        'pisspissk': 'piss.piss_k',
        'pissconsistencylambda': 'piss.consistency_lambda',

        # Model Parameters (Hyperparameters)
        'modelhiddendim': 'model.hidden_dim',
        'modellayers': 'model.num_layers',
        'modelnumlayers': 'model.num_layers',  # Alias
        'modeldropout': 'model.dropout',
        'modelfeaturedim': 'model.feature_dim',
        'modeljktype': 'model.jk_type',
        'modelactivationfn': 'model.activation_fn'
    }

    parts = exp_name.split('_')
    start_idx = 0
    if parts[0] == 'exp':
        start_idx = 1

    name_body = '_'.join(parts[start_idx:])
    tokens = name_body.split('_')

    i = 0
    while i < len(tokens) - 1:
        key_short = tokens[i]

        if key_short in key_mapping:
            full_key = key_mapping[key_short]
            val_str = tokens[i + 1]
            value = None

            if val_str.lower() == 'true':
                value = True
            elif val_str.lower() == 'false':
                value = False
            else:
                try:
                    value = int(val_str)
                except ValueError:
                    try:
                        value = float(val_str)
                    except ValueError:
                        value = val_str

            params[full_key] = value
            i += 2
        else:
            i += 1

    return params


def run_sensitivity_exp(task_config):
    global_settings = task_config['global_settings']
    model_source = task_config['model_source']
    test_dataset = task_config['test_dataset']

    model_dir = os.path.join(PROJECT_ROOT, global_settings['models_root_dir'], model_source['path'])

    # Locate actual experiment folder
    if not glob.glob(os.path.join(model_dir, "model_run_*.pt")):
        sub_exps = glob.glob(os.path.join(model_dir, "exp_*"))
        if sub_exps:
            model_dir = sub_exps[0]

    exp_name = os.path.basename(model_dir)
    print(f"\n- Testing Sensitivity Config: {exp_name}")
    print(f"  - On Dataset Group: {test_dataset['name']}")

    # 1. Parse Parameters from Folder Name
    parsed_params = parse_exp_name_to_params(exp_name)
    current_model_params = deepcopy(task_config['model_params'])
    # Also update piss_params if present, although TCR_GIN main logic mostly uses model_params arg object

    # 2. Apply Parsed Parameters
    for key, value in parsed_params.items():
        if key.startswith('model.'):
            param_name = key.split('.')[1]
            current_model_params[param_name] = value

    model_files = sorted(glob.glob(os.path.join(model_dir, "model_run_*.pt")))
    if not model_files:
        print(f"  [Warning] No models found in {model_dir}. Skipping.")
        return None

    device = torch.device("cuda" if global_settings['device'] == 'auto' and torch.cuda.is_available() else "cpu")
    models = []
    for path in model_files:
        try:
            args = argparse.Namespace(**current_model_params)
            # Ensure input_dim matches feature_dim logic unless specified
            args.input_dim = current_model_params['feature_dim']

            model = TCR_GIN(args).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"  [Warning] Failed to load model {path}: {e}")
    if not models:
        print("  [ERROR] No models were successfully loaded. Skipping.")
        return None

    sample_limit = task_config.get('sampling', {}).get('test_limit', -1)

    maes_by_run = [[] for _ in models]
    times_by_run = [[] for _ in models]

    dataset_paths = test_dataset['paths']

    for d_path in dataset_paths:
        dataset_dir = os.path.join(global_settings['datasets_root_dir'], d_path)
        graph_ids = get_graph_ids(dataset_dir, limit=sample_limit)

        labels_file_exists = lambda gid: os.path.exists(os.path.join(dataset_dir, f"{gid}_label.json"))
        labels = {
            gid: json.load(open(os.path.join(dataset_dir, f"{gid}_label.json")))['critical_threshold']
            for gid in graph_ids if labels_file_exists(gid)
        }

        if not labels:
            continue

        for i, model in enumerate(models):
            preds, truths, total_times = [], [], []

            for gid, truth in labels.items():
                try:
                    prefix = os.path.join(dataset_dir, gid)

                    # 1. Feature Calc Time
                    t_feat_start = time.time()
                    edges = np.load(f"{prefix}_edges.npz", allow_pickle=True)['edges']
                    G = nx.Graph()
                    G.add_edges_from(edges)
                    feature_set = get_feature_set_from_dim(current_model_params['feature_dim'])
                    _ = calculate_node_features(G, feature_set)
                    t_feat_end = time.time()
                    feat_time = t_feat_end - t_feat_start

                    # 2. Inference
                    features_loaded = np.load(f"{prefix}_features.npy")[:, :current_model_params['feature_dim']]
                    x = torch.tensor(features_loaded, dtype=torch.float32)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                    data = Data(x=x, edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1))
                    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                    data = data.to(device)

                    t_inf_start = time.time()
                    with torch.no_grad():
                        y_pred = torch.clamp(model(data), 0.0, 1.0).item()
                    t_inf_end = time.time()
                    inf_time = t_inf_end - t_inf_start

                    preds.append(y_pred)
                    truths.append(truth)
                    total_times.append(feat_time + inf_time)

                except Exception:
                    continue

            if truths:
                maes_by_run[i].append(mean_absolute_error(truths, preds))
                times_by_run[i].append(np.mean(total_times))

    final_maes = []
    final_times = []

    for i in range(len(models)):
        if maes_by_run[i]:
            final_maes.append(np.mean(maes_by_run[i]))
            final_times.append(np.mean(times_by_run[i]))

    if not final_maes:
        print("  [ERROR] No valid results produced.")
        return None

    mae_mean, mae_std = np.mean(final_maes), np.std(final_maes)
    time_mean = np.mean(final_times)

    result = {
        'Dataset Name': test_dataset['name'],
        'Constituents': test_dataset['constituents_display'],
        'MAE': f"{mae_mean:.4f} ± {mae_std:.4f}",
        'MAE (mean)': mae_mean,
        'MAE (std)': mae_std,
        'Time (s)': f"{time_mean:.4f}"
    }

    for key, value in parsed_params.items():
        short_key = key.split('.')[-1]
        result[short_key] = value

    return result


def generate_sensitivity_tasks(config, base_config):
    tasks = []
    experiments = config.get('experiments', [])

    for exp_config in experiments:
        print(f"\nParsing Group: '{exp_config.get('name', 'Unnamed')}'")
        templates = exp_config.get('templates', {})
        m_path_template = templates.get('m_path', '')
        d_path_template = templates.get('d_path', '')

        for instance in exp_config.get('instances', []):
            task = deepcopy(base_config)
            task = merge_configs(
                task,
                {k: v for k, v in instance.items() if k not in ['source_scales', 'source_generators', 'constituents']}
            )

            d_paths = []
            display_str = ""

            if 'source_scales' in instance and 'source_generators' in instance:
                scales = instance['source_scales']
                generators = instance['source_generators']
                for s in scales:
                    for g in generators:
                        d_paths.append(d_path_template.format(scale=s, generator=g))
                display_str = f"Scales: {scales}, Gens: {generators}"
                dataset_name_key = instance.get('mix_id', 'Unknown')

            elif 'constituents' in instance:
                for c in instance['constituents']:
                    d_paths.append(d_path_template.format(constituent_name=c))
                display_str = f"Constituents: {instance['constituents']}"
                dataset_name_key = instance.get('dataset_name', 'Unknown')
            else:
                continue

            relative_model_path = m_path_template.format(mix_id=dataset_name_key, dataset_name=dataset_name_key)

            task['model_source'] = {'name': dataset_name_key, 'path': relative_model_path}
            task['test_dataset'] = {
                'name': dataset_name_key,
                'constituents_display': display_str,
                'paths': d_paths
            }

            full_model_base_path = os.path.join(
                PROJECT_ROOT,
                base_config['global_settings']['models_root_dir'],
                relative_model_path
            )

            # Sensitivity runs produce many exp_* folders (one for each param combination)
            exp_folders = glob.glob(os.path.join(full_model_base_path, "exp_*"))

            if not exp_folders:
                if os.path.exists(full_model_base_path):
                    exp_folders = [full_model_base_path]

            if not exp_folders:
                print(f"  [Warning] No experiment folders found in {full_model_base_path}")
                continue

            for exp_folder_path in exp_folders:
                sub_task = deepcopy(task)
                rel_path = os.path.relpath(
                    exp_folder_path,
                    os.path.join(PROJECT_ROOT, base_config['global_settings']['models_root_dir'])
                )
                sub_task['model_source']['path'] = rel_path
                tasks.append(sub_task)

    print(f"  - Generated {len(tasks)} total sensitivity testing tasks.")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="GNN Sensitivity Study Testing Framework")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    print("=" * 60 + "\nGNN Sensitivity Study Testing Framework\n" + "=" * 60)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    all_results = []
    with manage_test_data(config) as temp_data_root:
        config['base_config']['global_settings']['datasets_root_dir'] = temp_data_root
        all_tasks = generate_sensitivity_tasks(config, config['base_config'])
        if not all_tasks:
            print("\n[INFO] No valid tasks were generated.")
        else:
            all_results = [
                result for task in tqdm(all_tasks, desc="Executing sensitivity tasks")
                if (result := run_sensitivity_exp(task))
            ]

    if not all_results:
        print("\n[INFO] No results were generated.")
        return

    print("\n[INFO] Aggregating results into the final table...")
    df = pd.DataFrame(all_results)

    fixed_cols = ['Dataset Name', 'Constituents']
    metric_cols = ['MAE', 'MAE (mean)', 'MAE (std)', 'Time (s)']
    param_cols = sorted([col for col in df.columns if col not in fixed_cols + metric_cols])

    column_order = fixed_cols + param_cols + metric_cols
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    if 'MAE (mean)' in df.columns:
        df['MAE (mean)'] = df['MAE (mean)'].map('{:.4f}'.format)
    if 'MAE (std)' in df.columns:
        df['MAE (std)'] = df['MAE (std)'].map('{:.4f}'.format)

    sort_cols = ['Dataset Name'] + param_cols
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    output_path = os.path.join(
        PROJECT_ROOT,
        config['base_config']['global_settings']['output_dir'],
        config['base_config']['global_settings']['output_filename']
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] All jobs completed. Sensitivity study results saved to:\n{output_path}")
    print("\n" + "=" * 60 + "\nTest Run Finished\n" + "=" * 60)


if __name__ == "__main__":
    main()
