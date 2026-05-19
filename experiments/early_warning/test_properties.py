# experiments/early_warning/test_properties.py
#
# TI-GIN model property evaluation framework.  Evaluates trained models on
# dismantling remnant graphs, computing monotonicity, smoothness, accuracy,
# and additivity (consistency) metrics.  Generates publication-quality
# comparison plots against traditional baseline algorithms.
#
# Usage:
#   python experiments/early_warning/test_properties.py \
#       --config experiments/early_warning/configs/<config>.yaml

# === 0. IMPORTS & GLOBAL CONFIGURATION ===
import os
import sys
import argparse
import yaml
import re
import json
from pathlib import Path
from glob import glob
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from itertools import cycle

# ==============================================================================
# Default scale factor used when label_stats.json is not found.
# A value of 1.0 means no scaling is assumed.
# ==============================================================================
DEFAULT_SCALE_FACTOR = 1.0

# --- Dynamic project module imports ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from torch_geometric.data import Batch
    from torch_geometric.utils import to_networkx
    from data_loader import load_single_graph
    from model.tcr_gin import TCR_GIN
except ImportError as e:
    print(f"Module import failed: {e}\nEnsure the script is located under the correct project directory."); sys.exit(1)

# --- Plotting and scientific computing imports ---
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    import seaborn as sns
    matplotlib.use('Agg')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8, 'axes.labelsize': 10, 'axes.titlesize': 12,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'figure.titlesize': 14, 'axes.linewidth': 1, 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.major.width': 1, 'ytick.major.width': 1,
    })
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("Warning: matplotlib or seaborn not found — visualization disabled.")

try:
    from scipy.stats import pearsonr
    SCIPY_ENABLED = True
except ImportError:
    SCIPY_ENABLED = False
    if PLOTTING_ENABLED: print("Warning: scipy not found — Pearson correlation and some plots disabled.")

try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    NETWORKX_ENABLED = False
    print("Warning: networkx not found — 'component_aware' prediction strategy unavailable.")

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --- Algorithm name mapping and visual styles ---
FILENAME_TO_SHORT_NAME = {
    'CollectiveInfluenceL1': 'CI1', 'CollectiveInfluenceL2': 'CI2', 'CollectiveInfluenceL3': 'CI3', 'GDM': 'GDM', 'GDMR': 'GDMR', 'CoreGDM': 'CoreGDM', 'CoreHD': 'CoreHD', 'EGND': 'EGND', 'EI_s1': 'EIs1', 'EI_s2': 'EIs2', 'GND': 'GND', 'GNDR': 'GNDR', 'MS': 'MS', 'MSR': 'MSR', 'network_entanglement_small': 'NES', 'network_entanglement_small_reinsertion': 'NESR', 'network_entanglement_mid': 'NEM', 'network_entanglement_mid_reinsertion': 'NEMR', 'network_entanglement_large': 'NEL', 'network_entanglement_large_reinsertion': 'NELR', 'vertex_entanglement': 'VE', 'vertex_entanglement_reinsertion': 'VER', 'degree_T': 'DC', 'degree_F': 'DCR', 'betweenness_centrality_T': 'BC', 'betweenness_centrality_F': 'BCR', 'eigenvector_centrality_T': 'EC', 'eigenvector_centrality_F': 'ECR', 'FINDER_CN': 'FINDER', 'Domirank': 'DomiRank'
}
try:
    COLORS = sns.color_palette("Paired", 12)
except (ImportError, NameError):
    COLORS = plt.get_cmap('tab20').colors
ALGO_STYLES = {
    'TI-GIN (Ours)': {'color': COLORS[5], 'marker': 'o', 'linestyle': '-', 'zorder': 20, 'markersize': 1.5, 'linewidth': 1.0}, 'BCR':    {'color': COLORS[1], 'marker': 's', 'linestyle': '--', 'markersize': 2.0, 'fillstyle': 'none', 'linewidth': 0.6}, 'DCR':    {'color': COLORS[3], 'marker': '^', 'linestyle': ':', 'markersize': 2.0, 'fillstyle': 'none', 'linewidth': 0.6}, 'CoreGDM':{'color': COLORS[7], 'marker': 'D', 'linestyle': '-.', 'markersize': 2.0, 'fillstyle': 'none', 'linewidth': 0.6}, 'CoreHD': {'color': COLORS[9], 'marker': 'p', 'linestyle': (0, (3, 5, 1, 5)), 'markersize': 2.5, 'fillstyle': 'none', 'linewidth': 0.6}, 'CI1':    {'color': COLORS[11], 'marker': '+', 'linestyle': '--', 'markersize': 2.5, 'linewidth': 0.6}, 'GDM':    {'color': 'saddlebrown', 'marker': 'x', 'linestyle': ':', 'markersize': 2.5, 'linewidth': 0.6}, 'default':{'color': 'grey', 'marker': 'None', 'linestyle': '--', 'linewidth': 0.5}
}
other_algos = [name for name in FILENAME_TO_SHORT_NAME.values() if name not in ALGO_STYLES]
color_cycle = cycle(sns.color_palette("tab20", 20))
custom_linestyles = [ '--', ':', '-.', (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (1, 1)), (0, (5, 1)), (0, (1, 5)), (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (1, 10)), (0, (3, 3, 1, 3, 1, 3)), (5, (10, 3)), (0, (10, 5, 3, 5)), (0, (8, 2, 2, 2, 2, 2)), ]
linestyle_cycle = cycle(custom_linestyles)
for algo in sorted(other_algos):
    ALGO_STYLES[algo] = {'color': next(color_cycle), 'marker': 'None', 'linestyle': next(linestyle_cycle), 'linewidth': 0.5}


# === 1. SETUP & ENVIRONMENT ===

def setup_environment(config_path_str):
    """Parse CLI arguments, load and process the config file, return the full config dict."""
    config_path = Path(config_path_str).resolve()
    config_dir = config_path.parent

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['dismantling_path'] = (config_dir / config['dismantling_data_dir']).resolve()
    config['output_dir'] = (config_dir / config['output_dir']).resolve()
    config['dataset_name'] = config['dismantling_path'].name
    config['summary_csv_path'] = config['output_dir'] / f"summary-{config['dataset_name']}-{config_path.stem}.csv"

    config['prediction_strategy'] = config.get('prediction_strategy', 'holistic')
    if config['prediction_strategy'] == 'component_aware' and not NETWORKX_ENABLED:
        print("Warning: 'component_aware' strategy requested but networkx is not installed. Falling back to 'holistic'.")
        config['prediction_strategy'] = 'holistic'

    return config

# === 2. MODEL & DATA LOADING ===

def _get_short_algo_name(filename_stem):
    sorted_keys = sorted(FILENAME_TO_SHORT_NAME.keys(), key=len, reverse=True)
    for long_name in sorted_keys:
        if filename_stem.endswith(long_name):
            return FILENAME_TO_SHORT_NAME[long_name]
    return filename_stem

def parse_params_from_folder_name(dir_name):
    params = {}
    param_str = re.sub(r'^exp_\d+_', '', dir_name)
    parts = param_str.split('_')
    i = 0
    while i < len(parts) - 1:
        key, value = parts[i], parts[i+1]
        params[key] = value
        i += 2
    return params

def load_model_suite(suite_config, base_model_params, device):
    """Load a collection of models from disk, automatically locating label_stats.json for each."""
    suite = []
    for model_info in suite_config:
        node_range = model_info['node_range']
        model_paths = glob(model_info['path'])
        if not model_paths:
            print(f"Warning: no models found matching path: {model_info['path']}")
            continue

        # Smart label_stats.json discovery:
        # 1. If found, read its scale_factor.
        # 2. If not found, assume no scaling was applied (scale_factor = 1.0).
        model_run_dir = Path(model_paths[0]).parent  # e.g., .../exp_001

        candidates = []
        # Try string replacement: models -> outputs
        try:
            path_str = str(model_run_dir)
            if 'models' in path_str:
                output_path_str = path_str.replace('models', 'outputs')
                candidates.append(Path(output_path_str) / 'label_stats.json')
        except: pass
        # Fallback: look directly in the model directory
        candidates.append(model_run_dir / 'label_stats.json')

        label_stats = None
        loaded_path = "N/A"

        for p in candidates:
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        label_stats = json.load(f)
                    loaded_path = str(p)
                    break
                except Exception as e:
                    print(f"  [Error] Found stats at {p} but failed to read: {e}")

        # Determine final scale factor
        if label_stats is None:
            label_stats = {'scale_factor': DEFAULT_SCALE_FACTOR}
            print(f"  [Info] label_stats.json NOT FOUND. Assuming NO SCALING (Scale Factor = {DEFAULT_SCALE_FACTOR})")
        else:
            if 'scale_factor' not in label_stats:
                label_stats['scale_factor'] = DEFAULT_SCALE_FACTOR
                print(f"  [Warning] label_stats.json found but missing 'scale_factor'. Using default {DEFAULT_SCALE_FACTOR}")
            else:
                print(f"  [Success] Loaded label stats from: {loaded_path} (Factor = {label_stats['scale_factor']})")

        models = []
        for model_path in sorted(model_paths):
            try:
                current_model_params = base_model_params.copy()
                if 'params' in model_info: current_model_params.update(model_info['params'])

                final_params = {}
                key_map = {'modelactivationfn': 'activation_fn', 'modeljktype': 'jk_type', 'modelusevirtualnode': 'use_virtual_node', 'pissconsistencylambda': 'consistency_lambda', 'pisspissk': 'piss_k', 'modelfeaturedim': 'feature_dim'}

                for key, value in current_model_params.items():
                    short_key = key_map.get(key.lower(), key)
                    if isinstance(value, str):
                        if value.lower() == 'true': converted_value = True
                        elif value.lower() == 'false': converted_value = False
                        else:
                            try:
                                if '.' in value: converted_value = float(value)
                                else: converted_value = int(value)
                            except ValueError: converted_value = value
                    else:
                        converted_value = value
                    final_params[short_key] = converted_value

                if 'feature_dim' in final_params and final_params['feature_dim'] is not None:
                    final_params['input_dim'] = final_params['feature_dim']

                model_args = argparse.Namespace(**final_params)
                model = TCR_GIN(model_args).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")

        if models:
            suite.append({
                'node_range': node_range,
                'models': models,
                'label_stats': label_stats
            })
    return suite


def get_model_and_stats_for_graph(graph, model_suite, run_idx):
    """Select the appropriate model and label stats for a graph based on its node count."""
    n_nodes = graph.num_nodes
    selected_group = None

    for model_group in model_suite:
        min_nodes, max_nodes = model_group['node_range']
        is_in_range = (min_nodes <= n_nodes < max_nodes)
        if max_nodes == -1: is_in_range = (min_nodes <= n_nodes)

        if is_in_range:
            selected_group = model_group
            break

    if selected_group is None and model_suite:
        selected_group = model_suite[-1]

    if selected_group:
        models_list = selected_group['models']
        stats = selected_group['label_stats']
        num_available = len(models_list)
        if run_idx < num_available: model_index = run_idx
        else: model_index = 0
        return models_list[model_index], stats

    # Fallback if no suite loaded
    return None, {'scale_factor': DEFAULT_SCALE_FACTOR}


def load_and_prepare_traditional_data(dismantling_path, for_plotting=False):
    """Load and clean baseline algorithm Excel result data."""
    purpose = "plotting" if for_plotting else "metric computation"
    print(f"Loading baseline algorithm data for {purpose}...")
    dataset_name = dismantling_path.name
    results_path = dismantling_path / f"{dataset_name}-Remnants" / "results_final"
    data_cache = {}
    if not results_path.exists():
        print(f"Warning: baseline results directory not found: '{results_path}'")
        return data_cache
    loaded_algos = []
    for xlsx_file in results_path.glob('*.xlsx'):
        try:
            short_algo_name = _get_short_algo_name(xlsx_file.stem)
            if not short_algo_name: continue
            df = pd.read_excel(xlsx_file)
            if 'network' not in df.columns: continue
            numeric_cols = ['critical_threshold', 'network_size']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)
            df['seq_id'] = df['network'].str.rsplit('_', n=1).str[0]
            df['step'] = pd.to_numeric(df['network'].str.rsplit('_', n=1).str[1], errors='coerce')
            df.dropna(subset=['step'], inplace=True)
            df['step'] = df['step'].astype(int)
            df.drop_duplicates(subset=['seq_id', 'step'], keep='first', inplace=True)
            data_cache[short_algo_name] = df
            loaded_algos.append(short_algo_name)
        except Exception as e:
            print(f"Failed to load file {xlsx_file.name}: {e}")
    if loaded_algos:
        print(f"  Successfully loaded {len(set(loaded_algos))} baseline algorithms: {', '.join(sorted(list(set(loaded_algos))))}")
    else:
        print("  No baseline algorithm data loaded.")
    return data_cache

# === 3. CORE ANALYSIS STAGES ===

def prepare_and_predict_all_graphs(model_suite, device, run_idx, config, model_args):
    """
    Stage 1: Iterate over all remnant graphs and compute predictions.

    Rescaling: uses scale_factor via division (Real = Norm / Scale).
    If scale_factor is missing, defaults to 1.0 (no rescaling).
    """
    dataset_name = config['dataset_name']
    remnants_path = config['dismantling_path'] / f"{dataset_name}-Remnants" / f"{dataset_name}-Remnants"
    prediction_strategy = config.get('prediction_strategy', 'holistic')
    force_component_aware = (prediction_strategy == 'component_aware')

    remnant_files = list(remnants_path.glob('*_edges.npz'))
    if not remnant_files:
        print("Warning: no graph files found in the Remnants directory.")
        return {}

    prediction_cache = {}
    print(f"  Stage 1: Preparing data and running dual prediction (strategy: {prediction_strategy})...")

    if force_component_aware and not NETWORKX_ENABLED:
        print("    [Warning] Strategy is component_aware but networkx is not installed; falling back to holistic.")

    with torch.no_grad():
        for remnant_path in tqdm(remnant_files, desc="    Decomposing & predicting", leave=False):
            graph_id = remnant_path.name.replace('_edges.npz', '')
            g_remnant = load_single_graph(str(remnant_path).replace('_edges.npz', ''), feature_dim=model_args.input_dim)
            if g_remnant is None or g_remnant.num_nodes == 0: continue

            # 1. Holistic prediction
            model_remnant, stats = get_model_and_stats_for_graph(g_remnant, model_suite, run_idx)
            pred_holistic = 0.0
            if model_remnant:
                batch_remnant = Batch.from_data_list([g_remnant]).to(device)
                pred_norm = model_remnant(batch_remnant).item()

                # Rescale: Real = Norm / Scale
                scale = float(stats.get('scale_factor', DEFAULT_SCALE_FACTOR))
                pred_holistic = pred_norm / scale

            # 2. Aggregated (component-aware) prediction
            pred_aggregated = pred_holistic
            if force_component_aware and NETWORKX_ENABLED and g_remnant.num_edges > 0:
                g_nx = to_networkx(g_remnant, to_undirected=True)
                components = list(nx.connected_components(g_nx))
                if len(components) > 1:
                    total_weighted_pred, total_nodes = 0.0, g_remnant.num_nodes
                    for node_indices in components:
                        node_idx_tensor = torch.tensor(list(node_indices), dtype=torch.long)
                        sub_g = g_remnant.subgraph(node_idx_tensor)
                        if sub_g.num_nodes == 0: continue

                        model_comp, stats_comp = get_model_and_stats_for_graph(sub_g, model_suite, run_idx)
                        if not model_comp: continue
                        batch_comp = Batch.from_data_list([sub_g]).to(device)
                        pred_comp_norm = model_comp(batch_comp).item()

                        # Component-level rescaling
                        scale = float(stats_comp.get('scale_factor', DEFAULT_SCALE_FACTOR))
                        pred_comp_real = pred_comp_norm / scale

                        total_weighted_pred += pred_comp_real * sub_g.num_nodes
                    pred_aggregated = total_weighted_pred / total_nodes if total_nodes > 0 else 0.0

            prediction_cache[graph_id] = {
                'graph': g_remnant,
                'pred_holistic': pred_holistic,
                'pred_aggregated': pred_aggregated,
                'y': g_remnant.y
            }
    return prediction_cache

def _compute_sequence_metrics(steps, values, n_initial):
    """Compute monotonicity and smoothness metrics using physics-based thresholds."""
    if len(steps) < 2 or n_initial <= 0: return {}
    sorted_pairs = sorted(zip(steps, values))
    steps = np.array([p[0] for p in sorted_pairs])
    vals = np.array([p[1] for p in sorted_pairs])

    unit_cost = 1.0 / n_initial
    val_diffs = np.diff(vals)
    step_spans = np.diff(steps)
    threshold_mono = step_spans * unit_cost
    threshold_smooth = step_spans * 2.0 * unit_cost

    total_steps = len(val_diffs)
    epsilon = 1e-9
    mono_violation_mask = val_diffs > (threshold_mono + epsilon)
    m_count = np.sum(mono_violation_mask)
    m_freq = m_count / total_steps if total_steps > 0 else 0.0
    m_excess = np.maximum(0, val_diffs - threshold_mono)
    m_int = np.mean(m_excess)

    smooth_violation_mask = val_diffs < -(threshold_smooth + epsilon)
    s_count = np.sum(smooth_violation_mask)
    s_freq = s_count / total_steps if total_steps > 0 else 0.0
    s_excess = np.maximum(0, -val_diffs - threshold_smooth)
    s_int = np.mean(s_excess)

    return {'M_freq': m_freq, 'M_int': m_int, 'S_freq': s_freq, 'S_int': s_int}

def test_monotonicity(prediction_cache, config):
    """Stage 2: Compute monotonicity and smoothness metrics (hardcodes N=6474 for route datasets)."""
    print("  Stage 2: Analysing monotonicity and smoothness...")
    prediction_strategy = config.get('prediction_strategy', 'holistic')
    if not prediction_cache: return {}, {}
    sequences = defaultdict(list)
    for graph_id in prediction_cache.keys():
        match = re.match(r'(.+)_(\d+)$', graph_id)
        if match:
            seq_id, step = match.groups()
            sequences[seq_id].append(int(step))
    all_metrics = defaultdict(list)
    plot_data = defaultdict(lambda: {'steps': [], 'd_norm': []})
    all_abs_errors = []
    is_route_dataset = 'route' in config['dataset_name'].lower()

    for seq_id in sorted(sequences.keys()):
        steps = sorted(sequences[seq_id])
        if not steps: continue
        n_initial = 0
        if is_route_dataset or 'route' in seq_id.lower():
            n_initial = 6474
        else:
            initial_graph_id = f"{seq_id}_0"
            if initial_graph_id in prediction_cache:
                n_initial = prediction_cache[initial_graph_id]['graph'].num_nodes
        if n_initial == 0: continue

        seq_steps = []
        seq_vals = []
        for step in steps:
            graph_id = f"{seq_id}_{step}"
            if graph_id not in prediction_cache: continue
            data = prediction_cache[graph_id]
            g = data['graph']
            pred_val = data['pred_aggregated'] if prediction_strategy == 'component_aware' else data['pred_holistic']

            if data['y'] is not None:
                abs_err = abs(pred_val - data['y'].item())
                all_abs_errors.append(abs_err)

            d_norm = pred_val * (g.num_nodes / n_initial)
            seq_steps.append(step)
            seq_vals.append(d_norm)

        plot_data[seq_id]['steps'] = seq_steps
        plot_data[seq_id]['d_norm'] = seq_vals
        metrics = _compute_sequence_metrics(seq_steps, seq_vals, n_initial)
        for k, v in metrics.items(): all_metrics[k].append(v)

    final_metrics = {}
    for k, v_list in all_metrics.items():
        if v_list:
            final_metrics[f"{k}_mean"] = np.mean(v_list)
            final_metrics[f"{k}_std"] = np.std(v_list)
        else:
            final_metrics[f"{k}_mean"] = -1.0; final_metrics[f"{k}_std"] = 0.0

    if all_abs_errors:
        final_metrics['monotonicity_accuracy_mae_mean'] = np.mean(all_abs_errors)
        final_metrics['monotonicity_accuracy_mae_std'] = np.std(all_abs_errors)
    else:
        final_metrics['monotonicity_accuracy_mae_mean'] = -1.0
        final_metrics['monotonicity_accuracy_mae_std'] = 0.0

    return final_metrics, dict(plot_data)

def test_additivity(prediction_cache, config):
    """Stage 3: Analyse additivity (consistency between holistic and component-aggregated predictions)."""
    print("  Stage 3: Analysing additivity (consistency)...")
    default_metrics = {'additivity_consistency_mae_mean': -1.0, 'additivity_consistency_mae_std': 0.0}
    if not prediction_cache: return default_metrics, {}
    all_abs_errors = []
    plot_data = defaultdict(lambda: {'remnant_norm': [], 'components_norm': [], 'metric_val': 0.0})

    for graph_id, data in prediction_cache.items():
        if data['pred_aggregated'] is not None:
            pred_holistic = data['pred_holistic']
            pred_aggregated = data['pred_aggregated']
            error = abs(pred_holistic - pred_aggregated)
            all_abs_errors.append(error)
            if error > 1e-6:
                base_network_id = graph_id.split('-')[0]
                plot_data[base_network_id]['remnant_norm'].append(pred_holistic)
                plot_data[base_network_id]['components_norm'].append(pred_aggregated)
                plot_data['global']['remnant_norm'].append(pred_holistic)
                plot_data['global']['components_norm'].append(pred_aggregated)

    if all_abs_errors:
        mean_val = np.mean(all_abs_errors)
        std_val = np.std(all_abs_errors)
    else:
        mean_val, std_val = 0.0, 0.0

    if 'global' in plot_data:
        plot_data['global']['metric_val'] = mean_val
    for k in plot_data:
        if k != 'global':
            diffs = np.abs(np.array(plot_data[k]['remnant_norm']) - np.array(plot_data[k]['components_norm']))
            plot_data[k]['metric_val'] = np.mean(diffs) if len(diffs) > 0 else 0.0

    metrics = {'additivity_consistency_mae_mean': mean_val, 'additivity_consistency_mae_std': std_val}
    return metrics, dict(plot_data)

# === 4. VISUALIZATION ===

def visualize_additivity_publication(plot_data, output_path, dataset_type, dataset_name):
    if not PLOTTING_ENABLED or not plot_data: return
    if 'global' in plot_data and len(plot_data['global']['remnant_norm']) > 0: groups = ['global']
    else: groups = sorted([k for k in plot_data.keys() if k != 'global' and len(plot_data[k].get('remnant_norm',[])) > 0])
    if not groups: print("  Additivity plot: insufficient data points."); return
    print(f"Generating additivity (consistency) plot -> {output_path}")
    num_plots = len(groups)
    if num_plots == 1: nrows, ncols = 1, 1; fig_size = (4.5, 4.5)
    else: ncols = 3; nrows = math.ceil(num_plots / ncols); fig_size = (10, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, squeeze=False, sharex=True, sharey=True)
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
    for i, group_name in enumerate(groups):
        ax = axes[i]; data = plot_data[group_name]; mae_val = data.get('metric_val', 0.0)
        ax.plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', linewidth=1.5, zorder=1, label='Perfect Consistency')
        ax.scatter(data['components_norm'], data['remnant_norm'], alpha=0.4, s=15, facecolors='#3498db', edgecolors='none', zorder=2, label='Predictions')
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
        ax.text(0.05, 0.95, f"MAE = {mae_val:.4f}", transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#bdc3c7', lw=1, alpha=0.9))
        ax.label_outer()
    for j in range(num_plots, len(axes)): axes[j].set_visible(False)
    fig.supxlabel('Component network collapse distance', fontsize=12, y=0.02)
    fig.supylabel('Remnant network collapse distance', fontsize=12, x=0.02)
    if num_plots == 1: axes[0].legend(loc='lower right', frameon=True, fontsize=9, framealpha=0.9)
    else: handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.98]); plt.savefig(output_path, dpi=600, bbox_inches='tight'); plt.close(fig)

def visualize_monotonicity_scheme1_enhanced(plot_data, traditional_data, output_dir, dataset_type):
    if not PLOTTING_ENABLED or not traditional_data: return
    AX_WIDTH = 2.2; AX_HEIGHT = 1.8; GAP_WIDTH = 0.2; GAP_HEIGHT = 0.3
    MARGIN_LEFT = 0.5; MARGIN_RIGHT = 0.1; MARGIN_BOTTOM = 0.5; MARGIN_TOP_BASE = 0.3
    LEGEND_HEIGHT = 0.7; TITLE_SPACE = 0.25; LEGEND_GAP = 0.05
    FIXED_ZOOM_BASELINES = ['CoreHD', 'DCR', 'BCR', 'GDMR', 'FINDER', 'DomiRank']
    ORDERED_ALGOS = [
        'DC', 'BC', 'R1', 'DCR', 'BCR', 'R2',
        'DomiRank', 'FINDER', 'CoreHD', 'GDM', 'GDMR', 'CoreGDM',
        'MS', 'MSR', 'CI1', 'CI2', 'CI3', 'EGND',
        'EIs1', 'EIs2', 'GND', 'GNDR', 'VE', 'VER',
        'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR'
    ]
    algo_order_map = {name: i for i, name in enumerate(ORDERED_ALGOS)}
    algo_order_map['TI-GIN (Ours)'] = -1
    def get_algo_sort_key(algo_name): return algo_order_map.get(algo_name, 999)
    def get_seq_sort_key(seq_id): return algo_order_map.get(seq_id.rsplit('-', 1)[-1], 999)
    def get_subplot_label(i): return chr(ord('a') + i) if i < 26 else f"a{chr(ord('a') + i - 26)}"
    def _draw_equidistant_diagonal_grid(ax, slope, num_lines=6, **line_kwargs):
        x_min, x_max = ax.get_xlim(); y_min, y_max = ax.get_ylim()
        corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
        c_vals = [y - slope * x for x, y in corners]
        c_min, c_max = min(c_vals), max(c_vals)
        c_range = np.linspace(c_min, c_max, num=num_lines)
        for c in c_range: ax.axline(xy1=(0, c), slope=slope, **line_kwargs)
    def _find_strict_zoom_region(seq_id, plot_data, traditional_data, num_neighbors=4):
        if seq_id not in plot_data or not plot_data[seq_id]['steps']: return None, 1
        tcr_gin_df = pd.DataFrame(plot_data[seq_id]).set_index('steps')['d_norm']
        if tcr_gin_df.empty: return None, 1
        steps = sorted(plot_data[seq_id]['steps'])
        step_interval = max(1, int(np.median(np.diff(steps)))) if len(steps) > 1 else 1
        n_initial = None
        for df in traditional_data.values():
            initial_rows = df[(df['seq_id'] == seq_id) & (df['step'] == 0)]
            if not initial_rows.empty: n_initial = initial_rows['network_size'].iloc[0]; break
        if n_initial is None or n_initial == 0: return None, step_interval
        theoretical_slope_abs = 1.0 / n_initial
        search_start_idx = int(len(steps) * 0.1); search_end_idx = int(len(steps) * 0.9)
        best_strict = (None, -float('inf')); best_relaxed = (None, -float('inf'))
        for i in range(search_start_idx, search_end_idx):
            start_idx = max(0, i - num_neighbors); end_idx = min(len(steps) - 1, i + num_neighbors)
            win_steps = steps[start_idx : end_idx+1]; win_vals = tcr_gin_df.loc[win_steps]
            if len(win_vals) < 2: continue
            diffs = win_vals.diff().dropna()
            if (diffs > 1e-9).any(): continue
            current_score = 0; count_valid_baselines = 0
            for algo in FIXED_ZOOM_BASELINES:
                if algo in traditional_data:
                    df = traditional_data[algo]
                    seq_df = df[df['seq_id'] == seq_id].set_index('step')
                    common_idx = seq_df.index.intersection(win_steps)
                    if len(common_idx) < 2: continue
                    base_vals = seq_df.loc[common_idx]
                    base_d_norm = base_vals['critical_threshold'] * base_vals['network_size'] / n_initial
                    gap = (base_d_norm - tcr_gin_df.loc[common_idx]).mean()
                    base_diffs = base_d_norm.diff().dropna()
                    volatility = base_diffs[base_diffs > 0].sum()
                    current_score += (gap * 1.0 + volatility * 10.0)
                    count_valid_baselines += 1
            if count_valid_baselines == 0: continue
            total_drop = win_vals.iloc[0] - win_vals.iloc[-1]
            total_span = win_steps[-1] - win_steps[0]
            avg_slope_abs = total_drop / total_span if total_span > 0 else 0
            region = (win_steps[0], win_steps[-1])
            if avg_slope_abs < theoretical_slope_abs:
                if current_score > best_strict[1]: best_strict = (region, current_score)
            else:
                if current_score > best_relaxed[1]: best_relaxed = (region, current_score)
        if best_strict[0] is not None: return best_strict[0], step_interval
        elif best_relaxed[0] is not None: return best_relaxed[0], step_interval
        else: return None, step_interval
    all_seq_ids = set(plot_data.keys())
    for algo_df in traditional_data.values(): all_seq_ids.update(algo_df['seq_id'].unique())
    sorted_sequences = sorted(list(all_seq_ids), key=get_seq_sort_key)
    if not sorted_sequences: return
    base_network_id = sorted_sequences[0].rsplit('-', 1)[0]
    GLOBAL_MAX_X = 0
    for seq_id in sorted_sequences:
        if seq_id in plot_data: GLOBAL_MAX_X = max(GLOBAL_MAX_X, max(plot_data[seq_id]['steps'], default=0))
        for df in traditional_data.values():
            seq_df = df[df['seq_id'] == seq_id]
            if not seq_df.empty: GLOBAL_MAX_X = max(GLOBAL_MAX_X, seq_df['step'].max())
    baseline_colors = cycle(['#666666', '#888888', '#aaaaaa', '#cccccc'])
    baseline_styles = {name: {**ALGO_STYLES.get(name, ALGO_STYLES['default']), 'color': next(baseline_colors), 'zorder': 10} for name in traditional_data.keys()}
    layouts = [(3, 3), (4, 3), (3, 3)]; sequence_chunks = []; start_idx = 0
    for rows, cols in layouts:
        count = rows * cols; sequence_chunks.append(sorted_sequences[start_idx : start_idx + count]); start_idx += count
    global_subplot_index = 0; all_handles = {}

    for fig_idx, (layout, seq_chunk) in enumerate(zip(layouts, sequence_chunks)):
        if not seq_chunk: continue
        nrows, ncols = layout
        output_path = output_dir / f"monotonicity_{base_network_id}_scheme1_part{fig_idx + 1}.png"
        print(f"  Generating monotonicity plot (Part {fig_idx + 1}/{len(layouts)}) -> {output_path}")
        zoom_meta_data = {}; global_max_y_span = 0.05
        for seq_id in seq_chunk:
            zoom_region, step_interval = _find_strict_zoom_region(seq_id, plot_data, traditional_data, num_neighbors=4)
            if zoom_region:
                x_start, x_end = zoom_region; local_y_min, local_y_max = 1.0, 0.0; has_data = False
                if seq_id in plot_data:
                    ti_df = pd.DataFrame(plot_data[seq_id])
                    z_ti = ti_df[(ti_df['steps'] >= x_start) & (ti_df['steps'] <= x_end)]
                    if not z_ti.empty: local_y_min = min(local_y_min, z_ti['d_norm'].min()); local_y_max = max(local_y_max, z_ti['d_norm'].max()); has_data = True
                n_initial = None
                for df in traditional_data.values():
                    ir = df[(df['seq_id'] == seq_id) & (df['step'] == 0)]
                    if not ir.empty: n_initial = ir['network_size'].iloc[0]; break
                if n_initial:
                    for algo_name in FIXED_ZOOM_BASELINES:
                        if algo_name in traditional_data:
                            df = traditional_data[algo_name]; seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                            z_b = seq_df[(seq_df['step'] >= x_start) & (seq_df['step'] <= x_end)]
                            if not z_b.empty:
                                d_n = z_b['critical_threshold'] * z_b['network_size'] / n_initial
                                local_y_min = min(local_y_min, d_n.min()); local_y_max = max(local_y_max, d_n.max()); has_data = True
                if has_data:
                    global_max_y_span = max(global_max_y_span, local_y_max - local_y_min)
                    zoom_meta_data[seq_id] = {'region': zoom_region, 'y_center': (local_y_min + local_y_max) / 2.0, 'n_initial': n_initial}
        global_max_y_span *= 1.2
        content_width = ncols * AX_WIDTH + (ncols - 1) * GAP_WIDTH; content_height = nrows * AX_HEIGHT + (nrows - 1) * GAP_HEIGHT
        current_top_margin = TITLE_SPACE + LEGEND_GAP + LEGEND_HEIGHT + 0.1 if fig_idx == 0 else MARGIN_TOP_BASE
        fig_width = MARGIN_LEFT + content_width + MARGIN_RIGHT; fig_height = MARGIN_BOTTOM + content_height + current_top_margin
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height)); axes = axes.flatten()
        plot_area_top = 1.0 - (current_top_margin / fig_height); plot_area_left = MARGIN_LEFT / fig_width; plot_area_right = 1.0 - (MARGIN_RIGHT / fig_width); plot_area_center_x = (plot_area_left + plot_area_right) / 2.0
        plt.subplots_adjust(left=plot_area_left, right=plot_area_right, bottom=MARGIN_BOTTOM / fig_height, top=plot_area_top, wspace=GAP_WIDTH / AX_WIDTH, hspace=GAP_HEIGHT / AX_HEIGHT)

        for i, seq_id in enumerate(seq_chunk):
            ax = axes[i]; attack_strategy = seq_id.rsplit('-', 1)[-1]
            ax.set_title(f"({get_subplot_label(global_subplot_index)}) {attack_strategy} Attack", loc='left', fontsize=10); global_subplot_index += 1
            tcr_gin_style = {**ALGO_STYLES.get('TI-GIN (Ours)', {}), 'linewidth': 1.5, 'zorder': 20, 'label': 'TI-GIN (Ours)'}
            if seq_id in plot_data and plot_data[seq_id]['steps']:
                h, = ax.plot(plot_data[seq_id]['steps'], plot_data[seq_id]['d_norm'], **tcr_gin_style)
                if 'TI-GIN (Ours)' not in all_handles: all_handles['TI-GIN (Ours)'] = h
            n_initial = None
            for df in traditional_data.values():
                initial_rows = df[(df['seq_id'] == seq_id) & (df['step'] == 0)]
                if not initial_rows.empty: n_initial = initial_rows['network_size'].iloc[0]; break
            if n_initial is None or n_initial == 0: continue
            for algo_name, df in traditional_data.items():
                seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                if not seq_df.empty:
                    d_norm = seq_df['critical_threshold'] * seq_df['network_size'] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES['default']); h, = ax.plot(seq_df['step'], d_norm, label=algo_name, **style)
                    if algo_name not in all_handles: all_handles[algo_name] = h
            if seq_id in zoom_meta_data:
                meta = zoom_meta_data[seq_id]; x_start, x_end = meta['region']; y_center = meta['y_center']; n_init = meta['n_initial']
                ax_inset = ax.inset_axes([0.45, 0.45, 0.52, 0.52])
                if seq_id in plot_data: ax_inset.plot(plot_data[seq_id]['steps'], plot_data[seq_id]['d_norm'], **tcr_gin_style)
                for algo_name in FIXED_ZOOM_BASELINES:
                    if algo_name not in traditional_data: continue
                    df = traditional_data[algo_name]; seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                    z_b = seq_df[(seq_df['step'] >= x_start) & (seq_df['step'] <= x_end)]
                    if not z_b.empty:
                        d_n = z_b['critical_threshold'] * z_b['network_size'] / n_init
                        base_style = baseline_styles.get(algo_name, ALGO_STYLES['default']).copy(); base_style['linewidth'] = 0.8; ax_inset.plot(z_b['step'], d_n, **base_style)
                target_bottom = y_center - global_max_y_span / 2; target_top = y_center + global_max_y_span / 2
                if target_bottom < 0: offset = -target_bottom; target_bottom += offset; target_top += offset
                elif target_top > 1.05: offset = target_top - 1.05; target_bottom -= offset; target_top -= offset
                ax_inset.set_xlim(x_start, x_end); ax_inset.set_ylim(max(0, target_bottom), target_top)
                _draw_equidistant_diagonal_grid(ax=ax_inset, slope=-1.0/n_init, num_lines=6, color='gainsboro', linewidth=0.5, zorder=0, alpha=0.8)
                ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3)); ax_inset.tick_params(axis='both', which='major', labelsize=6); ax_inset.set_xticklabels([]); ax_inset.set_yticklabels([])
                for spine in ax_inset.spines.values(): spine.set_edgecolor('#555555'); spine.set_linewidth(0.8)
                ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=0.8)
            ax.set_ylim(0, 0.4); ax.set_xlim(0, GLOBAL_MAX_X * 1.02); ax.label_outer()

        for j in range(len(seq_chunk), len(axes)): axes[j].set_visible(False)
        fig.supxlabel('Attack step', fontsize=10, y=0.02); fig.supylabel('Collapse distance', fontsize=10, x=0.02)
        if fig_idx == 0 and all_handles:
            sorted_handles = [all_handles[k] for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            sorted_labels = [k for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            offset_inches = TITLE_SPACE + LEGEND_GAP; bbox_y_pos = plot_area_top + (offset_inches / fig_height)
            fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(plot_area_center_x, bbox_y_pos), ncol=10, frameon=False, fontsize=7, handlelength=1.5, columnspacing=0.9, bbox_transform=fig.transFigure)
        plt.savefig(output_path, dpi=600, bbox_inches='tight'); plt.close(fig)

def visualize_monotonicity_scheme2_residual(plot_data, traditional_data, output_dir, dataset_type):
    if not PLOTTING_ENABLED or not traditional_data: return
    baseline_colors = cycle(['#666666', '#888888', '#aaaaaa', '#cccccc'])
    baseline_styles = {}
    for algo_name in traditional_data.keys():
        original_style = ALGO_STYLES.get(algo_name, ALGO_STYLES['default']).copy(); original_style['color'] = next(baseline_colors); baseline_styles[algo_name] = original_style
    all_seq_ids = set(plot_data.keys())
    for algo_df in traditional_data.values(): all_seq_ids.update(algo_df['seq_id'].unique())
    if not all_seq_ids: return
    network_sequences = defaultdict(set)
    for seq_id in all_seq_ids: network_sequences[seq_id.rsplit('-', 1)[0]].add(seq_id)
    for base_network_id, sequences in network_sequences.items():
        sorted_sequences = sorted(list(sequences))
        output_path = output_dir / f"monotonicity_{base_network_id}_scheme2_residual.png"
        print(f"  Generating scheme-2 residual plot -> {output_path}")
        num_plots = len(sorted_sequences)
        if num_plots == 0: continue
        ncols, nrows = 3, math.ceil(num_plots / 3)
        fig = plt.figure(figsize=(7.2, 3.5 * nrows)); outer_gs = fig.add_gridspec(nrows, ncols, hspace=0.1, wspace=0.1)
        all_handles, max_x_val = {}, 0
        for seq_id in sorted_sequences:
            if seq_id in plot_data and plot_data[seq_id]['steps']: max_x_val = max(max_x_val, max(plot_data[seq_id]['steps']))
            for df in traditional_data.values():
                seq_df = df[df['seq_id'] == seq_id];
                if not seq_df.empty: max_x_val = max(max_x_val, seq_df['step'].max())
        for i, seq_id in enumerate(sorted_sequences):
            inner_gs = outer_gs[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(inner_gs[0]); ax_residual = fig.add_subplot(inner_gs[1], sharex=ax_main)
            ax_main.set_title(f"({chr(97 + i)}) {seq_id.rsplit('-', 1)[-1]} Attack", loc='left', fontsize=10)
            tcr_gin_map = {}
            if seq_id in plot_data and plot_data[seq_id]['steps']:
                style = ALGO_STYLES.get('TI-GIN (Ours)').copy(); style['linewidth'] = 1.5
                h, = ax_main.plot(plot_data[seq_id]['steps'], plot_data[seq_id]['d_norm'], label='TI-GIN (Ours)', **style)
                if 'TI-GIN (Ours)' not in all_handles: all_handles['TI-GIN (Ours)'] = h
                tcr_gin_map = pd.DataFrame(plot_data[seq_id]).set_index('steps')['d_norm'].to_dict()
            n_initial = None
            for df in traditional_data.values():
                initial_rows = df[(df['seq_id'] == seq_id) & (df['step'] == 0)];
                if not initial_rows.empty: n_initial = initial_rows['network_size'].iloc[0]; break
            if n_initial is None or n_initial == 0: continue
            for algo_name, df in traditional_data.items():
                seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                if not seq_df.empty:
                    seq_df['d_norm'] = seq_df['critical_threshold'] * seq_df['network_size'] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES['default']); h, = ax_main.plot(seq_df['step'], seq_df['d_norm'], label=algo_name, **style)
                    if algo_name not in all_handles: all_handles[algo_name] = h
                    if tcr_gin_map:
                        residuals = [row_data['d_norm'] - tcr_gin_map.get(row_data['step'], np.nan) for _, row_data in seq_df.iterrows()]
                        ax_residual.plot(seq_df['step'], residuals, **style)
            plt.setp(ax_main.get_xticklabels(), visible=False); ax_main.set_ylim(bottom=0); ax_residual.axhline(0, color='red', linestyle='--', linewidth=1.0, zorder=20)
            ax_main.set_xlim(left=0, right=max_x_val * 1.02)
            row_idx, col_idx = i // ncols, i % ncols
            if col_idx == 0: ax_main.set_ylabel('Collapse dist.'); ax_residual.set_ylabel('Residual')
            else: plt.setp(ax_main.get_yticklabels(), visible=False); plt.setp(ax_residual.get_yticklabels(), visible=False)
            if row_idx < nrows - 1: plt.setp(ax_residual.get_xticklabels(), visible=False)
        sorted_handles_labels = sorted(all_handles.items(), key=lambda item: (0, item[0]) if item[0] == 'TI-GIN (Ours)' else (1, item[0]))
        sorted_handles, sorted_labels = [item[1] for item in sorted_handles_labels], [item[0] for item in sorted_handles_labels]
        if sorted_handles:
            legend = fig.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=9, frameon=True, fontsize=7, facecolor='white', edgecolor='lightgrey', framealpha=0.6); legend.get_frame().set_linewidth(0.5)
        fig.supxlabel('Attack step', fontsize=10, y=0.06); fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.89); plt.savefig(output_path, dpi=600, bbox_inches='tight'); plt.close(fig)

def visualize_monotonicity_large_scale(plot_data, traditional_data, output_dir, dataset_type, config=None):
    if not PLOTTING_ENABLED or not traditional_data: return
    AX_WIDTH = 2.2; AX_HEIGHT = 1.8; GAP_WIDTH = 0.2; GAP_HEIGHT = 0.3
    MARGIN_LEFT = 0.65; MARGIN_RIGHT = 0.1; MARGIN_BOTTOM = 0.5; MARGIN_TOP_BASE = 0.3
    LEGEND_HEIGHT = 0.7; TITLE_SPACE = 0.25; LEGEND_GAP = 0.05
    FIXED_ZOOM_BASELINES = ['CoreHD', 'DCR', 'BCR', 'GDMR', 'FINDER', 'DomiRank']
    ORDERED_ALGOS = [
        'DC', 'BC', 'R1', 'DCR', 'BCR', 'R2',
        'DomiRank', 'FINDER', 'CoreHD', 'GDM', 'GDMR', 'CoreGDM',
        'MS', 'MSR', 'CI1', 'CI2', 'CI3', 'EGND',
        'EIs1', 'EIs2', 'GND', 'GNDR', 'VE', 'VER',
        'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR'
    ]
    algo_order_map = {name: i for i, name in enumerate(ORDERED_ALGOS)}
    algo_order_map['TI-GIN (Ours)'] = -1
    def get_algo_sort_key(algo_name): return algo_order_map.get(algo_name, 999)
    def get_seq_sort_key(seq_id): return algo_order_map.get(seq_id.rsplit('-', 1)[-1], 999)
    def get_subplot_label(i): return chr(ord('a') + i) if i < 26 else f"a{chr(ord('a') + i - 26)}"
    def _draw_equidistant_diagonal_grid(ax, slope, num_lines=6, **line_kwargs):
        x_min, x_max = ax.get_xlim(); y_min, y_max = ax.get_ylim()
        corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
        c_vals = [y - slope * x for x, y in corners]
        c_min, c_max = min(c_vals), max(c_vals); c_range = np.linspace(c_min, c_max, num=num_lines)
        for c in c_range: ax.axline(xy1=(0, c), slope=slope, **line_kwargs)
    def _find_strict_zoom_region(seq_id, plot_data, traditional_data, num_neighbors=4):
        if seq_id not in plot_data or not plot_data[seq_id]['steps']: return None
        tcr_gin_df = pd.DataFrame(plot_data[seq_id]).set_index('steps')['d_norm']
        if tcr_gin_df.empty: return None
        steps = sorted(plot_data[seq_id]['steps']); n_initial = None
        for df in traditional_data.values():
            initial_rows = df[(df['seq_id'] == seq_id) & (df['step'] == 0)]
            if not initial_rows.empty: n_initial = initial_rows['network_size'].iloc[0]; break
        if n_initial is None or n_initial == 0: return None
        theoretical_slope_abs = 1.0 / n_initial; search_start_idx = int(len(steps) * 0.1); search_end_idx = int(len(steps) * 0.9); best_strict = (None, -float('inf')); best_relaxed = (None, -float('inf'))
        for i in range(search_start_idx, search_end_idx):
            start_idx = max(0, i - num_neighbors); end_idx = min(len(steps) - 1, i + num_neighbors)
            win_steps = steps[start_idx : end_idx+1]; win_vals = tcr_gin_df.loc[win_steps]
            if len(win_vals) < 2: continue
            diffs = win_vals.diff().dropna()
            if (diffs > 1e-9).any(): continue
            current_score = 0; count_valid_baselines = 0
            for algo in FIXED_ZOOM_BASELINES:
                if algo in traditional_data:
                    df = traditional_data[algo]; seq_df = df[df['seq_id'] == seq_id].set_index('step'); common_idx = seq_df.index.intersection(win_steps)
                    if len(common_idx) < 2: continue
                    base_vals = seq_df.loc[common_idx]; base_d_norm = base_vals['critical_threshold'] * base_vals['network_size'] / n_initial
                    gap = (base_d_norm - tcr_gin_df.loc[common_idx]).mean(); base_diffs = base_d_norm.diff().dropna(); volatility = base_diffs[base_diffs > 0].sum()
                    current_score += (gap * 1.0 + volatility * 10.0); count_valid_baselines += 1
            if count_valid_baselines == 0: continue
            total_drop = win_vals.iloc[0] - win_vals.iloc[-1]; total_span = win_steps[-1] - win_steps[0]; avg_slope_abs = total_drop / total_span if total_span > 0 else 0; region = (win_steps[0], win_steps[-1])
            if avg_slope_abs < theoretical_slope_abs:
                if current_score > best_strict[1]: best_strict = (region, current_score)
            else:
                if current_score > best_relaxed[1]: best_relaxed = (region, current_score)
        if best_strict[0] is not None: return best_strict[0]
        elif best_relaxed[0] is not None: return best_relaxed[0]
        else: return None
    all_seq_ids = set(plot_data.keys())
    for algo_df in traditional_data.values(): all_seq_ids.update(algo_df['seq_id'].unique())
    if not all_seq_ids: return
    sorted_sequences = sorted(list(all_seq_ids), key=get_seq_sort_key); base_network_id = sorted_sequences[0].rsplit('-', 1)[0]
    baseline_colors = cycle(['#666666', '#888888', '#aaaaaa', '#cccccc'])
    baseline_styles = {name: {**ALGO_STYLES.get(name, ALGO_STYLES['default']), 'color': next(baseline_colors), 'zorder': 10} for name in traditional_data.keys()}
    layouts = [(3, 3), (4, 3), (3, 3)]; sequence_chunks = []; start_idx = 0
    for rows, cols in layouts:
        count = rows * cols; sequence_chunks.append(sorted_sequences[start_idx : start_idx + count]); start_idx += count
    global_subplot_index = 0; all_handles = {}

    for fig_idx, (layout, seq_chunk) in enumerate(zip(layouts, sequence_chunks)):
        if not seq_chunk: continue
        nrows, ncols = layout
        output_path = output_dir / f"monotonicity_large_scale_{base_network_id}_part{fig_idx + 1}.png"
        print(f"  Generating large-scale monotonicity plot (Part {fig_idx + 1}) -> {output_path}")
        zoom_meta_data = {}; global_inset_y_span = 0.05
        for seq_id in seq_chunk:
            zoom_region = _find_strict_zoom_region(seq_id, plot_data, traditional_data)
            if zoom_region:
                x_start, x_end = zoom_region; local_y_min, local_y_max = 1.0, 0.0; has_data = False
                if seq_id in plot_data:
                    ti_df = pd.DataFrame(plot_data[seq_id]); z_ti = ti_df[(ti_df['steps'] >= x_start) & (ti_df['steps'] <= x_end)]
                    if not z_ti.empty: local_y_min = min(local_y_min, z_ti['d_norm'].min()); local_y_max = max(local_y_max, z_ti['d_norm'].max()); has_data = True
                n_initial = None
                for df in traditional_data.values():
                    ir = df[(df['seq_id'] == seq_id) & (df['step'] == 0)];
                    if not ir.empty: n_initial = ir['network_size'].iloc[0]; break
                if n_initial:
                    for algo_name in FIXED_ZOOM_BASELINES:
                        if algo_name in traditional_data:
                            df = traditional_data[algo_name]; seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                            z_b = seq_df[(seq_df['step'] >= x_start) & (seq_df['step'] <= x_end)]
                            if not z_b.empty:
                                d_n = z_b['critical_threshold'] * z_b['network_size'] / n_initial
                                local_y_min = min(local_y_min, d_n.min()); local_y_max = max(local_y_max, d_n.max()); has_data = True
                if has_data:
                    global_inset_y_span = max(global_inset_y_span, local_y_max - local_y_min)
                    zoom_meta_data[seq_id] = {'region': zoom_region, 'y_center': (local_y_min + local_y_max) / 2.0, 'n_initial': n_initial}
        global_inset_y_span *= 1.2
        page_max_x = 0
        for seq_id in seq_chunk:
            if seq_id in plot_data and plot_data[seq_id]['steps']: page_max_x = max(page_max_x, max(plot_data[seq_id]['steps']))
            for df in traditional_data.values():
                seq_df = df[df['seq_id'] == seq_id];
                if not seq_df.empty: page_max_x = max(page_max_x, seq_df['step'].max())
        page_target_xlim_right = max(10, page_max_x * 1.02)
        content_width = ncols * AX_WIDTH + (ncols - 1) * GAP_WIDTH; content_height = nrows * AX_HEIGHT + (nrows - 1) * GAP_HEIGHT
        current_top_margin = TITLE_SPACE + LEGEND_GAP + LEGEND_HEIGHT + 0.1 if fig_idx == 0 else MARGIN_TOP_BASE
        fig_width = MARGIN_LEFT + content_width + MARGIN_RIGHT; fig_height = MARGIN_BOTTOM + content_height + current_top_margin
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height)); axes = axes.flatten()
        plot_area_top = 1.0 - (current_top_margin / fig_height); plot_area_left = MARGIN_LEFT / fig_width; plot_area_right = 1.0 - (MARGIN_RIGHT / fig_width); plot_area_center_x = (plot_area_left + plot_area_right) / 2.0
        plt.subplots_adjust(left=plot_area_left, right=plot_area_right, bottom=MARGIN_BOTTOM/fig_height, top=plot_area_top, wspace=GAP_WIDTH/AX_WIDTH, hspace=GAP_HEIGHT/AX_HEIGHT)

        for i, seq_id in enumerate(seq_chunk):
            ax = axes[i]; attack_strategy = seq_id.rsplit('-', 1)[-1]; ax.set_title(f"({get_subplot_label(global_subplot_index)}) {attack_strategy} Attack", loc='left', fontsize=10); global_subplot_index += 1
            n_initial = 6474 if 'route' in base_network_id.lower() else None
            if n_initial is None:
                for df in traditional_data.values():
                    ir = df[(df['seq_id'] == seq_id) & (df['step'] == 0)];
                    if not ir.empty: n_initial = ir['network_size'].iloc[0]; break
                if n_initial is None:
                     for df in traditional_data.values():
                        seq_df = df[df['seq_id'] == seq_id];
                        if not seq_df.empty: n_initial = max(n_initial or 0, (seq_df['network_size'] + seq_df['step']).max())
            tcr_gin_style = ALGO_STYLES.get('TI-GIN (Ours)', {}).copy(); tcr_gin_style.update({'linewidth': 1.8, 'zorder': 20, 'label': 'TI-GIN (Ours)'})
            if seq_id in plot_data and plot_data[seq_id]['steps']:
                h, = ax.plot(plot_data[seq_id]['steps'], plot_data[seq_id]['d_norm'], **tcr_gin_style)
                if 'TI-GIN (Ours)' not in all_handles: all_handles['TI-GIN (Ours)'] = h
            for algo_name, df in traditional_data.items():
                seq_df = df[df['seq_id'] == seq_id].sort_values('step')
                if not seq_df.empty:
                    d_norm = seq_df['critical_threshold'] * seq_df['network_size'] / n_initial
                    style = baseline_styles.get(algo_name, ALGO_STYLES['default']); h, = ax.plot(seq_df['step'], d_norm, label=algo_name, **style)
                    if algo_name not in all_handles: all_handles[algo_name] = h
            ax.set_ylim(0, 1.05); linthresh_val = 0.07; ax.set_yscale('symlog', linthresh=linthresh_val, linscale=1.0); major_ticks = [0, 0.035, 0.07, 1.0]; ax.set_yticks(major_ticks); ax.yaxis.set_major_formatter(ScalarFormatter())
            try:
                trans = ax.get_yaxis_transform(); inv_trans = ax.transAxes.inverted(); y_disp = trans.transform([0, linthresh_val])[1]; y_axes_coord = inv_trans.transform([0, y_disp])[1]
                if 0 < y_axes_coord < 1:
                    d = 0.015; kwargs = dict(transform=ax.transAxes, color='k', clip_on=False); ax.plot((-d, +d), (y_axes_coord-d, y_axes_coord+d), **kwargs); ax.plot((-d, +d), (y_axes_coord-d-0.015, y_axes_coord+d-0.015), **kwargs)
            except: pass
            local_min = float('inf')
            if seq_id in plot_data and plot_data[seq_id]['steps']: local_min = min(local_min, min(plot_data[seq_id]['steps']))
            start_xlim = 0
            if local_min != float('inf') and local_min > (page_target_xlim_right * 0.02):
                start_xlim = max(0, local_min - page_target_xlim_right * 0.02); d = 0.015; kwargs = dict(transform=ax.transAxes, color='k', clip_on=False); ax.plot((-d, +d), (-d, +d), **kwargs); ax.plot((-d, +d), (-d-0.015, +d-0.015), **kwargs)
            ax.set_xlim(start_xlim, page_target_xlim_right); ax.grid(False); ax.label_outer()
            if seq_id in zoom_meta_data:
                meta = zoom_meta_data[seq_id]; x_start, x_end = meta['region']; y_center = meta['y_center']; n_init = meta['n_initial']
                ax_inset = ax.inset_axes([0.45, 0.45, 0.52, 0.52])
                if seq_id in plot_data: style = ALGO_STYLES.get('TI-GIN (Ours)', {}).copy(); style.update({'linewidth': 1.5, 'zorder': 20}); ax_inset.plot(plot_data[seq_id]['steps'], plot_data[seq_id]['d_norm'], **style)
                for algo_name in FIXED_ZOOM_BASELINES:
                    if algo_name not in traditional_data: continue
                    df = traditional_data[algo_name]; seq_df = df[df['seq_id'] == seq_id].sort_values('step'); z_b = seq_df[(seq_df['step'] >= x_start) & (seq_df['step'] <= x_end)]
                    if not z_b.empty:
                        d_n = z_b['critical_threshold'] * z_b['network_size'] / n_init
                        base_style = baseline_styles.get(algo_name, ALGO_STYLES['default']).copy(); base_style['linewidth'] = 0.8; ax_inset.plot(z_b['step'], d_n, **base_style)
                target_bottom = y_center - global_inset_y_span / 2; target_top = y_center + global_inset_y_span / 2
                if target_bottom < 0: offset = -target_bottom; target_bottom += offset; target_top += offset
                elif target_top > 1.05: offset = target_top - 1.05; target_bottom -= offset; target_top -= offset
                ax_inset.set_xlim(x_start, x_end); ax_inset.set_ylim(max(0, target_bottom), target_top)
                if n_init: _draw_equidistant_diagonal_grid(ax=ax_inset, slope=-1.0/n_init, num_lines=6, color='gainsboro', linewidth=0.5, zorder=0, alpha=0.8)
                ax_inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3)); ax_inset.tick_params(axis='both', which='major', labelsize=6); ax_inset.set_xticklabels([]); ax_inset.set_yticklabels([])
                for spine in ax_inset.spines.values(): spine.set_edgecolor('#555555'); spine.set_linewidth(0.8)
                ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=0.8)
        for j in range(len(seq_chunk), len(axes)): axes[j].set_visible(False)
        fig.supxlabel('Attack step', fontsize=10, y=0.02); fig.supylabel('Collapse distance', fontsize=10, x=0.01)
        if fig_idx == 0 and all_handles:
            sorted_handles = [all_handles[k] for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            sorted_labels = [k for k in sorted(all_handles, key=get_algo_sort_key) if k in all_handles]
            offset_inches = TITLE_SPACE + LEGEND_GAP; bbox_y_pos = plot_area_top + (offset_inches / fig_height)
            fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(plot_area_center_x, bbox_y_pos), ncol=10, frameon=False, fontsize=7, handlelength=1.5, columnspacing=0.9, bbox_transform=fig.transFigure)
        plt.savefig(output_path, dpi=600, bbox_inches='tight'); plt.close(fig)

# === 5. JOB EXECUTION & BENCHMARKING ===

def discover_model_jobs(config, base_model_params, device, config_dir):
    """Discover and set up TI-GIN model evaluation jobs based on the config."""
    test_type = config['test_type']
    jobs = []

    if test_type == 'single_param':
        print("Discovering model jobs: 'single_param' mode")
        suite_to_load = []
        for model_info in config['model_suite']:
            if 'base_dir' not in model_info: continue
            search_path = (config_dir / model_info['base_dir']).resolve()
            exp_dirs = list(search_path.glob('exp_*'))
            if not exp_dirs: continue
            model_path_pattern = str(exp_dirs[0] / 'model_run_*.pt')
            suite_to_load.append({'node_range': model_info['node_range'], 'path': model_path_pattern})

        if suite_to_load:
            model_suite = load_model_suite(suite_to_load, base_model_params, device)
            if model_suite:
                num_runs = max((len(m['models']) for m in model_suite), default=0)
                jobs.append({'name': 'TI-GIN (Ours)', 'params': {}, 'suite': model_suite, 'num_runs': num_runs, 'do_plotting': config.get('plotting', {}).get('enabled', True)})

    elif test_type == 'multi_param':
        print("Discovering model jobs: 'multi_param' mode")
        exp_dirs_by_range = defaultdict(list)
        for model_info in config['model_suite']:
            if 'base_dir' not in model_info: continue
            node_range_tuple = tuple(model_info['node_range'])
            search_path = (config_dir / model_info['base_dir']).resolve()
            exp_dirs_by_range[node_range_tuple].extend(search_path.glob('exp_*'))
        param_groups = defaultdict(lambda: defaultdict(list))
        for node_range, exp_dirs in exp_dirs_by_range.items():
            for exp_dir in exp_dirs:
                param_str = re.sub(r'^exp_\d+_', '', exp_dir.name); param_groups[param_str][node_range].append(exp_dir)
        for param_str, range_dirs in param_groups.items():
            job_params = parse_params_from_folder_name(f"exp_0_{param_str}")
            suite_to_load, num_runs_per_range = [], []
            for model_info in config['model_suite']:
                node_range = tuple(model_info['node_range'])
                if node_range in range_dirs:
                    model_path_pattern = str(range_dirs[node_range][0] / 'model_run_*.pt')
                    model_paths = glob(model_path_pattern)
                    if not model_paths: continue
                    num_runs_per_range.append(len(model_paths))
                    suite_to_load.append({'node_range': list(node_range), 'path': model_path_pattern, 'params': job_params})
            if not suite_to_load: continue
            model_suite = load_model_suite(suite_to_load, base_model_params, device)
            if model_suite:
                num_runs = max(num_runs_per_range) if num_runs_per_range else 0
                if num_runs == 0: continue
                jobs.append({'name': 'TI-GIN', 'params': job_params, 'suite': model_suite, 'num_runs': num_runs, 'do_plotting': False})
    return jobs

def calculate_benchmarks(dismantling_path, config):
    """Calculate baseline algorithm metrics (hardcodes N=6474 for route datasets)."""
    data_cache = load_and_prepare_traditional_data(dismantling_path, for_plotting=False)
    if not data_cache: return pd.DataFrame()
    dataset_name = config['dataset_name']; is_route_dataset = 'route' in dataset_name.lower()
    path_options = [dismantling_path / f"{dataset_name}-Remnants" / f"{dataset_name}-Remnants", dismantling_path / f"{dataset_name}-Remnants"]
    remnants_dir = None
    for p in path_options:
        if p.exists() and list(p.glob('*.npz')): remnants_dir = p; break
    input_dim = config.get('base_model_params', {}).get('input_dim', 7); all_results = []
    print("Computing baseline algorithm metrics...")
    for short_algo_name, df in tqdm(data_cache.items(), desc="  Processing baselines"):
        algo_metrics = defaultdict(list); all_mae_errors = []
        for seq_id, group in df.groupby('seq_id'):
            group = group.sort_values('step').reset_index(drop=True)
            if group.empty: continue
            n_initial = 0
            if is_route_dataset or 'route' in str(seq_id).lower(): n_initial = 6474
            else:
                n_initial_row = group[group['step'] == 0]
                if not n_initial_row.empty: n_initial = n_initial_row['network_size'].iloc[0]
            if n_initial == 0: continue
            vals = group['critical_threshold'].values * group['network_size'].values / n_initial
            steps = group['step'].values
            seq_metrics = _compute_sequence_metrics(steps, vals, n_initial)
            for k, v in seq_metrics.items(): algo_metrics[k].append(v)
            if remnants_dir:
                for idx, row in group.iterrows():
                    graph_name = row.get('network'); recorded_val = row.get('critical_threshold')
                    if graph_name and pd.notna(recorded_val):
                        file_path = remnants_dir / f"{graph_name}_edges.npz"
                        if file_path.exists():
                            try:
                                load_path_str = str(file_path).replace('_edges.npz', ''); g = load_single_graph(load_path_str, feature_dim=input_dim)
                                if g is not None and g.y is not None: all_mae_errors.append(abs(recorded_val - g.y.item()))
                            except: pass
        row = {'algorithm': short_algo_name}
        for k, v_list in algo_metrics.items():
            if v_list: row[f"{k}_mean"] = np.mean(v_list); row[f"{k}_std"] = np.std(v_list)
            else: row[f"{k}_mean"] = -1.0; row[f"{k}_std"] = 0.0
        row['additivity_consistency_mae_mean'] = 0.0; row['additivity_consistency_mae_std'] = 0.0
        if all_mae_errors: row['monotonicity_accuracy_mae_mean'] = np.mean(all_mae_errors); row['monotonicity_accuracy_mae_std'] = np.std(all_mae_errors)
        else: row['monotonicity_accuracy_mae_mean'] = -1.0; row['monotonicity_accuracy_mae_std'] = 0.0
        all_results.append(row)
    return pd.DataFrame(all_results)

def run_evaluation_job(job, config, device):
    """Run a single TI-GIN model evaluation job, including all analysis stages and plotting."""
    key_map = {'modelactivationfn': 'activation_fn', 'modeljktype': 'jk_type', 'modelusevirtualnode': 'use_virtual_node', 'pissconsistencylambda': 'consistency_lambda', 'pisspissk': 'piss_k', 'modelfeaturedim': 'feature_dim'}
    processed_job_params = {}
    for key, value in job['params'].items():
        short_key = key_map.get(key.lower(), key)
        if isinstance(value, str):
            if value.lower() == 'true': converted_value = True
            elif value.lower() == 'false': converted_value = False
            else:
                try:
                    if '.' in value: converted_value = float(value)
                    else: converted_value = int(value)
                except ValueError: converted_value = value
        else: converted_value = value
        processed_job_params[short_key] = converted_value
    job_params = config.get('base_model_params', {}).copy(); job_params.update(processed_job_params)
    if 'feature_dim' in job_params and job_params['feature_dim'] is not None: job_params['input_dim'] = job_params['feature_dim']
    model_args = argparse.Namespace(**job_params)
    all_run_results = []; monotonicity_plot_data_final, additivity_plot_data_final = None, None
    job_desc = job['name']
    if job['params']: job_desc += f" ({', '.join(f'{k}={v}' for k, v in job['params'].items())})"
    run_iterator = tqdm(range(job['num_runs']), desc=f"  Running {job_desc}", leave=False)
    for run_idx in run_iterator:
        prediction_cache = prepare_and_predict_all_graphs(job['suite'], device, run_idx, config, model_args)
        mono_metrics, mono_plot = test_monotonicity(prediction_cache, config)
        addi_metrics, addi_plot = test_additivity(prediction_cache, config)
        run_results = {**mono_metrics, **addi_metrics}
        all_run_results.append(run_results)
        if run_idx == 0: monotonicity_plot_data_final = mono_plot; additivity_plot_data_final = addi_plot
    if not all_run_results: return None
    results_df = pd.DataFrame(all_run_results); agg_metrics = {}
    if job['num_runs'] > 1:
        mean_series, std_series = results_df.mean(), results_df.std()
        for col in results_df.columns: agg_metrics[col], agg_metrics[f"{col}_std"] = mean_series[col], std_series[col]
    elif not results_df.empty: agg_metrics = results_df.iloc[0].to_dict()
    agg_metrics['algorithm'] = job['name']; agg_metrics.update(job['params'])
    if job['do_plotting']:
        print("Generating plots...")
        trad_dfs_for_plot = load_and_prepare_traditional_data(config['dismantling_path'], for_plotting=True)
        scheme1_dir = config['output_dir'] / "monotonicity_plots_scheme1_enhanced"; scheme1_dir.mkdir(parents=True, exist_ok=True)
        visualize_monotonicity_scheme1_enhanced(monotonicity_plot_data_final, trad_dfs_for_plot, scheme1_dir, config['dataset_type'])
        scheme2_dir = config['output_dir'] / "monotonicity_plots_scheme2_residual"; scheme2_dir.mkdir(parents=True, exist_ok=True)
        visualize_monotonicity_scheme2_residual(monotonicity_plot_data_final, trad_dfs_for_plot, scheme2_dir, config['dataset_type'])
        large_scale_dir = config['output_dir'] / "monotonicity_plots_large_scale"; large_scale_dir.mkdir(parents=True, exist_ok=True)
        visualize_monotonicity_large_scale(monotonicity_plot_data_final, trad_dfs_for_plot, large_scale_dir, config['dataset_type'])
        addi_plot_dir = config['output_dir'] / "additivity_plots"; addi_plot_dir.mkdir(parents=True, exist_ok=True)
        addi_plot_path = addi_plot_dir / f"{config['dataset_name']}_additivity.png"
        visualize_additivity_publication(additivity_plot_data_final, addi_plot_path, config['dataset_type'], config['dataset_name'])
    return agg_metrics

# === 6. MAIN WORKFLOW ===

def run_all_evaluations(config, device):
    base_model_params = config.get('base_model_params', {}); config_dir = Path(config['__file_path__']).parent
    jobs_to_run = discover_model_jobs(config, base_model_params, device, config_dir); all_job_summary_results = []
    if jobs_to_run:
        job_iterator = tqdm(jobs_to_run, desc="Processing all model jobs")
        for job in job_iterator:
            summary_result = run_evaluation_job(job, config, device)
            if summary_result: all_job_summary_results.append(summary_result)
    else: print("No valid TI-GIN model jobs found.")
    tgin_results_df = pd.DataFrame(all_job_summary_results)
    benchmark_df = calculate_benchmarks(config['dismantling_path'], config)
    return tgin_results_df, benchmark_df

def process_and_save_results(tgin_results_df, benchmark_df, config):
    if not tgin_results_df.empty:
        model_scales = str(sorted([Path(m['base_dir']).name for m in config.get('model_suite', [])]))
        tgin_results_df['test_dataset'] = config['dataset_name']; tgin_results_df['model_scales'] = model_scales; tgin_results_df['prediction_strategy'] = config['prediction_strategy']
    if not benchmark_df.empty:
        benchmark_df['test_dataset'] = config['dataset_name']; benchmark_df['model_scales'] = 'N/A'; benchmark_df['prediction_strategy'] = 'N/A'
    final_df = pd.concat([tgin_results_df, benchmark_df], ignore_index=True)
    if final_df.empty: print("\nEvaluation complete, but no valid results were collected."); return
    meta_cols_def = ['test_dataset', 'algorithm', 'prediction_strategy', 'model_scales']
    metric_roots = ['M_freq', 'M_int', 'S_freq', 'S_int', 'additivity_consistency_mae', 'monotonicity_accuracy_mae']
    all_cols = list(final_df.columns); found_meta = []; found_metrics = []; found_params = []
    for col in all_cols:
        if col in meta_cols_def: found_meta.append(col); continue
        is_metric = False
        for root in metric_roots:
            if col.startswith(root): is_metric = True; break
        if is_metric: found_metrics.append(col)
        else: found_params.append(col)
    found_meta.sort(key=lambda x: meta_cols_def.index(x)); found_params.sort(); found_metrics.sort()
    ordered_cols = found_meta + found_params + found_metrics; final_df = final_df.reindex(columns=ordered_cols)
    config['summary_csv_path'].parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(config['summary_csv_path'], index=False, float_format='%.6f')
    print(f"\nEvaluation complete! Aggregated results saved to: {config['summary_csv_path']}")
    print("\n--- Final Results Preview ---")
    with pd.option_context('display.max_rows', 40, 'display.max_columns', None, 'display.width', 200): print(final_df)

def main():
    parser = argparse.ArgumentParser(description="TI-GIN Model Property Evaluation Framework")
    parser.add_argument('--config', type=str, required=True, help="Path to the evaluation config file")
    cli_args = parser.parse_args()
    config = setup_environment(cli_args.config); config['__file_path__'] = cli_args.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-" * 50)
    print(f"Evaluation dataset: {config['dataset_name']} (type: {config.get('dataset_type', 'N/A')})")
    print(f"Test type: {config['test_type']}")
    print(f"Prediction strategy (for monotonicity): {config['prediction_strategy']}")
    print(f"Device: {device}")
    print(f"Output directory: {config['output_dir']}")
    print("-" * 50)
    tgin_results_df, benchmark_df = run_all_evaluations(config, device)
    process_and_save_results(tgin_results_df, benchmark_df, config)

if __name__ == '__main__':
    main()
