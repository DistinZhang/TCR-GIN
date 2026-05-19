#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/early_warning/calculate_decision_window.py
(Modified version v5 — legend arrow consistent across all figures)
"""

import os
import sys
import re
import time
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
from pathlib import Path
from tqdm import tqdm
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.legend_handler import HandlerBase


# ==============================================================================
# Plot styling
# ==============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 2.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
})

ORDERED_ALGOS = [
    'DC', 'DCR', 'BC', 'BCR', 'R1', 'R2', 'MS', 'MSR',
    'DomiRank', 'FINDER',
    r'CI $\ell$-1', r'CI $\ell$-2', r'CI $\ell$-3', 'EGND', 'GND', 'GNDR',
    r'EI ${\sigma _1}$', r'EI ${\sigma _2}$',
    'GDM', 'GDMR', 'CoreGDM', 'CoreHD',
    'VE', 'VER',
    'NEL', 'NELR', 'NEM', 'NEMR', 'NES', 'NESR'
]

algo_order_map = {name: i for i, name in enumerate(ORDERED_ALGOS)}
algo_order_map['TCR-GIN'] = -1
algo_order_map['CI1']  = algo_order_map[r'CI $\ell$-1']
algo_order_map['CI2']  = algo_order_map[r'CI $\ell$-2']
algo_order_map['CI3']  = algo_order_map[r'CI $\ell$-3']
algo_order_map['EIs1'] = algo_order_map[r'EI ${\sigma _1}$']
algo_order_map['EIs2'] = algo_order_map[r'EI ${\sigma _2}$']

FILENAME_TO_SHORT_NAME = {
    'CollectiveInfluenceL1': r'CI $\ell$-1',
    'CollectiveInfluenceL2': r'CI $\ell$-2',
    'CollectiveInfluenceL3': r'CI $\ell$-3',
    'GDM': 'GDM', 'GDMR': 'GDMR', 'CoreGDM': 'CoreGDM', 'CoreHD': 'CoreHD',
    'EGND': 'EGND',
    'EI_s1': r'EI ${\sigma _1}$', 'EI_s2': r'EI ${\sigma _2}$',
    'GND': 'GND', 'GNDR': 'GNDR',
    'MS': 'MS', 'MSR': 'MSR',
    'network_entanglement_small': 'NES',
    'network_entanglement_small_reinsertion': 'NESR',
    'network_entanglement_mid': 'NEM',
    'network_entanglement_mid_reinsertion': 'NEMR',
    'network_entanglement_large': 'NEL',
    'network_entanglement_large_reinsertion': 'NELR',
    'vertex_entanglement': 'VE',
    'vertex_entanglement_reinsertion': 'VER',
    'degree_T': 'DC', 'degree_F': 'DCR',
    'betweenness_centrality_T': 'BC', 'betweenness_centrality_F': 'BCR',
    'eigenvector_centrality_T': 'EC', 'eigenvector_centrality_F': 'ECR',
    'FINDER_CN': 'FINDER', 'Domirank': 'DomiRank',
    'degree': 'DC', 'betweenness_centrality': 'BC', 'betweenness': 'BC',
    'eigenvector_centrality': 'EC', 'eigenvector': 'EC',
    'CI1': r'CI $\ell$-1', 'CI2': r'CI $\ell$-2', 'CI3': r'CI $\ell$-3',
    'EIs1': r'EI ${\sigma _1}$', 'EIs2': r'EI ${\sigma _2}$',
    'R1': 'R1', 'R2': 'R2',
    'random1': 'R1', 'random2': 'R2',
    'random_T': 'R1', 'random_F': 'R2',
}

DEFAULT_SCALE_FACTOR = 1.0
MODEL_INDEX_TO_USE   = 0

try:
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from torch_geometric.data import Batch
    from data_loader import load_single_graph
    from model.tcr_gin import TCR_GIN
    IMPORTS_OK = True
except ImportError as e:
    print(f"[Warning] Failed to import project dependencies: {e}")
    IMPORTS_OK = False

warnings.filterwarnings('ignore')


# ==============================================================================
# Helper functions
# ==============================================================================
def get_alpha_label(n):
    if n < 26:
        return chr(97 + n)
    return get_alpha_label(n // 26 - 1) + get_alpha_label(n % 26)


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
    return True if str(x).strip().lower() in ['true', '1', 't', 'yes'] else False


def infer_algo_from_results_row(heuristic_raw, static_raw):
    h   = to_short_algo(heuristic_raw)
    st  = _parse_bool(static_raw)
    raw = str(heuristic_raw).strip().lower() if heuristic_raw is not None else ""
    if raw in ['degree', 'deg']:
        return 'DCR' if st is False else 'DC'
    if raw in ['betweenness_centrality', 'betweenness', 'bc']:
        return 'BCR' if st is False else 'BC'
    return h


def extract_and_normalize_algo(raw_name):
    s = str(raw_name).strip().replace('.npz', '').replace('_edges', '')
    candidates = sorted(
        list(set(list(FILENAME_TO_SHORT_NAME.keys()) + ORDERED_ALGOS)),
        key=len, reverse=True
    )
    for cand in candidates:
        if s == cand:
            return to_short_algo(cand)
        if s.endswith(cand):
            idx = len(s) - len(cand)
            if idx > 0 and s[idx - 1] in ['_', '-']:
                return to_short_algo(cand)
    return "Unknown"


def parse_filename_info(filename: str):
    stem  = filename.replace('_edges.npz', '').replace('.npz', '').replace('_label.json', '')
    parts = stem.split('_')
    nums  = []
    while parts and re.fullmatch(r'-?\d+', parts[-1]):
        nums.append(int(parts.pop()))
        if len(nums) >= 2:
            break
    if not nums:
        return "Unknown", -1
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


def calc_lcc_global(G, initial_network_size):
    if G is None or G.number_of_nodes() == 0 or initial_network_size <= 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len) if not nx.is_empty(G) else []
    return len(largest_cc) / initial_network_size


def get_results_file_path(input_root_path: Path):
    path         = Path(input_root_path)
    dataset_name = path.name
    prefix       = dataset_name.split('_')[0]
    candidates   = [
        path / f"{prefix}-results.csv",       path / f"{prefix}-results.xlsx",
        path / f"{dataset_name}-results.csv", path / f"{dataset_name}-results.xlsx",
        path.parent / f"{prefix}-results.csv",       path.parent / f"{prefix}-results.xlsx",
        path.parent / f"{dataset_name}-results.csv", path.parent / f"{dataset_name}-results.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c
    pool = (
        list(path.glob("*-results.csv")) + list(path.glob("*-results.xlsx")) +
        list(path.parent.glob("*-results.csv")) + list(path.parent.glob("*-results.xlsx"))
    )
    pool = [p for p in pool if p.exists()]
    if pool:
        pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return pool[0]
    return None


def _clean_removals_string(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = re.sub(r'\bnan\b', 'None', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTRUE\b', 'True', s)
    s = re.sub(r'\bFALSE\b', 'False', s)
    return s


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
    for j in range(len(item) - 1, 0, -1):
        try:
            v = float(item[j])
            return v if (not math.isnan(v) and 0.0 <= v <= 1.1) else None
        except Exception:
            continue
    return None


def replay_order_to_lcc(initial_graph, order_list, threshold, initial_size):
    """No early break — curve runs to the last node in order_list."""
    if initial_graph is None or initial_graph.number_of_nodes() == 0:
        return pd.DataFrame({'step': [0], 'LCC': [0.0]})
    G       = initial_graph.copy()
    records = [{'step': 0, 'LCC': 1.0}]
    step    = 0
    for n in order_list:
        step += 1
        try:
            n_int = int(n)
        except Exception:
            continue
        if G.has_node(n_int):
            G.remove_node(n_int)
        lcc = (
            len(max(nx.connected_components(G), key=len)) / initial_size
            if G.number_of_nodes() > 0 else 0.0
        )
        records.append({'step': step, 'LCC': float(lcc)})
    return pd.DataFrame(records)


def parse_csv_result_file(file_path: Path, initial_graph, initial_size: int,
                          default_threshold: float):
    file_path = Path(file_path)
    try:
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path)

        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'heuristic' not in df.columns or 'removals' not in df.columns:
            return {}, []

        results = {}
        for _, row in df.iterrows():
            algo_short = infer_algo_from_results_row(row.get('heuristic'), row.get('static'))
            if algo_short == "Unknown":
                continue
            raw = row.get('removals')
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

            if all(isinstance(x, (int, np.integer)) or
                   (isinstance(x, str) and x.isdigit()) for x in data_list):
                if initial_graph:
                    df_curve = replay_order_to_lcc(
                        initial_graph, list(data_list), default_threshold, initial_size)
                    results[algo_short] = df_curve
                continue

            steps, lccs = [], []
            for item in data_list:
                sv, lcc = None, None
                if isinstance(item, dict):
                    sv  = item.get('step', item.get('k', item.get('t')))
                    lcc = item.get('lcc', item.get('LCC', item.get('giant')))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    sv  = int(item[0])
                    lcc = extract_lcc_from_removal_item(item)
                if sv is not None and lcc is not None:
                    steps.append(int(sv))
                    lccs.append(float(lcc))

            if steps:
                df_curve = pd.DataFrame({'step': steps, 'LCC': lccs}).sort_values('step')
                if df_curve.iloc[0]['step'] != 0:
                    df_curve = pd.concat([pd.DataFrame([{'step': 0, 'LCC': 1.0}]), df_curve])
                results[algo_short] = df_curve

        return results, []
    except Exception as e:
        print(f" [Error] Failed to parse result file: {e}")
        return {}, []


def load_baseline_predictions_df(input_root, initial_size):
    results_final_dir = input_root / f"{input_root.name}-Remnants" / "results_final"
    if not results_final_dir.exists():
        return pd.DataFrame()

    files   = list(results_final_dir.glob("*.xlsx")) + list(results_final_dir.glob("*.csv"))
    records = []
    for f in files:
        try:
            df = pd.read_excel(f) if f.suffix == '.xlsx' else pd.read_csv(f)
            for _, row in df.iterrows():
                if 'network' not in row:
                    continue
                algo_raw, step = parse_filename_info(str(row['network']))
                attack_algo    = extract_and_normalize_algo(algo_raw)
                if attack_algo == "Unknown":
                    attack_algo = algo_raw
                predictor_algo = infer_algo_from_results_row(
                    row.get('heuristic'), row.get('static'))
                if predictor_algo == "Unknown":
                    continue
                ct = float(row.get('critical_threshold', 0.0))
                dt = float(row.get('dismantle_time', 0.0))
                ns = float(row.get('network_size', initial_size)) or initial_size
                records.append({
                    'algorithm': attack_algo, 'step': step,
                    'predictor': predictor_algo,
                    'DC': ct * (ns / initial_size), 'Time': dt,
                })
        except Exception:
            pass

    if not records:
        return pd.DataFrame()
    b_df     = pd.DataFrame(records)
    pivot_df = b_df.pivot_table(
        index=['algorithm', 'step'], columns='predictor',
        values=['DC', 'Time'], aggfunc='first'
    )
    pivot_df.columns = [f"Predictor_{col[1]}_{col[0]}" for col in pivot_df.columns.values]
    return pivot_df.reset_index()


# ==============================================================================
# TCR-GIN predictor
# ==============================================================================
class TCRGINPredictor:
    def __init__(self, config_path_str, device):
        self.device           = device
        self.models_map       = []
        self.enabled          = False
        self.input_dim_global = 3

        if not config_path_str or not IMPORTS_OK:
            return
        try:
            self.config_path = Path(config_path_str).resolve()
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            base_params = self.config.get('base_model_params', {})
            self.input_dim_global = base_params.get('feature_dim',
                                                     base_params.get('input_dim', 7))
            self._load_suite()
            if self.models_map:
                self.enabled = True
                print(f"[TCR-GIN] Loaded {len(self.models_map)} model(s).")
        except Exception as e:
            print(f"[TCR-GIN] Init error: {e}")

    def _resolve_paths(self, model_info, config_dir, project_root):
        if 'path' in model_info:
            for cand in [model_info['path'],
                         str(config_dir / model_info['path']),
                         str(project_root / model_info['path'])]:
                if glob.glob(cand):
                    return cand
        if 'base_dir' in model_info:
            for base in [config_dir / model_info['base_dir'],
                         project_root / model_info['base_dir']]:
                if base.exists():
                    exp_dirs = sorted(list(base.glob('exp_*')))
                    if exp_dirs:
                        return str(exp_dirs[0] / 'model_run_*.pt')
        return None

    def _load_suite(self):
        base_params        = self.config.get('base_model_params', {})
        config_dir         = self.config_path.parent
        project_root_guess = config_dir.parents[1]
        key_map = {
            'modelactivationfn': 'activation_fn', 'modeljktype': 'jk_type',
            'modelusevirtualnode': 'use_virtual_node',
            'pissconsistencylambda': 'consistency_lambda',
            'pisspissk': 'piss_k', 'modelfeaturedim': 'feature_dim',
        }
        for model_info in self.config.get('model_suite', []):
            node_range   = model_info.get('node_range', [0, -1])
            path_pattern = self._resolve_paths(model_info, config_dir, project_root_guess)
            if not path_pattern:
                continue
            model_files = sorted(glob.glob(path_pattern))
            if not model_files:
                continue
            target = Path(
                model_files[-1] if len(model_files) <= MODEL_INDEX_TO_USE
                else model_files[MODEL_INDEX_TO_USE]
            )
            scale_factor = DEFAULT_SCALE_FACTOR
            for p in [target.parent / 'label_stats.json',
                      target.parent.parent / 'outputs' / 'label_stats.json']:
                if p.exists():
                    try:
                        scale_factor = json.load(open(p)).get(
                            'scale_factor', DEFAULT_SCALE_FACTOR)
                        break
                    except Exception:
                        pass
            params = {**base_params, **model_info.get('params', {})}
            fp = {}
            for k, v in params.items():
                sk = key_map.get(k.lower(), k)
                if isinstance(v, str):
                    if v.lower() == 'true':    v = True
                    elif v.lower() == 'false': v = False
                    elif '.' in v:
                        try: v = float(v)
                        except Exception: pass
                    else:
                        try: v = int(v)
                        except Exception: pass
                fp[sk] = v
            if 'feature_dim' in fp:
                fp['input_dim'] = fp['feature_dim']
            try:
                model = TCR_GIN(argparse.Namespace(**fp)).to(self.device)
                model.load_state_dict(torch.load(target, map_location=self.device))
                model.eval()
                self.models_map.append({
                    'range': node_range, 'model': model,
                    'scale': scale_factor, 'input_dim': fp.get('input_dim', 7),
                })
            except Exception as e:
                print(f"[TCR-GIN] Failed to load {target}: {e}")

    def predict(self, file_path):
        t0 = time.time()
        if not self.enabled:
            return 0.001, 0.0
        input_dim = self.models_map[0]['input_dim'] if self.models_map else self.input_dim_global
        try:
            stem = str(file_path).replace('_edges.npz', '')
            data = load_single_graph(stem, feature_dim=input_dim)
            if data is None:
                return 0.001, time.time() - t0
            N        = data.num_nodes
            selected = None
            for m in self.models_map:
                lo, hi = m['range']
                if (hi == -1 and N >= lo) or (hi != -1 and lo <= N < hi):
                    selected = m
                    break
            if selected is None and self.models_map:
                selected = self.models_map[-1]
            if selected is None:
                return 0.001, time.time() - t0
            batch = Batch.from_data_list([data]).to(self.device)
            with torch.no_grad():
                pred_norm = selected['model'](batch).item()
            return max(0.001, pred_norm / selected['scale']), time.time() - t0
        except Exception:
            return 0.001, time.time() - t0


# ==============================================================================
# Visualization helpers
# ==============================================================================
def _add_tick(ax, value: float):
    ticks = list(ax.get_yticks())
    if not any(np.isclose(t, value, rtol=0, atol=1e-12) for t in ticks):
        ax.set_yticks(sorted(ticks + [value]))


def _bold_tick_label(ax, value: float):
    ticks = np.array(ax.get_yticks(), dtype=float)
    if ticks.size == 0:
        return
    idx = int(np.argmin(np.abs(ticks - value)))
    if np.isclose(ticks[idx], value, rtol=0, atol=1e-12):
        labels = ax.get_yticklabels()
        if idx < len(labels):
            labels[idx].set_fontweight("bold")


def _enforce_integer_xticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _ensure_dc_zero_at_xmax(df_plot: pd.DataFrame, dc_col: str, x_max: int) -> pd.DataFrame:
    """
    Guarantee the DC curve terminates at x_max with value 0.

    For BC-type algos: the "-1" virtual row already exists at step_plot=x_max
    with DC=NaN.  Set it to 0 in-place (no append) so the line is continuous.

    For R1/R2 after virtual-collapse injection: no row exists yet at x_max,
    so append one built by forward-filling.
    """
    if dc_col not in df_plot.columns:
        return df_plot

    step_col     = 'step_plot' if 'step_plot' in df_plot.columns else 'step'
    mask_at_xmax = df_plot[step_col] == x_max

    if mask_at_xmax.any():
        df_plot = df_plot.copy()
        df_plot.loc[mask_at_xmax, dc_col] = 0.0
        return df_plot

    # No row at x_max — append (forward-fill for consistent marker style)
    valid_mask = df_plot[dc_col].notna()
    if not valid_mask.any():
        return df_plot

    pad_row           = df_plot.ffill().iloc[[-1]].copy()
    pad_row[step_col] = x_max
    pad_row['step']   = x_max
    pad_row[dc_col]   = 0.0

    df_plot = pd.concat([df_plot, pad_row], ignore_index=True)
    df_plot = df_plot.sort_values(step_col).reset_index(drop=True)
    return df_plot


def _inject_virtual_collapse(df_algo: pd.DataFrame,
                              collapse_target: float,
                              dc_col: str) -> pd.DataFrame:
    """
    For algos (e.g. R1/R2) whose LCC never drops below collapse_target,
    inject a synthetic collapse row at max_step+1 so a right boundary exists.

    The injected filename contains '_-1_' so existing detection logic
    identifies it correctly.
    """
    lcc_col = 'LCC'
    if lcc_col not in df_algo.columns:
        return df_algo

    has_collapse = (df_algo[lcc_col] < collapse_target).any()
    has_minus1   = df_algo['filename'].astype(str).str.contains(r'_-1_', regex=True).any()

    if has_collapse or has_minus1:
        return df_algo   # nothing to do

    max_step     = int(df_algo['step'].max())
    virtual_step = max_step + 1

    last_row  = df_algo.iloc[[-1]].copy()
    sample_fn = str(df_algo['filename'].iloc[0])
    fn_stem   = sample_fn.replace('_edges.npz', '').replace('.npz', '')
    fn_prefix = re.sub(r'_-?\d+$', '', fn_stem)
    virtual_fn = f"{fn_prefix}_-1_edges.npz"

    last_row['filename'] = virtual_fn
    last_row['step']     = virtual_step
    last_row[lcc_col]    = collapse_target * 0.9   # just below τ
    if dc_col in last_row.columns:
        last_row[dc_col] = 0.0

    df_injected = pd.concat([df_algo, last_row], ignore_index=True)
    df_injected = df_injected.sort_values('step').reset_index(drop=True)
    return df_injected


# ==============================================================================
# Legend: custom handler for lead-time <-> arrow
# [v5] fixed_width_pts ensures identical arrow length regardless of fontsize
# ==============================================================================
class HandlerLeadTimeArrow(HandlerBase):
    """
    Draw a FancyArrowPatch <-> identical to in-plot annotation arrows.

    fixed_width_pts: if provided, overrides the renderer-computed swatch width
                     so that arrow length is the same regardless of the legend
                     fontsize used in different figures.
    """
    def __init__(self, arrow_lw=2.2, mutation_scale=10, fixed_width_pts=None):
        super().__init__()
        self._lw              = arrow_lw
        self._mutation_scale  = mutation_scale
        self._fixed_width_pts = fixed_width_pts   # [v5] absolute override

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # [v5] Use fixed width if specified; otherwise fall back to renderer value
        draw_width = self._fixed_width_pts if self._fixed_width_pts is not None else width
        y   = height / 2
        pad = draw_width * 0.05
        arrow = FancyArrowPatch(
            posA=(xdescent + pad,              y),
            posB=(xdescent + draw_width - pad, y),
            arrowstyle="<->",
            color=orig_handle.get_color(),
            lw=self._lw,
            mutation_scale=self._mutation_scale,
            transform=trans,
            zorder=5,
        )
        return [arrow]


# [v5] Single shared handler instance used by ALL legend calls.
#      fixed_width_pts=30 matches the natural swatch width of the scheme3 figure
#      (fontsize=12 × handlelength=2.5), so both figures render identically.
_LEAD_TIME_HANDLER = HandlerLeadTimeArrow(
    arrow_lw=2.2, mutation_scale=10, fixed_width_pts=30
)


def _build_legend_handles(collapse_target: float, colors: dict) -> list:
    """Shared legend handles for ALL figure types."""
    lead_time_proxy = Line2D([0], [0], color=colors["alert"], lw=0, label="Lead time")
    return [
        Line2D([0], [0], color=colors["lcc"],   lw=1.6,
               label="LCC size"),
        Line2D([0], [0], color=colors["dc"],    lw=1.4, marker="o", markersize=4,
               label="Collapse distance"),

        Line2D([0], [0], color=colors["dc"],    lw=1.0, ls="-.",
               label="Warning target (3 steps)"),
        Line2D([0], [0], color=colors["lcc"],   lw=1.0, ls="--",
               label=f"Collapse target ($\\tau={collapse_target}$)"),
        Line2D([0], [0], color=colors["alert"], lw=1.4, ls="--",
               label="Warning step"),
        lead_time_proxy,
    ]


# ==============================================================================
# Core metric computation
# ==============================================================================
def compute_decision_window(df_panel: pd.DataFrame, dc_col_name: str,
                            collapse_target: float = 0.3,
                            initial_size_global=None):
    if df_panel is None or df_panel.empty:
        return None

    # Inject virtual collapse row for algos that never reach τ (e.g. R1/R2)
    df_panel = _inject_virtual_collapse(df_panel, collapse_target, dc_col_name)

    fn          = df_panel["filename"].astype(str)
    minus1_mask = fn.str.contains(r"_-1_", regex=True)
    df_last     = df_panel.loc[minus1_mask].dropna(subset=["step"]).copy()
    df_base     = df_panel.loc[~minus1_mask].dropna(subset=["step"]).copy()

    if not df_last.empty:
        x_max = int(df_last["step"].iloc[0])
    else:
        if df_base.empty:
            return None
        below_target = df_base[df_base['LCC'] < collapse_target]
        x_max = (int(below_target.iloc[0]['step']) if not below_target.empty
                 else int(df_base["step"].max()))

    collapse_step = x_max   # rightmost step = collapse step

    df_plot = pd.concat([df_base, df_last], ignore_index=True)
    df_plot["step_plot"] = df_plot["step"]
    df_plot.loc[
        df_plot["filename"].astype(str).str.contains(r"_-1_", regex=True),
        "step_plot"
    ] = x_max
    df_plot = df_plot.sort_values("step_plot").reset_index(drop=True)

    n0 = initial_size_global
    if n0 is None:
        n0 = (df_base["network_size"].max() if not df_base.empty
              else df_panel["network_size"].max())
    if n0 is None or n0 == 0:
        n0 = 100

    warning_threshold = 3 / n0

    warning_step = collapse_step
    cd_valid     = df_base.dropna(subset=[dc_col_name]) if not df_base.empty else pd.DataFrame()
    warn_rows    = (cd_valid[cd_valid[dc_col_name] < warning_threshold]
                    if not cd_valid.empty else pd.DataFrame())
    if len(warn_rows):
        warning_step = int(warn_rows.iloc[0]["step"])

    decision_start  = warning_step
    decision_end    = collapse_step
    lead_time_steps = decision_end - decision_start

    # Ensure DC = 0 at x_max (in-place on existing row, never creates duplicate)
    if dc_col_name in df_plot.columns:
        df_plot = _ensure_dc_zero_at_xmax(df_plot, dc_col_name, x_max)

    return {
        "df_plot":           df_plot,
        "df_base":           df_base,
        "collapse_target":   collapse_target,
        "collapse_step":     collapse_step,
        "x_min":             0,
        "x_max":             x_max,
        "warning_threshold": warning_threshold,
        "warning_step":      warning_step,
        "decision_start":    decision_start,
        "decision_end":      decision_end,
        "lead_time_steps":   lead_time_steps,
        "n0":                n0,
    }


# ==============================================================================
# Single panel
# ==============================================================================
def plot_panel(ax, df_panel, panel_title, colors,
               show_xlabel, show_ylabel_left, show_ylabel_right,
               collapse_target, initial_size, dc_col="Predicted DC"):

    p = compute_decision_window(df_panel, dc_col, collapse_target, initial_size)
    if p is None:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return None, None

    color_lcc   = colors["lcc"]
    color_dc    = colors["dc"]
    color_alert = colors["alert"]

    ax.set_title(panel_title, loc='left', fontsize=9, fontweight='bold')

    ax.plot(p["df_plot"]["step_plot"], p["df_plot"]["LCC"],
            color=color_lcc, linewidth=1.6, label="_nolegend_")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelcolor=color_lcc)
    ax.axhline(p["collapse_target"], color=color_lcc, linestyle="--",
               linewidth=1.0, alpha=0.95)
    _add_tick(ax, p["collapse_target"])
    ax.figure.canvas.draw()
    _bold_tick_label(ax, p["collapse_target"])
    ax.set_xlim(p["x_min"], p["x_max"])
    _enforce_integer_xticks(ax)

    if show_xlabel:
        ax.set_xlabel("Attack step")
    if show_ylabel_left:
        ax.set_ylabel("Network function (LCC size)", color=color_lcc, fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(
        p["df_plot"]["step_plot"], p["df_plot"][dc_col],
        color=color_dc, linewidth=1.4, marker="o", markersize=2.6, markeredgewidth=0
    )
    ax2.tick_params(axis="y", labelcolor=color_dc)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))

    if show_ylabel_right:
        ax2.set_ylabel("Collapse distance", color=color_dc, fontsize=8)

    dc_vals = p["df_base"][dc_col].dropna().values
    if len(dc_vals):
        dc_max = float(np.max(dc_vals))
        pad    = max(0.0005, 0.08 * (dc_max - float(np.min(dc_vals))))
        ax2.set_ylim(0, dc_max + pad)

    ax2.axhline(p["warning_threshold"], color=color_dc, linestyle="-.",
                linewidth=1.0, alpha=0.85)
    _add_tick(ax2, p["warning_threshold"])
    ax2.figure.canvas.draw()
    _bold_tick_label(ax2, p["warning_threshold"])

    ax.axvline(p["warning_step"], color=color_alert, linestyle="--", linewidth=1.4)

    mid_y = 0.58
    if p["lead_time_steps"] > 0:
        ax.annotate(
            "", xy=(p["decision_end"], mid_y), xytext=(p["decision_start"], mid_y),
            arrowprops=dict(arrowstyle="<->", color=color_alert, lw=2.2, mutation_scale=14),
            zorder=5
        )
        ax.text(
            (p["decision_start"] + p["decision_end"]) / 2, mid_y + 0.04,
            f"{p['lead_time_steps']} steps",
            color=color_alert, alpha=0.95, ha="center", va="bottom",
            fontsize=7.8, fontweight="bold", zorder=6
        )

    return p, ax2


# ==============================================================================
# generate_summary_plots
# ==============================================================================
def generate_summary_plots(df, output_dir, dataset_name, collapse_target, initial_size):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "lcc": "#1f77b4", "dc": "#d62728",
        "alert": "#2ca02c", "collapse": "#8b0000",
    }

    AX_W, AX_H   = 2.8, 1.65
    GAP_W, GAP_H = 0.6, 0.4
    ML, MR, MB   = 0.7, 0.6, 0.6
    MT_BASE      = 0.3
    LEG_H, TS, LG = 0.7, 0.3, 0.1

    summary_stats   = []
    predictors_info = [{'name': 'TCR-GIN', 'dc_col': 'Predicted DC', 'time_col': 'TCR-GIN_Time'}]
    if 'Label_DC' in df.columns:
        predictors_info.append({'name': 'GroundTruth', 'dc_col': 'Label_DC', 'time_col': None})
    for col in df.columns:
        if col.startswith('Predictor_') and col.endswith('_DC'):
            pn = col.replace('Predictor_', '').replace('_DC', '')
            predictors_info.append({'name': pn, 'dc_col': col,
                                    'time_col': f"Predictor_{pn}_Time"})

    baseline_time_cols = [
        p['time_col'] for p in predictors_info
        if p['name'] not in ['TCR-GIN', 'GroundTruth'] and p['time_col']
    ]

    chunk_size = 10
    num_chunks = math.ceil(len(ORDERED_ALGOS) / chunk_size)
    nrows, ncols = 5, 2

    for i in range(num_chunks):
        algos_chunk = ORDERED_ALGOS[i * chunk_size:(i + 1) * chunk_size]
        content_w   = ncols * AX_W + (ncols - 1) * GAP_W
        content_h   = nrows * AX_H + (nrows - 1) * GAP_H
        cur_top     = (TS + LG + LEG_H + 0.1) if i == 0 else MT_BASE
        fig_w       = ML + content_w + MR
        fig_h       = MB + content_h + cur_top

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.flatten()

        pt = 1.0 - cur_top / fig_h
        pl = ML / fig_w
        pr = 1.0 - MR / fig_w
        cx = (pl + pr) / 2.0

        fig.subplots_adjust(
            left=pl, right=pr, bottom=MB / fig_h, top=pt,
            wspace=GAP_W / AX_W, hspace=GAP_H / AX_H
        )

        has_data = False
        for j, algo in enumerate(algos_chunk):
            ax      = axes[j]
            df_algo = df[df['algorithm'] == algo].copy()
            letter  = get_alpha_label(i * chunk_size + j)
            title   = f"({letter}) {algo} attack"
            show_xl = (j >= 8)
            show_yl = (j % 2 == 0)
            show_yr = (j % 2 == 1)

            if not df_algo.empty:
                has_data = True
                plot_panel(ax, df_algo, title, colors,
                           show_xl, show_yl, show_yr,
                           collapse_target, initial_size)

                for p_info in predictors_info:
                    pname, dc_col, tc = p_info['name'], p_info['dc_col'], p_info['time_col']
                    if dc_col not in df_algo.columns:
                        continue
                    pp = compute_decision_window(df_algo, dc_col, collapse_target, initial_size)
                    if pp:
                        avg_t = 0.0
                        if pname == 'GroundTruth':
                            for bt in baseline_time_cols:
                                if bt in df_algo.columns:
                                    avg_t += df_algo[bt].mean()
                        elif tc and tc in df_algo.columns:
                            avg_t = df_algo[tc].mean()
                        summary_stats.append({
                            "Algorithm":         algo,
                            "Predictor":         pname,
                            "Collapse Target":   collapse_target,
                            "Collapse Step":     pp['collapse_step'],
                            "Warning Threshold": pp['warning_threshold'],
                            "Warning Step":      pp['warning_step'],
                            "Lead Time (steps)": pp['lead_time_steps'],
                            "Average Time (s)":  avg_t,
                        })
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, f"{algo}: No Data", ha='center', va='center')

        for k in range(len(algos_chunk), chunk_size):
            axes[k].axis('off')

        if i == 0 and has_data:
            legend_handles = _build_legend_handles(collapse_target, colors)
            lead_proxy     = legend_handles[-1]
            bbox_y = pt + (TS + LG) / fig_h - 0.02
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(cx, bbox_y),
                ncol=3, frameon=False,
                handlelength=2.5, columnspacing=1.0, fontsize=8,
                bbox_transform=fig.transFigure,
                # [v5] shared handler — fixed width, consistent with scheme3
                handler_map={lead_proxy: _LEAD_TIME_HANDLER},
            )

        out_base = plots_dir / f"{dataset_name}_panel_{i + 1}"
        if has_data:
            fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
            fig.savefig(f"{out_base}.svg", bbox_inches="tight")
            fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
            print(f"[Plot] Saved panel {i + 1} → {out_base}.png")
        plt.close(fig)

    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(
            output_dir / f"{dataset_name}_decision_summary.csv", index=False)
        print("[Summary] Saved lead-time statistics.")


# ==============================================================================
# generate_scheme3_summary_plots  (3×2 layout)
# ==============================================================================
def generate_scheme3_summary_plots(df, output_dir, dataset_name, collapse_target, initial_size):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "lcc": "#1f77b4", "dc": "#d62728",
        "alert": "#2ca02c", "collapse": "#8b0000",
    }

    AX_W, AX_H    = 2.8, 2.0
    GAP_W, GAP_H  = 0.9, 0.5
    ML, MR, MB    = 0.8, 0.2, 0.7
    LEG_H, TS, LG = 0.6, 0.2, 0.1
    FS_MAIN = 12
    FS_TICK = 11

    target_algos = ['DC', 'DCR', 'BC', 'BCR', 'R1', 'R2']
    nrows, ncols = 3, 2

    content_w = ncols * AX_W + (ncols - 1) * GAP_W
    content_h = nrows * AX_H + (nrows - 1) * GAP_H
    cur_top   = TS + LG + LEG_H + 0.1
    fig_w     = ML + content_w + MR
    fig_h     = MB + content_h + cur_top

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()

    pt = 1.0 - cur_top / fig_h
    pl = ML / fig_w
    pr = 1.0 - MR / fig_w
    cx = (pl + pr) / 2.0

    fig.subplots_adjust(
        left=pl, right=pr, bottom=MB / fig_h, top=pt,
        wspace=GAP_W / AX_W, hspace=GAP_H / AX_H
    )

    def subplot_label(i):
        return chr(ord('a') + i) if i < 26 else f"a{chr(ord('a') + i - 26)}"

    has_data = False

    for j, algo in enumerate(target_algos):
        ax      = axes[j]
        df_algo = df[df['algorithm'] == algo].copy()

        show_xlabel = (j >= 4)
        # y-axis labels only on the middle row (j=2 left, j=3 right)
        show_yl = (j == 2)
        show_yr = (j == 3)

        if not df_algo.empty:
            has_data = True
            p = compute_decision_window(df_algo, "Predicted DC", collapse_target, initial_size)
            if p is not None:
                color_lcc   = colors["lcc"]
                color_dc    = colors["dc"]
                color_alert = colors["alert"]

                ax.set_title(subplot_label(j), loc='left',
                             fontsize=FS_MAIN, fontweight='bold')

                ax.plot(p["df_plot"]["step_plot"], p["df_plot"]["LCC"],
                        color=color_lcc, linewidth=1.6, label="_nolegend_")
                ax.set_ylim(0, 1)
                ax.tick_params(axis="y", labelcolor=color_lcc, labelsize=FS_TICK)
                ax.axhline(p["collapse_target"], color=color_lcc, linestyle="--",
                           linewidth=1.0, alpha=0.95)
                _add_tick(ax, p["collapse_target"])
                ax.figure.canvas.draw()
                _bold_tick_label(ax, p["collapse_target"])
                ax.set_xlim(p["x_min"], p["x_max"])
                ax.tick_params(axis="x", labelsize=FS_TICK)
                _enforce_integer_xticks(ax)

                if show_xlabel:
                    ax.set_xlabel("Attack step", fontsize=FS_MAIN)
                if show_yl:
                    ax.set_ylabel("Network function (LCC size)",
                                  color=color_lcc, fontsize=FS_MAIN)

                ax2 = ax.twinx()
                ax2.plot(
                    p["df_plot"]["step_plot"], p["df_plot"]["Predicted DC"],
                    color=color_dc, linewidth=1.4,
                    marker="o", markersize=2.6, markeredgewidth=0
                )
                ax2.tick_params(axis="y", labelcolor=color_dc, labelsize=FS_TICK)
                if show_yr:
                    ax2.set_ylabel("Collapse distance", color=color_dc, fontsize=FS_MAIN)

                dc_vals = p["df_base"]["Predicted DC"].dropna().values
                if len(dc_vals):
                    dc_max = float(np.max(dc_vals))
                    pad    = max(0.0005, 0.08 * (dc_max - float(np.min(dc_vals))))
                    ax2.set_ylim(0, dc_max + pad)

                ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))
                ax2.axhline(p["warning_threshold"], color=color_dc, linestyle="-.",
                            linewidth=1.0, alpha=0.85)
                _add_tick(ax2, p["warning_threshold"])
                ax2.figure.canvas.draw()
                _bold_tick_label(ax2, p["warning_threshold"])

                ax.axvline(p["warning_step"], color=color_alert,
                           linestyle="--", linewidth=1.4)

                mid_y = 0.58
                if p["lead_time_steps"] > 0:
                    ax.annotate(
                        "", xy=(p["decision_end"], mid_y),
                        xytext=(p["decision_start"], mid_y),
                        arrowprops=dict(arrowstyle="<->", color=color_alert,
                                        lw=2.2, mutation_scale=14),
                        zorder=5
                    )
                    ax.text(
                        (p["decision_start"] + p["decision_end"]) / 2,
                        mid_y + 0.04,
                        f"{p['lead_time_steps']} steps",
                        color=color_alert, alpha=0.95, ha="center", va="bottom",
                        fontsize=FS_TICK, fontweight="bold", zorder=6
                    )
        else:
            ax.axis('off')

    if has_data:
        legend_handles = _build_legend_handles(collapse_target, colors)
        lead_proxy     = legend_handles[-1]
        bbox_y = pt + (TS + LG) / fig_h - 0.01
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(cx, bbox_y),
            ncol=3, frameon=False,
            handlelength=2.5, columnspacing=1.0, fontsize=FS_MAIN,
            bbox_transform=fig.transFigure,
            # [v5] shared handler — identical arrow to panel figures
            handler_map={lead_proxy: _LEAD_TIME_HANDLER},
        )

        out_base = plots_dir / f"{dataset_name}_decision_scheme3"
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
        fig.savefig(f"{out_base}.svg", bbox_inches="tight")
        fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
        print(f"[Plot] Saved scheme-3 → {out_base}.pdf")

    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root',      type=str, required=True)
    parser.add_argument('--output_dir',      type=str, required=True)
    parser.add_argument('--initial_size',    type=int, required=True)
    parser.add_argument('--collapse_target', type=float, default=0.3)
    parser.add_argument('--model_config',    type=str, default=None)
    args = parser.parse_args()

    input_root   = Path(args.input_root)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    dataset_name = input_root.name
    csv_path     = out_dir / f"{dataset_name}_metrics.csv"

    need_calc = True
    if csv_path.exists():
        print(f" -> [Check] Found existing data file: {csv_path}")
        try:
            df_exist = pd.read_csv(csv_path)
            if (not df_exist.empty
                    and all(c in df_exist.columns
                            for c in ['Predicted DC', 'LCC', 'TCR-GIN_Time', 'Label_DC'])
                    and df_exist['algorithm'].nunique() > 1):
                print(" -> [Skip] Existing data complete. Skipping recomputation.")
                rename_map = {
                    'CI1': r'CI $\ell$-1', 'CI2': r'CI $\ell$-2',
                    'CI3': r'CI $\ell$-3',
                    'EIs1': r'EI ${\sigma _1}$', 'EIs2': r'EI ${\sigma _2}$',
                }
                df_exist['algorithm'] = df_exist['algorithm'].replace(rename_map)
                need_calc = False
                df_final  = df_exist
        except Exception as e:
            print(f" -> [Warning] Could not read cached file: {e}. Recomputing.")

    if need_calc:
        print(" -> [Calc] Computing LCC, Predicted DC, Label DC ...")
        rn = f"{dataset_name}-Remnants"
        base_dir = None
        for cand in [input_root / rn / rn, input_root / rn]:
            if cand.exists():
                base_dir = cand
                break
        all_files = list(base_dir.glob("*.npz")) if base_dir else []
        if not all_files:
            all_files = list(input_root.glob("*.npz")) + list(input_root.rglob("*.npz"))
        if not all_files:
            print(f" -> [Error] No .npz files found under {input_root}.")
            return

        print(f" -> [Info] Found {len(all_files)} graph files.")
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = TCRGINPredictor(args.model_config, device)

        data_records = []
        for f in tqdm(all_files, desc="Processing Graphs"):
            algo_raw, step = parse_filename_info(f.name)
            algo = extract_and_normalize_algo(algo_raw)
            if algo == "Unknown":
                algo = algo_raw
            G = load_graph_robust(f)
            if G is None:
                continue
            curr_size = G.number_of_nodes()
            lcc       = calc_lcc_global(G, args.initial_size)

            pred_dc, ti_time = 0.0, 0.0
            if predictor.enabled:
                raw_pred, ti_time = predictor.predict(f)
                pred_dc = raw_pred * (curr_size / args.initial_size)

            label_dc   = 0.0
            label_path = (f.parent /
                          f.name.replace('_edges.npz', '_label.json')
                                .replace('.npz', '_label.json'))
            if label_path.exists():
                try:
                    with open(label_path) as lf:
                        lr       = json.load(lf).get('critical_threshold', 0.0)
                        label_dc = lr * (curr_size / args.initial_size)
                except Exception:
                    pass

            data_records.append({
                'filename': f.name, 'algorithm': algo, 'step': step,
                'network_size': curr_size, 'LCC': lcc,
                'Predicted DC': pred_dc, 'TCR-GIN_Time': ti_time, 'Label_DC': label_dc,
            })

        df_updated = pd.DataFrame(data_records)
        if df_updated.empty:
            print(" -> [Error] No records generated.")
            return
    else:
        df_updated = df_exist.copy()

    print(" -> [Merge] Baseline predictor outputs ...")
    baseline_df = load_baseline_predictions_df(input_root, args.initial_size)
    if not baseline_df.empty:
        drop_cols = [c for c in df_updated.columns if c.startswith('Predictor_')]
        if drop_cols:
            df_updated = df_updated.drop(columns=drop_cols)
        df_updated = pd.merge(df_updated, baseline_df, on=['algorithm', 'step'], how='left')

    initial_graph = None
    step0_files   = []
    rem_dir = input_root / f"{dataset_name}-Remnants"
    if rem_dir.exists():
        step0_files = (list(rem_dir.glob("*_0_edges.npz")) +
                       list(rem_dir.glob("*_0.npz")))
        if not step0_files:
            nested = rem_dir / f"{dataset_name}-Remnants"
            if nested.exists():
                step0_files = (list(nested.glob("*_0_edges.npz")) +
                               list(nested.glob("*_0.npz")))
    if step0_files:
        initial_graph = load_graph_robust(step0_files[0])

    results_file = get_results_file_path(input_root)
    if results_file and initial_graph:
        print(f" -> [Merge] LCC curves (step=-1): {results_file.name}")
        curves, _ = parse_csv_result_file(
            results_file, initial_graph, args.initial_size, args.collapse_target)
        curve_rows = [
            {'algorithm': algo, 'step': int(r['step']), 'LCC_curve': float(r['LCC'])}
            for algo, cdf in curves.items()
            for _, r in cdf.iterrows()
        ]
        df_curve = pd.DataFrame(curve_rows)
        if not df_updated.empty:
            df_updated['step'] = df_updated['step'].fillna(0).astype(int)

        prefix = (step0_files[0].name.split('-')[0]
                  if step0_files and '-' in step0_files[0].name else dataset_name)

        def make_vf(algo):
            return f"{prefix}-{algo}_-1_edges.npz"

        if not df_curve.empty:
            if df_updated.empty:
                merged = df_curve.rename(columns={'LCC_curve': 'LCC'})
                merged['filename'] = merged['algorithm'].apply(make_vf)
            else:
                merged = pd.merge(df_updated, df_curve,
                                  on=['algorithm', 'step'], how='outer',
                                  suffixes=('', '_c'))
                merged['LCC']      = merged['LCC'].combine_first(merged['LCC_curve'])
                merged['filename'] = merged.apply(
                    lambda r: r['filename'] if pd.notna(r['filename'])
                    else make_vf(r['algorithm']), axis=1)
                merged = merged.drop(columns=['LCC_curve'])
            df_final = merged
        else:
            df_final = df_updated
    else:
        df_final = df_updated
        print(" -> [Warning] No results file or initial graph. Skipping step=-1 merge.")

    df_final.sort_values(by=['algorithm', 'step'], inplace=True)
    df_final.to_csv(csv_path, index=False)
    print(f" -> [Done] Saved: {csv_path}")

    if df_final.empty:
        print(" -> [Error] Empty DataFrame. Aborting.")
        return

    df_final['step']         = pd.to_numeric(df_final['step'],         errors='coerce')
    df_final['LCC']          = pd.to_numeric(df_final['LCC'],          errors='coerce')
    df_final['Predicted DC'] = pd.to_numeric(df_final['Predicted DC'], errors='coerce')

    print(" -> [Plot] Panel figures ...")
    generate_summary_plots(df_final, out_dir, dataset_name, args.collapse_target, args.initial_size)

    print(" -> [Plot] Scheme-3 figure ...")
    generate_scheme3_summary_plots(
        df_final, out_dir, dataset_name, args.collapse_target, args.initial_size)

    print(" -> [All Done]")


if __name__ == "__main__":
    main()
