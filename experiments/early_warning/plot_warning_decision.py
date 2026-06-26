#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the network structure and decision plan at an early-warning step.
Designed for journal-style figures.

Features:
1. Left panel: residual network at the warning step, top collapse-node candidates, and remaining attack sequence.
2. Right panel: robustness-metric time series with warning step and decision window annotations.

Usage:
    python plot_warning_decision.py --network transport-111 --algorithm DC --warning-step 12
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.sparse import load_npz

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # TCR-GIN project root

# Journal-style colors compatible with ColorBrewer
COLORS = {
    'node_default': '#4e79a7',      # blue - regular node
    'node_priority1': '#e15759',    # red - first-priority collapse node
    'node_priority2': '#f28e2b',    # orange - second-priority collapse node
    'node_priority3': '#59a14f',    # green - third-priority collapse node
    'node_articulation': '#ff6b6b', # pink - articulation node
    'edge_normal': '#cccccc',       # light gray - regular edge
    'edge_bridge': '#d62728',       # dark red - bridge edge
    'warning_line': '#e15759',      # red - warning line
    'decision_window': '#ffdd89',   # pale yellow - decision-window shading
    'lcc_curve': '#1f77b4',         # LCC curve
    'pred_curve': '#9467bd',        # prediction curve
}

# Node marker styles
NODE_MARKERS = {
    'priority1': '*',  # star
    'priority2': 's',  # square
    'priority3': '^',  # triangle
}

# ============================================================================
# Data loading helpers
# ============================================================================
def normalize_network_name(network_name):
    """Normalize a network name by replacing hyphens with underscores."""
    return network_name.replace('-', '_')


def load_network_at_step(network_name, algorithm, step):
    """Load the network structure for a specified step."""
    network_normalized = normalize_network_name(network_name)
    
    # Try several possible paths
    possible_paths = [
        PROJECT_ROOT / f"webapp/examples/{network_normalized}_demo/network/{network_normalized}_multiplex_aggr_edges.npz",
        PROJECT_ROOT / f"webapp/examples/transport_demo/network/transport_multiplex_aggr_edges.npz",
        PROJECT_ROOT / f"data/{network_name}/{network_normalized}_aggr.npz",
        PROJECT_ROOT / f"experiments/early_warning/data/{network_normalized}.npz",
    ]
    
    network_file = None
    for path in possible_paths:
        if path.exists():
            network_file = path
            print(f"Found network file: {path}")
            break
    
    if network_file is None:
        raise FileNotFoundError(f"Network file not found for {network_name}; tried:\n" + 
                              "\n".join(str(p) for p in possible_paths))
    
    # Load original network
    adj_matrix = load_npz(network_file)
    G = nx.from_scipy_sparse_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    
    # Read metrics to get the network size at this step and infer removed-node count
    metrics_file = SCRIPT_DIR / f"results/decision_window/{network_name}/{network_normalized}_metrics.csv"
    
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        df_step = df[(df['algorithm'] == algorithm) & (df['step'] == step)]
        if not df_step.empty:
            remaining_size = int(df_step.iloc[0]['network_size'])
            original_size = G.number_of_nodes()
            removed_count = original_size - remaining_size
            print(f"Original network: {original_size} nodes, at step {step}: {remaining_size} nodes ({removed_count} removed)")
            
            # Note: return the full network because the exact removed nodes are unknown here
            # If an attack-sequence file is available, load it and remove the corresponding nodes
    
    return G, []


def get_metrics_data(network_name, algorithm):
    """Load robustness metric data."""
    network_normalized = normalize_network_name(network_name)
    metrics_file = SCRIPT_DIR / f"results/decision_window/{network_name}/{network_normalized}_metrics.csv"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    df = pd.read_csv(metrics_file)
    df = df[df['algorithm'] == algorithm].sort_values('step')
    return df


def get_warning_info(network_name, algorithm):
    """Load warning information."""
    network_normalized = normalize_network_name(network_name)
    summary_file = SCRIPT_DIR / f"results/decision_window/{network_name}/{network_normalized}_decision_summary.csv"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    row = df[(df['Algorithm'] == algorithm) & (df['Predictor'] == 'TCR-GIN')].iloc[0]
    
    return {
        'warning_step': int(row['Warning Step']),
        'collapse_step': int(row['Collapse Step']),
        'lead_time': int(row['Lead Time (steps)']),
        'threshold': float(row['Warning Threshold']),
        'target': float(row['Collapse Target']),
    }


def get_top_nodes_for_removal(G, top_k=3, method='degree'):
    """Get top collapse-node candidates.
    
    method: 'degree', 'betweenness', 'closeness'
    """
    if method == 'degree':
        centrality = nx.degree_centrality(G)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.degree_centrality(G)
    
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:top_k]]


# ============================================================================
# Plotting functions
# ============================================================================
def draw_network_panel(ax, G, warning_step, top_nodes, removed_nodes, algorithm):
    """Draw the network-structure panel."""
    ax.set_title(f'(a) Network at Warning Step {warning_step}', 
                 fontsize=10, fontweight='bold', loc='left', pad=8)
    ax.axis('off')
    
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center', fontsize=10)
        return
    
    # Compute layout
    if G.number_of_nodes() < 1000:
        pos = nx.spring_layout(G, seed=42, iterations=200, k=1.5/np.sqrt(max(G.number_of_nodes(), 1)))
    else:
        # Use a faster layout for large networks
        pos = nx.kamada_kawai_layout(G)
    
    # Identify bridge edges and articulation points
    bridges = set(tuple(sorted(e)) for e in nx.bridges(G)) if G.number_of_edges() > 0 else set()
    articulation_points = set(nx.articulation_points(G)) if G.number_of_nodes() > 0 else set()
    
    # Draw edges
    regular_edges = [e for e in G.edges() if tuple(sorted(e)) not in bridges]
    bridge_edges = [e for e in G.edges() if tuple(sorted(e)) in bridges]
    
    if regular_edges:
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax,
                              edge_color=COLORS['edge_normal'], width=0.5, alpha=0.4)
    if bridge_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bridge_edges, ax=ax,
                              edge_color=COLORS['edge_bridge'], width=1.2, alpha=0.9)
    
    # Draw regular nodes
    normal_nodes = [n for n in G.nodes() if n not in top_nodes and n not in articulation_points]
    if normal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, ax=ax,
                              node_color=COLORS['node_default'], node_size=30, alpha=0.7)
    
    # Draw articulation points outside the top candidates
    cut_nodes = [n for n in articulation_points if n not in top_nodes]
    if cut_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=cut_nodes, ax=ax,
                              node_color=COLORS['node_articulation'], 
                              node_size=50, alpha=0.8, edgecolors='white', linewidths=1)
    
    # Draw top collapse-node candidates with distinct markers
    markers_config = [
        (0, NODE_MARKERS['priority1'], COLORS['node_priority1'], 200, '1st'),
        (1, NODE_MARKERS['priority2'], COLORS['node_priority2'], 150, '2nd'),
        (2, NODE_MARKERS['priority3'], COLORS['node_priority3'], 120, '3rd'),
    ]
    
    for idx, marker, color, size, label in markers_config:
        if idx < len(top_nodes):
            node = top_nodes[idx]
            if node in pos:  # Ensure the node exists
                nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                      node_shape=marker, node_color=color, 
                                      node_size=size, edgecolors='black', linewidths=1.5,
                                      alpha=0.95)
                # Add node rank label
                x, y = pos[node]
                ax.text(x, y-0.08, f'{idx+1}', fontsize=8, ha='center', 
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.8))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLORS['node_priority1'], label='1st priority'),
        mpatches.Patch(color=COLORS['node_priority2'], label='2nd priority'),
        mpatches.Patch(color=COLORS['node_priority3'], label='3rd priority'),
        mpatches.Patch(color=COLORS['node_articulation'], label='Articulation pt'),
        plt.Line2D([0], [0], color=COLORS['edge_bridge'], linewidth=2, label='Bridge'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, 
             framealpha=0.9, edgecolor='gray')
    
    # Add network statistics
    stats_text = f"n={G.number_of_nodes()}, m={G.number_of_edges()}\n" \
                 f"Components: {nx.number_connected_components(G)}\n" \
                 f"Bridges: {len(bridges)}, Cut points: {len(articulation_points)}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=7,
           verticalalignment='bottom', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))


def draw_metrics_panel(ax, df, warning_info):
    """Draw the robustness-metrics panel."""
    ax.set_title('(b) Robustness Metrics Over Time', 
                fontsize=10, fontweight='bold', loc='left', pad=8)
    
    warning_step = warning_info['warning_step']
    collapse_step = warning_info['collapse_step']
    target = warning_info['target']
    
    # Draw LCC curve
    ax.plot(df['step'], df['LCC'], color=COLORS['lcc_curve'], 
           linewidth=1.5, label='LCC', alpha=0.9)
    
    # Draw predicted DC curve
    if 'Predicted DC' in df.columns:
        pred_data = df[['step', 'Predicted DC']].dropna()
        if not pred_data.empty:
            ax.plot(pred_data['step'], pred_data['Predicted DC'], 
                   color=COLORS['pred_curve'],
                   linewidth=1.2, label='Predicted DC', linestyle='--', alpha=0.8)
    
    # Annotate target line
    ax.axhline(target, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(df['step'].max() * 0.98, target, f'Target={target}', 
           fontsize=8, va='bottom', ha='right', color='gray', style='italic')
    
    # Annotate warning step
    ax.axvline(warning_step, color=COLORS['warning_line'], 
              linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)
    ax.text(warning_step, ax.get_ylim()[1] * 0.95, 'Warning', 
           fontsize=9, ha='center', va='top', color=COLORS['warning_line'],
           fontweight='bold', rotation=0)
    
    # Annotate decision window (next three steps)
    decision_window_end = min(warning_step + 3, collapse_step)
    ax.axvspan(warning_step, decision_window_end, 
              color=COLORS['decision_window'], alpha=0.3, zorder=1)
    
    # Add decision-window label
    window_center = (warning_step + decision_window_end) / 2
    ax.text(window_center, ax.get_ylim()[1] * 0.85, 'Decision\nWindow', 
           fontsize=8, ha='center', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor=COLORS['decision_window'], alpha=0.8))
    
    # Configure axes
    ax.set_xlabel('Attack Step', fontsize=9)
    ax.set_ylabel('Metric Value', fontsize=9)
    ax.set_xlim(0, df['step'].max() * 1.05)
    ax.set_ylim(0, 1.05)
    
    # Grid and ticks
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax.tick_params(labelsize=8)
    
    # Legend
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================================
# Main function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network', type=str, default='transport-111',
                       help='Network name, e.g. transport-111')
    parser.add_argument('--algorithm', type=str, default='DC',
                       help='Attack algorithm: DC or BC')
    parser.add_argument('--warning-step', type=int, default=None,
                       help='Warning step; read from summary if omitted')
    parser.add_argument('--centrality', type=str, default='degree',
                       choices=['degree', 'betweenness', 'closeness'],
                       help='Node centrality metric')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path; generated automatically if omitted')
    parser.add_argument('--dpi', type=int, default=600,
                       help='Image resolution')
    parser.add_argument('--figsize', type=str, default='7.2,3.5',
                       help='Figure size in inches, formatted as width,height')
    
    args = parser.parse_args()
    
    # Parse figure size
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Configure matplotlib parameters for journal-style output
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.8,
        'pdf.fonttype': 42,  # TrueType fonts
        'ps.fonttype': 42,
    })
    
    # Load warning information
    print(f"Loading data for {args.network} ({args.algorithm})...")
    warning_info = get_warning_info(args.network, args.algorithm)
    warning_step = args.warning_step or warning_info['warning_step']
    warning_info['warning_step'] = warning_step
    
    print(f"Warning step: {warning_step}")
    print(f"Collapse step: {warning_info['collapse_step']}")
    print(f"Lead time: {warning_info['lead_time']} steps")
    
    # Load network and metric data
    G, removed_nodes = load_network_at_step(args.network, args.algorithm, warning_step)
    df = get_metrics_data(args.network, args.algorithm)
    
    # Get top collapse-node candidates
    top_nodes = get_top_nodes_for_removal(G, top_k=3, method=args.centrality)
    print(f"Top 3 nodes for removal (by {args.centrality}): {top_nodes}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Draw left network panel
    draw_network_panel(axes[0], G, warning_step, top_nodes, removed_nodes, args.algorithm)
    
    # Draw right metrics panel
    draw_metrics_panel(axes[1], df, warning_info)
    
    # Adjust layout
    plt.tight_layout(pad=1.5, w_pad=2.0)
    
    # Save figure
    if args.output is None:
        network_normalized = normalize_network_name(args.network)
        output_dir = SCRIPT_DIR / f"results/decision_window/{args.network}/plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{network_normalized}_warning_decision_{args.algorithm}_step{warning_step}.pdf"
    else:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF and PNG
    for ext in ['pdf', 'png']:
        save_path = output_file.with_suffix(f'.{ext}')
        fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()