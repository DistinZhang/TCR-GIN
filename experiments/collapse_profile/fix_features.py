#!/usr/bin/env python3
"""
Fix feature files: ensure feature rows cover ALL node IDs in edges.

For isolated nodes (not in any edge), features are set to 0.

Usage:
    python fix_features.py --data_dir /root/autodl-tmp/ERGM/split-ergm/london --feature_set full
    python fix_features.py --data_dir /root/autodl-tmp/ERGM/split-ergm/london --feature_set full --dry_run
"""

import argparse
import os
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import random


def calculate_node_features_safe(G, num_nodes, feature_set='basic'):
    """
    Calculate node features for ALL nodes 0..num_nodes-1,
    including isolated nodes that might not appear in any edge.

    Args:
        G: NetworkX graph (may only contain connected nodes)
        num_nodes: Total number of nodes (features will have this many rows)
        feature_set: 'basic', 'extended', or 'full'

    Returns:
        feature_matrix: (num_nodes, num_features)
        feature_names: list of feature names
    """
    # Ensure graph has all node IDs 0..num_nodes-1
    for i in range(num_nodes):
        if i not in G:
            G.add_node(i)

    nodes = list(range(num_nodes))  # Ensure ordered 0,1,...,num_nodes-1
    features_dict = {}

    # Basic features
    deg_dict = dict(G.degree())
    features_dict['degree'] = np.array([deg_dict.get(u, 0) for u in nodes], dtype=np.float64)
    features_dict['clustering'] = np.array([nx.clustering(G, u) for u in nodes], dtype=np.float64)

    core = nx.core_number(G)
    features_dict['kcore'] = np.array([core.get(u, 0) for u in nodes], dtype=np.float64)

    # Extended features
    if feature_set in ['extended', 'full']:
        features_dict['avg_neighbor_deg'] = np.array([
            sum(G.degree(v) for v in G.neighbors(u)) / max(1, G.degree(u))
            for u in nodes
        ], dtype=np.float64)

        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100)
            features_dict['pagerank'] = np.array([pr.get(u, 0.0) for u in nodes], dtype=np.float64)
        except Exception:
            features_dict['pagerank'] = np.full(num_nodes, 1.0 / max(1, num_nodes), dtype=np.float64)

    # Full features
    if feature_set == 'full':
        try:
            bc = nx.betweenness_centrality(G, k=min(30, len(G)), seed=42)
            features_dict['betweenness'] = np.array([bc.get(u, 0.0) for u in nodes], dtype=np.float64)
        except Exception:
            features_dict['betweenness'] = np.zeros(num_nodes, dtype=np.float64)

        try:
            ec = nx.eigenvector_centrality_numpy(G)
            features_dict['eigenvector'] = np.array([ec.get(u, 0.0) for u in nodes], dtype=np.float64)
        except Exception:
            max_deg = max(features_dict['degree']) if features_dict['degree'].max() > 0 else 1.0
            features_dict['eigenvector'] = features_dict['degree'] / max(1.0, max_deg)

    feature_matrix = np.column_stack([features_dict[f] for f in features_dict])
    return feature_matrix, list(features_dict.keys())


def load_graph_from_npz(npz_path, num_nodes=None):
    """Load graph from npz, ensuring all nodes 0..num_nodes-1 exist."""
    data = np.load(npz_path, allow_pickle=True)
    edges = None
    for key in ['edges', 'edge_list', 'edgelist', 'data']:
        if key in data:
            edges = data[key]
            break
    if edges is None:
        edges = data[list(data.keys())[0]]

    G = nx.Graph()

    # Add all nodes first if num_nodes specified
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))

    G.add_edges_from(edges)

    # Infer num_nodes from edges if not specified
    if num_nodes is None:
        max_node = int(edges.max()) + 1 if edges.size > 0 else 0
        num_nodes = max(G.number_of_nodes(), max_node)
        for i in range(num_nodes):
            if i not in G:
                G.add_node(i)

    return G, num_nodes


def fix_directory(data_dir, feature_set, dry_run=False):
    """Fix all feature files in a directory."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*_edges.npz"))

    if not npz_files:
        print(f"  No edge files found in {data_dir}")
        return 0, 0, 0

    fixed = 0
    skipped = 0
    errors = 0

    for npz_path in tqdm(npz_files, desc=f"  {data_dir.name}", leave=False):
        graph_id = npz_path.name.replace("_edges.npz", "")
        feat_path = data_dir / f"{graph_id}_features.npy"

        try:
            # Load edges to determine actual num_nodes needed
            with np.load(npz_path, allow_pickle=True) as loader:
                edges = loader.get("edges", loader.get("data"))
                if edges is None:
                    edges = loader[list(loader.keys())[0]]

            max_edge_node = int(edges.max()) + 1 if edges.size > 0 else 0

            # Load existing features
            if feat_path.exists():
                old_features = np.load(feat_path)
                num_feat_rows = old_features.shape[0]
            else:
                num_feat_rows = 0

            # Check if fix is needed
            if num_feat_rows >= max_edge_node:
                skipped += 1
                continue

            # Need to fix
            num_nodes = max_edge_node  # At minimum need this many rows

            if dry_run:
                print(f"    [DRY RUN] Would fix {graph_id}: {num_feat_rows} → {num_nodes} rows")
                fixed += 1
                continue

            # Rebuild graph and compute features
            G, num_nodes = load_graph_from_npz(npz_path, num_nodes=num_nodes)
            new_features, feature_names = calculate_node_features_safe(G, num_nodes, feature_set)

            # Save
            np.save(feat_path, new_features)
            fixed += 1

        except Exception as e:
            print(f"    [ERROR] {graph_id}: {e}")
            errors += 1

    return fixed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Fix feature files for ERGM datasets")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory (e.g., /root/autodl-tmp/ERGM/split-ergm/london)")
    parser.add_argument("--feature_set", type=str, default="full",
                        choices=["basic", "extended", "full"],
                        help="Feature set to compute (must match what training expects)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report what would be fixed, don't modify files")
    parser.add_argument("--subdirs", type=str, nargs="*", default=None,
                        help="Specific subdirectories to fix (e.g., 'configuration/train erdos_renyi/train')")
    args = parser.parse_args()

    root = Path(args.data_dir)

    if args.subdirs:
        dirs_to_fix = [root / s for s in args.subdirs]
    else:
        # Auto-discover: find all directories containing _edges.npz files
        dirs_to_fix = set()
        for npz in root.rglob("*_edges.npz"):
            dirs_to_fix.add(npz.parent)
        dirs_to_fix = sorted(dirs_to_fix)

    print(f"{'='*70}")
    print(f"Feature Fix Script")
    print(f"{'='*70}")
    print(f"Root directory: {root}")
    print(f"Feature set: {args.feature_set}")
    print(f"Dry run: {args.dry_run}")
    print(f"Directories to process: {len(dirs_to_fix)}")
    print()

    total_fixed = 0
    total_skipped = 0
    total_errors = 0

    for d in dirs_to_fix:
        if not d.exists():
            print(f"[SKIP] Not found: {d}")
            continue

        rel_path = d.relative_to(root) if d.is_relative_to(root) else d
        print(f"Processing: {rel_path}")

        fixed, skipped, errors = fix_directory(d, args.feature_set, args.dry_run)
        total_fixed += fixed
        total_skipped += skipped
        total_errors += errors

        print(f"  → Fixed: {fixed}, Already OK: {skipped}, Errors: {errors}")

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total fixed:      {total_fixed}")
    print(f"  Total already OK: {total_skipped}")
    print(f"  Total errors:     {total_errors}")
    if args.dry_run:
        print(f"\n  ⚠ This was a DRY RUN. No files were modified.")
        print(f"  Remove --dry_run to apply fixes.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
