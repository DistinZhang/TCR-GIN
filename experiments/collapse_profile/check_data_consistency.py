#!/usr/bin/env python3
"""Check consistency of edges.npz and features.npy for all graphs in a directory."""

import os
import sys
import numpy as np
from pathlib import Path

def check_directory(data_dir):
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*_edges.npz"))
    
    issues = []
    for npz_path in npz_files:
        graph_id = npz_path.name.replace("_edges.npz", "")
        feat_path = data_dir / f"{graph_id}_features.npy"
        
        if not feat_path.exists():
            issues.append(f"[MISSING] {graph_id}: features file not found")
            continue
        
        with np.load(npz_path, allow_pickle=True) as loader:
            edges = loader.get("edges", loader.get("data"))
        
        features = np.load(feat_path)
        
        num_feat_nodes = features.shape[0]
        max_edge_node = int(edges.max()) + 1 if edges.size > 0 else 0
        
        if max_edge_node > num_feat_nodes:
            issues.append(
                f"[MISMATCH] {graph_id}: features={num_feat_nodes} rows, "
                f"edges reference node {max_edge_node - 1} "
                f"(need at least {max_edge_node} rows, short by {max_edge_node - num_feat_nodes})"
            )
    
    return issues

if __name__ == "__main__":
    dirs_to_check = [
        "/root/autodl-tmp/ERGM/split-ergm/power/configuration/train",
        "/root/autodl-tmp/ERGM/split-ergm/power/configuration/valid",
        "/root/autodl-tmp/ERGM/split-ergm/power/configuration/test",
        "/root/autodl-tmp/ERGM/split-ergm/power/erdos_renyi/train",
        "/root/autodl-tmp/ERGM/split-ergm/power/erdos_renyi/valid",
        "/root/autodl-tmp/ERGM/split-ergm/power/erdos_renyi/test",
        "/root/autodl-tmp/ERGM/split-ergm/power/stochastic_block/train",
        "/root/autodl-tmp/ERGM/split-ergm/power/stochastic_block/valid",
        "/root/autodl-tmp/ERGM/split-ergm/power/stochastic_block/test",
    ]
    
    total_issues = 0
    for d in dirs_to_check:
        if not os.path.isdir(d):
            print(f"[SKIP] Directory not found: {d}")
            continue
        
        issues = check_directory(d)
        n_files = len(list(Path(d).glob("*_edges.npz")))
        
        if issues:
            print(f"\n[{d}] — {len(issues)} issues in {n_files} graphs:")
            for issue in issues:
                print(f"  {issue}")
            total_issues += len(issues)
        else:
            print(f"[OK] {d} — {n_files} graphs, all consistent")
    
    print(f"\n{'='*60}")
    if total_issues:
        print(f"Total issues found: {total_issues}")
    else:
        print("All data is consistent!")
