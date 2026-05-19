#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/data_loader.py

Dataset and dataloader utilities for cached graph loading and PISS-style
training / evaluation workflows.

Function
--------
This module provides:
1. A cached graph dataset that loads raw graph files and stores them as `.pt`
   files for faster reuse
2. A PISS-style dataset wrapper that returns `(anchor, positive)` graph pairs
3. A custom collate function for batching valid graph pairs
4. Dataloader builders for train / validation / test splits
5. A single-graph loader for standalone evaluation or inference

Usage
-----
Example:
    from data_loader import get_piss_dataloaders, load_single_graph

    train_loader, val_loader, test_loader = get_piss_dataloaders(
        train_path="data/train",
        val_path="data/valid",
        test_path="data/test",
        batch_size=32,
        num_workers=4,
        cache_path="cache",
        rebuild_cache=False,
        piss_k=4,
        use_gpu=True,
        feature_dim=7,
    )
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


# ==============================================================================
# Core dataset classes
# ==============================================================================
class PISSGraphDataset(Dataset):
    """
    Dataset wrapper for the PISS framework.

    This wrapper converts a standard graph dataset into `(anchor, positive)`
    pairs.

    Notes
    -----
    - In the current training pipeline, the positive sample returned here is not
      used during training.
    - Training generates K dynamic single-node-deletion positive candidates
      online and selects the worst one.
    - Therefore, this dataset usually uses `k=0`, which makes the returned
      positive sample identical to the anchor and avoids unnecessary subgraph
      construction.
    """

    def __init__(self, data_list: list, k: int = 0):
        super(PISSGraphDataset, self).__init__()
        self.data_list = [d for d in data_list if d is not None and d.num_nodes > k]
        self.k = k

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> tuple[PyGData, PyGData]:
        anchor_data = self.data_list[idx]

        if self.k <= 0:
            return anchor_data, anchor_data

        positive_data = anchor_data.clone()
        num_nodes = anchor_data.num_nodes

        nodes_to_remove = min(self.k, num_nodes - 1) if num_nodes > 1 else 0
        if nodes_to_remove > 0:
            perm = torch.randperm(num_nodes)
            nodes_to_drop_indices = perm[:nodes_to_remove]
            keep_mask = torch.ones(num_nodes, dtype=torch.bool)
            keep_mask[nodes_to_drop_indices] = False
            positive_data = positive_data.subgraph(keep_mask)

        return anchor_data, positive_data


class CachedGraphDataset(Dataset):
    """Load graph data from raw files and cache it as `.pt` files."""

    def __init__(
        self,
        dataset_path: str,
        graph_ids: List[str],
        cache_path: str = "cache/",
        rebuild_cache: bool = False,
        skip_invalid: bool = True,
        feature_dim: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.graph_ids = graph_ids
        self.cache_path = Path(cache_path)
        self.skip_invalid = skip_invalid
        self.feature_dim = feature_dim

        feature_suffix = (
            f"fd{self.feature_dim}"
            if self.feature_dim is not None and self.feature_dim > 0
            else "fdAll"
        )
        self.cache_suffix = f"_{feature_suffix}_cached.pt"

        self.cache_path.mkdir(parents=True, exist_ok=True)

        if rebuild_cache:
            print(f"[*] Building / validating cache for dataset '{self.dataset_path.name}' @ {self.cache_path}")
            desc = f"Building cache(feat_dim={self.feature_dim or 'All'})"
            for gid in tqdm(self.graph_ids, desc=desc):
                self._build_and_cache_if_not_exist(gid)

    def _find_files_for_graph(self, graph_id: str) -> Dict[str, Path]:
        """Robustly locate the full file set for one graph."""
        possible_prefixes = [
            self.dataset_path / f"net_{graph_id}",
            self.dataset_path / graph_id,
        ]

        for prefix in possible_prefixes:
            if (prefix.parent / f"{prefix.name}_label.json").exists():
                return {
                    'edges': prefix.parent / f"{prefix.name}_edges.npz",
                    'features': prefix.parent / f"{prefix.name}_features.npy",
                    'label': prefix.parent / f"{prefix.name}_label.json",
                }

        raise FileNotFoundError(
            f"Could not find the complete file set for graph {graph_id} in {self.dataset_path}"
        )

    def _build_and_cache_if_not_exist(self, graph_id: str):
        """Create a cache file if it does not already exist."""
        save_path = self.cache_path / f"{graph_id}{self.cache_suffix}"
        if save_path.exists():
            return

        try:
            self._build_and_cache(graph_id, save_path)
        except Exception as e:
            if self.skip_invalid:
                print(f"[!] Warning: failed to build or cache graph {graph_id}; skipping. Error: {e}")
            else:
                raise e

    def _build_and_cache(self, graph_id: str, save_path: Path) -> Optional[PyGData]:
        """Load one graph from raw files and write its cached `.pt` version."""
        file_paths = self._find_files_for_graph(graph_id)

        with open(file_paths['label'], 'r') as f:
            label_data = json.load(f)
        threshold = label_data['critical_threshold']

        with np.load(file_paths['edges'], allow_pickle=True) as loader:
            edges = loader.get('edges', loader.get('data'))

        row, col = edges[:, 0], edges[:, 1]
        edge_index = torch.from_numpy(
            np.array([np.concatenate([row, col]), np.concatenate([col, row])])
        ).long()

        features = np.load(file_paths['features'])

        if self.feature_dim is not None and self.feature_dim > 0:
            if self.feature_dim > features.shape[1]:
                print(
                    f"[!] Warning: feature_dim({self.feature_dim}) > actual feature count({features.shape[1]}). "
                    f"Using all features."
                )
            else:
                features = features[:, :self.feature_dim]

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor([[threshold]], dtype=torch.float32)

        data = PyGData(x=x, edge_index=edge_index, y=y)
        torch.save(data, save_path)
        return data

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx: int) -> Optional[PyGData]:
        graph_id = self.graph_ids[idx]
        data_path = self.cache_path / f"{graph_id}{self.cache_suffix}"

        try:
            if not data_path.exists():
                self._build_and_cache(graph_id, data_path)
            return torch.load(data_path)
        except Exception as e:
            if self.skip_invalid:
                print(f"[!] Warning: failed to load graph {graph_id}; skipping. Error: {e}")
                return None
            raise e


# ==============================================================================
# Helper functions
# ==============================================================================
def get_graph_ids(dataset_path: str) -> List[str]:
    """Extract all unique graph IDs from a dataset directory."""
    if not os.path.isdir(dataset_path):
        return []

    label_files = [f for f in os.listdir(dataset_path) if f.endswith('_label.json')]
    graph_ids = {f.replace('_label.json', '') for f in label_files}
    return sorted(list(graph_ids))


def piss_collate(data_list: List[Tuple[PyGData, PyGData]]):
    """Custom collate function for PISS graph pairs."""
    valid_pairs = [pair for pair in data_list if pair is not None]
    if not valid_pairs:
        return None, None

    anchor_list, positive_list = zip(*valid_pairs)
    return Batch.from_data_list(list(anchor_list)), Batch.from_data_list(list(positive_list))


# ==============================================================================
# Dataloader entry points
# ==============================================================================
def get_piss_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int,
    cache_path: str,
    rebuild_cache: bool,
    piss_k: int,
    use_gpu: bool,
    feature_dim: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders for PISS model training.

    Notes
    -----
    - `piss_k` is no longer used here to delete K nodes when building positive
      samples.
    - During training, consistency positives are generated dynamically inside
      the training loop as K single-node-deletion candidates.
    - Therefore, the dataset wrapper here uses `k=0` for train / val / test to
      avoid unnecessary positive-sample construction overhead.
    """
    train_ids = get_graph_ids(train_path)
    val_ids = get_graph_ids(val_path)
    test_ids = get_graph_ids(test_path)

    if not train_ids:
        raise RuntimeError(f"No graph files were found in training path '{train_path}'.")

    def load_data(path, ids, rebuild):
        dataset = CachedGraphDataset(
            path,
            ids,
            cache_path,
            rebuild,
            feature_dim=feature_dim,
        )
        return [data for data in dataset]

    train_data_list = load_data(train_path, train_ids, rebuild_cache)
    val_data_list = load_data(val_path, val_ids, False)
    test_data_list = load_data(test_path, test_ids, False)

    train_dataset = PISSGraphDataset(train_data_list, k=0)
    val_dataset = PISSGraphDataset(val_data_list, k=0)
    test_dataset = PISSGraphDataset(test_data_list, k=0)

    loader_kwargs = {'num_workers': num_workers, 'pin_memory': use_gpu}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=piss_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=piss_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=piss_collate,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def load_single_graph(graph_prefix: str, feature_dim: Optional[int] = None) -> Optional[PyGData]:
    """Load a single graph for standalone evaluation or inference."""
    try:
        edges_path = f"{graph_prefix}_edges.npz"
        features_path = f"{graph_prefix}_features.npy"
        label_path = f"{graph_prefix}_label.json"

        with np.load(edges_path) as loader:
            edges = loader.get('edges', loader.get('data'))

        row, col = edges[:, 0], edges[:, 1]
        edge_index = torch.from_numpy(
            np.array([np.concatenate([row, col]), np.concatenate([col, row])])
        ).long()

        features = np.load(features_path)

        if feature_dim is not None and feature_dim > 0:
            if feature_dim <= features.shape[1]:
                features = features[:, :feature_dim]

        x = torch.from_numpy(features).float()

        y = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                y_scalar = json.load(f)['critical_threshold']
            y = torch.tensor([y_scalar], dtype=torch.float).view(1, -1)

        return PyGData(x=x, edge_index=edge_index, y=y)

    except Exception as e:
        print(f"Failed to load graph {Path(graph_prefix).name}: {e}")
        return None
