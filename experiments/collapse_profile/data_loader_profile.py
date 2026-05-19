#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading utilities for collapse-profile training.

Function
--------
This module loads single-source or multi-source collapse-profile graph datasets,
caches processed PyTorch Geometric graph objects, and builds train/validation/test
dataloaders for profile-valued PISS training.

Inputs
------
- train/validation/test/cache paths as a string, comma-separated string, or list.
- Graph triplets containing `*_edges.npz`, `*_features.npy`, and profile label
    JSON files.
- Tau values and label keys used to slice profile targets.

Outputs
-------
- Cached graph objects.
- `PISSGraphDataset` instances and PyTorch Geometric dataloaders.

Usage
-----
Example:
        train_loader, val_loader, test_loader = get_profile_dataloaders(
                train_path=["data/source_a/train", "data/source_b/train"],
                val_path=["data/source_a/valid", "data/source_b/valid"],
                test_path=["data/source_a/test", "data/source_b/test"],
                cache_path="cache/profile",
                batch_size=32,
                tau_values=[0.1, 0.2, 0.3],
        )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _to_path_list(val: Union[None, str, list]) -> List[str]:
    """Convert supported path inputs into a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(p).strip() for p in val if str(p).strip()]
    if isinstance(val, str):
        return [p.strip() for p in val.split(",") if p.strip()]
    return [str(val).strip()]


def _align_cache_paths(
    cache_paths: List[str],
    ref_paths: List[str],
    fallback_prefix: str = "cache",
) -> List[str]:
    """Align cache_paths to the length of ref_paths."""
    n = len(ref_paths)
    if len(cache_paths) == 0:
        return [os.path.join(fallback_prefix, f"split_{i}") for i in range(n)]
    if len(cache_paths) == 1 and n > 1:
        base = cache_paths[0]
        return [os.path.join(base, f"split_{i}") for i in range(n)]
    if len(cache_paths) == n:
        return cache_paths
    raise ValueError(
        f"cache_path length ({len(cache_paths)}) does not match data path length ({n}).\n"
        f"  cache_path : {cache_paths}\n"
        f"  data_path  : {ref_paths}"
    )


# ---------------------------------------------------------------------------
# Dataset: single source with cache
# ---------------------------------------------------------------------------

class PISSGraphDataset(Dataset):
    def __init__(self, data_list: List[PyGData], k: int = 0):
        super().__init__()
        self.data_list = [d for d in data_list if d is not None and d.num_nodes > k]
        self.k = k

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        anchor_data = self.data_list[idx]
        if self.k <= 0:
            return anchor_data, anchor_data

        positive_data = anchor_data.clone()
        num_nodes = anchor_data.num_nodes
        nodes_to_remove = min(self.k, num_nodes - 1) if num_nodes > 1 else 0
        if nodes_to_remove > 0:
            perm = torch.randperm(num_nodes)
            nodes_to_drop = perm[:nodes_to_remove]
            keep_mask = torch.ones(num_nodes, dtype=torch.bool)
            keep_mask[nodes_to_drop] = False
            positive_data = positive_data.subgraph(keep_mask)
        return anchor_data, positive_data


class CachedProfileGraphDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        graph_ids: List[str],
        tau_values: List[float],
        cache_path: str = "cache/",
        rebuild_cache: bool = False,
        skip_invalid: bool = True,
        feature_dim: Optional[int] = None,
        label_suffix: str = "_profile_label.json",
        label_tau_key: str = "tau_grid_full",
        label_profile_key: str = "collapse_profile_full",
    ):
        self.dataset_path = Path(dataset_path)
        self.graph_ids = graph_ids
        self.tau_values = [float(x) for x in tau_values]
        self.cache_path = Path(cache_path)
        self.skip_invalid = skip_invalid
        self.feature_dim = feature_dim
        self.label_suffix = label_suffix
        self.label_tau_key = label_tau_key
        self.label_profile_key = label_profile_key

        feature_suffix = (
            f"fd{self.feature_dim}"
            if self.feature_dim is not None and self.feature_dim > 0
            else "fdAll"
        )
        tau_suffix = "tau_" + "-".join(
            f"{x:.2f}".replace(".", "p") for x in self.tau_values
        )
        self.cache_suffix = f"_{feature_suffix}_{tau_suffix}_cached.pt"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if rebuild_cache:
            for gid in self.graph_ids:
                self._build_and_cache_if_not_exist(gid)

    # --- File discovery ---

    def _find_files_for_graph(self, graph_id: str) -> Dict[str, Path]:
        edge_candidates = [
            self.dataset_path / f"{graph_id}_edges.npz",
            self.dataset_path / f"net_{graph_id}_edges.npz",
        ]
        feat_candidates = [
            self.dataset_path / f"{graph_id}_features.npy",
            self.dataset_path / f"net_{graph_id}_features.npy",
        ]
        label_candidates = [
            self.dataset_path / f"{graph_id}{self.label_suffix}",
            self.dataset_path / f"net_{graph_id}{self.label_suffix}",
        ]

        edges_path   = next((p for p in edge_candidates  if p.exists()), None)
        features_path= next((p for p in feat_candidates  if p.exists()), None)
        label_path   = next((p for p in label_candidates if p.exists()), None)

        if edges_path and features_path and label_path:
            return {"edges": edges_path, "features": features_path, "label": label_path}
        raise FileNotFoundError(
            f"Could not find edges / features / profile label "
            f"for graph '{graph_id}' in {self.dataset_path}"
        )

    # --- Build and cache ---

    def _build_and_cache_if_not_exist(self, graph_id: str):
        save_path = self.cache_path / f"{graph_id}{self.cache_suffix}"
        if save_path.exists():
            return
        try:
            self._build_and_cache(graph_id, save_path)
        except Exception as e:
            if self.skip_invalid:
                print(f"[!] Warning: failed to build/cache graph '{graph_id}'; "
                      f"skipping. Error: {e}")
            else:
                raise

    def _select_tau_slice(
        self,
        full_tau_grid: List[float],
        full_profile: List[float],
    ) -> Tuple[List[float], List[int]]:
        full_tau_grid = [round(float(x), 8) for x in full_tau_grid]
        tau_to_idx = {round(float(t), 8): i for i, t in enumerate(full_tau_grid)}
        selected, selected_indices = [], []
        for tau in self.tau_values:
            key = round(float(tau), 8)
            if key not in tau_to_idx:
                raise KeyError(
                    f"Requested tau={tau:.4f} is not present in label tau grid. "
                    f"Available: {full_tau_grid}"
                )
            idx = tau_to_idx[key]
            selected.append(float(full_profile[idx]))
            selected_indices.append(idx)
        return selected, selected_indices

    def _build_and_cache(self, graph_id: str, save_path: Path) -> PyGData:
        file_paths = self._find_files_for_graph(graph_id)

        with open(file_paths["label"], "r", encoding="utf-8") as f:
            label_data = json.load(f)

        full_tau_grid = label_data[self.label_tau_key]
        full_profile  = label_data[self.label_profile_key]
        selected_profile, selected_indices = self._select_tau_slice(
            full_tau_grid, full_profile
        )

        with np.load(file_paths["edges"], allow_pickle=True) as loader:
            edges = loader.get("edges", loader.get("data"))

        row, col = edges[:, 0], edges[:, 1]
        edge_index = torch.from_numpy(
            np.array([
                np.concatenate([row, col]),
                np.concatenate([col, row]),
            ])
        ).long()

        features = np.load(file_paths["features"])
        if (
            self.feature_dim is not None
            and self.feature_dim > 0
            and self.feature_dim <= features.shape[1]
        ):
            features = features[:, : self.feature_dim]

        x         = torch.tensor(features, dtype=torch.float32)
        y_profile = torch.tensor(selected_profile, dtype=torch.float32).view(1, -1)

        data = PyGData(
            x=x,
            edge_index=edge_index,
            y=y_profile,
            y_profile=y_profile,
            tau_values=torch.tensor(self.tau_values, dtype=torch.float32),
            tau_indices=torch.tensor(selected_indices, dtype=torch.long),
            num_nodes=int(x.shape[0]),
        )
        torch.save(data, save_path)
        return data

    # --- Dataset interface ---

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx: int):
        graph_id  = self.graph_ids[idx]
        data_path = self.cache_path / f"{graph_id}{self.cache_suffix}"
        try:
            if not data_path.exists():
                self._build_and_cache(graph_id, data_path)
            return torch.load(data_path)
        except Exception as e:
            if self.skip_invalid:
                print(f"[!] Warning: failed to load graph '{graph_id}'; "
                      f"skipping. Error: {e}")
                return None
            raise


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_graph_ids(
    dataset_path: str,
    label_suffix: str = "_profile_label.json",
) -> List[str]:
    """Return all graph IDs matching label_suffix in a directory."""
    if not os.path.isdir(dataset_path):
        return []
    label_files = [f for f in os.listdir(dataset_path) if f.endswith(label_suffix)]
    graph_ids = {f[: -len(label_suffix)] for f in label_files}
    return sorted(graph_ids)


def piss_collate(data_list):
    """Filter None values and return (anchor_batch, positive_batch)."""
    valid_pairs = [pair for pair in data_list if pair is not None]
    if not valid_pairs:
        return None, None
    anchor_list, positive_list = zip(*valid_pairs)
    return (
        Batch.from_data_list(list(anchor_list)),
        Batch.from_data_list(list(positive_list)),
    )


# ---------------------------------------------------------------------------
# Multi-source path loading
# ---------------------------------------------------------------------------

def _load_split_data(
    data_paths: List[str],
    cache_bases: List[str],
    split_name: str,
    tau_values: List[float],
    rebuild_cache: bool,
    feature_dim: Optional[int],
    label_suffix: str,
    label_tau_key: str,
    label_profile_key: str,
) -> List[PyGData]:
    """Load one split from multiple source paths and merge samples."""
    all_data: List[PyGData] = []

    for src_idx, (data_path, cache_base) in enumerate(zip(data_paths, cache_bases)):
        ids = get_graph_ids(data_path, label_suffix=label_suffix)
        if not ids:
            print(
                f"[!] Warning: No '{label_suffix}' files found in "
                f"[{split_name}] source {src_idx}: {data_path}  — skipping."
            )
            continue

        # Use one cache subdirectory per source to avoid graph_id collisions.
        split_cache = str(Path(cache_base) / split_name)

        dataset = CachedProfileGraphDataset(
            dataset_path=data_path,
            graph_ids=ids,
            tau_values=tau_values,
            cache_path=split_cache,
            rebuild_cache=rebuild_cache,
            feature_dim=feature_dim,
            label_suffix=label_suffix,
            label_tau_key=label_tau_key,
            label_profile_key=label_profile_key,
        )

        loaded = [data for data in dataset if data is not None]
        print(
            f"[✓] [{split_name}] source {src_idx} ({Path(data_path).name}): "
            f"{len(loaded)} graphs loaded."
        )
        all_data.extend(loaded)

    return all_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile_dataloaders(
    train_path: Union[str, List[str]],
    val_path: Union[str, List[str]],
    test_path: Union[str, List[str]],
    batch_size: int,
    num_workers: int,
    cache_path: Union[str, List[str]],
    rebuild_cache: bool,
    piss_k: int,
    use_gpu: bool,
    tau_values: List[float],
    feature_dim: Optional[int] = None,
    label_suffix: str = "_profile_label.json",
    label_tau_key: str = "tau_grid_full",
    label_profile_key: str = "collapse_profile_full",
):
    """Build train/validation/test dataloaders with optional multi-source mixing."""
    # --- Normalize inputs to lists ---
    train_paths = _to_path_list(train_path)
    val_paths   = _to_path_list(val_path)
    test_paths  = _to_path_list(test_path)
    cache_list  = _to_path_list(cache_path)

    if not train_paths:
        raise RuntimeError("train_path cannot be empty. Please check the configuration.")

    # --- Align cache directories ---
    # Match train sources and cache roots one-to-one.
    train_caches = _align_cache_paths(cache_list, train_paths, fallback_prefix="cache")

    # val/test reuse train cache roots when source counts match; split
    # subdirectories keep the cached data distinct.
    if len(val_paths) == len(train_paths):
        val_caches = train_caches
    else:
        val_caches = _align_cache_paths(cache_list, val_paths, fallback_prefix="cache")

    if len(test_paths) == len(train_paths):
        test_caches = train_caches
    else:
        test_caches = _align_cache_paths(cache_list, test_paths, fallback_prefix="cache")

    # --- Common loading parameters ---
    shared_kwargs = dict(
        tau_values=tau_values,
        feature_dim=feature_dim,
        label_suffix=label_suffix,
        label_tau_key=label_tau_key,
        label_profile_key=label_profile_key,
    )

    # --- Load each split ---
    print(f"[*] Loading train data from {len(train_paths)} source(s) ...")
    train_data_list = _load_split_data(
        train_paths, train_caches, "train", rebuild_cache=rebuild_cache, **shared_kwargs
    )

    print(f"[*] Loading val data from {len(val_paths)} source(s) ...")
    val_data_list = _load_split_data(
        val_paths, val_caches, "val", rebuild_cache=False, **shared_kwargs
    )

    print(f"[*] Loading test data from {len(test_paths)} source(s) ...")
    test_data_list = _load_split_data(
        test_paths, test_caches, "test", rebuild_cache=False, **shared_kwargs
    )

    if not train_data_list:
        raise RuntimeError(
            f"Training set is empty. Please check train_path: {train_paths}"
        )

    print(
        f"[*] Dataset sizes — "
        f"train: {len(train_data_list)}, "
        f"val: {len(val_data_list)}, "
        f"test: {len(test_data_list)}"
    )

    # --- Build PISSGraphDataset ---
    train_dataset = PISSGraphDataset(train_data_list, k=piss_k)
    val_dataset   = PISSGraphDataset(val_data_list,   k=0)
    test_dataset  = PISSGraphDataset(test_data_list,  k=0)

    # --- Build DataLoader ---
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_gpu}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=piss_collate, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=piss_collate, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=piss_collate, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
