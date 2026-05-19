#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Argument parsing for collapse-profile experiments.

Function
--------
This module parses command-line arguments, loads optional YAML defaults,
normalizes single-source and multi-source dataset paths, validates core paths,
and prepares tau values for collapse-profile training and evaluation.

Inputs
------
- `--config`: optional YAML configuration file.
- Train/validation/test/cache paths provided as single strings,
  comma-separated strings, or YAML lists.

Outputs
-------
- An argparse namespace with normalized path lists and validated settings.

Usage
-----
YAML list:
    train_path:
      - /path/to/dataset_a/train
      - /path/to/dataset_b/train

Command-line comma-separated paths:
    python experiments/collapse_profile/train_profile.py \
        --train_path /path/a/train,/path/b/train

Backward-compatible single path:
    python experiments/collapse_profile/train_profile.py \
        --train_path /path/a/train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


# ---------------------------------------------------------------------------
# YAML configuration loading
# ---------------------------------------------------------------------------

def load_and_flatten_config(config_path: str) -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    flat_config: Dict[str, Any] = {}

    if "base_config" in config_data and isinstance(config_data["base_config"], dict):
        for section, params in config_data["base_config"].items():
            if isinstance(params, dict):
                flat_config.update(params)
            else:
                flat_config[section] = params

    ignore_keys = {"base_config", "experiments", "instances", "templates"}
    for key, value in config_data.items():
        if key not in ignore_keys:
            flat_config[key] = value

    tau_cfg = flat_config.pop("tau", None)
    if isinstance(tau_cfg, dict) and "values" in tau_cfg:
        flat_config["tau_values"] = tau_cfg["values"]

    return flat_config


# ---------------------------------------------------------------------------
# tau-value normalization
# ---------------------------------------------------------------------------

def normalize_tau_values(tau_values) -> List[float]:
    if tau_values is None:
        raise ValueError("tau_values must be provided.")
    if isinstance(tau_values, str):
        vals = [float(x.strip()) for x in tau_values.split(",") if x.strip()]
    elif isinstance(tau_values, (list, tuple)):
        vals = [float(x) for x in tau_values]
    else:
        raise TypeError(f"Unsupported tau_values type: {type(tau_values)}")
    vals = sorted(vals)
    if not vals:
        raise ValueError("tau_values is empty.")
    return vals


# ---------------------------------------------------------------------------
# Path-argument normalization
# ---------------------------------------------------------------------------

def _to_path_list(val: Union[None, str, list]) -> List[str]:
    """Convert any supported path value into a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(p).strip() for p in val if str(p).strip()]
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return parts
    return [str(val).strip()]


def _normalize_path_args(args) -> None:
    """Normalize path fields in place and align cache_path length."""

    # --- Dataset paths ---
    for attr in ("train_path", "val_path", "test_path"):
        setattr(args, attr, _to_path_list(getattr(args, attr, None)))

    # --- cache_path normalization aligned with train_path ---
    n_train = len(args.train_path)

    cache_raw = getattr(args, "cache_path", None)
    cache_list = _to_path_list(cache_raw)

    if len(cache_list) == 0:
        # No cache configured: create one cache subdirectory per training path.
        cache_list = [
            os.path.join("cache", f"split_{i}") for i in range(max(n_train, 1))
        ]
    elif len(cache_list) == 1 and n_train > 1:
        # One cache root is broadcast into per-source cache subdirectories.
        base = cache_list[0]
        cache_list = [os.path.join(base, f"split_{i}") for i in range(n_train)]
    elif len(cache_list) != n_train and n_train > 0:
        raise ValueError(
            f"cache_path length ({len(cache_list)}) does not match train_path length ({n_train})."
            f"\n  cache_path : {cache_list}"
            f"\n  train_path : {args.train_path}"
        )

    args.cache_path = cache_list


# ---------------------------------------------------------------------------
# Main parsing entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Collapse-profile TCR-GIN training")

    # --- Core settings ---
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--use_tqdm", type=lambda x: str(x).lower() == "true")

    # --- Dataset paths: keep type=str for CLI compatibility; normalization handles lists. ---
    parser.add_argument(
        "--train_path", type=str,
        help="Training-set path. Use comma-separated paths or a YAML list for multiple sources.",
    )
    parser.add_argument(
        "--val_path", type=str,
        help="Validation-set path. Use comma-separated paths or a YAML list for multiple sources.",
    )
    parser.add_argument(
        "--test_path", type=str,
        help="Test-set path. Use comma-separated paths or a YAML list for multiple sources.",
    )
    parser.add_argument(
        "--cache_path", type=str,
           help="Cache path. Use comma-separated paths or a YAML list; if one root is provided, "
               "split_0, split_1, ... subdirectories are generated automatically.",
    )
    parser.add_argument("--rebuild_cache", type=lambda x: str(x).lower() == "true")
    parser.add_argument("--label_suffix", type=str)
    parser.add_argument("--label_tau_key", type=str)
    parser.add_argument("--label_profile_key", type=str)

    # --- Model architecture ---
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--jk_type", type=str)
    parser.add_argument("--use_virtual_node", type=lambda x: str(x).lower() == "true")
    parser.add_argument("--use_residual", type=lambda x: str(x).lower() == "true")
    parser.add_argument("--activation_fn", type=str)
    parser.add_argument("--tau_values", type=str)

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2_reg", type=float)
    parser.add_argument("--label_scale", type=float, default=100.0)
    parser.add_argument("--use_lr_scheduler", type=lambda x: str(x).lower() == "true")
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--monitor_metric", type=str)
    parser.add_argument("--early_stop_patience", type=int)

    # --- Consistency / monotonicity ---
    parser.add_argument("--consistency_lambda", type=float)
    parser.add_argument("--piss_k", type=int)
    parser.add_argument("--monotonic_lambda", type=float)

    # --- Outputs ---
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_path", type=str)

    # ------------------------------------------------------------------
    # Two-stage parsing: read config defaults first, then apply CLI overrides.
    # ------------------------------------------------------------------
    temp_args, _ = parser.parse_known_args()
    defaults: Dict[str, Any] = {}
    if temp_args.config:
        defaults = load_and_flatten_config(temp_args.config)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Path normalization must run after set_defaults.
    # ------------------------------------------------------------------
    _normalize_path_args(args)

    # Basic path validation
    for attr in ("train_path", "val_path", "test_path"):
        paths: List[str] = getattr(args, attr)
        if not paths:
            raise ValueError(f"{attr} cannot be empty. Set it in the config file or command line.")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{attr} path does not exist: {p}")

    # ------------------------------------------------------------------
    # input_dim / feature_dim alignment
    # ------------------------------------------------------------------
    if args.feature_dim is not None and args.feature_dim > 0:
        args.input_dim = args.feature_dim
    elif not hasattr(args, "input_dim") or args.input_dim is None:
        raise ValueError("input_dim or feature_dim must be provided.")

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------
    if args.experiment_name is None:
        args.experiment_name = (
            Path(args.config).stem if args.config else "collapse_profile_run"
        )

    # ------------------------------------------------------------------
    # tau-value normalization
    # ------------------------------------------------------------------
    args.tau_values = normalize_tau_values(args.tau_values)

    # ------------------------------------------------------------------
    # Default label fields
    # ------------------------------------------------------------------
    if args.label_suffix is None:
        args.label_suffix = "_profile_label.json"
    if args.label_tau_key is None:
        args.label_tau_key = "tau_grid_full"
    if args.label_profile_key is None:
        args.label_profile_key = "collapse_profile_full"

    return args
