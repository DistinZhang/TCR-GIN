#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility helpers for collapse-profile training."""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import numpy as np
import torch



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def setup_logger(log_dir: str):
    logger = logging.getLogger("collapse_profile")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode="min", metric="profile_mae", save_path=None):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.metric = metric
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_metrics = {}

    def __call__(self, epoch, metrics, model=None):
        score = metrics.get(self.metric)
        if score is None:
            raise ValueError(f"Monitored metric '{self.metric}' not found. Available: {sorted(metrics.keys())}")

        compare_score = -score if self.mode == "min" else score
        is_best = False
        if self.best_score is None:
            self.best_score = compare_score
            self.best_epoch = epoch
            self.best_metrics = metrics
            is_best = True
        elif compare_score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = compare_score
            self.best_epoch = epoch
            self.best_metrics = metrics
            self.counter = 0
            is_best = True

        if is_best and model is not None and self.save_path:
            torch.save(model.state_dict(), self.save_path)
        return is_best


class LRScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr_mult = float(epoch) / max(1, self.warmup_epochs)
        else:
            progress = float(epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            lr_mult = 0.5 * (1 + np.cos(np.pi * progress))

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = max(self.min_lr, lr_mult * self.base_lrs[i])



def monotonicity_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] <= 1:
        return pred.new_tensor(0.0)
    violation = torch.relu(pred[:, 1:] - pred[:, :-1])
    return (violation ** 2).mean()



def compute_profile_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, tau_values: List[float]) -> Dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    diff = y_true_np - y_pred_np

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # Same as MAE/RMSE here but semantically explicit for vector targets.
    metrics = {
        "profile_mse": mse,
        "profile_rmse": rmse,
        "profile_mae": mae,
    }

    tau_mae = np.mean(np.abs(diff), axis=0)
    tau_rmse = np.sqrt(np.mean(diff ** 2, axis=0))

    for tau, tau_err in zip(tau_values, tau_mae):
        metrics[f"tau_mae_{tau:.2f}"] = float(tau_err)
    for tau, tau_err in zip(tau_values, tau_rmse):
        metrics[f"tau_rmse_{tau:.2f}"] = float(tau_err)

    # Profile correlation averaged over graphs.
    corr_vals = []
    for yt, yp in zip(y_true_np, y_pred_np):
        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            continue
        corr_vals.append(np.corrcoef(yt, yp)[0, 1])
    metrics["profile_corr"] = float(np.mean(corr_vals)) if corr_vals else 0.0

    mono_viol = np.maximum(y_pred_np[:, 1:] - y_pred_np[:, :-1], 0.0)
    metrics["monotonicity_violation_mean"] = float(np.mean(mono_viol)) if mono_viol.size else 0.0
    return metrics



def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj
