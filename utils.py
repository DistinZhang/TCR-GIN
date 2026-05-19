#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/utils.py

Utility helpers for training, evaluation, logging, scheduling, and metric
post-processing.

Function
--------
This module provides:
1. Random-seed initialization for reproducibility
2. Logger setup for console and file output
3. Early stopping based on a monitored metric
4. Learning-rate scheduling with warmup + cosine decay
5. Regression metric computation
6. Recursive conversion of NumPy-specific types to standard Python types

Usage
-----
Example:
    from utils import (
        set_seed,
        setup_logger,
        EarlyStopping,
        LRScheduler,
        compute_metrics,
        convert_numpy_types,
    )
"""

import os
import random
import logging
from typing import Dict

import numpy as np
import torch


# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Logging
# ==============================================================================
def setup_logger(log_dir: str):
    """Create a logger with console output and optional file output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'train.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# Early stopping
# ==============================================================================
class EarlyStopping:
    """Stop training when a monitored metric no longer improves."""

    def __init__(self, patience=10, delta=0, mode='min', metric='mae', save_path=None):
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
        if not isinstance(metrics, dict):
            raise TypeError(
                f"EarlyStopping expects a metric dictionary, but received {type(metrics)}"
            )

        score = metrics.get(self.metric)
        if score is None:
            raise ValueError(
                f"Monitored metric '{self.metric}' was not found in the metric dictionary: "
                f"{metrics.keys()}"
            )

        compare_score = -score if self.mode == 'min' else score
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


# ==============================================================================
# Learning-rate scheduler
# ==============================================================================
class LRScheduler:
    """Warmup followed by cosine learning-rate decay."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr_mult = float(epoch) / max(1, self.warmup_epochs)
        else:
            progress = float(epoch - self.warmup_epochs) / float(
                self.max_epochs - self.warmup_epochs
            )
            lr_mult = 0.5 * (1 + np.cos(np.pi * progress))

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.min_lr, lr_mult * self.base_lrs[i])


# ==============================================================================
# Metric computation
# ==============================================================================
def compute_metrics(y_true, y_pred):
    """Compute regression metrics from prediction and target tensors."""
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()

    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_np - y_pred_np))

    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true_np, y_pred_np)
    except ImportError:
        r2 = 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


# ==============================================================================
# NumPy-to-Python conversion
# ==============================================================================
def convert_numpy_types(obj):
    """Recursively convert NumPy-specific objects into Python-native types."""
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
