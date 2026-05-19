#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/train_piss.py

Train TCR-GIN with PISS-style consistency regularization using dynamically
generated positive samples.

Function
--------
This script:
1. Loads training, validation, and test datasets for PISS training
2. Trains TCR-GIN with supervised anchor loss and consistency loss
3. For each anchor graph, randomly generates K single-node-deletion positive
   candidates and selects the worst-consistency candidate (max-of-K)
4. Applies early stopping and optional learning-rate scheduling
5. Saves the best model checkpoint for each run
6. Evaluates the best checkpoint on the test set
7. Exports per-run training results as JSON

Usage
-----
Example:
    python train_piss.py \
        --train_path path/to/train \
        --val_path path/to/val \
        --test_path path/to/test \
        --experiment_name my_experiment
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from arguments import parse_args
from data_loader import get_piss_dataloaders
from model.tcr_gin import TCR_GIN, init_weights
from utils import (
    set_seed,
    setup_logger,
    LRScheduler,
    EarlyStopping,
    compute_metrics,
    convert_numpy_types,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ==============================================================================
# Training / validation / testing
# ==============================================================================
def train_epoch_piss(model, dataloader, optimizer, scheduler, device, epoch, args, use_tqdm=True):
    """
    Train one epoch with dynamically generated positive samples.

    For each anchor graph:
    - Randomly sample K distinct nodes
    - Create K positive candidates by deleting one node per candidate
    - In eval + no_grad mode, score all candidates and compute:
          e_i = min((diff)^2, (diff - 1/N0)^2)
    - Select the candidate with the largest e_i (worst-of-K)
    - Run a gradient-enabled forward pass only on the selected positives
      to compute the consistency loss

    Notes
    -----
    - `args.piss_k` is interpreted as the number of candidate positives K per
      anchor graph, not the number of deleted nodes in one sample.
    """
    model.train()
    total_loss = 0.0
    total_anchor_loss = 0.0
    total_consist_loss = 0.0
    all_preds, all_targets = [], []

    if use_tqdm and tqdm:
        desc = f"Epoch {epoch + 1:03d}/{args.epochs}"
        dataloader_iter = tqdm(dataloader, desc=desc, total=len(dataloader))
    else:
        dataloader_iter = dataloader

    eps = 1e-8
    K = max(1, int(getattr(args, "piss_k", 1)))

    for batch_anchor, _ in dataloader_iter:
        if batch_anchor is None:
            continue

        batch_anchor = batch_anchor.to(device)
        optimizer.zero_grad()

        # 1) Anchor prediction and supervised loss
        pred_anchor = model(batch_anchor)
        target = batch_anchor.y
        anchor_loss = F.mse_loss(pred_anchor, target)

        # 2) Count nodes per anchor graph
        with torch.no_grad():
            N0 = scatter(
                torch.ones_like(batch_anchor.batch, dtype=torch.float),
                batch_anchor.batch,
                reduce='sum'
            ).to(device)

        # 3) Construct K single-node-deletion positive candidates per graph on CPU
        anchor_list = batch_anchor.to_data_list()
        pos_candidates_lists = []
        counts = []

        for data in anchor_list:
            data_cpu = data.to('cpu')
            num_nodes = data_cpu.num_nodes

            if num_nodes <= 1:
                pos_candidates_lists.append([])
                counts.append(0)
                continue

            m = min(K, num_nodes)
            perm = torch.randperm(num_nodes)
            chosen = perm[:m]
            candidates = []

            for v in chosen.tolist():
                keep_mask = torch.ones(num_nodes, dtype=torch.bool)
                keep_mask[v] = False
                pos_data = data_cpu.clone().subgraph(keep_mask)
                candidates.append(pos_data)

            pos_candidates_lists.append(candidates)
            counts.append(m)

        total_candidates = sum(counts)

        if total_candidates == 0:
            consist_loss = torch.tensor(0.0, device=device)
            loss = (1 - args.consistency_lambda) * anchor_loss + args.consistency_lambda * consist_loss

            if args.consistency_lambda > 0:
                loss.backward()
            else:
                anchor_loss.backward()

            optimizer.step()

            B = int(batch_anchor.num_graphs)
            total_loss += loss.item() * B
            total_anchor_loss += anchor_loss.item() * B
            total_consist_loss += 0.0

            all_preds.append(pred_anchor.detach())
            all_targets.append(target.detach())

            if use_tqdm and tqdm:
                dataloader_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    anc=f"{anchor_loss.item():.4f}",
                    con=f"{0.0:.4f}"
                )
            continue

        # 4) Score all candidates once in eval + no_grad mode
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            pos_batch_all = Batch.from_data_list(
                [p for plist in pos_candidates_lists for p in plist]
            ).to(device)
            pred_pos_all = model(pos_batch_all)

        if model_was_training:
            model.train()

        # 5) Select the worst candidate for each anchor graph
        worst_indices = []
        valid_anchor_idx = []
        sel_pos_datas = []
        offset = 0

        for i, m in enumerate(counts):
            if m == 0:
                continue

            preds_chunk = pred_pos_all[offset: offset + m]
            offset += m

            scale_i = (N0[i] - 1.0) / (N0[i] + eps)
            scaled_preds = preds_chunk * scale_i
            target_i = target[i].expand_as(scaled_preds)

            diff = target_i - scaled_preds
            err1 = diff.pow(2)
            err2 = (diff - (1.0 / (N0[i] + eps))).pow(2)
            err_min = torch.min(err1, err2).squeeze(1)

            _, max_j = torch.max(err_min, dim=0)
            worst_indices.append(int(max_j.item()))
            valid_anchor_idx.append(i)
            sel_pos_datas.append(pos_candidates_lists[i][int(max_j.item())])

        # 6) Re-run selected positives with gradients enabled
        if len(sel_pos_datas) > 0:
            sel_pos_batch = Batch.from_data_list(sel_pos_datas).to(device)
            pred_pos_sel = model(sel_pos_batch)

            N0_valid = N0[valid_anchor_idx]
            Npos = scatter(
                torch.ones_like(sel_pos_batch.batch, dtype=torch.float),
                sel_pos_batch.batch,
                reduce='sum'
            ).to(device)

            scaling = Npos / (N0_valid + eps)
            scaled_pred_pos = pred_pos_sel * scaling.unsqueeze(1)

            target_valid = target[valid_anchor_idx]
            diff = target_valid - scaled_pred_pos
            err1 = diff.pow(2)
            err2 = (diff - (1.0 / (N0_valid.unsqueeze(1) + eps))).pow(2)
            min_err = torch.min(err1, err2)
            consist_loss = min_err.mean()
            valid_count = len(sel_pos_datas)
        else:
            consist_loss = torch.tensor(0.0, device=device)
            valid_count = 0

        # 7) Backpropagation
        loss = (1 - args.consistency_lambda) * anchor_loss + args.consistency_lambda * consist_loss

        if args.consistency_lambda > 0:
            loss.backward()
        else:
            anchor_loss.backward()

        optimizer.step()

        # 8) Statistics
        B = int(batch_anchor.num_graphs)
        total_loss += loss.item() * B
        total_anchor_loss += anchor_loss.item() * B
        total_consist_loss += consist_loss.item() * valid_count

        all_preds.append(pred_anchor.detach())
        all_targets.append(target.detach())

        if use_tqdm and tqdm:
            dataloader_iter.set_postfix(
                loss=f"{loss.item():.4f}",
                anc=f"{anchor_loss.item():.4f}",
                con=f"{consist_loss.item():.4f}"
            )

    if scheduler:
        scheduler.step(epoch)

    num_samples = len(dataloader.dataset)
    if num_samples == 0:
        return 0, 0, 0, {}

    avg_loss = total_loss / num_samples
    avg_anchor_loss = total_anchor_loss / num_samples
    avg_consist_loss = total_consist_loss / num_samples

    metrics = compute_metrics(torch.cat(all_targets), torch.cat(all_preds))
    return avg_loss, avg_anchor_loss, avg_consist_loss, metrics


def validate(model, dataloader, device):
    """Run validation on a dataloader."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_anchor = batch_data[0]
            else:
                batch_anchor = batch_data

            if batch_anchor is None:
                continue

            batch_anchor = batch_anchor.to(device)
            pred = model(batch_anchor)
            target = batch_anchor.y
            loss = F.mse_loss(pred, target)

            total_loss += loss.item() * batch_anchor.num_graphs
            all_preds.append(pred)
            all_targets.append(target)

    num_samples = len(dataloader.dataset)
    if num_samples == 0:
        return 0, {}

    avg_loss = total_loss / num_samples
    metrics = compute_metrics(torch.cat(all_targets), torch.cat(all_preds))
    return avg_loss, metrics


def test(model, dataloader, device):
    """Run testing with the same logic as validation."""
    return validate(model, dataloader, device)


# ==============================================================================
# Main workflow
# ==============================================================================
def main():
    args = parse_args()

    if args.feature_dim is not None and args.feature_dim > 0:
        args.input_dim = args.feature_dim
    elif not hasattr(args, 'input_dim') or args.input_dim is None:
        raise ValueError("Model input dimension must be specified via 'input_dim' or 'feature_dim'.")

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"All outputs for experiment '{args.experiment_name}' will be saved to: {output_dir}")
    logger.info(f"Configuration:\n{json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_piss_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        piss_k=args.piss_k,
        use_gpu=(device.type == 'cuda'),
        feature_dim=args.feature_dim
    )
    logger.info(
        f"Dataset loading completed "
        f"(training will sample K={args.piss_k} single-node-deletion positives per graph; "
        f"feature_dim={args.input_dim})."
    )

    use_tqdm_final = args.use_tqdm and sys.stdout.isatty()
    if args.use_tqdm and not use_tqdm_final:
        logger.info("A non-interactive environment was detected; tqdm has been disabled for cleaner logs.")

    for run in range(1, args.num_runs + 1):
        logger.info(f"======== Starting run {run}/{args.num_runs} ({args.experiment_name}) ========")
        set_seed(args.seed + run)

        model = TCR_GIN(args).to(device)
        model.apply(init_weights)
        if run == 1:
            logger.info(f"Model architecture: {model}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        scheduler = (
            LRScheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr)
            if args.use_lr_scheduler else None
        )

        model_save_path = os.path.join(model_dir, f'model_run_{run}.pt')
        early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            metric=args.monitor_metric,
            save_path=model_save_path
        )

        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss, anchor_loss, consist_loss, train_metrics = train_epoch_piss(
                model, train_loader, optimizer, scheduler, device, epoch, args, use_tqdm=use_tqdm_final
            )

            if (epoch + 1) % args.eval_steps == 0:
                val_loss, val_metrics = validate(model, val_loader, device)
                epoch_time = time.time() - start_time

                log_interval = 10
                if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                    logger.info(
                        f"Run {run}, Ep {epoch + 1:03d} | "
                        f"L_total: {train_loss:.4f} (anc: {anchor_loss:.4f}, con: {consist_loss:.4f}) | "
                        f"Train MAE: {train_metrics['mae']:.4f} | "
                        f"Val MAE: {val_metrics['mae']:.4f} | "
                        f"Time: {epoch_time:.2f}s"
                    )

                early_stopping(epoch, val_metrics, model)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered. Best epoch: {early_stopping.best_epoch + 1}")
                    break

        logger.info(f"Loading best model from epoch {early_stopping.best_epoch + 1}")
        model.load_state_dict(torch.load(model_save_path))
        test_loss, test_metrics = test(model, test_loader, device)

        logger.info(
            f"Test results - "
            f"Loss: {test_loss:.4f}, "
            f"RMSE: {test_metrics['rmse']:.4f}, "
            f"MAE: {test_metrics['mae']:.4f}, "
            f"R²: {test_metrics['r2']:.4f}"
        )

        results = {
            'run': run,
            'best_epoch': early_stopping.best_epoch + 1,
            'best_val_metrics': early_stopping.best_metrics,
            'test_metrics': test_metrics,
        }

        with open(os.path.join(output_dir, f'run_{run}_results.json'), 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=4)

    if os.path.exists(args.cache_path) and args.rebuild_cache:
        shutil.rmtree(args.cache_path)
        logger.info(f"Cache directory removed: {args.cache_path}")


if __name__ == "__main__":
    main()
