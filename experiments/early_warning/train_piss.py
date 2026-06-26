#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/early_warning/train_piss.py

Training entry for the early-warning PISS/TI-GIN pipeline.

Example:
    python experiments/early_warning/train_piss.py --config path/to/config.yaml
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
import time
import json
import shutil
from pathlib import Path

# ==============================================================================
# Path import fix
# ==============================================================================
# Current file:
#   TCR-GIN/experiments/early_warning/train_piss.py
# current_dir       -> TCR-GIN/experiments/early_warning
# current_dir.parent -> TCR-GIN/experiments
# current_dir.parent.parent -> TCR-GIN
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Batch
from torch.cuda.amp import autocast, GradScaler

from arguments import parse_args
from data_loader import get_piss_dataloaders
from model.tcr_gin import TCR_GIN, init_weights
from utils import set_seed, setup_logger, LRScheduler, EarlyStopping, compute_metrics, convert_numpy_types

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ==============================================================================
# Helper: Compute Label Statistics (Modified to accept args)
# ==============================================================================
def compute_label_stats(dataloader, args):
    """
    Sets the label scaling factor based on args.label_scale_factor.
    """
    scale_factor = float(args.label_scale_factor)

    print(f"[*] Using Configured Label Scale Factor: {scale_factor}")

    # Optional: Log statistics for reference
    all_labels = []
    for batch_data in dataloader:
        if isinstance(batch_data, (list, tuple)):
            batch_anchor = batch_data[0]
        else:
            batch_anchor = batch_data
        if batch_anchor is not None:
            all_labels.append(batch_anchor.y)

    if all_labels:
        avg_abs = torch.cat(all_labels).view(-1).abs().mean().item()
        print(f"    Original Data Mean Abs: {avg_abs:.6e}")
        print(f"    Scaled Data Mean Abs:   {avg_abs * scale_factor:.6e}")

    # Set mean=0, std=1 for compatibility, store real factor in 'scale_factor'
    return {'mean': 0.0, 'std': 1.0, 'scale_factor': scale_factor}


# ==============================================================================
# Training Function (Modified for Scaled Consistency)
# ==============================================================================
def train_epoch_piss(model, dataloader, optimizer, scheduler, device, epoch, args, label_stats, scaler, use_tqdm=True):
    """
    Train for one epoch with AMP and SCALED Consistency Loss.
    """
    model.train()
    total_loss, total_anchor_loss, total_consist_loss = 0.0, 0.0, 0.0
    all_preds_real, all_targets_real = [], []

    scale_factor = label_stats['scale_factor']
    eps = 1e-8

    if use_tqdm and tqdm:
        desc = f"Epoch {epoch+1:03d}/{args.epochs}"
        dataloader_iter = tqdm(dataloader, desc=desc, total=len(dataloader))
    else:
        dataloader_iter = dataloader

    K = max(1, int(getattr(args, "piss_k", 1)))

    for batch_anchor, _ in dataloader_iter:
        if batch_anchor is None:
            continue

        batch_anchor = batch_anchor.to(device)
        optimizer.zero_grad()

        # --- [AMP Context Start] ---
        with autocast(enabled=(device.type == 'cuda')):
            # --- [Scaling Step] ---
            target_real = batch_anchor.y  # [B, 1]
            target_norm = target_real * scale_factor

            pred_anchor_norm = model(batch_anchor)
            anchor_loss = F.mse_loss(pred_anchor_norm, target_norm)

            # 3. Prepare Consistency Loss
            with torch.no_grad():
                N0 = scatter(
                    torch.ones_like(batch_anchor.batch, dtype=torch.float),
                    batch_anchor.batch, reduce='sum'
                ).to(device)

            # Construct K positive candidates
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

            # Handle case with no valid positive candidates
            if total_candidates == 0:
                loss = (1 - args.consistency_lambda) * anchor_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Restore true values for logging and metrics: pred / S
                pred_real = pred_anchor_norm.detach() / scale_factor
                all_preds_real.append(pred_real.float())
                all_targets_real.append(target_real.detach().float())

                total_loss += loss.item() * batch_anchor.num_graphs
                total_anchor_loss += anchor_loss.item() * batch_anchor.num_graphs
                continue

            # 4) Evaluate candidates
            model_was_training = model.training
            model.eval()
            with torch.no_grad():
                pos_batch_all = Batch.from_data_list([p for plist in pos_candidates_lists for p in plist]).to(device)
                pred_pos_all_norm = model(pos_batch_all)

            if model_was_training:
                model.train()

            worst_indices = []
            valid_anchor_idx = []
            sel_pos_datas = []
            offset = 0

            expected_change_norm = (1.0 / (N0 + eps)) * scale_factor

            for i, m in enumerate(counts):
                if m == 0:
                    continue
                preds_chunk_norm = pred_pos_all_norm[offset: offset + m]
                offset += m

                scale_i = (N0[i] - 1.0) / (N0[i] + eps)
                scaled_preds_norm = preds_chunk_norm * scale_i

                target_i_norm = target_norm[i].expand_as(scaled_preds_norm)
                diff_norm = target_i_norm - scaled_preds_norm

                err1 = diff_norm.pow(2)
                err2 = (diff_norm - expected_change_norm[i]).pow(2)

                err_min = torch.min(err1, err2).squeeze(1)

                max_val, max_j = torch.max(err_min, dim=0)
                worst_indices.append(int(max_j.item()))
                valid_anchor_idx.append(i)
                sel_pos_datas.append(pos_candidates_lists[i][int(max_j.item())])

            # 5) Calculate Consistency Loss
            if len(sel_pos_datas) > 0:
                sel_pos_batch = Batch.from_data_list(sel_pos_datas).to(device)
                pred_pos_sel_norm = model(sel_pos_batch)

                N0_valid = N0[valid_anchor_idx]
                Npos = scatter(
                    torch.ones_like(sel_pos_batch.batch, dtype=torch.float),
                    sel_pos_batch.batch, reduce='sum'
                ).to(device)

                scaling = (Npos) / (N0_valid + eps)
                scaled_pred_pos_norm = pred_pos_sel_norm * scaling.unsqueeze(1)

                target_valid_norm = target_norm[valid_anchor_idx]
                diff_norm = target_valid_norm - scaled_pred_pos_norm

                err1 = diff_norm.pow(2)
                expected_change_valid_norm = (1.0 / (N0_valid.unsqueeze(1) + eps)) * scale_factor
                err2 = (diff_norm - expected_change_valid_norm).pow(2)

                min_err = torch.min(err1, err2)
                consist_loss = min_err.mean()
                valid_count = len(sel_pos_datas)
            else:
                consist_loss = torch.tensor(0.0, device=device)
                valid_count = 0

            # 6) Combine Losses
            loss = (1 - args.consistency_lambda) * anchor_loss + args.consistency_lambda * consist_loss

        # --- [AMP Backward] ---
        if args.consistency_lambda > 0:
            scaler.scale(loss).backward()
        else:
            scaler.scale(anchor_loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # 7) Statistics
        B = int(batch_anchor.num_graphs)
        total_loss += loss.item() * B
        total_anchor_loss += anchor_loss.item() * B
        total_consist_loss += consist_loss.item() * valid_count

        # restore: pred / S
        pred_anchor_real = pred_anchor_norm.detach() / scale_factor
        all_preds_real.append(pred_anchor_real.float())
        all_targets_real.append(target_real.detach().float())

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

    metrics = compute_metrics(torch.cat(all_targets_real), torch.cat(all_preds_real))
    return avg_loss, avg_anchor_loss, avg_consist_loss, metrics


# ==============================================================================
# Validation Function
# ==============================================================================
def validate(model, dataloader, device, label_stats):
    model.eval()
    total_loss = 0
    all_preds_real, all_targets_real = [], []

    # [Modified] get scaling factor from label_stats
    scale_factor = label_stats['scale_factor']

    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_anchor = batch_data[0]
            else:
                batch_anchor = batch_data

            if batch_anchor is None:
                continue
            batch_anchor = batch_anchor.to(device)

            # Predict (Scaled)
            pred_norm = model(batch_anchor)

            # Target (Scaled)
            target_real = batch_anchor.y
            target_norm = target_real * scale_factor

            loss = F.mse_loss(pred_norm, target_norm)
            total_loss += loss.item() * batch_anchor.num_graphs

            # Unscale: pred / S
            pred_real = pred_norm / scale_factor
            all_preds_real.append(pred_real.float())
            all_targets_real.append(target_real.float())

    num_samples = len(dataloader.dataset)
    if num_samples == 0:
        return 0, {}

    avg_loss = total_loss / num_samples
    metrics = compute_metrics(torch.cat(all_targets_real), torch.cat(all_preds_real))
    return avg_loss, metrics


def test(model, dataloader, device, label_stats):
    return validate(model, dataloader, device, label_stats)


# ==============================================================================
# Main Function
# ==============================================================================
def main():
    args = parse_args()

    if args.feature_dim is not None and args.feature_dim > 0:
        args.input_dim = args.feature_dim
    elif not hasattr(args, 'input_dim') or args.input_dim is None:
        raise ValueError("Must specify 'input_dim' or 'feature_dim'!")

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Experiment '{args.experiment_name}' outputs: {output_dir}")
    logger.info(f"Config: \n{json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Load Data
    train_loader, val_loader, test_loader = get_piss_dataloaders(
        train_path=args.train_path, val_path=args.val_path, test_path=args.test_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cache_path=args.cache_path, rebuild_cache=args.rebuild_cache,
        piss_k=args.piss_k, use_gpu=(device.type == 'cuda'),
        feature_dim=args.feature_dim
    )
    logger.info(f"Data loaded. Train samples: {len(train_loader.dataset)}")

    # [Modified] Compute Stats (pass args)
    label_stats = compute_label_stats(train_loader, args)

    with open(os.path.join(output_dir, 'label_stats.json'), 'w') as f:
        json.dump(label_stats, f)
    logger.info("Label stats saved to label_stats.json")

    use_tqdm_final = args.use_tqdm and sys.stdout.isatty()
    if args.use_tqdm and not use_tqdm_final:
        logger.info("Non-interactive env detected, TQDM disabled.")

    for run in range(1, args.num_runs + 1):
        logger.info(f"======== Run {run}/{args.num_runs} ({args.experiment_name}) ========")
        set_seed(args.seed + run)

        model = TCR_GIN(args).to(device)
        model.apply(init_weights)
        if run == 1:
            logger.info(f"Model Structure: {model}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        scheduler = LRScheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr) if args.use_lr_scheduler else None

        model_save_path = os.path.join(model_dir, f'model_run_{run}.pt')
        early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            metric=args.monitor_metric,
            save_path=model_save_path
        )

        for epoch in range(args.epochs):
            start_time = time.time()

            # Train
            train_loss, anchor_loss, consist_loss, train_metrics = train_epoch_piss(
                model, train_loader, optimizer, scheduler, device, epoch, args,
                label_stats=label_stats, scaler=scaler, use_tqdm=use_tqdm_final
            )

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if (epoch + 1) % args.eval_steps == 0:
                val_loss, val_metrics = validate(model, val_loader, device, label_stats)
                epoch_time = time.time() - start_time

                log_interval = 10
                if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                    logger.info(
                        f"Run {run}, Ep {epoch+1:03d} | "
                        f"L_tot: {train_loss:.4f} (anc: {anchor_loss:.4f}, con: {consist_loss:.4f}) | "
                        f"Tr MAE: {train_metrics['mae']:.4f} | "
                        f"Val MAE: {val_metrics['mae']:.4f} | "
                        f"Time: {epoch_time:.2f}s"
                    )

                early_stopping(epoch, val_metrics, model)
                if early_stopping.early_stop:
                    logger.info(f"Early Stopping! Best Epoch: {early_stopping.best_epoch + 1}")
                    break

        logger.info(f"Loading best model (Epoch {early_stopping.best_epoch + 1})")
        model.load_state_dict(torch.load(model_save_path))

        test_loss, test_metrics = test(model, test_loader, device, label_stats)

        logger.info(
            f"Test Results - RMSE: {test_metrics['rmse']:.4f}, "
            f"MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}"
        )

        results = {
            'run': run,
            'best_epoch': early_stopping.best_epoch + 1,
            'best_val_metrics': early_stopping.best_metrics,
            'test_metrics': test_metrics,
            'label_stats': label_stats
        }
        with open(os.path.join(output_dir, f'run_{run}_results.json'), 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=4)

    if os.path.exists(args.cache_path) and args.rebuild_cache:
        shutil.rmtree(args.cache_path)
        logger.info(f"Cache cleared: {args.cache_path}")


if __name__ == "__main__":
    main()
