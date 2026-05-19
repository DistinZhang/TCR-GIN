#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train TCR-GIN for collapse-profile prediction.

Key differences from the scalar version:
1. y is now a profile vector over explicitly listed tau values
2. PISS consistency is vectorized over tau and averaged across the profile

Additional scaling behavior:
3. Targets can be scaled during training/eval loss computation via args.label_scale
4. Metrics are always computed on the original scale

PISS consistency uses the same target-based formula as the scalar TCR-GIN:
    scaling = Npos / N0   (i.e. (N-1)/N)
    diff = target_scaled - pred_pos_scaled * scaling
    err = min(diff², (diff - label_scale/N0)²)

Loss combination (three-part, coefficients sum to 1):
    α = 1 - λ_con - λ_mono
    loss = α * supervised_loss + λ_con * consistency_loss + λ_mono * monotonic_loss

Monotonicity prior:
    As tau increases the collapse distance should decrease.
    We penalise any *increase* between consecutive tau slots:
        violations = ReLU(pred[:, t+1] - pred[:, t])   for t = 0 .. T-2
        mono_loss  = mean(violations²)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

THIS_DIR = Path(__file__).resolve().parent
PROFILE_ROOT = THIS_DIR.parent
PROJECT_ROOT = PROFILE_ROOT.parent.parent

for p in [str(THIS_DIR), str(PROFILE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

from arguments_profile import parse_args
from data_loader_profile import get_profile_dataloaders
from model.tcr_gin_profile import TCR_GIN_Profile, init_weights
from utils_profile import (
    EarlyStopping,
    LRScheduler,
    compute_profile_metrics,
    convert_numpy_types,
    set_seed,
    setup_logger,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ─────────────────────────────────────────────────────────────────────────────
#  Monotonicity loss
# ─────────────────────────────────────────────────────────────────────────────

def monotonicity_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalise violations of the monotone-decreasing prior over tau.

    Args:
        pred: [B, T] predictions ordered by *increasing* tau.
              Under the prior, pred[:, t] >= pred[:, t+1].

    Returns:
        Scalar loss (mean of squared positive differences).
    """
    # diff[:, t] = pred[:, t+1] - pred[:, t];  should be <= 0
    diff = pred[:, 1:] - pred[:, :-1]          # [B, T-1]
    violations = F.relu(diff)                   # only positive (violating) part
    return violations.pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_profile(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    args,
    use_tqdm=True,
):
    """Train one epoch.

    Training losses are computed in scaled label space:
        scaled_target = target * label_scale

    Returned metrics are computed in the original label space.

    Loss combination (three-part, coefficients sum to 1):
        α = 1 - λ_con - λ_mono
        loss = α * supervised_loss + λ_con * consistency_loss + λ_mono * monotonic_loss
    """
    model.train()
    total_loss = 0.0
    total_supervised = 0.0
    total_consistency = 0.0
    total_monotonic = 0.0
    all_preds_unscaled, all_targets_unscaled = [], []

    dataloader_iter = dataloader
    if use_tqdm and tqdm:
        dataloader_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1:03d}/{args.epochs}",
            total=len(dataloader),
        )

    eps = 1e-8
    K = max(1, int(getattr(args, "piss_k", 1)))
    label_scale = float(getattr(args, "label_scale", 1.0))
    consistency_lambda = float(args.consistency_lambda)
    monotonic_lambda = float(getattr(args, "monotonic_lambda", 0.0))

    # ── Validate that coefficients are sane ───────────────────────────────
    assert consistency_lambda + monotonic_lambda <= 1.0 + 1e-7, (
        f"consistency_lambda ({consistency_lambda}) + monotonic_lambda ({monotonic_lambda}) > 1"
    )
    supervised_lambda = 1.0 - consistency_lambda - monotonic_lambda

    for batch_data in dataloader_iter:
        batch_anchor = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        if batch_anchor is None:
            continue

        batch_anchor = batch_anchor.to(device)
        optimizer.zero_grad()

        # ── Forward (anchor, train mode) ──────────────────────────────────
        pred_anchor_scaled = model(batch_anchor)          # [B, T], scaled space

        target = batch_anchor.y                           # [B, T], original space
        target_scaled = target * label_scale              # [B, T], scaled space

        supervised_loss = F.mse_loss(pred_anchor_scaled, target_scaled)

        # ── Monotonicity loss (in scaled space) ──────────────────────────
        if monotonic_lambda > 0.0:
            mono_loss = monotonicity_loss(pred_anchor_scaled)
        else:
            mono_loss = torch.tensor(0.0, device=device)

        # ── PISS consistency (target-based, matching scalar version) ──────
        if consistency_lambda > 0.0:
            with torch.no_grad():
                N0 = scatter(
                    torch.ones_like(batch_anchor.batch, dtype=torch.float),
                    batch_anchor.batch,
                    reduce="sum",
                ).to(device)                              # [B]

            anchor_list = batch_anchor.to_data_list()
            pos_candidates_lists = []
            counts = []

            for data in anchor_list:
                data_cpu = data.to("cpu")
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
                consistency_loss = torch.tensor(0.0, device=device)
            else:
                # ── Candidate selection: eval + no_grad to protect BN stats ──
                model.eval()
                with torch.no_grad():
                    pos_batch_all = Batch.from_data_list(
                        [p for plist in pos_candidates_lists for p in plist]
                    ).to(device)
                    pred_pos_all_scaled = model(pos_batch_all)   # [total_cand, T]

                    # ── Target-based hardest-positive selection ───────────
                    sel_pos_datas = []
                    valid_anchor_idx = []
                    offset = 0

                    for i, m in enumerate(counts):
                        if m == 0:
                            continue

                        preds_chunk_scaled = pred_pos_all_scaled[offset: offset + m]
                        offset += m

                        scale_i = (N0[i] - 1.0) / (N0[i] + eps)
                        scaled_preds = preds_chunk_scaled * scale_i

                        target_i_scaled = target_scaled[i].unsqueeze(0).expand_as(scaled_preds)
                        diff = target_i_scaled - scaled_preds

                        tol_i = label_scale / (N0[i] + eps)
                        err1 = diff.pow(2)
                        err2 = (diff - tol_i).pow(2)
                        err_min = torch.minimum(err1, err2).mean(dim=1)

                        _, max_j = torch.max(err_min, dim=0)
                        sel_pos_datas.append(pos_candidates_lists[i][int(max_j.item())])
                        valid_anchor_idx.append(i)

                # ── Restore model to train mode before gradient forward ───
                model.train()

                if sel_pos_datas:
                    sel_pos_batch = Batch.from_data_list(sel_pos_datas).to(device)

                    pred_pos_sel_scaled = model(sel_pos_batch)

                    Npos = scatter(
                        torch.ones_like(sel_pos_batch.batch, dtype=torch.float),
                        sel_pos_batch.batch,
                        reduce="sum",
                    ).to(device)

                    N0_valid = N0[valid_anchor_idx]

                    scaling = Npos / (N0_valid + eps)
                    scaled_pred_pos = pred_pos_sel_scaled * scaling.unsqueeze(1)

                    target_valid_scaled = target_scaled[valid_anchor_idx]
                    diff = target_valid_scaled - scaled_pred_pos

                    tol = label_scale / (N0_valid.unsqueeze(1) + eps)
                    err1 = diff.pow(2)
                    err2 = (diff - tol).pow(2)
                    consistency_loss = torch.minimum(err1, err2).mean()
                else:
                    consistency_loss = torch.tensor(0.0, device=device)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # ── Total loss: three-part convex combination (coefficients sum to 1) ─
        loss = (
            supervised_lambda * supervised_loss
            + consistency_lambda * consistency_loss
            + monotonic_lambda * mono_loss
        )

        loss.backward()
        optimizer.step()

        B = int(batch_anchor.num_graphs)
        total_loss += loss.item() * B
        total_supervised += supervised_loss.item() * B
        total_consistency += consistency_loss.item() * B
        total_monotonic += mono_loss.item() * B

        # Convert back to original scale for metric accumulation
        pred_anchor = pred_anchor_scaled / label_scale
        all_preds_unscaled.append(pred_anchor.detach())
        all_targets_unscaled.append(target.detach())

    if scheduler:
        scheduler.step(epoch)

    num_samples = len(dataloader.dataset)
    if num_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, {}

    avg_loss = total_loss / num_samples
    avg_supervised = total_supervised / num_samples
    avg_consistency = total_consistency / num_samples
    avg_monotonic = total_monotonic / num_samples

    metrics = compute_profile_metrics(
        torch.cat(all_targets_unscaled),
        torch.cat(all_preds_unscaled),
        args.tau_values,
    )
    return avg_loss, avg_supervised, avg_consistency, avg_monotonic, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_profile(model, dataloader, device, tau_values, label_scale=1.0):
    """Evaluate model.

    Loss is computed in scaled space; metrics are in original space.
    """
    model.eval()
    total_loss = 0.0
    all_preds_unscaled, all_targets_unscaled = [], []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_anchor = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            if batch_anchor is None:
                continue

            batch_anchor = batch_anchor.to(device)
            pred_scaled = model(batch_anchor)
            target = batch_anchor.y
            target_scaled = target * label_scale

            loss = F.mse_loss(pred_scaled, target_scaled)
            total_loss += loss.item() * batch_anchor.num_graphs

            pred = pred_scaled / label_scale
            all_preds_unscaled.append(pred)
            all_targets_unscaled.append(target)

    num_samples = len(dataloader.dataset)
    if num_samples == 0:
        return 0.0, {}

    avg_loss = total_loss / num_samples
    metrics = compute_profile_metrics(
        torch.cat(all_targets_unscaled),
        torch.cat(all_preds_unscaled),
        tau_values,
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not hasattr(args, "label_scale"):
        args.label_scale = 1.0

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Experiment output directory: {output_dir}")
    logger.info(
        f"Configuration:\n{json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True)}"
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Active tau grid: {args.tau_values}")
    logger.info(f"Label scale for training/eval loss: {args.label_scale}")

    # ── Log the three-part loss weighting ─────────────────────────────────
    _cl = float(args.consistency_lambda)
    _ml = float(getattr(args, "monotonic_lambda", 0.0))
    _sl = 1.0 - _cl - _ml
    logger.info(
        f"Loss weights — supervised: {_sl:.2f}, consistency: {_cl:.2f}, monotonic: {_ml:.2f}"
    )

    train_loader, val_loader, test_loader = get_profile_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        piss_k=args.piss_k,
        use_gpu=(device.type == "cuda"),
        tau_values=args.tau_values,
        feature_dim=args.feature_dim,
        label_suffix=args.label_suffix,
        label_tau_key=args.label_tau_key,
        label_profile_key=args.label_profile_key,
    )

    use_tqdm_final = args.use_tqdm and sys.stdout.isatty()
    if args.use_tqdm and not use_tqdm_final:
        logger.info("Detected non-interactive environment; tqdm disabled.")

    for run in range(1, args.num_runs + 1):
        logger.info(
            f"======== Starting run {run}/{args.num_runs} ({args.experiment_name}) ========"
        )
        set_seed(args.seed + run)

        model = TCR_GIN_Profile(args).to(device)
        model.apply(init_weights)
        if run == 1:
            logger.info(f"Model architecture: {model}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        scheduler = (
            LRScheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr)
            if args.use_lr_scheduler
            else None
        )

        model_save_path = os.path.join(model_dir, f"model_run_{run}.pt")
        early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            metric=args.monitor_metric,
            save_path=model_save_path,
        )

        for epoch in range(args.epochs):
            start_time = time.time()
            should_print = (epoch == 0) or ((epoch + 1) % 10 == 0)

            train_loss, sup_loss, con_loss, mono_loss, train_metrics = train_epoch_profile(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args,
                use_tqdm=(use_tqdm_final and should_print),
            )

            if (epoch + 1) % args.eval_steps == 0:
                val_loss, val_metrics = evaluate_profile(
                    model,
                    val_loader,
                    device,
                    args.tau_values,
                    label_scale=float(args.label_scale),
                )
                epoch_time = time.time() - start_time

                if should_print:
                    logger.info(
                        f"Run {run}, Ep {epoch + 1:03d} | "
                        f"L_total_scaled: {train_loss:.4f} "
                        f"(sup: {sup_loss:.4f}, con: {con_loss:.4f}, mono: {mono_loss:.4f}) | "
                        f"Train profile_MAE(orig): {train_metrics['profile_mae']:.6f} | "
                        f"Val profile_MAE(orig): {val_metrics['profile_mae']:.6f} | "
                        f"Time: {epoch_time:.2f}s"
                    )

                early_stopping(epoch, val_metrics, model)
                if early_stopping.early_stop:
                    logger.info(
                        f"Early stopping triggered. Best epoch: {early_stopping.best_epoch + 1}"
                    )
                    break

        logger.info(f"Loading best model from epoch {early_stopping.best_epoch + 1}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))

        # ── Test evaluation ───────────────────────────────────────────────
        if test_loader is not None and len(test_loader.dataset) > 0:
            test_loss, test_metrics = evaluate_profile(
                model,
                test_loader,
                device,
                args.tau_values,
                label_scale=float(args.label_scale),
            )

            logger.info(
                f"Test results - Loss(scaled): {test_loss:.4f}, "
                f"profile_RMSE(orig): {test_metrics['profile_rmse']:.6f}, "
                f"profile_MAE(orig): {test_metrics['profile_mae']:.6f}, "
                f"profile_corr(orig): {test_metrics['profile_corr']:.6f}"
            )
        else:
            logger.info("No test data available — skipping test evaluation.")
            test_metrics = {}

        results = {
            "run": run,
            "label_scale": float(args.label_scale),
            "active_tau_values": args.tau_values,
            "best_epoch": early_stopping.best_epoch + 1,
            "best_val_metrics": early_stopping.best_metrics,
            "test_metrics": test_metrics,
        }

        with open(
            os.path.join(output_dir, f"run_{run}_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(convert_numpy_types(results), f, indent=4, ensure_ascii=False)

    if args.rebuild_cache:
        cache_paths = (
            args.cache_path if isinstance(args.cache_path, list) else [args.cache_path]
        )
        for cp in cache_paths:
            if os.path.exists(cp):
                shutil.rmtree(cp)
                logger.info(f"Cache directory removed: {cp}")


if __name__ == "__main__":
    main()
