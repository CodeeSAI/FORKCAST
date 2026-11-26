"""CPU training pipeline (1-step + autoregressive multi-step) for FORKCAST."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (
    CHANNELS,
    LAT,
    ForkcastNetCDFDataset,
    compute_train_climatology,
    load_stats,
)
from src.evaluate import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    DILATE,
    LOCAL_MIN_FRAC,
    MAX_FAR_RESPONSE_FRAC,
    MAX_SPEED_CELLS_PER_DAY,
    NONZERO_MIN,
    OUTWARD_ENERGY_FRACTION,
    OUTWARD_GROWTH_FACTOR,
    STABLE_MAX_STD,
    box_mask,
    causality_checks,
    evaluate_full,
    make_shock,
    near_mask,
    validity,
)
from src.model import SCORE_STEPS, ForkcastResCNN


def _get_process_memory_mb() -> Optional[Dict[str, float]]:
    """Measure current and peak process RAM in MB (Windows API with POSIX fallback)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return {
                "working_set_mb": round(pmc.WorkingSetSize / (1024 * 1024), 2),
                "peak_working_set_mb": round(pmc.PeakWorkingSetSize / (1024 * 1024), 2),
            }
    except Exception:
        pass
    try:
        import resource  # type: ignore

        peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"working_set_mb": round(peak_kb / 1024.0, 2), "peak_working_set_mb": round(peak_kb / 1024.0, 2)}
    except Exception:
        return None


def _lat_weight_tensor() -> torch.Tensor:
    w = np.cos(np.deg2rad(LAT))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32).reshape(1, 1, 1, len(LAT), 1)


def compute_rainfall_aware_loss(
    pred_phys: torch.Tensor,
    pred_raw: torch.Tensor,
    target_phys: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lat_w: torch.Tensor,
    rain_alpha: float = 1.5,
    grad_weight: float = 0.15,
    neg_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Latitude-weighted, rainfall-aware, anti-blur loss over [B, K, 6, 32, 36].

    - Base term: latitude-weighted MSE in normalized channel units.
    - Rain-aware term: emphasizes rainy & heavy-rain (>=10 mm/day) cells on tp (ch=1).
    - Anti-blur term: spatial gradient (dx, dy) matching on normalized fields.
    - Non-negativity hinge: penalizes negative pre-ReLU tp and tcwv raw predictions.
    """
    std_5 = std.reshape(1, 1, 6, 1, 1)
    mean_5 = mean.reshape(1, 1, 6, 1, 1)

    err_norm = (pred_phys - target_phys) / std_5
    sq_err = err_norm.pow(2) * lat_w

    # Upweight precipitation cells smoothly up to heavy-rain storms (>= 30 mm/day)
    tp_target = target_phys[:, :, 1:2]
    tp_weight = 1.0 + float(rain_alpha) * torch.clamp(tp_target / 10.0, min=0.0, max=3.0)
    weights = torch.ones_like(sq_err)
    weights[:, :, 1:2] = tp_weight

    mse_loss = (sq_err * weights).mean()

    pred_norm = (pred_phys - mean_5) / std_5
    targ_norm = (target_phys - mean_5) / std_5
    dx_p = pred_norm[..., :, 1:] - pred_norm[..., :, :-1]
    dx_t = targ_norm[..., :, 1:] - targ_norm[..., :, :-1]
    dy_p = pred_norm[..., 1:, :] - pred_norm[..., :-1, :]
    dy_t = targ_norm[..., 1:, :] - targ_norm[..., :-1, :]
    grad_loss = 0.5 * (
        ((dx_p - dx_t).pow(2) * lat_w).mean()
        + ((dy_p - dy_t).pow(2) * lat_w[..., :-1, :]).mean()
    )

    neg_tp = F.relu(-pred_raw[:, :, 1] / std_5[:, :, 1]).pow(2).mean()
    neg_tcwv = F.relu(-pred_raw[:, :, 5] / std_5[:, :, 5]).pow(2).mean()
    neg_loss = neg_tp + 0.1 * neg_tcwv

    total = mse_loss + float(grad_weight) * grad_loss + float(neg_weight) * neg_loss
    parts = {
        "total": float(total.detach().item()),
        "mse": float(mse_loss.detach().item()),
        "grad": float(grad_loss.detach().item()),
        "neg": float(neg_loss.detach().item()),
    }
    return total, parts


def counterfactual_response_loss(
    model: ForkcastResCNN,
    init_states: torch.Tensor,
    doy_sc: torch.Tensor,
    baseline_phys: torch.Tensor,
    shock_name: str,
) -> torch.Tensor:
    """Contract-aligned training regularizer for learned, local shock propagation.

    The alternate rollout is produced by the same trained network. This loss does
    not insert a forecast pattern or alter inference outputs: it trains the model
    to retain a local Day-1 response and develop a stable, spatially reachable
    response outside the official shock neighbourhood by Days 4-8.
    """
    if shock_name not in ("warm_ocean", "monsoon_stall"):
        raise ValueError(f"Unsupported counterfactual shock: {shock_name!r}")
    if init_states.ndim != 5 or init_states.shape[1:] != (3, 6, 32, 36):
        raise ValueError(f"init_states must have shape [B,3,6,32,36], got {tuple(init_states.shape)}")
    if baseline_phys.ndim != 5 or baseline_phys.shape[1] < 8:
        raise ValueError("baseline_phys must contain at least 8 forecast days")

    last_np = init_states[:, -1].detach().cpu().numpy()
    perturb_np = np.stack([make_shock(shock_name, state) for state in last_np], axis=0)
    perturb = torch.as_tensor(perturb_np, dtype=init_states.dtype, device=init_states.device)
    shocked_phys = model.rollout(init_states, doy_sc, steps=8, perturbation=perturb)
    delta_norm = (shocked_phys - baseline_phys[:, :8]) / model.std.reshape(1, 1, 6, 1, 1)
    energy = delta_norm.square().sum(dim=2)  # [B, 8, 32, 36]

    box = BOX_BAY_OF_BENGAL if shock_name == "warm_ocean" else BOX_ARABIAN_SEA
    near_np = near_mask(box, dilate=DILATE)
    near = torch.as_tensor(near_np, dtype=energy.dtype, device=energy.device).reshape(1, 1, 32, 36)
    outside = 1.0 - near

    day1_total = energy[:, 0].sum(dim=(-2, -1))
    day1_near = (energy[:, 0] * near[:, 0]).sum(dim=(-2, -1))
    local_fraction = day1_near / (day1_total + 1e-12)
    local_loss = F.relu(float(LOCAL_MIN_FRAC) - local_fraction).square()

    outside_energy = (energy * outside).sum(dim=(-2, -1))
    day1_outside = outside_energy[:, 0]
    days4_8_outside = outside_energy[:, 3:8].mean(dim=1)
    max_total_energy = energy.sum(dim=(-2, -1)).amax(dim=1)
    required_outside = float(OUTWARD_ENERGY_FRACTION) * max_total_energy
    threshold_loss = F.relu(required_outside - days4_8_outside) / (required_outside + 1e-8)
    growth_ratio = days4_8_outside / (day1_outside + 1e-8)
    growth_loss = F.relu(float(OUTWARD_GROWTH_FACTOR) - growth_ratio) / float(OUTWARD_GROWTH_FACTOR)

    max_response_std = delta_norm.abs().amax(dim=(1, 2, 3, 4))
    nonzero_loss = F.relu(float(NONZERO_MIN) - max_response_std) / float(NONZERO_MIN)
    stable_loss = F.relu(max_response_std - float(STABLE_MAX_STD)) / float(STABLE_MAX_STD)

    core_np = box_mask(box)
    iy, ix = np.indices(core_np.shape)
    cy, cx = np.where(core_np)
    distance = np.sqrt((iy[..., None] - cy) ** 2 + (ix[..., None] - cx) ** 2).min(axis=-1)
    far_masks_np = np.stack([
        distance > (float(DILATE) + float(MAX_SPEED_CELLS_PER_DAY) * (day + 1))
        for day in range(8)
    ], axis=0)
    far_masks = torch.as_tensor(far_masks_np, dtype=energy.dtype, device=energy.device).unsqueeze(0)
    far_fraction = (energy * far_masks).sum(dim=(-2, -1)) / (energy.sum(dim=(-2, -1)) + 1e-12)
    teleport_loss = F.relu(far_fraction.amax(dim=1) - float(MAX_FAR_RESPONSE_FRAC)).square()

    return (
        local_loss
        + threshold_loss
        + growth_loss
        + nonzero_loss
        + stable_loss
        + teleport_loss
    ).mean()


def _run_epoch(
    model: ForkcastResCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lat_w: torch.Tensor,
    rollout_steps: int,
    rain_alpha: float,
    grad_weight: float,
    neg_weight: float,
    shock_weight: float = 0.0,
    shock_focus: str = "alternate",
    max_grad_norm: float = 1.0,
    max_batches: int = 0,
    max_mem_mb: float = 5500.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.train()
    sums = {"total": 0.0, "mse": 0.0, "grad": 0.0, "neg": 0.0, "shock_causality": 0.0}
    n_batches = 0
    total_batches = len(loader)
    t0 = time.perf_counter()

    for init_states, doy_sc, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        pred_phys, pred_raw = model.rollout(
            init_states, doy_sc, steps=rollout_steps, return_raw=True
        )
        loss, parts = compute_rainfall_aware_loss(
            pred_phys,
            pred_raw,
            targets,
            model.mean,
            model.std,
            lat_w,
            rain_alpha=rain_alpha,
            grad_weight=grad_weight,
            neg_weight=neg_weight,
        )
        shock_penalty = loss.new_zeros(())
        if shock_weight > 0.0 and rollout_steps >= 8:
            if shock_focus == "alternate":
                shock_name = "warm_ocean" if n_batches % 2 == 0 else "monsoon_stall"
            elif shock_focus in ("warm_ocean", "monsoon_stall"):
                shock_name = shock_focus
            else:
                raise ValueError(f"Unsupported shock_focus: {shock_focus!r}")
            shock_penalty = counterfactual_response_loss(
                model, init_states, doy_sc, pred_phys, shock_name
            )
            loss = loss + float(shock_weight) * shock_penalty
        parts = dict(parts)
        parts["shock_causality"] = float(shock_penalty.detach().item())
        parts["total"] = float(loss.detach().item())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at batch {n_batches + 1}: {parts}"
            )
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradient in {name} at batch {n_batches + 1}"
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        mem = _get_process_memory_mb()
        if mem and max_mem_mb > 0 and mem["working_set_mb"] > max_mem_mb:
            raise MemoryError(
                f"Working set memory exceeded limit: {mem['working_set_mb']} MB > {max_mem_mb} MB"
            )

        for k, v in parts.items():
            sums[k] += v
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    train_elapsed_s = max(time.perf_counter() - t0, 1e-9)
    batches_per_sec = n_batches / train_elapsed_s
    est_full_epoch_s = total_batches / max(batches_per_sec, 1e-9)
    avg_loss = {k: v / max(n_batches, 1) for k, v in sums.items()}
    timing = {
        "batches_run": int(n_batches),
        "total_batches_in_epoch": int(total_batches),
        "train_elapsed_s": round(train_elapsed_s, 3),
        "batches_per_sec": round(batches_per_sec, 4),
        "est_full_epoch_s": round(est_full_epoch_s, 2),
    }
    return avg_loss, timing


def _evaluate_val_loss(
    model: ForkcastResCNN,
    val_ds: ForkcastNetCDFDataset,
    lat_w: torch.Tensor,
    rollout_steps: int,
    batch_size: int,
    rain_alpha: float,
    grad_weight: float,
    neg_weight: float,
) -> Dict[str, float]:
    """Compute validation loss on the same objective and rollout horizon as training."""
    prev_steps = val_ds.rollout_steps
    if prev_steps != rollout_steps:
        val_ds.set_rollout_steps(rollout_steps)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    sums = {"total": 0.0, "mse": 0.0, "grad": 0.0, "neg": 0.0}
    n_batches = 0
    try:
        with torch.no_grad():
            for init_states, doy_sc, targets in loader:
                pred_phys, pred_raw = model.rollout(
                    init_states, doy_sc, steps=rollout_steps, return_raw=True
                )
                _, parts = compute_rainfall_aware_loss(
                    pred_phys,
                    pred_raw,
                    targets,
                    model.mean,
                    model.std,
                    lat_w,
                    rain_alpha=rain_alpha,
                    grad_weight=grad_weight,
                    neg_weight=neg_weight,
                )
                for k, v in parts.items():
                    sums[k] += v
                n_batches += 1
    finally:
        if val_ds.rollout_steps != prev_steps:
            val_ds.set_rollout_steps(prev_steps)
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def _save_checkpoint(
    path: Path,
    model: ForkcastResCNN,
    epoch: int,
    stage: str,
    val_metrics: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "hidden_channels": model.hidden_channels,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "train_args": vars(args),
        },
        path,
    )


def run_smoke_test(args: argparse.Namespace) -> Dict[str, object]:
    """Verify 1-step & multi-step forward/backward pass, locality, null-shock, and memory."""
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(int(args.threads))

    stats = load_stats(args.stats)
    train_ds = ForkcastNetCDFDataset(
        args.nc, split="train", rollout_steps=1, chunk_cache_mb=args.chunk_cache_mb
    )
    val_ds = ForkcastNetCDFDataset(
        args.nc, split="val", rollout_steps=10, chunk_cache_mb=args.chunk_cache_mb
    )
    holdout_ds = ForkcastNetCDFDataset(
        args.nc, split="holdout", rollout_steps=10, chunk_cache_mb=args.chunk_cache_mb
    )

    try:
        model = ForkcastResCNN(
            mean=stats["mean"],
            std=stats["std"],
            static_fields=train_ds.static,
            hidden_channels=int(args.hidden),
        )
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        lat_w = _lat_weight_tensor()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay)

        # 1. Single-step forward + backward smoke test
        loader1 = DataLoader(train_ds, batch_size=min(8, args.batch_size), shuffle=True, num_workers=0)
        t0 = time.perf_counter()
        init1, doy1, targ1 = next(iter(loader1))
        opt.zero_grad(set_to_none=True)
        pred1, raw1 = model.rollout(init1, doy1, steps=1, return_raw=True)
        loss1, parts1 = compute_rainfall_aware_loss(
            pred1, raw1, targ1, model.mean, model.std, lat_w,
            args.rain_alpha, args.grad_weight, args.neg_weight,
        )
        loss1.backward()
        grad_norms1 = [
            float(p.grad.detach().norm().item())
            for p in model.parameters()
            if p.grad is not None
        ]
        opt.step()
        step1_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Multi-step (10-step) autoregressive forward + backward smoke test
        train_ds.set_rollout_steps(10)
        loader10 = DataLoader(train_ds, batch_size=min(8, args.batch_size), shuffle=True, num_workers=0)
        t0 = time.perf_counter()
        init10, doy10, targ10 = next(iter(loader10))
        opt.zero_grad(set_to_none=True)
        pred10, raw10 = model.rollout(init10, doy10, steps=10, return_raw=True)
        loss10, parts10 = compute_rainfall_aware_loss(
            pred10, raw10, targ10, model.mean, model.std, lat_w,
            args.rain_alpha, args.grad_weight, args.neg_weight,
        )
        loss10.backward()
        grad_norms10 = [
            float(p.grad.detach().norm().item())
            for p in model.parameters()
            if p.grad is not None
        ]
        opt.step()
        step10_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Inference contract, zero-shock & Day-1 shock locality check
        model.eval()
        with torch.no_grad():
            v_init, v_doy, _ = val_ds[0]
            t0 = time.perf_counter()
            base = model.rollout(v_init, v_doy, steps=10).numpy()
            infer_10_ms = (time.perf_counter() - t0) * 1000.0

            zeros = torch.zeros((6, 32, 36), dtype=torch.float32)
            null_out = model.rollout(v_init, v_doy, steps=10, perturbation=zeros).numpy()
            null_max_diff = float(np.abs(null_out - base).max())

            warm_p = torch.from_numpy(make_shock("warm_ocean", v_init[-1].numpy()))
            warm_out = model.rollout(v_init, v_doy, steps=10, perturbation=warm_p).numpy()
            stall_p = torch.from_numpy(make_shock("monsoon_stall", v_init[-1].numpy()))
            stall_out = model.rollout(v_init, v_doy, steps=10, perturbation=stall_p).numpy()

        std_np = np.asarray(stats["std"], dtype=np.float32)
        null_std_diff = float((np.abs(null_out - base) / std_np[None, :, None, None]).max())
        warm_chk = causality_checks(base, warm_out, BOX_BAY_OF_BENGAL, std_np)
        stall_chk = causality_checks(base, stall_out, BOX_ARABIAN_SEA, std_np)
        shock_validity = {
            "baseline": validity(base, 10),
            "warm_ocean": validity(warm_out, 10),
            "monsoon_stall": validity(stall_out, 10),
            "null": validity(null_out, 10),
        }

        rss_mb = _get_process_memory_mb()

        return {
            "split_days": {
                "train": train_ds.split_days,
                "val": val_ds.split_days,
                "holdout": holdout_ds.split_days,
            },
            "window_counts_k10": {
                "train": len(train_ds),
                "val": len(val_ds),
                "holdout": len(holdout_ds),
            },
            "architecture": {
                "class": "ForkcastResCNN",
                "hidden_channels": model.hidden_channels,
                "param_count": param_count,
                "param_kb": round(param_bytes / 1024.0, 2),
                "buffer_kb": round(buffer_bytes / 1024.0, 2),
                "conv3x3_depth_per_step": 3,
                "padding_mode": "replicate",
                "spatial_normalization": None,
            },
            "stage1_k1_smoke": {
                "init_shape": list(init1.shape),
                "doy_shape": list(doy1.shape),
                "target_shape": list(targ1.shape),
                "pred_shape": list(pred1.shape),
                "loss": parts1,
                "min_grad_norm": round(min(grad_norms1), 6),
                "max_grad_norm": round(max(grad_norms1), 6),
                "all_grads_finite": all(np.isfinite(grad_norms1)),
                "min_pred_tp": float(pred1[:, :, 1].min().item()),
                "min_pred_tcwv": float(pred1[:, :, 5].min().item()),
                "step_ms": round(step1_ms, 2),
            },
            "stage2_k10_smoke": {
                "init_shape": list(init10.shape),
                "target_shape": list(targ10.shape),
                "pred_shape": list(pred10.shape),
                "loss": parts10,
                "min_grad_norm": round(min(grad_norms10), 6),
                "max_grad_norm": round(max(grad_norms10), 6),
                "all_grads_finite": all(np.isfinite(grad_norms10)),
                "min_pred_tp": float(pred10[:, :, 1].min().item()),
                "min_pred_tcwv": float(pred10[:, :, 5].min().item()),
                "step_ms": round(step10_ms, 2),
            },
            "inference_checks": {
                "rollout_10step_shape": list(base.shape),
                "rollout_10step_ms": round(infer_10_ms, 2),
                "all_finite": bool(np.isfinite(base).all()),
                "per_channel_min": {c: float(base[:, i].min()) for i, c in enumerate(CHANNELS)},
                "per_channel_max": {c: float(base[:, i].max()) for i, c in enumerate(CHANNELS)},
                "t2m_in_range": bool(float(base[:, 0].min()) >= 233.15 and float(base[:, 0].max()) <= 333.15),
                "tp_nonneg": bool(float(base[:, 1].min()) >= 0.0),
                "tcwv_nonneg": bool(float(base[:, 5].min()) >= 0.0),
                "null_shock_max_abs_diff": null_max_diff,
                "null_shock_max_std_diff": null_std_diff,
                "shock_validity": shock_validity,
                "shock_validity_ok": all(len(v) == 0 for v in shock_validity.values()),
                "warm_ocean": warm_chk,
                "monsoon_stall": stall_chk,
            },
            "memory": rss_mb,
        }
    finally:
        train_ds.close()
        val_ds.close()
        holdout_ds.close()


def competition_proxy_score(val_metrics: Dict[str, object]) -> Dict[str, float]:
    """Approximate the model-dependent 70-point organiser score for checkpoint selection.

    This uses validation aggregates as a selection proxy; the official scorer still
    decides the final score on its five hidden weather windows. It deliberately
    scores RMSE against the better of persistence/climatology, heavy-rain CSI,
    Day-8 anti-blur, and both shock responses rather than persistence skill alone.
    """
    ss_value = val_metrics.get("mean_ss_vs_best_ref")
    csi_value = val_metrics.get("mean_csi_skill")
    ss = float(ss_value) if ss_value is not None and np.isfinite(float(ss_value)) else float("nan")
    csi_skill = float(csi_value) if csi_value is not None and np.isfinite(float(csi_value)) else float("nan")

    rmse_points = 30.0 * float(np.clip(ss / 0.12, 0.0, 1.0)) if np.isfinite(ss) else 0.0
    csi_points = 15.0 * float(np.clip(csi_skill / 0.10, 0.0, 1.0)) if np.isfinite(csi_skill) else 0.0

    day8_ratios = val_metrics.get("leads", {}).get("day_8", {}).get("variance_ratio", {})
    blur_scores = []
    for channel in CHANNELS:
        value = day8_ratios.get(channel)
        if value is None or not np.isfinite(float(value)):
            blur_scores.append(0.0)
        else:
            ratio = float(value)
            blur_scores.append(float(np.clip(ratio / 0.40, 0.0, 1.0) * (ratio <= 2.50)))
    anti_blur_points = 10.0 * float(np.mean(blur_scores))

    shock_points = 0.0
    for name in ("warm_ocean", "monsoon_stall"):
        shock = val_metrics.get(name, {})
        quarters = 0
        if shock.get("nonzero", False):
            quarters += 1
            if shock.get("localized", False):
                quarters += 1 + int(bool(shock.get("propagates", False))) + int(bool(shock.get("stable", False)))
        shock_points += 7.5 * quarters / 4.0
    if not bool(val_metrics.get("null_shock_ok", False)):
        shock_points *= 0.5

    return {
        "total": rmse_points + csi_points + anti_blur_points + shock_points,
        "rmse_points": rmse_points,
        "csi_points": csi_points,
        "anti_blur_points": anti_blur_points,
        "shock_points": shock_points,
    }


def _execute_and_log_epoch(
    stage_name: str,
    epoch: int,
    k_steps: int,
    model: ForkcastResCNN,
    train_loader: DataLoader,
    val_ds: ForkcastNetCDFDataset,
    train_clim: np.ndarray,
    stats: Dict[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    lat_w: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
    log_path: Path,
    best_score: float,
) -> float:
    """Run one stage epoch, compute full pilot/validation telemetry, log JSONL, and checkpoint."""
    t_ep0 = time.perf_counter()
    tr_loss, timing = _run_epoch(
        model,
        train_loader,
        optimizer,
        lat_w,
        k_steps,
        args.rain_alpha,
        args.grad_weight,
        args.neg_weight,
        shock_weight=getattr(args, "shock_weight", 0.0),
        shock_focus=getattr(args, "shock_focus", "alternate"),
        max_batches=args.max_batches,
        max_mem_mb=getattr(args, "max_mem_mb", 5500.0),
    )
    scheduler.step()

    t_val0 = time.perf_counter()
    val_loss = _evaluate_val_loss(
        model,
        val_ds,
        lat_w,
        rollout_steps=k_steps,
        batch_size=args.batch_size,
        rain_alpha=args.rain_alpha,
        grad_weight=args.grad_weight,
        neg_weight=args.neg_weight,
    )
    val_metrics = evaluate_full(
        model,
        args.nc,
        stats,
        split="val",
        steps=SCORE_STEPS,
        batch_size=args.batch_size,
        climatology=train_clim,
        dataset=val_ds,
    )
    val_elapsed_s = time.perf_counter() - t_val0
    total_elapsed_s = time.perf_counter() - t_ep0
    rss_mb = _get_process_memory_mb()

    csi_summary = {
        lead: {
            "csi_model": m["csi_model"],
            "csi_persist": m["csi_persist"],
            "csi_clim": m["csi_clim"],
            "csi_skill": m["csi_skill"],
            "mean_ss_vs_persist": m["mean_ss_vs_persist"],
            "mean_ss_vs_clim": m["mean_ss_vs_clim"],
            "mean_ss_vs_best_ref": m["mean_ss_vs_best_ref"],
        }
        for lead, m in val_metrics["leads"].items()
    }

    record = {
        "stage": stage_name,
        "epoch": epoch,
        "k_steps": k_steps,
        "train_elapsed_s": timing["train_elapsed_s"],
        "val_elapsed_s": round(val_elapsed_s, 3),
        "elapsed_s": round(total_elapsed_s, 2),
        "batches_run": timing["batches_run"],
        "total_batches_in_epoch": timing["total_batches_in_epoch"],
        "batches_per_sec": timing["batches_per_sec"],
        "est_full_epoch_s": timing["est_full_epoch_s"],
        "memory": rss_mb,
        "train_loss": tr_loss,
        "val_loss": val_loss,
        "val_day4_ss_vs_persist": val_metrics["day4_ss_vs_persist"],
        "val_mean_ss_vs_persist": val_metrics["mean_ss_vs_persist"],
        "val_mean_ss_vs_clim": val_metrics["mean_ss_vs_clim"],
        "val_mean_ss_vs_best_ref": val_metrics["mean_ss_vs_best_ref"],
        "val_mean_csi_skill": val_metrics["mean_csi_skill"],
        "val_leads_summary": csi_summary,
        "val_invalid_windows": val_metrics["invalid_windows"],
        "shock_validity_ok": val_metrics["shock_validity_ok"],
        "null_shock_ok": val_metrics["null_shock_ok"],
        "val_competition_proxy": competition_proxy_score(val_metrics),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)

    score_parts = competition_proxy_score(val_metrics)
    score = float(score_parts["total"])
    day4_skill = val_metrics.get("day4_ss_vs_persist")
    day4_gate = day4_skill is not None and np.isfinite(float(day4_skill)) and float(day4_skill) > 0.0
    localized_shock_gate = any(
        bool(val_metrics.get(name, {}).get("nonzero", False))
        and bool(val_metrics.get(name, {}).get("localized", False))
        for name in ("warm_ocean", "monsoon_stall")
    )
    if (
        val_metrics["invalid_windows"] == 0
        and val_metrics["shock_validity_ok"]
        and day4_gate
        and localized_shock_gate
        and score > best_score
    ):
        best_score = score
        _save_checkpoint(out_dir / "best.pt", model, epoch, stage_name, val_metrics, args)
    _save_checkpoint(out_dir / "latest.pt", model, epoch, stage_name, val_metrics, args)
    return best_score


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(int(args.threads))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "history.jsonl"

    stats = load_stats(args.stats)
    train_clim = compute_train_climatology(
        args.nc, split="train", chunk_cache_mb=args.chunk_cache_mb
    )
    train_ds = ForkcastNetCDFDataset(
        args.nc, split="train", rollout_steps=1, chunk_cache_mb=args.chunk_cache_mb
    )
    val_ds = ForkcastNetCDFDataset(
        args.nc, split="val", rollout_steps=SCORE_STEPS, chunk_cache_mb=args.chunk_cache_mb
    )

    try:
        model = ForkcastResCNN(
            mean=stats["mean"],
            std=stats["std"],
            static_fields=train_ds.static,
            hidden_channels=int(args.hidden),
        )
        if getattr(args, "resume_ckpt", None):
            ckpt_path = Path(args.resume_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {ckpt_path}", flush=True)

        lat_w = _lat_weight_tensor()
        best_score = -float("inf")

        (out_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )

        # Stage 1: 1-step training
        opt1 = torch.optim.AdamW(
            model.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay
        )
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=max(args.epochs_stage1, 1)
        )
        train_ds.set_rollout_steps(1)
        loader1 = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        for epoch in range(1, args.epochs_stage1 + 1):
            best_score = _execute_and_log_epoch(
                stage_name="stage1_k1",
                epoch=epoch,
                k_steps=1,
                model=model,
                train_loader=loader1,
                val_ds=val_ds,
                train_clim=train_clim,
                stats=stats,
                optimizer=opt1,
                scheduler=sched1,
                lat_w=lat_w,
                args=args,
                out_dir=out_dir,
                log_path=log_path,
                best_score=best_score,
            )

        # Stage 2: Autoregressive multi-step fine-tuning curriculum
        if args.epochs_stage2 > 0:
            if not getattr(args, "resume_ckpt", None):
                best_ckpt = out_dir / "best.pt"
                if best_ckpt.exists():
                    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])

            schedule = [int(x) for x in str(args.stage2_schedule).split(",") if x.strip()]
            opt2 = torch.optim.AdamW(
                model.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay
            )
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=max(args.epochs_stage2, 1)
            )

            for epoch in range(1, args.epochs_stage2 + 1):
                stage_idx = min(
                    len(schedule) - 1,
                    int((epoch - 1) * len(schedule) / max(args.epochs_stage2, 1)),
                )
                k_steps = schedule[stage_idx]
                train_ds.set_rollout_steps(k_steps)
                loader_k = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
                )

                best_score = _execute_and_log_epoch(
                    stage_name=f"stage2_k{k_steps}",
                    epoch=epoch,
                    k_steps=k_steps,
                    model=model,
                    train_loader=loader_k,
                    val_ds=val_ds,
                    train_clim=train_clim,
                    stats=stats,
                    optimizer=opt2,
                    scheduler=sched2,
                    lat_w=lat_w,
                    args=args,
                    out_dir=out_dir,
                    log_path=log_path,
                    best_score=best_score,
                )
    finally:
        train_ds.close()
        val_ds.close()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train FORKCAST model on CPU")
    ap.add_argument("--nc", default=r"C:\FORKCAST_DATA\india_climate_train.nc")
    ap.add_argument("--stats", default=r"C:\FORKCAST_DATA\stats.json")
    ap.add_argument("--out-dir", default=r"C:\FORKCAST_DATA\runs\forkcast_rescnn")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-stage1", type=int, default=8)
    ap.add_argument("--epochs-stage2", type=int, default=12)
    ap.add_argument("--stage2-schedule", default="4,8,10")
    ap.add_argument("--lr-stage1", type=float, default=1e-3)
    ap.add_argument("--lr-stage2", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--rain-alpha", type=float, default=1.5)
    ap.add_argument("--grad-weight", type=float, default=0.15)
    ap.add_argument("--neg-weight", type=float, default=0.05)
    ap.add_argument(
        "--shock-weight",
        type=float,
        default=0.0,
        help="Weight for official localized-to-outward counterfactual response loss (active for K>=8)",
    )
    ap.add_argument(
        "--shock-focus",
        choices=("alternate", "warm_ocean", "monsoon_stall"),
        default="alternate",
        help="Train both official shock families alternately, or focus a single shock family",
    )
    ap.add_argument("--chunk-cache-mb", type=int, default=192)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="Cap batches per epoch (0=unlimited)")
    ap.add_argument("--resume-ckpt", default=None, help="Path to checkpoint to resume weights from")
    ap.add_argument("--max-mem-mb", type=float, default=5500.0, help="Abort training if working set memory exceeds this (MB)")
    ap.add_argument("--smoke-test", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke_test:
        res = run_smoke_test(args)
        print(json.dumps(res, indent=2))
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
