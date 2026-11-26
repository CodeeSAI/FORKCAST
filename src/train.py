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

from src.dataset import LAT, ForkcastNetCDFDataset, load_stats
from src.dataset import CHANNELS
from src.evaluate import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    causality_checks,
    evaluate_vs_persistence,
    make_shock,
)
from src.model import ForkcastResCNN


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


def _run_epoch(
    model: ForkcastResCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lat_w: torch.Tensor,
    rollout_steps: int,
    rain_alpha: float,
    grad_weight: float,
    neg_weight: float,
    max_grad_norm: float = 1.0,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    sums = {"total": 0.0, "mse": 0.0, "grad": 0.0, "neg": 0.0}
    n_batches = 0

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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k, v in parts.items():
            sums[k] += v
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

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
        warm_chk = causality_checks(base, warm_out, BOX_BAY_OF_BENGAL, std_np)
        stall_chk = causality_checks(base, stall_out, BOX_ARABIAN_SEA, std_np)

        # RSS memory measurement (Windows K32GetProcessMemoryInfo / Linux resource fallback)
        rss_mb = None
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
                rss_mb = {
                    "working_set_mb": round(pmc.WorkingSetSize / (1024 * 1024), 2),
                    "peak_working_set_mb": round(pmc.PeakWorkingSetSize / (1024 * 1024), 2),
                }
        except Exception:
            rss_mb = None

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
                "warm_ocean": warm_chk,
                "monsoon_stall": stall_chk,
            },
            "memory": rss_mb,
        }
    finally:
        train_ds.close()
        val_ds.close()
        holdout_ds.close()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(int(args.threads))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "history.jsonl"

    stats = load_stats(args.stats)
    train_ds = ForkcastNetCDFDataset(
        args.nc, split="train", rollout_steps=1, chunk_cache_mb=args.chunk_cache_mb
    )
    val_ds = ForkcastNetCDFDataset(
        args.nc, split="val", rollout_steps=10, chunk_cache_mb=args.chunk_cache_mb
    )

    try:
        model = ForkcastResCNN(
            mean=stats["mean"],
            std=stats["std"],
            static_fields=train_ds.static,
            hidden_channels=int(args.hidden),
        )
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
            t0 = time.perf_counter()
            tr_loss = _run_epoch(
                model, loader1, opt1, lat_w, 1, args.rain_alpha, args.grad_weight, args.neg_weight,
                max_batches=args.max_batches,
            )
            sched1.step()
            val_metrics = evaluate_vs_persistence(
                model, val_ds, steps=10, batch_size=args.batch_size
            )
            elapsed = time.perf_counter() - t0
            record = {
                "stage": "stage1_k1",
                "epoch": epoch,
                "elapsed_s": round(elapsed, 2),
                "train_loss": tr_loss,
                "val_day4_ss": val_metrics["day4_ss_vs_persist"],
                "val_mean_ss": val_metrics["mean_ss_vs_persist"],
                "val_invalid_windows": val_metrics["invalid_windows"],
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

            score = float(val_metrics["mean_ss_vs_persist"])
            if val_metrics["invalid_windows"] == 0 and score > best_score:
                best_score = score
                _save_checkpoint(out_dir / "best.pt", model, epoch, "stage1_k1", val_metrics, args)
            _save_checkpoint(out_dir / "latest.pt", model, epoch, "stage1_k1", val_metrics, args)

        # Stage 2: Autoregressive multi-step fine-tuning curriculum
        if args.epochs_stage2 > 0:
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
                # Map epoch across curriculum horizons
                stage_idx = min(
                    len(schedule) - 1,
                    int((epoch - 1) * len(schedule) / max(args.epochs_stage2, 1)),
                )
                k_steps = schedule[stage_idx]
                train_ds.set_rollout_steps(k_steps)
                loader_k = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
                )

                t0 = time.perf_counter()
                tr_loss = _run_epoch(
                    model,
                    loader_k,
                    opt2,
                    lat_w,
                    k_steps,
                    args.rain_alpha,
                    args.grad_weight,
                    args.neg_weight,
                    max_batches=args.max_batches,
                )
                sched2.step()
                val_metrics = evaluate_vs_persistence(
                    model, val_ds, steps=10, batch_size=args.batch_size
                )
                elapsed = time.perf_counter() - t0
                record = {
                    "stage": f"stage2_k{k_steps}",
                    "epoch": epoch,
                    "elapsed_s": round(elapsed, 2),
                    "train_loss": tr_loss,
                    "val_day4_ss": val_metrics["day4_ss_vs_persist"],
                    "val_mean_ss": val_metrics["mean_ss_vs_persist"],
                    "val_invalid_windows": val_metrics["invalid_windows"],
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)

                score = float(val_metrics["mean_ss_vs_persist"])
                if val_metrics["invalid_windows"] == 0 and score > best_score:
                    best_score = score
                    _save_checkpoint(
                        out_dir / "best.pt", model, epoch, f"stage2_k{k_steps}", val_metrics, args
                    )
                _save_checkpoint(
                    out_dir / "latest.pt", model, epoch, f"stage2_k{k_steps}", val_metrics, args
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
    ap.add_argument("--chunk-cache-mb", type=int, default=192)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="Cap batches per epoch (0=unlimited)")
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
