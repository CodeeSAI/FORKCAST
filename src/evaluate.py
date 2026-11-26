"""Offline evaluation against persistence, train climatology, validity, and shock causality."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    CHANNELS,
    LAT,
    LON,
    NC,
    NLAT,
    NLON,
    ForkcastNetCDFDataset,
    _doy_slot,
    compute_train_climatology,
    load_stats,
)
from src.model import ForkcastResCNN, MAX_STEPS, SCORE_STEPS

LEAD_DAYS: Tuple[int, ...] = (1, 4, 8)
RAIN_THRESH = 10.0
BLUR_MIN_RATIO = 0.40
BLUR_MAX_RATIO = 2.50
NULL_TOL = 1e-3
NONZERO_MIN = 1e-3
LOCAL_MIN_FRAC = 0.60
DILATE = 3
MAX_SPEED_CELLS_PER_DAY = 8.0
STABLE_MAX_STD = 5.0

BOX_BAY_OF_BENGAL = (8.0, 20.0, 82.0, 94.0)
BOX_ARABIAN_SEA = (8.0, 20.0, 65.0, 75.0)


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT))
    return (w / w.mean()).astype(np.float64)


def box_mask(box: Tuple[float, float, float, float]) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    return ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]


def rmse_lat(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Latitude-weighted RMSE per channel for [..., 6, 32, 36] arrays."""
    w = lat_weights()[:, None]
    se = (np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) ** 2
    return np.sqrt((se * w).mean(axis=(-2, -1)).reshape(-1, NC).mean(axis=0))


def csi(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float = RAIN_THRESH) -> float:
    p = pred_tp > thr
    t = true_tp > thr
    h = int((p & t).sum())
    m = int((~p & t).sum())
    f = int((p & ~t).sum())
    return float("nan") if (h + m + f) == 0 else float(h / (h + m + f))


def variance_ratio(pred_day: np.ndarray, true_day: np.ndarray) -> np.ndarray:
    """Per-channel spatial variance ratio pred / true for [..., 6, 32, 36] day slices."""
    if pred_day.ndim == 4:
        vp = pred_day.reshape(pred_day.shape[0], NC, -1).var(axis=-1).mean(axis=0)
        vt = true_day.reshape(true_day.shape[0], NC, -1).var(axis=-1).mean(axis=0)
    else:
        vp = pred_day.reshape(NC, -1).var(axis=1)
        vt = true_day.reshape(NC, -1).var(axis=1)
    return np.where(vt > 0, vp / np.maximum(vt, 1e-30), np.nan)


def validity(out: np.ndarray, steps: int) -> List[str]:
    bad: List[str] = []
    if out.shape != (steps, NC, NLAT, NLON):
        return [f"shape {out.shape} != {(steps, NC, NLAT, NLON)}"]
    if not np.isfinite(out).all():
        bad.append("NaN/Inf")
    else:
        if float(out[:, 1].min()) < -1e-6:
            bad.append(f"negative rain ({float(out[:, 1].min()):.6g})")
        t = out[:, 0]
        if float(t.min()) < 233.15 or float(t.max()) > 333.15:
            bad.append(f"t2m out of [-40, +60] C ({float(t.min()):.2f}..{float(t.max()):.2f})")
        if float(out[:, 5].min()) < -1e-6:
            bad.append(f"negative tcwv ({float(out[:, 5].min()):.6g})")
    return bad


def near_mask(box: Tuple[float, float, float, float], dilate: int = DILATE) -> np.ndarray:
    out = box_mask(box).copy()
    for _ in range(dilate):
        o = out.copy()
        o[1:] |= out[:-1]
        o[:-1] |= out[1:]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def make_shock(name: str, last_state: np.ndarray) -> np.ndarray:
    d = np.zeros((NC, NLAT, NLON), dtype=np.float32)
    if name == "warm_ocean":
        m = box_mask(BOX_BAY_OF_BENGAL)
        d[0][m] = 3.0
        d[5][m] = 0.15 * last_state[5][m]
    elif name == "monsoon_stall":
        m = box_mask(BOX_ARABIAN_SEA)
        d[2][m] = -0.4 * last_state[2][m]
        d[3][m] = -0.4 * last_state[3][m]
    elif name != "null":
        raise ValueError(f"Unknown shock name: {name!r}")
    return d


def causality_checks(
    base: np.ndarray,
    shock: np.ndarray,
    box: Tuple[float, float, float, float],
    std: np.ndarray,
) -> Dict[str, object]:
    d = shock - base
    if not np.isfinite(d).all():
        return {
            "finite": False,
            "nonzero": False,
            "localized": False,
            "propagates": False,
            "stable": False,
        }
    std4 = std.astype(np.float64)[None, :, None, None]
    d64 = d.astype(np.float64)
    z = np.abs(d64 / std4)
    nonzero = bool(z.max() >= NONZERO_MIN)
    stable = bool(z.max() <= STABLE_MAX_STD)
    energy = ((d64 / std4) ** 2).sum(axis=1)
    steps = energy.shape[0]
    near = near_mask(box)
    tot1 = float(energy[0].sum())
    near1 = float(energy[0][near].sum())
    near_frac_day1 = float(near1 / tot1) if tot1 > 0 else 0.0
    localized = bool(tot1 > 0 and near_frac_day1 >= LOCAL_MIN_FRAC)

    out_e = np.array([energy[k][~near].sum() for k in range(steps)], dtype=np.float64)
    late = float(out_e[3:8].mean()) if steps >= 8 else float(out_e[-1])
    grows = bool(
        late > 1.5 * float(out_e[0]) + 1e-12
        and late > 0.02 * max(float(energy.sum(axis=(1, 2)).max()), 1e-12)
    )

    ii, jj = np.indices((NLAT, NLON))
    bi, bj = np.where(box_mask(box))
    dist = np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1)
    tele = 0.0
    for k in range(steps):
        far = dist > (DILATE + MAX_SPEED_CELLS_PER_DAY * (k + 1))
        tele = max(tele, float(energy[k][far].sum() / max(float(energy[k].sum()), 1e-12)))
    no_teleport = bool(tele < 0.05)

    return {
        "finite": True,
        "nonzero": nonzero,
        "localized": localized,
        "propagates": bool(grows and no_teleport),
        "stable": stable,
        "max_abs_std": float(z.max()),
        "total_energy_day1": tot1,
        "near_energy_day1": near1,
        "near_frac_day1": near_frac_day1,
        "out_energy_day1": float(out_e[0]),
        "out_energy_days4_8": late,
        "out_energy_grows": grows,
        "far_frac_max": tele,
        "no_teleport": no_teleport,
    }


def evaluate_vs_persistence(
    model: ForkcastResCNN,
    dataset: ForkcastNetCDFDataset,
    steps: int = 10,
    batch_size: int = 32,
    leads: Sequence[int] = LEAD_DAYS,
) -> Dict[str, object]:
    """Fast multi-step validation benchmark against repeat-yesterday persistence."""
    if steps < 1 or steps > MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}], got {steps}")
    prev_steps = dataset.rollout_steps
    if prev_steps != steps:
        dataset.set_rollout_steps(steps)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds_all: List[np.ndarray] = []
    pers_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []

    with torch.no_grad():
        for init_states, doy_sc, targets in loader:
            out = model.rollout(init_states, doy_sc, steps=steps).cpu().numpy()
            last = init_states[:, -1:].cpu().numpy()
            pers = np.repeat(last, steps, axis=1)
            preds_all.append(out)
            pers_all.append(pers)
            true_all.append(targets.cpu().numpy())

    if prev_steps != steps:
        dataset.set_rollout_steps(prev_steps)

    preds = np.concatenate(preds_all, axis=0)
    pers = np.concatenate(pers_all, axis=0)
    trues = np.concatenate(true_all, axis=0)

    lead_metrics: Dict[str, Dict[str, object]] = {}
    ss_list: List[float] = []
    for d in leads:
        if d > steps:
            continue
        idx = d - 1
        r_m = rmse_lat(preds[:, idx], trues[:, idx])
        r_p = rmse_lat(pers[:, idx], trues[:, idx])
        ss_p = 1.0 - r_m / np.maximum(r_p, 1e-12)
        csi_m = csi(preds[:, idx, 1], trues[:, idx, 1], RAIN_THRESH)
        csi_p = csi(pers[:, idx, 1], trues[:, idx, 1], RAIN_THRESH)
        vr = variance_ratio(preds[:, idx], trues[:, idx])
        lead_metrics[f"day_{d}"] = {
            "rmse_model": {c: float(r_m[i]) for i, c in enumerate(CHANNELS)},
            "rmse_persist": {c: float(r_p[i]) for i, c in enumerate(CHANNELS)},
            "ss_vs_persist": {c: float(ss_p[i]) for i, c in enumerate(CHANNELS)},
            "mean_ss_vs_persist": float(np.mean(ss_p)),
            "csi_model": csi_m,
            "csi_persist": csi_p,
            "variance_ratio": {c: float(vr[i]) for i, c in enumerate(CHANNELS)},
        }
        ss_list.append(float(np.mean(ss_p)))

    valid_failures = 0
    for i in range(preds.shape[0]):
        if validity(preds[i], steps):
            valid_failures += 1

    return {
        "split": dataset.split,
        "num_windows": int(preds.shape[0]),
        "steps": int(steps),
        "mean_ss_vs_persist": float(np.mean(ss_list)) if ss_list else float("nan"),
        "day4_ss_vs_persist": lead_metrics["day_4"]["mean_ss_vs_persist"] if "day_4" in lead_metrics else None,
        "valid_windows": int(preds.shape[0] - valid_failures),
        "invalid_windows": int(valid_failures),
        "min_tp": float(preds[:, :, 1].min()),
        "min_tcwv": float(preds[:, :, 5].min()),
        "leads": lead_metrics,
    }


def evaluate_full(
    model: ForkcastResCNN,
    nc_path: str | Path,
    stats: Dict[str, object],
    split: str = "val",
    steps: int = SCORE_STEPS,
    batch_size: int = 32,
    climatology: Optional[np.ndarray] = None,
    dataset: Optional[ForkcastNetCDFDataset] = None,
) -> Dict[str, object]:
    """Full validation report: persistence & train-climatology skill, CSI, anti-blur, and causality."""
    import datetime as dt

    if steps < 1 or steps > MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}], got {steps}")

    owns_ds = dataset is None
    if owns_ds:
        ds = ForkcastNetCDFDataset(nc_path, split=split, rollout_steps=steps)
        prev_steps = steps
    else:
        ds = dataset
        prev_steps = ds.rollout_steps
        if prev_steps != steps:
            ds.set_rollout_steps(steps)
    try:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if climatology is None:
            climatology = compute_train_climatology(nc_path, split="train")

        model.eval()
        preds_all, pers_all, clim_all, true_all = [], [], [], []
        anchor_dates = [ds.get_anchor_date(i) for i in range(len(ds))]
        offset = 0

        with torch.no_grad():
            for init_states, doy_sc, targets in loader:
                bsz = init_states.shape[0]
                out = model.rollout(init_states, doy_sc, steps=steps).cpu().numpy()
                pers = np.repeat(init_states[:, -1:].cpu().numpy(), steps, axis=1)

                clim_b = np.zeros((bsz, steps, NC, NLAT, NLON), dtype=np.float32)
                for b in range(bsz):
                    d0 = anchor_dates[offset + b]
                    for k in range(steps):
                        clim_b[b, k] = climatology[_doy_slot(d0 + dt.timedelta(days=k + 1))]
                offset += bsz

                preds_all.append(out)
                pers_all.append(pers)
                clim_all.append(clim_b)
                true_all.append(targets.cpu().numpy())

        preds = np.concatenate(preds_all, axis=0)
        pers = np.concatenate(pers_all, axis=0)
        clims = np.concatenate(clim_all, axis=0)
        trues = np.concatenate(true_all, axis=0)

        leads_out: Dict[str, Dict[str, object]] = {}
        all_ss_persist: List[float] = []
        all_ss_clim: List[float] = []
        all_ss_best: List[float] = []
        all_csi_ss: List[float] = []
        for d in LEAD_DAYS:
            if d > steps:
                continue
            idx = d - 1
            rm = rmse_lat(preds[:, idx], trues[:, idx])
            rp = rmse_lat(pers[:, idx], trues[:, idx])
            rc = rmse_lat(clims[:, idx], trues[:, idx])
            best = np.minimum(rp, rc)
            ss_best = 1.0 - rm / np.maximum(best, 1e-12)
            ss_p = 1.0 - rm / np.maximum(rp, 1e-12)
            ss_c = 1.0 - rm / np.maximum(rc, 1e-12)

            cm = csi(preds[:, idx, 1], trues[:, idx, 1], RAIN_THRESH)
            cp = csi(pers[:, idx, 1], trues[:, idx, 1], RAIN_THRESH)
            cc = csi(clims[:, idx, 1], trues[:, idx, 1], RAIN_THRESH)
            has_ref = bool(np.isfinite([cp, cc]).any())
            ref_csi = float(np.nanmax([cp, cc])) if has_ref else float("nan")
            csi_ss = (
                float((cm - ref_csi) / (1.0 - ref_csi))
                if np.isfinite(cm) and np.isfinite(ref_csi) and ref_csi < 1.0
                else float("nan")
            )
            vr = variance_ratio(preds[:, idx], trues[:, idx])

            leads_out[f"day_{d}"] = {
                "rmse_model": {c: float(rm[i]) for i, c in enumerate(CHANNELS)},
                "rmse_persist": {c: float(rp[i]) for i, c in enumerate(CHANNELS)},
                "rmse_clim": {c: float(rc[i]) for i, c in enumerate(CHANNELS)},
                "ss_vs_persist": {c: float(ss_p[i]) for i, c in enumerate(CHANNELS)},
                "ss_vs_clim": {c: float(ss_c[i]) for i, c in enumerate(CHANNELS)},
                "ss_vs_best_ref": {c: float(ss_best[i]) for i, c in enumerate(CHANNELS)},
                "mean_ss_vs_persist": float(np.mean(ss_p)),
                "mean_ss_vs_clim": float(np.mean(ss_c)),
                "mean_ss_vs_best_ref": float(np.mean(ss_best)),
                "csi_model": cm,
                "csi_persist": cp,
                "csi_clim": cc,
                "csi_skill": csi_ss,
                "variance_ratio": {c: float(vr[i]) for i, c in enumerate(CHANNELS)},
            }
            all_ss_persist.append(float(np.mean(ss_p)))
            all_ss_clim.append(float(np.mean(ss_c)))
            all_ss_best.append(float(np.mean(ss_best)))
            if np.isfinite(csi_ss):
                all_csi_ss.append(csi_ss)

        invalid = sum(1 for i in range(preds.shape[0]) if validity(preds[i], steps))
        all_finite = bool(np.isfinite(preds).all())

        # Shock causality & latency check on a monsoon-season validation anchor if available
        monsoon_idx = 0
        for i, d0 in enumerate(anchor_dates):
            if d0.month in (6, 7, 8):
                monsoon_idx = i
                break

        init_s, doy_s, _ = ds[monsoon_idx]
        std_np = np.asarray(stats["std"], dtype=np.float32)
        with torch.no_grad():
            t0 = time.perf_counter()
            base_out = model.rollout(init_s, doy_s, steps=steps).cpu().numpy()
            single_ms = (time.perf_counter() - t0) * 1000.0

            zeros_pert = torch.zeros((NC, NLAT, NLON), dtype=torch.float32)
            null_out = model.rollout(init_s, doy_s, steps=steps, perturbation=zeros_pert).cpu().numpy()
            null_diff_std = float((np.abs(null_out - base_out) / std_np[None, :, None, None]).max())

            last_np = init_s[-1].numpy()
            warm_pert = torch.from_numpy(make_shock("warm_ocean", last_np))
            stall_pert = torch.from_numpy(make_shock("monsoon_stall", last_np))
            warm_out = model.rollout(init_s, doy_s, steps=steps, perturbation=warm_pert).cpu().numpy()
            stall_out = model.rollout(init_s, doy_s, steps=steps, perturbation=stall_pert).cpu().numpy()

        warm_chk = causality_checks(base_out, warm_out, BOX_BAY_OF_BENGAL, std_np)
        stall_chk = causality_checks(base_out, stall_out, BOX_ARABIAN_SEA, std_np)

        shock_validity = {
            "baseline": validity(base_out, steps),
            "warm_ocean": validity(warm_out, steps),
            "monsoon_stall": validity(stall_out, steps),
            "null": validity(null_out, steps),
        }
        shock_validity_ok = all(len(v) == 0 for v in shock_validity.values())
        shock_rollout_stats = {
            name: {
                "all_finite": bool(np.isfinite(arr).all()),
                "per_channel_min": {c: float(arr[:, i].min()) for i, c in enumerate(CHANNELS)},
                "per_channel_max": {c: float(arr[:, i].max()) for i, c in enumerate(CHANNELS)},
            }
            for name, arr in (
                ("baseline", base_out),
                ("warm_ocean", warm_out),
                ("monsoon_stall", stall_out),
                ("null", null_out),
            )
        }

        return {
            "split": ds.split,
            "num_windows": int(preds.shape[0]),
            "steps": int(steps),
            "day4_ss_vs_persist": leads_out["day_4"]["mean_ss_vs_persist"] if "day_4" in leads_out else None,
            "mean_ss_vs_persist": float(np.mean(all_ss_persist)) if all_ss_persist else None,
            "mean_ss_vs_clim": float(np.mean(all_ss_clim)) if all_ss_clim else None,
            "mean_ss_vs_best_ref": float(np.mean(all_ss_best)) if all_ss_best else None,
            "mean_csi_skill": float(np.mean(all_csi_ss)) if all_csi_ss else None,
            "all_finite": all_finite,
            "valid_windows": int(preds.shape[0] - invalid),
            "invalid_windows": int(invalid),
            "per_channel_min": {c: float(preds[:, :, i].min()) for i, c in enumerate(CHANNELS)},
            "per_channel_max": {c: float(preds[:, :, i].max()) for i, c in enumerate(CHANNELS)},
            "min_tp": float(preds[:, :, 1].min()),
            "min_tcwv": float(preds[:, :, 5].min()),
            "leads": leads_out,
            "single_10step_ms": float(single_ms),
            "null_shock_max_std_diff": null_diff_std,
            "null_shock_ok": bool(null_diff_std <= NULL_TOL),
            "causality_anchor_date": str(anchor_dates[monsoon_idx]),
            "shock_validity": shock_validity,
            "shock_validity_ok": shock_validity_ok,
            "shock_rollout_stats": shock_rollout_stats,
            "warm_ocean": warm_chk,
            "monsoon_stall": stall_chk,
        }
    finally:
        if owns_ds:
            ds.close()
        elif prev_steps != ds.rollout_steps:
            ds.set_rollout_steps(prev_steps)


def stability_diagnostics(
    model: ForkcastResCNN,
    init_states: torch.Tensor,
    doy_sincos: torch.Tensor,
    horizons: Sequence[int] = (20, 30),
) -> Dict[str, object]:
    """
    OFFLINE RESEARCH DIAGNOSTIC ONLY — not part of the API contract.

    Runs extended rollouts beyond MAX_STEPS by manually iterating model.step()
    to test long-horizon stability. Supports both unbatched [3,6,32,36] and
    batched [B,3,6,32,36] inputs. Must never be used in serve.py or API code.
    """
    if init_states.ndim not in (4, 5):
        raise ValueError(f"init_states must be 4D or 5D, got {tuple(init_states.shape)}")

    unbatched = init_states.ndim == 4
    if unbatched and doy_sincos.ndim != 1:
        raise ValueError(f"doy_sincos must be 1D for unbatched init_states, got {tuple(doy_sincos.shape)}")
    if not unbatched and (doy_sincos.ndim != 2 or doy_sincos.shape[0] != init_states.shape[0]):
        raise ValueError(
            f"doy_sincos shape {tuple(doy_sincos.shape)} incompatible with batch size {init_states.shape[0]}"
        )

    model.eval()
    results: Dict[str, object] = {}
    with torch.no_grad():
        for long_steps in horizons:
            if long_steps < 1:
                raise ValueError(f"horizon step count must be >= 1, got {long_steps}")
            if unbatched:
                window = init_states.unsqueeze(0).to(dtype=torch.float32).clone()
                cur_doy = doy_sincos.unsqueeze(0).to(dtype=torch.float32)
            else:
                window = init_states.to(dtype=torch.float32).clone()
                cur_doy = doy_sincos.to(dtype=torch.float32)

            preds_list = []
            for _ in range(int(long_steps)):
                pred = model.step(window, cur_doy)
                preds_list.append(pred)
                window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)
                cur_doy = model.advance_doy_sincos(cur_doy, steps=1)

            # Always [B, long_steps, 6, 32, 36] so channel axes never shift.
            out_batched = torch.stack(preds_list, dim=1).cpu().numpy()
            bsz = out_batched.shape[0]

            validity_errors: List[str] = []
            invalid_samples = 0
            for b in range(bsz):
                errs = validity(out_batched[b], int(long_steps))
                if errs:
                    invalid_samples += 1
                    if unbatched:
                        validity_errors.extend(errs)
                    else:
                        validity_errors.extend(f"sample_{b}: {e}" for e in errs)

            results[f"steps_{long_steps}"] = {
                "num_samples": int(bsz),
                "invalid_samples": int(invalid_samples),
                "validity_errors": validity_errors,
                "all_finite": bool(np.isfinite(out_batched).all()),
                "min_tp": float(out_batched[:, :, 1].min()),
                "min_tcwv": float(out_batched[:, :, 5].min()),
                "max_abs_t2m": float(np.abs(out_batched[:, :, 0]).max()),
                "t2m_range": [
                    float(out_batched[:, :, 0].min()),
                    float(out_batched[:, :, 0].max()),
                ],
            }
    return results


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    stats_path: str | Path,
    nc_path: str | Path,
) -> Tuple[ForkcastResCNN, Dict[str, object]]:
    from src.dataset import inspect_netcdf_metadata

    stats = load_stats(stats_path)
    _, static_fields = inspect_netcdf_metadata(nc_path)
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden_channels", 48))
    model = ForkcastResCNN(
        mean=stats["mean"],
        std=stats["std"],
        static_fields=static_fields,
        hidden_channels=hidden,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate FORKCAST checkpoint")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path outside repo")
    ap.add_argument("--nc", default=r"C:\FORKCAST_DATA\india_climate_train.nc")
    ap.add_argument("--stats", default=r"C:\FORKCAST_DATA\stats.json")
    ap.add_argument("--split", choices=("val", "holdout"), default="val")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None, help="Optional JSON output path outside repo")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    model, stats = load_model_from_checkpoint(args.ckpt, args.stats, args.nc)
    report = evaluate_full(
        model,
        args.nc,
        stats,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    txt = json.dumps(report, indent=2)
    print(txt)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
