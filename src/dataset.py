"""Lazy/indexed NetCDF dataset and chronological split utilities for FORKCAST."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

CHANNELS: Tuple[str, ...] = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
STATIC_CHANNELS: Tuple[str, ...] = ("land_sea_mask", "orography_m")
NC = 6
NLAT = 32
NLON = 36
T_IN = 3
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float64)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0, dtype=np.float64)

SPLIT_RANGES: Dict[str, Tuple[dt.date, dt.date]] = {
    "train": (dt.date(2005, 1, 1), dt.date(2018, 12, 31)),
    "val": (dt.date(2019, 1, 1), dt.date(2020, 12, 31)),
    "holdout": (dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
}


def load_stats(stats_path: str | Path) -> Dict[str, object]:
    """Load read-only channel normalization statistics from stats.json."""
    path = Path(stats_path)
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    if tuple(stats.get("channels", ())) != CHANNELS:
        raise ValueError(f"stats.json channels {stats.get('channels')} != expected {CHANNELS}")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (NC,) or std.shape != (NC,):
        raise ValueError("stats.json mean/std must have length 6")
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("stats.json mean/std must be finite with positive std")
    return {"channels": list(CHANNELS), "mean": mean, "std": std}


def doy_sincos_from_date(d: dt.date) -> np.ndarray:
    """Compute [sin(2*pi*doy/365.0), cos(2*pi*doy/365.0)] matching monsoon_r1.contract."""
    doy = d.timetuple().tm_yday
    angle = 2.0 * np.pi * float(doy) / 365.0
    return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)


def inspect_netcdf_metadata(nc_path: str | Path) -> Tuple[List[dt.date], np.ndarray]:
    """Validate NetCDF schema lazily using xarray metadata and extract dates + static [2,32,36]."""
    path = Path(nc_path)
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if tuple(ds.channel.values.tolist()) != CHANNELS:
            raise ValueError(f"Unexpected channels: {ds.channel.values.tolist()}")
        if tuple(ds.static_channel.values.tolist()) != STATIC_CHANNELS:
            raise ValueError(f"Unexpected static_channel: {ds.static_channel.values.tolist()}")
        data_shape = tuple(ds["data"].shape)
        if len(data_shape) != 4 or data_shape[1:] != (NC, NLAT, NLON):
            raise ValueError(f"Unexpected data shape: {data_shape}")
        if tuple(ds["static"].shape) != (2, NLAT, NLON):
            raise ValueError(f"Unexpected static shape: {ds['static'].shape}")
        if not np.allclose(ds["lat"].values, LAT, atol=1e-6):
            raise ValueError("Dataset lat coordinate mismatch")
        if not np.allclose(ds["lon"].values, LON, atol=1e-6):
            raise ValueError("Dataset lon coordinate mismatch")

        time_vals = ds["time"].values
        diffs = np.diff(time_vals).astype("timedelta64[D]").astype(int)
        if not np.all(diffs == 1):
            raise ValueError("Dataset time axis has non-daily gaps")

        dates = [
            dt.date.fromisoformat(str(np.datetime_as_string(t, unit="D")))
            for t in time_vals
        ]
        static = np.asarray(ds["static"].values, dtype=np.float32).copy()
        if not np.isfinite(static).all():
            raise ValueError("Static fields contain non-finite values")
    return dates, static


class ForkcastNetCDFDataset(Dataset):
    """Memory-efficient PyTorch Dataset backed by lazy indexed netCDF4 reads."""

    def __init__(
        self,
        nc_path: str | Path,
        split: str,
        rollout_steps: int = 1,
        input_days: int = T_IN,
        chunk_cache_mb: int = 192,
    ) -> None:
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {tuple(SPLIT_RANGES)}, got {split!r}")
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        if input_days != T_IN:
            raise ValueError(f"input_days must equal contract T_IN={T_IN}, got {input_days}")

        self.nc_path = str(Path(nc_path))
        self.split = split
        self.input_days = input_days
        self.chunk_cache_bytes = int(chunk_cache_mb) * 1024 * 1024

        self.dates, self.static = inspect_netcdf_metadata(self.nc_path)
        self.doy_sincos_all = np.stack(
            [doy_sincos_from_date(d) for d in self.dates], axis=0
        ).astype(np.float32)

        start_date, end_date = SPLIT_RANGES[split]
        split_idx = [
            i for i, d in enumerate(self.dates) if start_date <= d <= end_date
        ]
        if not split_idx:
            raise ValueError(f"No dates found for split {split!r}")
        if split_idx[-1] - split_idx[0] + 1 != len(split_idx):
            raise ValueError(f"Split {split!r} indices are not contiguous")

        self.split_start = split_idx[0]
        self.split_end = split_idx[-1]
        self.split_days = len(split_idx)

        self.rollout_steps = 1
        self.anchors = np.empty(0, dtype=np.int64)
        self.set_rollout_steps(rollout_steps)

        self._nc: Optional[netCDF4.Dataset] = None
        self._data_var = None

    def set_rollout_steps(self, rollout_steps: int) -> None:
        """Update forecast horizon while strictly enforcing split boundaries."""
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        first_anchor = self.split_start + (self.input_days - 1)
        last_anchor = self.split_end - rollout_steps
        if last_anchor < first_anchor:
            raise ValueError(
                f"Split {self.split!r} ({self.split_days} days) too short for "
                f"input_days={self.input_days} + rollout_steps={rollout_steps}"
            )
        self.rollout_steps = int(rollout_steps)
        self.anchors = np.arange(first_anchor, last_anchor + 1, dtype=np.int64)

    def _get_var(self):
        if self._nc is None:
            nc = netCDF4.Dataset(self.nc_path, "r")
            var = nc.variables["data"]
            var.set_var_chunk_cache(
                size=self.chunk_cache_bytes, nelems=521, preemption=0.75
            )
            self._nc = nc
            self._data_var = var
        return self._data_var

    def close(self) -> None:
        if self._nc is not None:
            self._nc.close()
            self._nc = None
            self._data_var = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(len(self.anchors))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = int(self.anchors[idx])
        t_start = anchor - (self.input_days - 1)
        t_end = anchor + self.rollout_steps + 1
        if t_start < self.split_start or (t_end - 1) > self.split_end:
            raise IndexError(
                f"Window [{t_start}, {t_end}) violates split {self.split!r} "
                f"[{self.split_start}, {self.split_end}]"
            )

        var = self._get_var()
        slab = np.asarray(var[t_start:t_end, :, :, :], dtype=np.float32)
        init_states = np.ascontiguousarray(slab[: self.input_days], dtype=np.float32)
        targets = np.ascontiguousarray(slab[self.input_days :], dtype=np.float32)
        doy_sc = np.ascontiguousarray(self.doy_sincos_all[anchor], dtype=np.float32)

        return (
            torch.from_numpy(init_states),
            torch.from_numpy(doy_sc),
            torch.from_numpy(targets),
        )

    def get_anchor_date(self, idx: int) -> dt.date:
        return self.dates[int(self.anchors[idx])]


def _doy_slot(d: dt.date) -> int:
    """0..365 leap-aware slot matching monsoon_r1.stats._doy_index."""
    ref = dt.date(2000, d.month, d.day)
    return (ref - dt.date(2000, 1, 1)).days


def compute_train_climatology(
    nc_path: str | Path,
    split: str = "train",
    smooth_days: int = 15,
    block_days: int = 365,
    chunk_cache_mb: int = 192,
) -> np.ndarray:
    """Compute [366,6,32,36] smoothed DOY climatology strictly from one split using chunked reads."""
    if split != "train":
        raise ValueError("Climatology baseline must only be computed from 'train' split")
    dates, _ = inspect_netcdf_metadata(nc_path)
    start_date, end_date = SPLIT_RANGES[split]
    split_idx = [i for i, d in enumerate(dates) if start_date <= d <= end_date]
    i0, i1 = split_idx[0], split_idx[-1] + 1

    clim = np.zeros((366, NC, NLAT, NLON), dtype=np.float64)
    cnt = np.zeros(366, dtype=np.float64)

    with netCDF4.Dataset(str(nc_path), "r") as nc:
        var = nc.variables["data"]
        var.set_var_chunk_cache(size=int(chunk_cache_mb) * 1024 * 1024, nelems=521, preemption=0.75)
        for pos in range(i0, i1, block_days):
            end = min(pos + block_days, i1)
            slab = np.asarray(var[pos:end, :, :, :], dtype=np.float32)
            for offset, t_idx in enumerate(range(pos, end)):
                slot = _doy_slot(dates[t_idx])
                clim[slot] += slab[offset]
                cnt[slot] += 1.0

    clim /= np.maximum(cnt, 1.0)[:, None, None, None]
    half = smooth_days // 2
    out = np.zeros_like(clim, dtype=np.float32)
    for k in range(366):
        out[k] = np.mean(
            [clim[(k + o) % 366] for o in range(-half, half + 1)], axis=0
        ).astype(np.float32)
    return out
