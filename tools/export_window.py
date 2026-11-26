"""Export a real three-day input window for the local dashboard.

Example:
  python tools/export_window.py --nc C:\FORKCAST_DATA\india_climate_train.nc --last-day 2021-08-15 --out C:\FORKCAST_DATA\demo_window.json

The JSON output must stay outside the Git repository. It contains only the selected input
window, not a historical archive or forecast lookup table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import netCDF4
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True, help="Path to india_climate_train.nc")
    parser.add_argument("--last-day", required=True, help="Last input day, YYYY-MM-DD")
    parser.add_argument("--out", required=True, help="Output JSON path; keep outside the repository")
    args = parser.parse_args()

    last_day = dt.date.fromisoformat(args.last_day)
    with netCDF4.Dataset(args.nc, "r") as ds:
        if "time" not in ds.variables or "data" not in ds.variables:
            raise ValueError("NetCDF must contain time and data variables")
        times = np.asarray(ds.variables["time"][:])
        units = getattr(ds.variables["time"], "units", None)
        calendar = getattr(ds.variables["time"], "calendar", "standard")
        if not units:
            raise ValueError("NetCDF time coordinate has no CF units")
        dates = netCDF4.num2date(
            times, units=units, calendar=calendar,
            only_use_cftime_datetimes=False, only_use_python_datetimes=True,
        )
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}
        required = [last_day - dt.timedelta(days=2), last_day - dt.timedelta(days=1), last_day]
        missing = [d.isoformat() for d in required if d not in date_to_idx]
        if missing:
            raise ValueError("Input dates not present in dataset: " + ", ".join(missing))
        indices = [date_to_idx[d] for d in required]
        if indices != list(range(indices[0], indices[0] + 3)):
            raise ValueError("The selected input days are not consecutive")
        window = np.asarray(ds.variables["data"][indices, :, :, :], dtype=np.float32)

    if window.shape != (3, 6, 32, 36) or not np.isfinite(window).all():
        raise ValueError(f"Invalid selected input window: shape={window.shape}")
    doy = last_day.timetuple().tm_yday
    angle = 2.0 * np.pi * doy / 365.0
    payload = {
        "init_states": window.tolist(),
        "doy_sincos": [float(np.sin(angle)), float(np.cos(angle))],
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "last_input_day": last_day.isoformat(),
        "shape": list(window.shape),
        "doy_sincos": payload["doy_sincos"],
        "bytes": out_path.stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
