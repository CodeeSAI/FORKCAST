"""FORKCAST offline inference server.

Run: python serve.py --port 8000
The server becomes ready only after a real trained checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep the submission checkout read-only: disable bytecode and cap CPU threads.
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.model import ForkcastResCNN, MAX_STEPS

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "model.pt"
CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
MODEL_NAME = "FORKCAST-ResCNN"
MODEL: ForkcastResCNN | None = None
LOAD_ERROR: str | None = None

app = FastAPI(title="FORKCAST", docs_url=None, redoc_url=None)
dashboard_dir = ROOT / "dashboard"
if dashboard_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir)), name="dashboard")


def load_checkpoint(path: Path) -> ForkcastResCNN:
    """Reconstruct the model exclusively from checkpoint-embedded buffers and weights."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint missing: {path}. Place the selected trained checkpoint at weights/model.pt."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict.")
    state = checkpoint["model_state_dict"]
    required = ("mean", "std", "static_raw")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Checkpoint is missing embedded model buffers: {missing}")

    mean = state["mean"].detach().cpu().reshape(6).numpy()
    std = state["std"].detach().cpu().reshape(6).numpy()
    static_raw = state["static_raw"].detach().cpu().numpy()
    if static_raw.shape == (1, 2, 32, 36):
        static_raw = static_raw[0]
    if static_raw.shape != (2, 32, 36):
        raise ValueError(f"Invalid embedded static fields shape: {static_raw.shape}")

    hidden = int(checkpoint.get("hidden_channels", 48))
    model = ForkcastResCNN(
        mean=mean,
        std=std,
        static_fields=static_raw,
        hidden_channels=hidden,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@app.on_event("startup")
def startup_load_model() -> None:
    global MODEL, LOAD_ERROR
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    weights_path = Path(os.environ.get("FORKCAST_WEIGHTS", str(DEFAULT_WEIGHTS)))
    try:
        MODEL = load_checkpoint(weights_path)
        LOAD_ERROR = None
        print(f"[FORKCAST] Loaded trained model from {weights_path}", flush=True)
    except Exception as exc:
        MODEL = None
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[FORKCAST] NOT READY: {LOAD_ERROR}", file=sys.stderr, flush=True)


@app.post("/health")
def health() -> Any:
    if MODEL is None:
        raise HTTPException(status_code=503, detail={"status": "loading_or_missing_weights", "reason": LOAD_ERROR})
    return {"status": "ready", "model_name": MODEL_NAME}


def _as_float32(value: Any, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        if isinstance(value, dict):
            if "b64" not in value or "shape" not in value:
                raise ValueError("encoded array needs b64 and shape")

            shape = value["shape"]
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("invalid encoded array shape")

            if tuple(shape) != expected_shape:
                raise ValueError(f"shape {tuple(shape)} does not match {expected_shape}")

            encoded = value["b64"]
            if not isinstance(encoded, str):
                raise ValueError("b64 must be a string")

            raw = base64.b64decode(encoded, validate=True)
            expected_bytes = int(np.prod(expected_shape)) * 4
            if len(raw) != expected_bytes:
                raise ValueError("base64 data length does not match shape")

            arr = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
        else:
            arr = np.asarray(value, dtype=np.float32)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be a valid numeric array: {exc}",
        ) from exc

    if arr.shape != expected_shape:
        raise HTTPException(
            status_code=422,
            detail=f"{name} shape {arr.shape} does not match required {expected_shape}",
        )

    if not np.isfinite(arr).all():
        raise HTTPException(status_code=422, detail=f"{name} must contain only finite values")

    return np.ascontiguousarray(arr, dtype=np.float32)


@app.post("/simulate")
async def simulate(request: Request) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": LOAD_ERROR})
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")

    raw_steps = body.get("steps")
    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
        raise HTTPException(status_code=422, detail="steps must be an integer")
    steps = raw_steps
    if steps < 1 or steps > MAX_STEPS:
        raise HTTPException(status_code=422, detail=f"steps must be between 1 and {MAX_STEPS}")

    init = _as_float32(body.get("init_states"), "init_states", (3, 6, 32, 36))
    doy = _as_float32(body.get("doy_sincos"), "doy_sincos", (2,))
    raw_pert = body.get("perturbation")
    perturbation = None if raw_pert is None else _as_float32(raw_pert, "perturbation", (6, 32, 36))

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            init_t = torch.from_numpy(init)
            doy_t = torch.from_numpy(doy)
            pert_t = None if perturbation is None else torch.from_numpy(perturbation)
            output = MODEL.rollout(
                init_t, doy_t, steps=steps, perturbation=pert_t
            ).cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {type(exc).__name__}: {exc}") from exc
    inference_ms = (time.perf_counter() - t0) * 1000.0

    if output.shape != (steps, 6, 32, 36) or not np.isfinite(output).all():
        raise HTTPException(status_code=500, detail="Model produced invalid shape or non-finite values")
    if float(output[:, 0].min()) < 233.15 or float(output[:, 0].max()) > 333.15:
        raise HTTPException(status_code=500, detail="Model temperature output is outside [-40C, +60C]")
    if float(output[:, 1].min()) < 0.0 or float(output[:, 5].min()) < 0.0:
        raise HTTPException(status_code=500, detail="Model produced negative precipitation or TCWV")

    return {"rollout": output.tolist(), "inference_ms": float(inference_ms)}


@app.get("/")
def dashboard_home() -> Any:
    index_path = dashboard_dir / "index.html"
    if not index_path.is_file():
        return {"status": "ok", "service": "FORKCAST", "dashboard": "not installed"}
    return FileResponse(index_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline FORKCAST inference server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()
    os.environ["FORKCAST_WEIGHTS"] = args.weights
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
