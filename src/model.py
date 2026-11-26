"""Compact, strictly-local residual CNN for FORKCAST offline inference and training."""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 6
NLAT = 32
NLON = 36
T_IN = 3
N_STATIC = 2
N_DOY = 2
IN_CHANNELS = T_IN * NC + N_STATIC + N_DOY  # 18 + 2 + 2 = 22
DAYS_PER_YEAR = 365.0
MAX_STEPS = 14
OROGRAPHY_SCALE = 5000.0


class LocalResBlock(nn.Module):
    """Strictly local residual block: one 3x3 replicate-padded conv + 1x1 mixing, no spatial norm."""

    def __init__(self, channels: int, init_res_scale: float = 0.5) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.res_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(init_res_scale), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv3(x))
        h = self.conv1(h)
        return x + self.res_scale * h


class ForkcastResCNN(nn.Module):
    """
    Compact CPU residual CNN predicting daily atmospheric state transitions.

    Architectural guarantees for Round 1 causality and validity:
    - Total 3x3 conv depth per step is 3 (stem + 2 LocalResBlocks), keeping Day-1 shock
      footprints strictly within radius 3 grid cells (100% inside DILATE=3).
    - Zero spatial normalization layers, zero global attention, zero circular padding.
    - Inputs and outputs are always in physical units; internal standardization uses
      the registered stats.json buffers and standardized static field buffers.
    - Physical non-negativity is enforced on precipitation (tp, ch=1) and moisture (tcwv, ch=5).
    """

    def __init__(
        self,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        static_fields: np.ndarray | torch.Tensor,
        hidden_channels: int = 48,
    ) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(1, NC, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(1, NC, 1, 1)
        static_t = torch.as_tensor(static_fields, dtype=torch.float32).reshape(
            1, N_STATIC, NLAT, NLON
        )
        if mean_t.shape != (1, NC, 1, 1) or std_t.shape != (1, NC, 1, 1):
            raise ValueError("mean and std must have 6 channels")
        if static_t.shape != (1, N_STATIC, NLAT, NLON):
            raise ValueError(f"static_fields must have shape (2, {NLAT}, {NLON})")

        # Static preprocessing: land_sea_mask stays in [0,1], orography_m / 5000.
        # This convention is baked into checkpoint buffers and must match at inference.
        static_preprocessed = static_t.clone()
        static_preprocessed[:, 1:2] = static_t[:, 1:2] / OROGRAPHY_SCALE

        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.register_buffer("static_raw", static_t)
        self.register_buffer("static_preprocessed", static_preprocessed)

        self.hidden_channels = int(hidden_channels)
        self.stem = nn.Conv2d(
            IN_CHANNELS,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.act = nn.SiLU()
        self.block1 = LocalResBlock(self.hidden_channels, init_res_scale=0.5)
        self.block2 = LocalResBlock(self.hidden_channels, init_res_scale=0.5)
        self.head_mix = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1)
        self.head_out = nn.Conv2d(self.hidden_channels, NC, kernel_size=1)

        nn.init.normal_(self.head_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head_out.bias)

    def build_input(
        self, window_phys: torch.Tensor, doy_sincos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B, 22, 32, 36] normalized input tensor and [B, 6, 32, 36] last-day normalized state."""
        bsz = window_phys.shape[0]
        if window_phys.shape != (bsz, T_IN, NC, NLAT, NLON):
            raise ValueError(
                f"Expected window_phys shape (B, {T_IN}, {NC}, {NLAT}, {NLON}), "
                f"got {tuple(window_phys.shape)}"
            )
        if doy_sincos.shape != (bsz, N_DOY):
            raise ValueError(
                f"Expected doy_sincos shape (B, {N_DOY}), got {tuple(doy_sincos.shape)}"
            )

        window_norm = (window_phys - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        last_norm = window_norm[:, -1]
        weather_flat = window_norm.reshape(bsz, T_IN * NC, NLAT, NLON)
        static_b = self.static_preprocessed.expand(bsz, -1, -1, -1)
        doy_b = doy_sincos[:, :, None, None].expand(bsz, N_DOY, NLAT, NLON)
        x = torch.cat([weather_flat, static_b, doy_b], dim=1)
        return x, last_norm

    def step(
        self,
        window_phys: torch.Tensor,
        doy_sincos: torch.Tensor,
        return_raw: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict next day's 6 physical weather channels [B, 6, 32, 36]."""
        x, last_norm = self.build_input(window_phys, doy_sincos)
        h = self.act(self.stem(x))
        h = self.block1(h)
        h = self.block2(h)
        delta_norm = self.head_out(self.act(self.head_mix(h)))
        pred_phys_raw = (last_norm + delta_norm) * self.std + self.mean

        tp_nonneg = F.relu(pred_phys_raw[:, 1:2])
        tcwv_nonneg = F.relu(pred_phys_raw[:, 5:6])
        pred_phys = torch.cat(
            [
                pred_phys_raw[:, 0:1],
                tp_nonneg,
                pred_phys_raw[:, 2:5],
                tcwv_nonneg,
            ],
            dim=1,
        )
        if return_raw:
            return pred_phys, pred_phys_raw
        return pred_phys

    @staticmethod
    def advance_doy_sincos(doy_sincos: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Advance [sin, cos] by `steps` days along the 365-day cycle."""
        theta = (2.0 * math.pi / DAYS_PER_YEAR) * float(steps)
        c = math.cos(theta)
        s = math.sin(theta)
        sin_cur = doy_sincos[..., 0]
        cos_cur = doy_sincos[..., 1]
        sin_next = sin_cur * c + cos_cur * s
        cos_next = cos_cur * c - sin_cur * s
        return torch.stack([sin_next, cos_next], dim=-1)

    def rollout(
        self,
        init_states: torch.Tensor,
        doy_sincos: torch.Tensor,
        steps: int,
        perturbation: Optional[torch.Tensor] = None,
        return_raw: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoregressive rollout from physical init_states ([3,6,32,36] or [B,3,6,32,36]).
        `perturbation` ([6,32,36] or [B,6,32,36]) is added once to init_states[-1] in physical units.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if steps > MAX_STEPS:
            raise ValueError(f"steps must be <= {MAX_STEPS} (contract MAX_STEPS), got {steps}")

        unbatched = init_states.ndim == 4
        if unbatched:
            window = init_states.unsqueeze(0).to(dtype=torch.float32).clone()
            cur_doy = doy_sincos.unsqueeze(0).to(dtype=torch.float32)
            pert = (
                None
                if perturbation is None
                else perturbation.unsqueeze(0).to(dtype=torch.float32)
            )
        elif init_states.ndim == 5:
            window = init_states.to(dtype=torch.float32).clone()
            cur_doy = doy_sincos.to(dtype=torch.float32)
            pert = None if perturbation is None else perturbation.to(dtype=torch.float32)
        else:
            raise ValueError(f"init_states must be 4D or 5D, got {tuple(init_states.shape)}")

        if pert is not None:
            if pert.shape != (window.shape[0], NC, NLAT, NLON):
                raise ValueError(
                    f"perturbation shape {tuple(pert.shape)} != {(window.shape[0], NC, NLAT, NLON)}"
                )
            window[:, -1] = window[:, -1] + pert

        preds = []
        raws = []
        for _ in range(int(steps)):
            pred_phys, pred_raw = self.step(window, cur_doy, return_raw=True)
            preds.append(pred_phys)
            raws.append(pred_raw)
            window = torch.cat([window[:, 1:], pred_phys.unsqueeze(1)], dim=1)
            cur_doy = self.advance_doy_sincos(cur_doy, steps=1)

        out_phys = torch.stack(preds, dim=1)
        out_raw = torch.stack(raws, dim=1)
        if unbatched:
            out_phys = out_phys.squeeze(0)
            out_raw = out_raw.squeeze(0)

        if return_raw:
            return out_phys, out_raw
        return out_phys

    def forward(
        self,
        init_states: torch.Tensor,
        doy_sincos: torch.Tensor,
        steps: int = 1,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.rollout(init_states, doy_sincos, steps=steps, perturbation=perturbation)
