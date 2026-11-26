"""Unit and contract verification tests for FORKCAST."""
from __future__ import annotations

import argparse
import datetime as dt
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import NC, NLAT, NLON, T_IN, inspect_netcdf_metadata, load_stats
from src.evaluate import (
    NULL_TOL,
    evaluate_full,
    evaluate_vs_persistence,
    load_model_from_checkpoint,
    make_shock,
    stability_diagnostics,
    validity,
)
from src.model import MAX_STEPS, OROGRAPHY_SCALE, SCORE_STEPS, ForkcastResCNN
from src.train import _can_promote_checkpoint, _save_checkpoint, counterfactual_response_loss

NC_PATH = Path(r"C:\FORKCAST_DATA\india_climate_train.nc")
STATS_PATH = Path(r"C:\FORKCAST_DATA\stats.json")


class _SyntheticForecastDataset(Dataset):
    """Lightweight Dataset matching ForkcastNetCDFDataset's evaluation interface."""

    def __init__(
        self,
        num_windows: int = 12,
        rollout_steps: int = 10,
        base_t2m: float = 296.0,
        tp_value: float = 2.0,
        start_date: dt.date = dt.date(2019, 7, 1),
    ) -> None:
        self.split = "val"
        self.rollout_steps = rollout_steps
        self.num_windows = num_windows
        self.base_t2m = base_t2m
        self.tp_value = tp_value
        self.start_date = start_date

    def set_rollout_steps(self, rollout_steps: int) -> None:
        self.rollout_steps = int(rollout_steps)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        total = T_IN + self.rollout_steps
        series = torch.zeros((total, NC, NLAT, NLON), dtype=torch.float32)
        series[:, 0] = self.base_t2m + 0.1 * (idx % 3)
        series[:, 1] = self.tp_value
        series[:, 2] = 3.0
        series[:, 3] = 1.5
        series[:, 4] = 100900.0
        series[:, 5] = 38.0
        init_states = series[:T_IN]
        targets = series[T_IN:]
        d0 = self.get_anchor_date(idx)
        doy = d0.timetuple().tm_yday
        angle = 2.0 * math.pi * float(doy) / 365.0
        doy_sc = torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32)
        return init_states, doy_sc, targets

    def get_anchor_date(self, idx: int) -> dt.date:
        return self.start_date + dt.timedelta(days=idx)


def _make_dummy_model(hidden_channels: int = 16) -> ForkcastResCNN:
    torch.manual_seed(7)
    mean = np.array([292.0, 3.6, 0.9, 0.2, 101122.0, 30.0], dtype=np.float32)
    std = np.array([13.5, 7.5, 2.9, 2.4, 619.0, 20.5], dtype=np.float32)
    lsm = np.linspace(0.0, 1.0, NLAT * NLON, dtype=np.float32).reshape(NLAT, NLON)
    oro = np.linspace(-0.2, 5200.0, NLAT * NLON, dtype=np.float32).reshape(NLAT, NLON)
    static_fields = np.stack([lsm, oro], axis=0)
    return ForkcastResCNN(mean=mean, std=std, static_fields=static_fields, hidden_channels=hidden_channels)


def _sample_inputs(batch_size: int = 2):
    base = torch.tensor([296.0, 4.0, 2.5, 1.2, 100800.0, 35.0], dtype=torch.float32)
    init_4d = base[None, :, None, None].expand(T_IN, NC, NLAT, NLON).clone()
    doy_1d = torch.tensor([math.sin(1.2), math.cos(1.2)], dtype=torch.float32)
    init_5d = init_4d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
    if batch_size > 1:
        init_5d[1, :, 0] -= 2.0
    doy_2d = doy_1d.unsqueeze(0).expand(batch_size, -1).clone()
    return init_4d, doy_1d, init_5d, doy_2d


class TestForkcastFixesAndContracts(unittest.TestCase):
    def test_static_buffer_values_synthetic_and_netcdf(self) -> None:
        self.assertEqual(OROGRAPHY_SCALE, 5000.0)
        model = _make_dummy_model()
        raw = model.static_raw
        prep = model.static_preprocessed
        self.assertTrue(torch.equal(prep[:, 0], raw[:, 0]))
        self.assertGreaterEqual(float(prep[:, 0].min()), 0.0)
        self.assertLessEqual(float(prep[:, 0].max()), 1.0)
        self.assertTrue(torch.allclose(prep[:, 1], raw[:, 1] / 5000.0, rtol=1e-6, atol=1e-6))

        if NC_PATH.exists() and STATS_PATH.exists():
            stats = load_stats(STATS_PATH)
            _, nc_static = inspect_netcdf_metadata(NC_PATH)
            nc_model = ForkcastResCNN(stats["mean"], stats["std"], nc_static, hidden_channels=16)
            self.assertTrue(
                torch.allclose(nc_model.static_preprocessed[0, 0], torch.from_numpy(nc_static[0]))
            )
            self.assertGreaterEqual(float(nc_model.static_preprocessed[:, 0].min()), 0.0)
            self.assertLessEqual(float(nc_model.static_preprocessed[:, 0].max()), 1.0)
            self.assertTrue(
                torch.allclose(
                    nc_model.static_preprocessed[0, 1],
                    torch.from_numpy(nc_static[1] / 5000.0),
                    rtol=1e-6,
                    atol=1e-6,
                )
            )

    def test_checkpoint_save_load_preserves_static_buffers(self) -> None:
        if not (NC_PATH.exists() and STATS_PATH.exists()):
            self.skipTest("Dataset/stats not present on disk")
        stats = load_stats(STATS_PATH)
        _, nc_static = inspect_netcdf_metadata(NC_PATH)
        model = ForkcastResCNN(stats["mean"], stats["std"], nc_static, hidden_channels=16)
        init_4d, doy_1d, _, _ = _sample_inputs()

        with torch.no_grad():
            ref_out = model.rollout(init_4d, doy_1d, steps=SCORE_STEPS)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test_ckpt.pt"
            _save_checkpoint(
                ckpt_path,
                model,
                epoch=1,
                stage="unit_test",
                val_metrics={"ok": True},
                args=argparse.Namespace(hidden=16),
            )
            loaded_model, _ = load_model_from_checkpoint(ckpt_path, STATS_PATH, NC_PATH)
            for buf_name in ("mean", "std", "static_raw", "static_preprocessed"):
                self.assertTrue(
                    torch.equal(getattr(model, buf_name), getattr(loaded_model, buf_name)),
                    msg=f"Buffer mismatch after checkpoint reload: {buf_name}",
                )
            with torch.no_grad():
                loaded_out = loaded_model.rollout(init_4d, doy_1d, steps=SCORE_STEPS)
            self.assertTrue(torch.equal(ref_out, loaded_out))

    def test_max_steps_limits(self) -> None:
        self.assertEqual(MAX_STEPS, 14)
        self.assertEqual(SCORE_STEPS, 10)
        model = _make_dummy_model()
        model.eval()
        init_4d, doy_1d, init_5d, doy_2d = _sample_inputs()

        with torch.no_grad():
            out_4d = model.rollout(init_4d, doy_1d, steps=14)
            out_5d = model.rollout(init_5d, doy_2d, steps=14)
        self.assertEqual(tuple(out_4d.shape), (14, NC, NLAT, NLON))
        self.assertEqual(tuple(out_5d.shape), (2, 14, NC, NLAT, NLON))
        self.assertTrue(torch.isfinite(out_4d).all())
        self.assertTrue(torch.isfinite(out_5d).all())

        for bad_steps in (0, 15, 20, 30):
            with self.assertRaises(ValueError):
                model.rollout(init_4d, doy_1d, steps=bad_steps)

    def test_perturbation_applied_exactly_once_in_physical_units(self) -> None:
        model = _make_dummy_model()
        model.eval()
        init_4d, doy_1d, _, _ = _sample_inputs()
        pert = torch.zeros((NC, NLAT, NLON), dtype=torch.float32)
        pert[0, 10:15, 10:15] = 3.0
        pert[5, 10:15, 10:15] = 5.25

        # 1. Spy on step() inputs to confirm pert appears in init window step 0 only
        seen_windows = []
        orig_step = model.step

        def spy_step(window_phys, cur_doy, return_raw=False):
            seen_windows.append(window_phys.detach().clone())
            return orig_step(window_phys, cur_doy, return_raw=return_raw)

        with mock.patch.object(model, "step", side_effect=spy_step):
            with torch.no_grad():
                out = model.rollout(init_4d, doy_1d, steps=4, perturbation=pert)

        self.assertEqual(len(seen_windows), 4)
        self.assertTrue(torch.allclose(seen_windows[0][0, -1], init_4d[-1] + pert, atol=1e-6))
        self.assertTrue(torch.allclose(seen_windows[1][0, -1], out[0], atol=1e-6))

        # 2. Zero residual weights -> pure identity dynamics proves no cumulative pert addition
        zero_res_model = _make_dummy_model()
        with torch.no_grad():
            zero_res_model.head_out.weight.zero_()
            zero_res_model.head_out.bias.zero_()
            out_zero = zero_res_model.rollout(init_4d, doy_1d, steps=4, perturbation=pert)
        expected_last = init_4d[-1] + pert
        for k in range(4):
            self.assertTrue(
                torch.allclose(out_zero[k], expected_last, atol=1e-5),
                msg=f"Step {k} diverged from single-application delta",
            )

    def test_physical_validity_and_null_shock_tolerance(self) -> None:
        model = _make_dummy_model()
        model.eval()
        init_4d, doy_1d, _, _ = _sample_inputs()
        std_np = model.std.reshape(NC).numpy()

        zeros = torch.zeros((NC, NLAT, NLON), dtype=torch.float32)
        warm = torch.from_numpy(make_shock("warm_ocean", init_4d[-1].numpy()))
        stall = torch.from_numpy(make_shock("monsoon_stall", init_4d[-1].numpy()))

        with torch.no_grad():
            base = model.rollout(init_4d, doy_1d, steps=SCORE_STEPS).numpy()
            null_out = model.rollout(init_4d, doy_1d, steps=SCORE_STEPS, perturbation=zeros).numpy()
            warm_out = model.rollout(init_4d, doy_1d, steps=SCORE_STEPS, perturbation=warm).numpy()
            stall_out = model.rollout(init_4d, doy_1d, steps=SCORE_STEPS, perturbation=stall).numpy()

        null_diff_std = float((np.abs(null_out - base) / std_np[None, :, None, None]).max())
        self.assertLessEqual(null_diff_std, NULL_TOL)

        for name, arr in (
            ("base", base),
            ("null", null_out),
            ("warm_ocean", warm_out),
            ("monsoon_stall", stall_out),
        ):
            self.assertTrue(np.isfinite(arr).all(), msg=f"Non-finite values in {name}")
            self.assertEqual(validity(arr, SCORE_STEPS), [], msg=f"Validity failed in {name}")
            self.assertGreaterEqual(float(arr[:, 0].min()), 233.15)
            self.assertLessEqual(float(arr[:, 0].max()), 333.15)
            self.assertGreaterEqual(float(arr[:, 1].min()), 0.0)
            self.assertGreaterEqual(float(arr[:, 5].min()), 0.0)

    def test_checkpoint_promotion_requires_both_propagating_shocks(self) -> None:
        good = {
            "day4_ss_vs_persist": 0.01,
            "invalid_windows": 0,
            "shock_validity_ok": True,
            "null_shock_ok": True,
            "warm_ocean": {"nonzero": True, "localized": True, "propagates": True, "stable": True},
            "monsoon_stall": {"nonzero": True, "localized": True, "propagates": True, "stable": True},
        }
        self.assertTrue(_can_promote_checkpoint(good, 10.0, 9.0))

        for failed_gate in ("nonzero", "localized", "propagates", "stable"):
            candidate = {**good, "warm_ocean": {**good["warm_ocean"], failed_gate: False}}
            self.assertFalse(_can_promote_checkpoint(candidate, 10.0, 9.0))
            candidate = {**good, "monsoon_stall": {**good["monsoon_stall"], failed_gate: False}}
            self.assertFalse(_can_promote_checkpoint(candidate, 10.0, 9.0))

        self.assertFalse(_can_promote_checkpoint({**good, "null_shock_ok": False}, 10.0, 9.0))
        self.assertFalse(_can_promote_checkpoint({**good, "invalid_windows": 1}, 10.0, 9.0))
        self.assertFalse(_can_promote_checkpoint({**good, "day4_ss_vs_persist": -0.01}, 10.0, 9.0))
        self.assertFalse(_can_promote_checkpoint(good, 8.0, 9.0))

    def test_counterfactual_response_loss_is_finite_and_differentiable(self) -> None:
        model = _make_dummy_model()
        init_4d, doy_1d, init_5d, doy_2d = _sample_inputs(batch_size=1)

        for shock_name in ("warm_ocean", "monsoon_stall"):
            model.zero_grad(set_to_none=True)
            baseline = model.rollout(init_5d, doy_2d, steps=10)
            loss = counterfactual_response_loss(
                model, init_5d, doy_2d, baseline, shock_name
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(loss.requires_grad)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_stability_diagnostics_batched_and_unbatched(self) -> None:
        model = _make_dummy_model()
        init_4d, doy_1d, init_5d, doy_2d = _sample_inputs(batch_size=2)

        res_unbatched = stability_diagnostics(model, init_4d, doy_1d, horizons=(20, 30))
        res_batched = stability_diagnostics(model, init_5d, doy_2d, horizons=(20, 30))

        for h in (20, 30):
            key = f"steps_{h}"
            u = res_unbatched[key]
            b = res_batched[key]
            self.assertEqual(u["num_samples"], 1)
            self.assertEqual(b["num_samples"], 2)
            self.assertEqual(u["invalid_samples"], 0)
            self.assertEqual(b["invalid_samples"], 0)
            self.assertEqual(u["validity_errors"], [])
            self.assertEqual(b["validity_errors"], [])
            self.assertTrue(u["all_finite"] and b["all_finite"])
            self.assertGreaterEqual(u["min_tp"], 0.0)
            self.assertGreaterEqual(b["min_tp"], 0.0)
            self.assertGreaterEqual(u["min_tcwv"], 0.0)
            self.assertGreaterEqual(b["min_tcwv"], 0.0)
            # Channel 0 is t2m (~294-296 K), catching any step/channel axis confusion
            self.assertGreater(b["t2m_range"][0], 233.15)
            self.assertLess(b["t2m_range"][1], 333.15)

        # Ensure sample-level invalid detection works across batches when sample 1 exceeds 333.15 K
        bad_init_5d = init_5d.clone()
        bad_init_5d[1, :, 0] = 340.0
        bad_res = stability_diagnostics(model, bad_init_5d, doy_2d, horizons=(20,))
        self.assertGreaterEqual(bad_res["steps_20"]["invalid_samples"], 1)
        self.assertTrue(any(err.startswith("sample_1:") for err in bad_res["steps_20"]["validity_errors"]))

    def test_evaluate_vs_persistence_all_horizons(self) -> None:
        model = _make_dummy_model()
        ds = _SyntheticForecastDataset(num_windows=4, rollout_steps=10)

        for steps, expect_leads in (
            (1, ["day_1"]),
            (4, ["day_1", "day_4"]),
            (8, ["day_1", "day_4", "day_8"]),
            (10, ["day_1", "day_4", "day_8"]),
        ):
            out = evaluate_vs_persistence(model, ds, steps=steps, batch_size=2)
            self.assertEqual(out["steps"], steps)
            self.assertEqual(list(out["leads"].keys()), expect_leads)
            self.assertEqual(out["invalid_windows"], 0)
            self.assertTrue(math.isfinite(out["mean_ss_vs_persist"]))
            if steps < 4:
                self.assertIsNone(out["day4_ss_vs_persist"])
            else:
                self.assertIsNotNone(out["day4_ss_vs_persist"])

    def test_evaluate_full_nan_csi_guard_and_shock_validity(self) -> None:
        model = _make_dummy_model()
        # No events > 10 mm/day -> both persistence & climatology CSI are NaN
        dry_ds = _SyntheticForecastDataset(num_windows=6, rollout_steps=10, tp_value=1.0)
        clim = np.zeros((366, NC, NLAT, NLON), dtype=np.float32)
        clim[:, 0] = 296.0
        clim[:, 1] = 1.0
        clim[:, 2] = 3.0
        clim[:, 3] = 1.5
        clim[:, 4] = 100900.0
        clim[:, 5] = 38.0
        stats = {
            "mean": model.mean.reshape(NC).numpy(),
            "std": model.std.reshape(NC).numpy(),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rep = evaluate_full(
                model,
                nc_path="unused.nc",
                stats=stats,
                split="val",
                steps=SCORE_STEPS,
                batch_size=3,
                climatology=clim,
                dataset=dry_ds,
            )
        nan_warnings = [w for w in caught if "All-NaN" in str(w.message)]
        self.assertEqual(nan_warnings, [])
        self.assertIsNone(rep["mean_csi_skill"])
        self.assertTrue(math.isnan(rep["leads"]["day_4"]["csi_skill"]))
        self.assertEqual(set(rep["shock_validity"].keys()), {"baseline", "warm_ocean", "monsoon_stall", "null"})
        self.assertTrue(rep["shock_validity_ok"])
        self.assertEqual(len(rep["per_channel_min"]), NC)
        self.assertEqual(len(rep["per_channel_max"]), NC)

        # Force warm_ocean shock to exceed 333.15 K and confirm shock_validity catches it
        hot_ds = _SyntheticForecastDataset(num_windows=4, rollout_steps=10, base_t2m=332.0, tp_value=1.0)
        hot_rep = evaluate_full(
            model,
            nc_path="unused.nc",
            stats=stats,
            split="val",
            steps=SCORE_STEPS,
            batch_size=2,
            climatology=clim,
            dataset=hot_ds,
        )
        self.assertFalse(hot_rep["shock_validity_ok"])
        self.assertTrue(len(hot_rep["shock_validity"]["warm_ocean"]) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
