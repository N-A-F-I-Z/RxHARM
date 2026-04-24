"""
tests/test_uncertainty.py
==========================
Tests for rxharm/uncertainty/ — MCUncertaintyEngine, BayesianCalibrator,
MorrisScreener, and rxharm/validation.py.
All pass without GEE or network access.
"""

from __future__ import annotations
import numpy as np
import pytest

SHAPE = (20, 20)


def _make_arrays(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "lst":           rng.uniform(28, 48,   SHAPE),
        "albedo":        rng.uniform(0.05, 0.35, SHAPE),
        "uhi":           rng.uniform(0, 8,    SHAPE),
        "population":    rng.uniform(100, 5000, SHAPE),
        "built_frac":    rng.uniform(0, 1,    SHAPE),
        "elderly_frac":  rng.uniform(0, 0.25, SHAPE),
        "child_frac":    rng.uniform(0, 0.15, SHAPE),
        "impervious":    rng.uniform(0, 1,    SHAPE),
        "cropland":      rng.uniform(0, 0.5,  SHAPE),
        "ndvi":          rng.uniform(0.1, 0.7, SHAPE),
        "ndwi":          rng.uniform(-0.2, 0.4, SHAPE),
        "tree_cover":    rng.uniform(0, 60,   SHAPE),
        "canopy_height": rng.uniform(0, 20,   SHAPE),
        "viirs_dnb":     rng.uniform(0, 50,   SHAPE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MCUncertaintyEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestMCUncertaintyEngine:

    @pytest.fixture
    def eng(self):
        from rxharm.uncertainty.monte_carlo import MCUncertaintyEngine
        return MCUncertaintyEngine(n_samples=20, random_seed=42)

    def test_perturb_returns_n_samples(self, eng):
        arrays = _make_arrays()
        samples = eng.perturb_indicators(arrays)
        assert len(samples) == 20

    def test_perturbed_samples_have_same_keys(self, eng):
        arrays  = _make_arrays()
        samples = eng.perturb_indicators(arrays)
        assert set(samples[0].keys()) == set(arrays.keys())

    def test_perturbed_samples_differ_from_original(self, eng):
        arrays  = _make_arrays()
        samples = eng.perturb_indicators(arrays)
        diffs = [np.mean(np.abs(samples[i]["lst"] - arrays["lst"]))
                 for i in range(5)]
        assert any(d > 0 for d in diffs)

    def test_no_nan_in_perturbed_output(self, eng):
        arrays  = _make_arrays()
        samples = eng.perturb_indicators(arrays)
        for s in samples[:5]:
            for name, arr in s.items():
                assert not np.any(np.isnan(arr)), f"NaN in {name}"

    def test_hvi_distribution_has_correct_keys(self, eng):
        from rxharm.index.hvi import HVIEngine
        arrays = _make_arrays()
        engine = HVIEngine()
        result = eng.compute_hvi_distribution(arrays, engine)
        for k in ("p10", "p50", "p90", "mean", "std", "confidence_width"):
            assert k in result

    def test_hvi_p10_le_p50_le_p90(self, eng):
        from rxharm.index.hvi import HVIEngine
        arrays = _make_arrays()
        result = eng.compute_hvi_distribution(arrays, HVIEngine())
        assert np.nanmean(result["p10"]) <= np.nanmean(result["p50"])
        assert np.nanmean(result["p50"]) <= np.nanmean(result["p90"])

    def test_confidence_width_nonnegative(self, eng):
        from rxharm.index.hvi import HVIEngine
        arrays = _make_arrays()
        result = eng.compute_hvi_distribution(arrays, HVIEngine())
        assert np.all(result["confidence_width"] >= -1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# BayesianCalibrator
# ══════════════════════════════════════════════════════════════════════════════

class TestBayesianCalibrator:

    @pytest.fixture
    def cal(self):
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        lst = np.random.uniform(28, 45, SHAPE)
        return BayesianCalibrator("A", lst)

    def test_calibrate_lambda_returns_dict(self, cal):
        result = cal.calibrate_lambda()
        assert isinstance(result, dict)

    def test_calibrate_lambda_returns_required_keys(self, cal):
        result = cal.calibrate_lambda()
        for k in ("lambda_map", "lambda_ci_low", "lambda_ci_high", "used_mcc_data"):
            assert k in result

    def test_lambda_within_prior_range(self, cal):
        from rxharm.config import LAMBDA_HVI_RANGE
        result = cal.calibrate_lambda()
        lo, hi = LAMBDA_HVI_RANGE
        assert lo <= result["lambda_map"] <= hi

    def test_no_mcc_data_uses_prior(self, cal):
        result = cal.calibrate_lambda()
        assert result["used_mcc_data"] is False

    def test_calibrate_mmt_returns_dict(self, cal):
        result = cal.calibrate_mmt(era5_p75=35.0)
        for k in ("mmt_posterior_mean", "mmt_posterior_sd", "mmt_95ci"):
            assert k in result

    def test_mmt_posterior_mean_near_prior(self, cal):
        result = cal.calibrate_mmt(era5_p75=35.0)
        assert abs(result["mmt_posterior_mean"] - 35.0) < 1.0

    def test_mmt_95ci_spans_prior(self, cal):
        result = cal.calibrate_mmt(era5_p75=35.0)
        lo, hi = result["mmt_95ci"]
        assert lo < 35.0 < hi


# ══════════════════════════════════════════════════════════════════════════════
# MorrisScreener
# ══════════════════════════════════════════════════════════════════════════════

class TestMorrisScreener:

    @pytest.fixture
    def screener(self):
        from rxharm.index.hvi import HVIEngine
        from rxharm.uncertainty.morris_screening import MorrisScreener
        return MorrisScreener(_make_arrays(), HVIEngine())

    def test_screen_returns_dataframe(self, screener):
        import pandas as pd
        df = screener.screen(n_trajectories=2)
        assert isinstance(df, pd.DataFrame)

    def test_screen_has_required_columns(self, screener):
        df = screener.screen(n_trajectories=2)
        for col in ("indicator", "sub_index", "mu_star", "sigma", "rank"):
            assert col in df.columns

    def test_screen_has_14_rows(self, screener):
        df = screener.screen(n_trajectories=2)
        assert len(df) == 14   # one row per indicator

    def test_mu_star_nonnegative(self, screener):
        df = screener.screen(n_trajectories=2)
        assert (df["mu_star"] >= 0).all()

    def test_ranks_are_unique(self, screener):
        df = screener.screen(n_trajectories=2)
        assert df["rank"].nunique() == len(df)


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:

    def test_validate_ahmedabad_passes_well_formed(self):
        from rxharm.validation import validate_ahmedabad
        hvi_results = {
            "HVI": np.random.beta(2, 5, SHAPE),
            "H_s": np.random.uniform(0, 1, SHAPE),
            "E":   np.random.uniform(0, 1, SHAPE),
            "S":   np.random.uniform(0, 1, SHAPE),
            "AC":  np.random.uniform(0, 1, SHAPE),
        }
        result = validate_ahmedabad(hvi_results, aoi_handler=None, verbose=False)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "checks" in result

    def test_validate_empty_hvi_fails(self):
        from rxharm.validation import validate_ahmedabad
        result = validate_ahmedabad({"HVI": np.array([])}, None, verbose=False)
        assert result["passed"] is False

    def test_validate_all_nan_fails(self):
        from rxharm.validation import validate_ahmedabad
        result = validate_ahmedabad(
            {"HVI": np.full(SHAPE, np.nan)}, None, verbose=False
        )
        assert result["passed"] is False

    def test_validate_out_of_range_fails(self):
        from rxharm.validation import validate_ahmedabad
        hvi = np.full(SHAPE, 5.0)  # clearly out of [0,1]
        result = validate_ahmedabad({"HVI": hvi}, None, verbose=False)
        assert result["checks"]["range_0_1"]["pass"] is False

    def test_print_hvi_summary_runs(self):
        from rxharm.validation import print_hvi_summary
        hvi_results = {
            "HVI": np.random.beta(2, 5, SHAPE),
            "H_s": np.random.uniform(0, 1, SHAPE),
            "E":   np.random.uniform(0, 1, SHAPE),
            "S":   np.random.uniform(0, 1, SHAPE),
            "AC":  np.random.uniform(0, 1, SHAPE),
        }
        hri_results = {"AD_baseline": np.random.uniform(0, 0.5, SHAPE)}

        class FakeAOI:
            source = "Ahmedabad"
            year   = 2023
        print_hvi_summary(hvi_results, hri_results, FakeAOI())  # should not raise

    def test_display_runtime_warning_runs(self):
        from rxharm.validation import display_runtime_warning
        display_runtime_warning("Test Op", "~5 min", can_skip=True)  # no raise

    def test_safe_gee_call_reraises(self):
        from rxharm.validation import safe_gee_call
        def bad_func(): raise RuntimeError("test error")
        with pytest.raises(RuntimeError):
            safe_gee_call(bad_func, operation_name="test_op")


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — Calibrator transparency (Group 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibratorFixes:
    """FIX v0.1.0: BayesianCalibrator must warn and set calibration_status."""

    def test_calibrate_lambda_returns_status_field(self):
        import numpy as np
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        cal = BayesianCalibrator("A", np.random.uniform(28, 45, (10, 10)))
        result = cal.calibrate_lambda()
        assert "calibration_status" in result, "Missing calibration_status field"

    def test_calibrate_lambda_prior_status_is_prior_only(self):
        import numpy as np
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        cal = BayesianCalibrator("A", np.random.uniform(28, 45, (10, 10)))
        result = cal.calibrate_lambda()
        assert result["calibration_status"] == "PRIOR_ONLY"
        assert result["used_mcc_data"] is False

    def test_calibrate_lambda_has_note_field(self):
        import numpy as np
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        cal = BayesianCalibrator("A", np.random.uniform(28, 45, (10, 10)))
        result = cal.calibrate_lambda()
        assert "note" in result
        assert len(result["note"]) > 10

    def test_calibrate_mmt_has_calibration_status(self):
        import numpy as np
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        cal = BayesianCalibrator("A", np.random.uniform(28, 45, (10, 10)))
        result = cal.calibrate_mmt(era5_p75=35.0)
        assert "calibration_status" in result
        assert result["calibration_status"] == "PRIOR_ONLY"

    def test_calibrate_mmt_has_note(self):
        import numpy as np
        from rxharm.uncertainty.bayesian_calibrate import BayesianCalibrator
        cal = BayesianCalibrator("A", np.random.uniform(28, 45, (10, 10)))
        result = cal.calibrate_mmt(era5_p75=33.5)
        assert "note" in result
        assert "33.5" in result["note"] or "ERA5" in result["note"]
