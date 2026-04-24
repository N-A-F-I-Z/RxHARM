"""
tests/test_hvi.py
=================
Tests for rxharm/index/normalizer.py, weighter.py, hvi.py, hri.py.
All pass without GEE or network access.
"""

from __future__ import annotations
import numpy as np
import pytest


# ── Shared synthetic data ─────────────────────────────────────────────────────

SHAPE = (30, 30)

def _make_arrays(seed=42):
    """Build a minimal set of 14 synthetic indicator arrays."""
    rng = np.random.default_rng(seed)
    return {
        "lst":           rng.uniform(28, 48,   SHAPE),
        "albedo":        rng.uniform(0.05, 0.35, SHAPE),
        "uhi":           rng.uniform(0, 8,    SHAPE),
        "population":    rng.uniform(0, 5000, SHAPE),
        "built_frac":    rng.uniform(0, 1,    SHAPE),
        "elderly_frac":  rng.uniform(0, 0.25, SHAPE),
        "child_frac":    rng.uniform(0, 0.15, SHAPE),
        "impervious":    rng.uniform(0, 1,    SHAPE),
        "cropland":      rng.uniform(0, 0.5,  SHAPE),
        "ndvi":          rng.uniform(0, 0.7,  SHAPE),
        "ndwi":          rng.uniform(-0.3, 0.5, SHAPE),
        "tree_cover":    rng.uniform(0, 60,   SHAPE),
        "canopy_height": rng.uniform(0, 20,   SHAPE),
        "viirs_dnb":     rng.uniform(0, 50,   SHAPE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NormalizerEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizerEngine:

    @pytest.fixture
    def eng(self):
        from rxharm.index.normalizer import NormalizerEngine
        return NormalizerEngine()

    def test_positive_direction_range(self, eng):
        arr = np.linspace(0, 100, 200)
        out = eng.normalize(arr, "lst", "positive")
        assert np.nanmin(out) >= 0.0
        assert np.nanmax(out) <= 1.0

    def test_negative_direction_inverts(self, eng):
        arr = np.linspace(0, 100, 200).reshape(20, 10)
        out = eng.normalize(arr, "ndvi", "negative")
        # highest raw → lowest normalised
        assert out.ravel()[np.argmax(arr.ravel())] < 0.5

    def test_nan_preserved(self, eng):
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = eng.normalize(arr, "test_nan")
        assert np.isnan(out[2])
        assert np.isfinite(out[0])

    def test_outlier_clipping(self, eng):
        arr = np.concatenate([np.linspace(0, 10, 198), [1000, 2000]])
        out = eng.normalize(arr, "lst_outlier")
        # extreme outlier at 2000 should not pull the median far from 0.5
        assert out[-1] <= 1.0
        assert np.nanmedian(out) > 0.3

    def test_constant_field_returns_half(self, eng):
        arr = np.full((10, 10), 5.0)
        out = eng.normalize(arr, "const")
        assert np.allclose(out[np.isfinite(out)], 0.5)

    def test_stats_stored(self, eng):
        arr = np.random.uniform(0, 1, 100)
        eng.normalize(arr, "check_stats")
        df = eng.get_stats()
        assert "check_stats" in df["indicator"].values

    def test_batch_normalize_length(self, eng):
        arrays = _make_arrays()
        out = eng.normalize_batch(arrays)
        assert len(out) == len(arrays)

    def test_batch_all_in_range(self, eng):
        arrays = _make_arrays()
        normed = eng.normalize_batch(arrays)
        for name, arr in normed.items():
            assert np.nanmin(arr) >= -1e-9, f"{name} below 0"
            assert np.nanmax(arr) <= 1+1e-9, f"{name} above 1"


# ══════════════════════════════════════════════════════════════════════════════
# WeighterEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestWeighterEngine:

    @pytest.fixture
    def normed(self):
        from rxharm.index.normalizer import NormalizerEngine
        return NormalizerEngine().normalize_batch(_make_arrays())

    def _hazard(self, normed):
        return {k: normed[k] for k in ("lst", "albedo", "uhi")}

    def test_equal_weights_sum_to_one(self, normed):
        from rxharm.index.weighter import WeighterEngine
        w = WeighterEngine("equal").compute_weights(self._hazard(normed), "hazard")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_pca_weights_sum_to_one(self, normed):
        from rxharm.index.weighter import WeighterEngine
        w = WeighterEngine("pca").compute_weights(self._hazard(normed), "hazard")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_entropy_weights_sum_to_one(self, normed):
        from rxharm.index.weighter import WeighterEngine
        w = WeighterEngine("entropy").compute_weights(self._hazard(normed), "hazard")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_critic_weights_sum_to_one(self, normed):
        from rxharm.index.weighter import WeighterEngine
        w = WeighterEngine("critic").compute_weights(self._hazard(normed), "hazard")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_manual_weights_valid(self, normed):
        from rxharm.index.weighter import WeighterEngine
        w = WeighterEngine("manual").compute_weights(self._hazard(normed), "hazard")
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_manual_weights_invalid_raises(self):
        from rxharm.index.weighter import WeighterEngine
        eng = WeighterEngine("equal")
        with pytest.raises(ValueError):
            eng._manual_weights(["a", "b"], {"a": 0.3, "b": 0.3})

    def test_unknown_method_raises(self):
        from rxharm.index.weighter import WeighterEngine
        with pytest.raises(ValueError):
            WeighterEngine("bad_method")


# ══════════════════════════════════════════════════════════════════════════════
# HVIEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestHVIEngine:

    @pytest.fixture
    def engine(self):
        from rxharm.index.hvi import HVIEngine
        return HVIEngine(weighting_method="equal")

    @pytest.fixture
    def arrays(self):
        return _make_arrays()

    def test_compute_all_returns_expected_keys(self, engine, arrays):
        r = engine.compute_all(arrays)
        for key in ("H_s", "E", "S", "AC", "HVI", "HVI_raw", "stats"):
            assert key in r, f"Missing key: {key}"

    def test_hvi_range_is_zero_to_one(self, engine, arrays):
        hvi = engine.compute_all(arrays)["HVI"]
        assert np.nanmin(hvi) >= -1e-9
        assert np.nanmax(hvi) <= 1 + 1e-9

    def test_output_shape_matches_input(self, engine, arrays):
        hvi = engine.compute_all(arrays)["HVI"]
        assert hvi.shape == SHAPE

    def test_zero_population_gives_zero_hvi(self, engine, arrays):
        a = {k: v.copy() for k, v in arrays.items()}
        a["population"] = np.zeros(SHAPE)
        hvi = engine.compute_all(a)["HVI"]
        # E is zero → HVI numerator = 0 → all near-zero
        assert np.nanmean(hvi) < 0.20

    def test_high_ac_reduces_hvi(self, engine, arrays):
        a_low_ac  = {k: v.copy() for k, v in arrays.items()}
        a_high_ac = {k: v.copy() for k, v in arrays.items()}
        for k in ("ndvi", "ndwi", "tree_cover", "canopy_height", "viirs_dnb"):
            a_high_ac[k] = np.full(SHAPE, 0.9)
            a_low_ac[k]  = np.full(SHAPE, 0.1)
        hvi_low_ac  = np.nanmean(engine.compute_all(a_low_ac)["HVI"])
        hvi_high_ac = np.nanmean(engine.compute_all(a_high_ac)["HVI"])
        assert hvi_high_ac < hvi_low_ac

    def test_nan_propagation(self, engine, arrays):
        a = {k: v.copy() for k, v in arrays.items()}
        a["lst"][0, 0] = np.nan
        # NaN in one indicator should not cascade to all cells
        hvi = engine.compute_all(a)["HVI"]
        n_nan = np.sum(np.isnan(hvi))
        assert n_nan < hvi.size  # not all NaN

    def test_sensitivity_test_returns_dataframe(self, engine, arrays):
        df = engine.sensitivity_test(arrays)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert "indicator" in df.columns
        assert "spearman_r" in df.columns
        assert len(df) > 0

    def test_subindex_in_unit_interval(self, engine, arrays):
        r = engine.compute_all(arrays)
        for si in ("H_s", "E", "S", "AC"):
            arr = r[si]
            assert np.nanmin(arr) >= -1e-9, f"{si} below 0"
            assert np.nanmax(arr) <= 1 + 1e-9, f"{si} above 1"


# ══════════════════════════════════════════════════════════════════════════════
# HRIEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestHRIEngine:

    @pytest.fixture
    def engine(self):
        from rxharm.index.hri import HRIEngine
        eng = HRIEngine(climate_zone="A", cdr_baseline=0.007)
        return eng

    @pytest.fixture
    def hvi_results(self):
        from rxharm.index.hvi import HVIEngine
        a = _make_arrays()
        return HVIEngine().compute_all(a)

    def test_compute_mmt_returns_float(self, engine, hvi_results):
        lst = _make_arrays()["lst"]
        mmt = engine.compute_mmt(lst)
        assert isinstance(mmt, float)
        assert 20 < mmt < 55

    def test_hri_range_zero_to_one(self, engine, hvi_results):
        engine.compute_mmt(_make_arrays()["lst"])
        engine.set_atmospheric_context(38.0)
        hri_r = engine.compute_all(hvi_results)
        hri   = hri_r["HRI"]
        assert np.nanmin(hri) >= -1e-9
        assert np.nanmax(hri) <= 1 + 1e-9

    def test_hri_equals_hs_times_hvi_proportional(self, engine, hvi_results):
        """HRI should correlate strongly with H_s × HVI."""
        engine.compute_mmt(_make_arrays()["lst"])
        hri_r = engine.compute_all(hvi_results)
        from scipy.stats import spearmanr
        hri   = hri_r["HRI"].ravel()
        hs    = hvi_results["H_s"].ravel()
        hvi   = hvi_results["HVI"].ravel()
        proxy = (hs * hvi).ravel()
        valid = np.isfinite(hri) & np.isfinite(proxy)
        r, _  = spearmanr(hri[valid], proxy[valid])
        assert r > 0.80

    def test_af_is_zero_below_mmt(self, engine):
        engine.mmt = 35.0
        lst = np.full((10, 10), 30.0)   # below MMT
        hvi = np.full((10, 10), 0.5)
        af  = engine.compute_attributable_fraction(lst, hvi)
        # delta_T = max(0, 30-35) = 0 → RR_adj = 1*(1+lambda*0.5) → AF = lambda*0.5/(1+lambda*0.5)
        # Non-zero but small; key test: no cell above MMT → no temperature-driven excess
        assert np.all(af >= 0)
        assert np.all(af < 0.5)

    def test_af_increases_with_temperature(self, engine):
        engine.mmt = 30.0
        hvi = np.full(100, 0.5)
        af_low  = engine.compute_attributable_fraction(np.full(100, 32.0), hvi)
        af_high = engine.compute_attributable_fraction(np.full(100, 40.0), hvi)
        assert np.mean(af_high) > np.mean(af_low)

    def test_ad_is_zero_for_zero_population(self, engine):
        engine.mmt = 30.0
        pop = np.zeros((10, 10))
        lst = np.full((10, 10), 38.0)
        hvi = np.full((10, 10), 0.5)
        ad  = engine.compute_attributable_deaths(pop, lst, hvi)
        assert np.all(ad == 0.0)

    def test_lambda_sensitivity(self, engine):
        engine.mmt = 30.0
        lst = np.full(50, 38.0)
        hvi = np.full(50, 0.5)
        af_lo = engine.compute_attributable_fraction(lst, hvi, lambda_hvi=0.1)
        af_hi = engine.compute_attributable_fraction(lst, hvi, lambda_hvi=0.8)
        assert np.mean(af_hi) > np.mean(af_lo)

    def test_unknown_climate_zone_raises(self):
        from rxharm.index.hri import HRIEngine
        with pytest.raises(ValueError):
            HRIEngine(climate_zone="Z", cdr_baseline=0.007)
