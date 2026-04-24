"""
tests/test_fetch.py
===================
Tests for rxharm/fetch modules.

Non-GEE tests (run anywhere):
    - VIIRSDownscaler end-to-end pipeline
    - _block_reduce shape and value correctness
    - HazardFetcher date-filter helper
    - Band naming contract (14 bands in correct order)

GEE tests (marked @pytest.mark.gee, skipped by default):
    - Actual indicator fetching against GEE

Run non-GEE tests:
    pytest tests/test_fetch.py -v -m "not gee"

Run GEE tests (requires ee.Authenticate() first):
    pytest tests/test_fetch.py -v -m gee
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# VIIRSDownscaler tests (no GEE required)
# ══════════════════════════════════════════════════════════════════════════════

class TestVIIRSDownscaler:
    """End-to-end and unit tests for the RFATPK downscaler."""

    @pytest.fixture
    def downscaler(self):
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        return VIIRSDownscaler(n_estimators=10, random_state=42)

    @pytest.fixture
    def synthetic_coarse(self):
        """10×10 synthetic VIIRS array at 500 m."""
        np.random.seed(42)
        return np.random.uniform(0, 50, (10, 10))

    @pytest.fixture
    def synthetic_fine_covariates(self):
        """50×50 covariate arrays at 100 m (5× upscale factor)."""
        np.random.seed(42)
        return {
            "population": np.random.uniform(0, 200, (50, 50)),
            "ndvi":        np.random.uniform(0, 0.8, (50, 50)),
            "built_frac":  np.random.uniform(0, 1, (50, 50)),
        }

    def test_output_shape_is_fine_resolution(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        result = downscaler.downscale(
            synthetic_coarse, synthetic_fine_covariates,
            coarse_resolution_m=500, fine_resolution_m=100,
        )
        assert result.shape == (50, 50), f"Expected (50, 50), got {result.shape}"

    def test_output_values_are_non_negative(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        result = downscaler.downscale(synthetic_coarse, synthetic_fine_covariates)
        assert np.all(result >= 0), "Radiance cannot be negative"

    def test_output_values_are_clipped_to_max(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        result = downscaler.downscale(synthetic_coarse, synthetic_fine_covariates)
        assert np.all(result <= 200), "Values exceed physical maximum 200 nW/cm²/sr"

    def test_output_has_no_nan(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        result = downscaler.downscale(synthetic_coarse, synthetic_fine_covariates)
        assert not np.any(np.isnan(result)), "Output contains NaN values"

    def test_output_dtype_is_float32(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        result = downscaler.downscale(synthetic_coarse, synthetic_fine_covariates)
        assert result.dtype == np.float32

    def test_downscaling_preserves_spatial_order(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        """Bright regions in coarse should remain relatively brighter in fine output."""
        from scipy.stats import spearmanr
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler

        result = downscaler.downscale(synthetic_coarse, synthetic_fine_covariates)
        result_coarse = VIIRSDownscaler._block_reduce(result, factor=5)
        corr, _ = spearmanr(synthetic_coarse.flatten(), result_coarse.flatten())
        assert corr > 0.70, (
            f"Spatial rank correlation too low: {corr:.3f} (expected > 0.70)"
        )

    def test_handles_nan_in_coarse_input(
        self, downscaler, synthetic_coarse, synthetic_fine_covariates
    ):
        """NaN pixels (cloud-masked) should not crash the downscaler."""
        coarse_with_nan = synthetic_coarse.copy()
        coarse_with_nan[0, 0] = np.nan
        result = downscaler.downscale(coarse_with_nan, synthetic_fine_covariates)
        assert result is not None
        assert result.shape == (50, 50)
        assert not np.any(np.isnan(result))

    def test_all_nan_falls_back_gracefully(
        self, downscaler, synthetic_fine_covariates
    ):
        """If most coarse pixels are NaN, result should still be valid."""
        all_nan = np.full((10, 10), np.nan)
        all_nan[5, 5] = 10.0  # one valid pixel
        result = downscaler.downscale(all_nan, synthetic_fine_covariates)
        assert result.shape == (50, 50)
        assert not np.any(np.isnan(result))


# ── _block_reduce unit tests ──────────────────────────────────────────────────

class TestBlockReduce:

    def test_correct_output_shape(self):
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        arr = np.ones((50, 50))
        reduced = VIIRSDownscaler._block_reduce(arr, factor=5)
        assert reduced.shape == (10, 10)

    def test_mean_value_preserved(self):
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        arr = np.full((10, 10), 4.0)
        reduced = VIIRSDownscaler._block_reduce(arr, factor=2)
        assert np.allclose(reduced, 4.0)

    def test_non_divisible_dimensions_handled(self):
        """Dimensions not divisible by factor must still work (via padding)."""
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        arr = np.ones((11, 13))  # not divisible by 5
        reduced = VIIRSDownscaler._block_reduce(arr, factor=5)
        assert reduced.shape == (3, 3)  # ceil(11/5)=3, ceil(13/5)=3

    def test_2x2_factor_on_4x4(self):
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        arr = np.arange(16, dtype=float).reshape(4, 4)
        reduced = VIIRSDownscaler._block_reduce(arr, factor=2)
        assert reduced.shape == (2, 2)
        # Top-left 2×2 block mean = (0+1+4+5)/4 = 2.5
        assert np.isclose(reduced[0, 0], 2.5)

    def test_preserves_spatial_gradient(self):
        """Block reduce should preserve large-scale brightness gradient."""
        from rxharm.fetch.viirs_downscaler import VIIRSDownscaler
        # Left half bright, right half dark
        arr = np.ones((10, 10))
        arr[:, 5:] = 0.0
        reduced = VIIRSDownscaler._block_reduce(arr, factor=5)
        assert reduced[0, 0] > reduced[0, 1], "Left should be brighter than right"


# ══════════════════════════════════════════════════════════════════════════════
# HazardFetcher date-filter tests (no GEE required)
# ══════════════════════════════════════════════════════════════════════════════

class TestHazardFetcherDateFilter:

    def test_single_month(self):
        from rxharm.fetch.hazard import HazardFetcher
        start, end = HazardFetcher._month_to_date_range(2023, [4])
        assert start == "2023-04-01"
        assert end   == "2023-05-01"

    def test_two_months(self):
        from rxharm.fetch.hazard import HazardFetcher
        start, end = HazardFetcher._month_to_date_range(2023, [4, 5])
        assert start == "2023-04-01"
        assert end   == "2023-06-01"

    def test_year_end_wrap_november_december(self):
        from rxharm.fetch.hazard import HazardFetcher
        start, end = HazardFetcher._month_to_date_range(2023, [11, 12])
        assert start == "2023-11-01"
        assert end   == "2024-01-01"

    def test_year_end_wrap_december_only(self):
        from rxharm.fetch.hazard import HazardFetcher
        start, end = HazardFetcher._month_to_date_range(2023, [12])
        assert start == "2023-12-01"
        assert end   == "2024-01-01"

    def test_unsorted_months_still_correct(self):
        """Input months in non-sorted order should give same result."""
        from rxharm.fetch.hazard import HazardFetcher
        start1, end1 = HazardFetcher._month_to_date_range(2023, [5, 4])
        start2, end2 = HazardFetcher._month_to_date_range(2023, [4, 5])
        assert start1 == start2
        assert end1   == end2

    def test_start_before_end(self):
        from rxharm.fetch.hazard import HazardFetcher
        for months in [[1], [6, 7], [11, 12], [3, 4, 5]]:
            start, end = HazardFetcher._month_to_date_range(2022, months)
            assert start < end, f"start >= end for months {months}"

    def test_date_format_is_iso(self):
        from rxharm.fetch.hazard import HazardFetcher
        start, end = HazardFetcher._month_to_date_range(2022, [6])
        assert len(start) == 10 and start[4] == "-" and start[7] == "-"
        assert len(end)   == 10 and end[4]   == "-" and end[7]   == "-"


# ══════════════════════════════════════════════════════════════════════════════
# Band naming contract tests (no GEE required)
# ══════════════════════════════════════════════════════════════════════════════

class TestBandNamingContract:
    """Verify that fetch_all_indicators encodes exactly 14 canonical band names."""

    EXPECTED_BANDS = [
        "lst", "albedo", "uhi",
        "population", "built_frac",
        "elderly_frac", "child_frac", "impervious", "cropland",
        "ndvi", "ndwi", "tree_cover", "canopy_height", "viirs_dnb_raw",
    ]

    def test_total_band_count_is_14(self):
        assert len(self.EXPECTED_BANDS) == 14

    def test_band_names_present_in_source(self):
        import inspect
        from rxharm.fetch import fetch_all_indicators
        source = inspect.getsource(fetch_all_indicators)
        for band in self.EXPECTED_BANDS:
            assert f'"{band}"' in source or f"'{band}'" in source, (
                f"Band '{band}' not found in fetch_all_indicators source"
            )

    def test_viirs_band_has_raw_suffix(self):
        """viirs_dnb_raw suffix signals downstream downscaling required."""
        assert "viirs_dnb_raw" in self.EXPECTED_BANDS
        assert "viirs_dnb" not in self.EXPECTED_BANDS  # no unsuffixed version

    def test_ndwi_is_vegetation_moisture_variant(self):
        """Verify NDWI comment in adaptive_capacity source."""
        import inspect
        from rxharm.fetch.adaptive_capacity import AdaptiveCapacityFetcher
        source = inspect.getsource(AdaptiveCapacityFetcher.get_ndwi)
        assert "B11" in source, "NDWI should use B11 (SWIR1) not Green band"
        assert "vegetation" in source.lower() or "Gao" in source, (
            "NDWI docstring should note it is the vegetation moisture variant"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Config additions tests (no GEE required)
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigAdditions:
    """Verify that Step III required constants are present in config.py."""

    def test_albedo_coeffs_present(self):
        import rxharm.config as cfg
        assert hasattr(cfg, "ALBEDO_COEFFS")
        assert isinstance(cfg.ALBEDO_COEFFS, dict)

    def test_albedo_coeffs_keys(self):
        import rxharm.config as cfg
        required = {"B2", "B4", "B5", "B6", "B7", "intercept"}
        assert required.issubset(set(cfg.ALBEDO_COEFFS.keys()))

    def test_albedo_coeffs_sum_approximately_one(self):
        """Band coefficients should sum to ~1 (physically: broadband from narrowband)."""
        import rxharm.config as cfg
        band_sum = sum(v for k, v in cfg.ALBEDO_COEFFS.items() if k != "intercept")
        assert 0.9 < band_sum < 1.1, (
            f"Albedo band coefficients sum to {band_sum:.4f} — check Liang (2000)"
        )

    def test_landsat_sr_scale_is_correct(self):
        import rxharm.config as cfg
        assert hasattr(cfg, "LANDSAT_SR_SCALE")
        assert abs(cfg.LANDSAT_SR_SCALE - 0.0000275) < 1e-10

    def test_landsat_sr_offset_is_correct(self):
        import rxharm.config as cfg
        assert hasattr(cfg, "LANDSAT_SR_OFFSET")
        assert abs(cfg.LANDSAT_SR_OFFSET - (-0.2)) < 1e-10

    def test_ghs_built_epochs_is_list(self):
        import rxharm.config as cfg
        assert hasattr(cfg, "GHS_BUILT_EPOCHS")
        assert isinstance(cfg.GHS_BUILT_EPOCHS, list)
        assert 2020 in cfg.GHS_BUILT_EPOCHS

    def test_s2_scl_mask_values_present(self):
        import rxharm.config as cfg
        assert hasattr(cfg, "S2_SCL_MASK_VALUES")
        assert isinstance(cfg.S2_SCL_MASK_VALUES, list)
        # Cloud shadow (3), cloud high prob (9) must be masked
        assert 3 in cfg.S2_SCL_MASK_VALUES
        assert 9 in cfg.S2_SCL_MASK_VALUES


# ══════════════════════════════════════════════════════════════════════════════
# Import chain test (no GEE required)
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchImports:

    def test_all_classes_importable(self):
        from rxharm.fetch import (
            HazardFetcher,
            ExposureFetcher,
            SensitivityFetcher,
            AdaptiveCapacityFetcher,
            VIIRSDownscaler,
            fetch_all_indicators,
        )
        assert HazardFetcher is not None
        assert ExposureFetcher is not None
        assert SensitivityFetcher is not None
        assert AdaptiveCapacityFetcher is not None
        assert VIIRSDownscaler is not None
        assert callable(fetch_all_indicators)

    def test_viirs_downscaler_does_not_import_ee(self):
        """VIIRSDownscaler must be importable without GEE."""
        import sys
        # Patch ee to raise ImportError to confirm it's not imported at module level
        original = sys.modules.get("ee")
        sys.modules["ee"] = None  # type: ignore
        try:
            from rxharm.fetch.viirs_downscaler import VIIRSDownscaler as VS
            ds = VS(n_estimators=5)
            assert ds is not None
        finally:
            if original is None:
                sys.modules.pop("ee", None)
            else:
                sys.modules["ee"] = original


# ══════════════════════════════════════════════════════════════════════════════
# GEE integration tests (skipped by default)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.gee
class TestHazardFetcherGEE:
    """Requires GEE authentication. Run with: pytest -m gee"""

    @pytest.fixture
    def ahmedabad_geometry(self):
        import ee
        return ee.Geometry.Point([72.5714, 23.0225]).buffer(3000)

    def test_lst_returns_correct_band(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.hazard import HazardFetcher
        fetcher = HazardFetcher(ahmedabad_geometry, 2022, [4, 5])
        lst = fetcher.get_lst()
        assert "lst" in lst.bandNames().getInfo()

    def test_lst_values_in_physical_range(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.hazard import HazardFetcher
        fetcher = HazardFetcher(ahmedabad_geometry, 2022, [4, 5])
        lst = fetcher.get_lst()
        stats = lst.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=ahmedabad_geometry,
            scale=1000,
        ).getInfo()
        assert stats.get("lst_min", 0) > -20
        assert stats.get("lst_max", 100) < 80

    def test_albedo_values_between_0_and_1(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.hazard import HazardFetcher
        fetcher = HazardFetcher(ahmedabad_geometry, 2022, [4, 5])
        albedo = fetcher.get_albedo()
        stats = albedo.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=ahmedabad_geometry,
            scale=1000,
        ).getInfo()
        assert stats.get("albedo_min", -1) >= 0
        assert stats.get("albedo_max", 2) <= 1

    def test_fetch_all_returns_three_bands(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.hazard import HazardFetcher
        fetcher = HazardFetcher(ahmedabad_geometry, 2022, [4, 5])
        img = fetcher.fetch_all()
        assert set(img.bandNames().getInfo()) == {"lst", "albedo", "uhi"}


@pytest.mark.gee
class TestExposureFetcherGEE:

    @pytest.fixture
    def ahmedabad_geometry(self):
        import ee
        return ee.Geometry.Point([72.5714, 23.0225]).buffer(3000)

    def test_population_values_non_negative(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.exposure import ExposureFetcher
        fetcher = ExposureFetcher(ahmedabad_geometry, 2020)
        pop = fetcher.get_population()
        stats = pop.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=ahmedabad_geometry,
            scale=1000,
        ).getInfo()
        assert stats.get("population", -1) >= 0

    def test_built_frac_between_0_and_1(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.exposure import ExposureFetcher
        fetcher = ExposureFetcher(ahmedabad_geometry, 2020)
        built = fetcher.get_built_fraction()
        stats = built.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=ahmedabad_geometry,
            scale=1000,
        ).getInfo()
        assert stats.get("built_frac_min", -1) >= 0
        assert stats.get("built_frac_max", 2) <= 1


@pytest.mark.gee
class TestSensitivityFetcherGEE:

    @pytest.fixture
    def ahmedabad_geometry(self):
        import ee
        return ee.Geometry.Point([72.5714, 23.0225]).buffer(3000)

    def test_age_fractions_between_0_and_1(self, ahmedabad_geometry):
        import ee
        from rxharm.fetch.sensitivity import SensitivityFetcher
        fetcher = SensitivityFetcher(ahmedabad_geometry, 2020, [4, 5])
        img = fetcher.get_age_fractions()
        for band in ["elderly_frac", "child_frac"]:
            stats = img.select(band).reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=ahmedabad_geometry,
                scale=1000,
            ).getInfo()
            assert stats.get(f"{band}_min", -1) >= 0
            assert stats.get(f"{band}_max", 2) <= 1

    def test_ndwi_is_vegetation_moisture_not_water(self, ahmedabad_geometry):
        """Confirm we are using (B8-B11)/(B8+B11) not (Green-NIR)/(Green+NIR)."""
        import ee
        from rxharm.fetch.adaptive_capacity import AdaptiveCapacityFetcher
        fetcher = AdaptiveCapacityFetcher(ahmedabad_geometry, 2022, [4, 5])
        ndwi = fetcher.get_ndwi()
        assert "ndwi" in ndwi.bandNames().getInfo()
        stats = ndwi.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=ahmedabad_geometry,
            scale=1000,
        ).getInfo()
        # Vegetation moisture NDWI for dry April in Ahmedabad is often negative
        assert stats.get("ndwi_min", 1) < 0.5, (
            "NDWI minimum suspiciously high — check vegetation moisture formula"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — Band Validator (Group 1)
# ══════════════════════════════════════════════════════════════════════════════

class TestBandValidator:
    """FIX v0.1.0: Tests for validate_indicator_arrays."""

    @staticmethod
    def _valid_arrays(shape=(30, 30)):
        import numpy as np
        rng = np.random.default_rng(99)
        return {
            "lst":           rng.uniform(25, 50, shape),
            "albedo":        rng.uniform(0.1, 0.5, shape),
            "uhi":           rng.uniform(0, 8, shape),
            "population":    rng.uniform(100, 5000, shape),
            "built_frac":    rng.uniform(0, 1, shape),
            "elderly_frac":  rng.uniform(0, 0.25, shape),
            "child_frac":    rng.uniform(0, 0.12, shape),
            "impervious":    rng.uniform(0, 1, shape),
            "cropland":      rng.uniform(0, 0.5, shape),
            "ndvi":          rng.uniform(0, 0.7, shape),
            "ndwi":          rng.uniform(-0.3, 0.3, shape),
            "tree_cover":    rng.uniform(0, 60, shape),
            "canopy_height": rng.uniform(0, 20, shape),
            "viirs_dnb":     rng.uniform(1, 50, shape),
        }

    def test_valid_arrays_pass(self):
        """Normal physical values should pass validation."""
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        report = validate_indicator_arrays(arrays)
        assert all(v["status"] == "ok" for v in report.values())

    def test_zero_filled_array_fails(self):
        """All-zero array (silent GEE failure) must raise ValueError."""
        import numpy as np, pytest
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        arrays["lst"] = np.zeros((30, 30))   # all zeros — failed GEE fetch
        with pytest.raises(ValueError, match="exactly zero"):
            validate_indicator_arrays(arrays)

    def test_mostly_nan_array_fails(self):
        """Heavily masked array (cloud-failed composite) must raise ValueError."""
        import numpy as np, pytest
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        lst = arrays["lst"].copy()
        mask = np.random.default_rng(0).random((30, 30)) < 0.95
        lst[mask] = np.nan
        arrays["lst"] = lst
        with pytest.raises(ValueError, match="valid pixels"):
            validate_indicator_arrays(arrays)

    def test_out_of_range_values_fail(self):
        """Values outside physical range must be caught."""
        import numpy as np, pytest
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        arrays["albedo"] = np.full((30, 30), 5.0)   # unit error: % not fraction
        with pytest.raises(ValueError, match="outside expected physical range"):
            validate_indicator_arrays(arrays)

    def test_canopy_height_failure_is_warning_not_error(self):
        """canopy_height is non-critical — a zero-filled band should NOT raise."""
        import numpy as np
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        arrays["canopy_height"] = np.zeros((30, 30))  # non-critical — should warn only
        # Should not raise — canopy_height is in _NON_CRITICAL set
        report = validate_indicator_arrays(arrays)
        assert report["canopy_height"]["status"] == "fail"   # flagged but not raised

    def test_extra_bands_ignored(self):
        """Bands not in BAND_EXPECTED_RANGES should be silently skipped."""
        from rxharm.fetch.validator import validate_indicator_arrays
        arrays = self._valid_arrays()
        import numpy as np
        arrays["viirs_dnb_raw"] = np.random.uniform(0, 50, (30, 30))  # pre-downscale name
        report = validate_indicator_arrays(arrays)
        assert "viirs_dnb_raw" not in report   # should be silently skipped

    def test_print_validation_report_runs(self):
        """print_validation_report should not raise."""
        from rxharm.fetch.validator import print_validation_report
        arrays = self._valid_arrays()
        print_validation_report(arrays)  # no raise

    def test_load_existing_export_import(self):
        """load_existing_export must be importable from rxharm.fetch."""
        from rxharm.fetch import load_existing_export
        assert callable(load_existing_export)


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — GEE Collections in config (Group 6a)
# ══════════════════════════════════════════════════════════════════════════════

class TestGEECollections:
    """FIX v0.1.0: GEE_COLLECTIONS dict in config.py."""

    def test_gee_collections_dict_present(self):
        from rxharm.config import GEE_COLLECTIONS
        assert isinstance(GEE_COLLECTIONS, dict)
        assert len(GEE_COLLECTIONS) >= 13

    def test_required_keys_present(self):
        from rxharm.config import GEE_COLLECTIONS
        for key in ["landsat8", "landsat9", "sentinel2", "dynamic_world",
                    "worldpop_pop", "ghs_built_s", "hansen_gfw", "viirs_dnb"]:
            assert key in GEE_COLLECTIONS, f"Missing key: {key}"

    def test_collection_ids_are_strings(self):
        from rxharm.config import GEE_COLLECTIONS
        for k, v in GEE_COLLECTIONS.items():
            assert isinstance(v, str) and len(v) > 5, f"{k}: invalid ID '{v}'"

    def test_fetch_modules_use_config_ids(self):
        """Fetch module constants must match config.GEE_COLLECTIONS values."""
        from rxharm.config import GEE_COLLECTIONS
        # exposure.py
        from rxharm.fetch.exposure import _WORLDPOP_NATIVE, _GHS_BUILT
        assert _WORLDPOP_NATIVE == GEE_COLLECTIONS["worldpop_pop"]
        assert _GHS_BUILT       == GEE_COLLECTIONS["ghs_built_s"]
        # sensitivity.py
        from rxharm.fetch.sensitivity import _WORLDPOP_AGE, _DW
        assert _WORLDPOP_AGE == GEE_COLLECTIONS["worldpop_agesex"]
        assert _DW           == GEE_COLLECTIONS["dynamic_world"]
        # adaptive_capacity.py
        from rxharm.fetch.adaptive_capacity import _S2_HARMONIZED, _VIIRS_ANNUAL
        assert _S2_HARMONIZED == GEE_COLLECTIONS["sentinel2"]
        assert _VIIRS_ANNUAL  == GEE_COLLECTIONS["viirs_dnb"]
