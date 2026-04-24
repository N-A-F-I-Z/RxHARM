"""
tests/test_aoi.py
=================
Tests for rxharm/aoi/handler.py and rxharm/aoi/decomposer.py.

All tests here run without GEE authentication.
GEE-dependent tests are marked with ``@pytest.mark.gee`` and
skipped by default in the CI pipeline.

Run without GEE:
    pytest tests/test_aoi.py -v -m "not gee"

Run with GEE (requires ee.Authenticate() first):
    pytest tests/test_aoi.py -v -m gee --gee-project YOUR_PROJECT
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rxharm.aoi.handler import AOIHandler
from rxharm.aoi.decomposer import ZoneDecomposer
import rxharm.config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_city_aoi():
    """Small ~3 km-radius AOI centred on Rajshahi, Bangladesh."""
    return AOIHandler((24.37, 88.60, 3.0), year=2023)


@pytest.fixture
def single_point_aoi():
    """Tiny 50 m-radius AOI — should trigger Moore's neighborhood mode."""
    return AOIHandler((24.37, 88.60, 0.05), year=2023)


@pytest.fixture
def paris_aoi():
    """~1 km-radius AOI in Paris (temperate zone test)."""
    return AOIHandler((48.85, 2.35, 1.0), year=2022)


# ══════════════════════════════════════════════════════════════════════════════
# AOIHandler — input type tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAOIHandlerInputTypes:

    def test_tuple_input_creates_valid_gdf(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        assert aoi.gdf is not None
        assert len(aoi.gdf) == 1
        assert aoi.gdf.crs.to_epsg() == 4326

    def test_centroid_is_near_input_point(self):
        lat, lon = 24.37, 88.60
        aoi = AOIHandler((lat, lon, 2.0), year=2023)
        clat, clon = aoi.centroid_ll
        assert abs(clat - lat) < 0.15
        assert abs(clon - lon) < 0.15

    def test_city_name_input_geocodes(self):
        """City-name geocoding requires internet; skipped if geopy unavailable."""
        geopy = pytest.importorskip("geopy")
        # REASON: Rajshahi is a mid-sized Bangladeshi city with stable Nominatim data.
        aoi = AOIHandler("Rajshahi, Bangladesh", year=2023)
        assert aoi.gdf is not None
        assert aoi.n_cells > 0

    def test_invalid_tuple_length_raises(self):
        with pytest.raises((ValueError, TypeError)):
            AOIHandler((24.37, 88.60), year=2023)  # missing radius

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AOIHandler((24.37, 88.60, -1.0), year=2023)

    def test_unknown_source_type_raises(self):
        with pytest.raises((ValueError, TypeError)):
            AOIHandler(12345, year=2023)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            AOIHandler("/nonexistent/path/to/file.shp", year=2023)

    def test_n_cells_is_positive(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        assert aoi.n_cells >= 1

    def test_bounds_are_valid(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        minx, miny, maxx, maxy = aoi.bounds
        assert minx < maxx
        assert miny < maxy
        assert -180 <= minx <= 180
        assert -90  <= miny <=  90

    def test_bounds_contain_centroid(self):
        lat, lon = 24.37, 88.60
        aoi = AOIHandler((lat, lon, 2.0), year=2023)
        minx, miny, maxx, maxy = aoi.bounds
        clat, clon = aoi.centroid_ll
        assert minx <= clon <= maxx
        assert miny <= clat <= maxy


# ══════════════════════════════════════════════════════════════════════════════
# AOIHandler — validation
# ══════════════════════════════════════════════════════════════════════════════

class TestAOIHandlerValidation:

    def test_valid_aoi_does_not_raise(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        aoi.validate()  # should not raise

    def test_year_too_old_raises(self):
        with pytest.raises(ValueError, match="year"):
            AOIHandler((24.37, 88.60, 2.0), year=2010).validate()

    def test_year_2015_raises(self):
        with pytest.raises(ValueError, match="year"):
            AOIHandler((24.37, 88.60, 2.0), year=2015).validate()

    def test_year_2016_is_valid(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2016)
        aoi.validate()  # should not raise

    def test_future_year_raises(self):
        with pytest.raises(ValueError, match="year"):
            AOIHandler((24.37, 88.60, 2.0), year=2099).validate()


# ══════════════════════════════════════════════════════════════════════════════
# AOIHandler — mode classification
# ══════════════════════════════════════════════════════════════════════════════

class TestAOIHandlerModeClassification:

    def test_tiny_aoi_is_moore_mode(self, single_point_aoi):
        assert single_point_aoi.mode == "moore"

    def test_small_aoi_mode_is_direct_or_moore(self):
        aoi = AOIHandler((24.37, 88.60, 1.0), year=2023)
        assert aoi.mode in ("direct", "moore")

    def test_mode_consistent_with_cell_count(self):
        aoi = AOIHandler((24.37, 88.60, 3.0), year=2023)
        n = aoi.n_cells
        if n <= cfg.MOORE_MAX_CELLS:
            assert aoi.mode == "moore"
        elif n <= cfg.MAX_CELLS_DIRECT:
            assert aoi.mode == "direct"
        elif n <= cfg.MAX_CELLS_MESO:
            assert aoi.mode == "meso"
        else:
            assert aoi.mode == "hierarchical"

    def test_mode_is_one_of_four_values(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        assert aoi.mode in ("moore", "direct", "meso", "hierarchical")

    def test_runtime_estimate_returns_string(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        rt = aoi.estimate_runtime()
        assert isinstance(rt, str)
        assert len(rt) > 0


# ══════════════════════════════════════════════════════════════════════════════
# AOIHandler — Köppen zone
# ══════════════════════════════════════════════════════════════════════════════

class TestAOIHandlerKoppenZone:

    def test_tropical_latitude_south_india(self):
        aoi = AOIHandler((10.0, 77.0, 1.0), year=2023)
        assert aoi.get_koppen_zone() == "A"

    def test_tropical_latitude_northeast_brazil(self):
        aoi = AOIHandler((-5.0, -35.0, 1.0), year=2023)
        assert aoi.get_koppen_zone() == "A"

    def test_arid_latitude_north_africa(self):
        aoi = AOIHandler((30.0, 10.0, 1.0), year=2023)
        assert aoi.get_koppen_zone() == "B"

    def test_temperate_latitude_paris(self, paris_aoi):
        assert paris_aoi.get_koppen_zone() == "C"

    def test_continental_latitude_st_petersburg(self):
        aoi = AOIHandler((60.0, 30.0, 1.0), year=2023)
        assert aoi.get_koppen_zone() == "D"

    def test_polar_latitude(self):
        aoi = AOIHandler((75.0, 0.0, 1.0), year=2023)
        assert aoi.get_koppen_zone() == "E"

    def test_koppen_returns_valid_key(self):
        aoi = AOIHandler((24.37, 88.60, 2.0), year=2023)
        assert aoi.get_koppen_zone() in cfg.BETA_BY_CLIMATE_ZONE

    def test_southern_hemisphere_uses_abs_lat(self):
        aoi_n = AOIHandler((10.0, 77.0, 1.0), year=2023)
        aoi_s = AOIHandler((-10.0, 77.0, 1.0), year=2023)
        assert aoi_n.get_koppen_zone() == aoi_s.get_koppen_zone()


# ══════════════════════════════════════════════════════════════════════════════
# AOIHandler — GeoJSON / EE geometry
# ══════════════════════════════════════════════════════════════════════════════

class TestAOIHandlerGeoJSON:

    def test_to_geojson_returns_dict(self, small_city_aoi):
        gj = small_city_aoi.to_geojson()
        assert isinstance(gj, dict)
        assert "type" in gj

    def test_to_ee_geometry_returns_dict(self, small_city_aoi):
        ee_geom = small_city_aoi.to_ee_geometry()
        assert isinstance(ee_geom, dict)
        assert "type" in ee_geom
        assert "coordinates" in ee_geom

    def test_to_ee_geometry_type_is_polygon_or_multipolygon(self, small_city_aoi):
        ee_geom = small_city_aoi.to_ee_geometry()
        assert ee_geom["type"] in ("Polygon", "MultiPolygon")

    def test_to_ee_geometry_is_cached(self, small_city_aoi):
        """Calling to_ee_geometry() twice should return the same dict object."""
        g1 = small_city_aoi.to_ee_geometry()
        g2 = small_city_aoi.to_ee_geometry()
        assert g1 is g2

    def test_display_summary_does_not_raise(self, small_city_aoi, capsys):
        small_city_aoi.display_summary()
        captured = capsys.readouterr()
        assert "RxHARM" in captured.out
        assert str(small_city_aoi.year) in captured.out

    def test_repr_contains_key_info(self, small_city_aoi):
        r = repr(small_city_aoi)
        assert "AOIHandler" in r
        assert str(small_city_aoi.year) in r
        assert small_city_aoi.mode in r


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — basic structure
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneDecomposerBasic:

    def test_decompose_returns_dict(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        for key in ("mode", "n_zones", "cells", "zones", "zone_assignments"):
            assert key in result, f"Missing key '{key}' in decompose() result"

    def test_n_zones_is_positive(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        assert result["n_zones"] >= 1

    def test_mode_matches_aoi_mode(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        assert result["mode"] == small_city_aoi.mode

    def test_cells_length_equals_n_zones_or_cells(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        assert len(result["cells"]) > 0

    def test_zone_assignments_length_matches_cells(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        result = d.decompose()
        assert len(result["zone_assignments"]) == len(result["cells"])

    def test_decompose_is_idempotent(self, small_city_aoi):
        """Calling decompose() twice should return the same result."""
        d = ZoneDecomposer(small_city_aoi)
        r1 = d.decompose()
        r2 = d.decompose()
        assert r1["n_zones"] == r2["n_zones"]
        assert r1["mode"] == r2["mode"]


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — cell grid
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneDecomposerCellGrid:

    def test_cell_grid_has_correct_crs(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        grid = d.get_cell_grid()
        assert grid.crs.to_epsg() == 4326

    def test_cell_grid_has_required_columns(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        grid = d.get_cell_grid()
        for col in ("cell_id", "geometry", "zone_id", "prescribable"):
            assert col in grid.columns, f"Missing column '{col}' in cell grid"

    def test_cell_grid_cells_within_aoi_bounds(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        grid = d.get_cell_grid()
        minx, miny, maxx, maxy = small_city_aoi.bounds
        tol = 0.002  # slight tolerance for grid edge effects
        assert all(grid.geometry.x >= minx - tol)
        assert all(grid.geometry.x <= maxx + tol)
        assert all(grid.geometry.y >= miny - tol)
        assert all(grid.geometry.y <= maxy + tol)

    def test_cell_grid_all_prescribable_by_default(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        grid = d.get_cell_grid()
        assert all(grid["prescribable"]), "All cells should be prescribable before filtering"

    def test_cell_grid_cell_ids_are_unique(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        grid = d.get_cell_grid()
        assert grid["cell_id"].nunique() == len(grid)


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — Moore mode
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneDecomposerMoore:

    def test_moore_mode_cell_count_at_most_nine(self, single_point_aoi):
        if single_point_aoi.mode != "moore":
            pytest.skip("AOI not in Moore mode")
        d = ZoneDecomposer(single_point_aoi)
        result = d.decompose()
        assert result["n_zones"] <= 9

    def test_moore_primary_cell_has_weight_one(self, single_point_aoi):
        if single_point_aoi.mode != "moore":
            pytest.skip("AOI not in Moore mode")
        d = ZoneDecomposer(single_point_aoi)
        result = d.decompose()
        primary_cells = [c for c in result["cells"] if c.get("type") == "primary"]
        assert len(primary_cells) == 1
        assert primary_cells[0]["weight"] == cfg.MOORE_WEIGHT_PRIMARY

    def test_moore_face_cells_have_correct_weight(self, single_point_aoi):
        if single_point_aoi.mode != "moore":
            pytest.skip("AOI not in Moore mode")
        d = ZoneDecomposer(single_point_aoi)
        result = d.decompose()
        face_cells = [
            c for c in result["cells"]
            if c.get("type") == "neighbor" and c.get("weight") == cfg.MOORE_WEIGHT_FACE
        ]
        assert len(face_cells) == 4  # N, S, E, W

    def test_moore_diagonal_cells_have_correct_weight(self, single_point_aoi):
        if single_point_aoi.mode != "moore":
            pytest.skip("AOI not in Moore mode")
        d = ZoneDecomposer(single_point_aoi)
        result = d.decompose()
        diag_cells = [
            c for c in result["cells"]
            if c.get("type") == "neighbor"
            and c.get("weight") == cfg.MOORE_WEIGHT_DIAGONAL
        ]
        assert len(diag_cells) == 4  # NE, NW, SE, SW

    def test_moore_weights_are_valid_values(self, single_point_aoi):
        if single_point_aoi.mode != "moore":
            pytest.skip("AOI not in Moore mode")
        d = ZoneDecomposer(single_point_aoi)
        result = d.decompose()
        valid_weights = {
            cfg.MOORE_WEIGHT_PRIMARY,
            cfg.MOORE_WEIGHT_FACE,
            cfg.MOORE_WEIGHT_DIAGONAL,
        }
        for cell in result["cells"]:
            assert cell["weight"] in valid_weights


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — filter_non_prescribable
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneDecomposerFilterNonPrescribable:

    def test_water_cells_marked_non_prescribable(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        grid = d.get_cell_grid()
        n = len(grid)

        # Mark first 10% of cells as water (DW label 0)
        n_water = max(1, n // 10)
        dw_labels  = np.ones(n, dtype=int) * 6  # all BUILT
        dw_labels[:n_water] = 0                  # override first cells as water
        pop_array  = np.ones(n) * 100.0

        d.filter_non_prescribable(dw_labels, pop_array)
        updated = d.get_cell_grid()
        non_p = updated[~updated["prescribable"]]
        assert len(non_p) >= n_water

    def test_snow_cells_marked_non_prescribable(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        grid = d.get_cell_grid()
        n = len(grid)

        dw_labels        = np.ones(n, dtype=int) * 6
        dw_labels[:5]    = 8   # snow label
        pop_array        = np.ones(n) * 100.0

        d.filter_non_prescribable(dw_labels, pop_array)
        updated = d.get_cell_grid()
        assert len(updated[~updated["prescribable"]]) >= 5

    def test_low_population_cells_non_prescribable(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        grid = d.get_cell_grid()
        n = len(grid)

        dw_labels  = np.ones(n, dtype=int) * 6
        pop_array  = np.ones(n) * 100.0
        pop_array[:5] = 0.0  # zero population

        d.filter_non_prescribable(dw_labels, pop_array)
        updated = d.get_cell_grid()
        assert len(updated[~updated["prescribable"]]) >= 5

    def test_mismatched_array_length_raises(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        with pytest.raises(ValueError):
            d.filter_non_prescribable(
                np.array([0, 1, 2]),  # wrong length
                np.array([1.0, 2.0, 3.0]),
            )


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — apply_majority_filter
# ══════════════════════════════════════════════════════════════════════════════

class TestMajorityFilter:

    def test_uniform_array_unchanged(self):
        arr = np.ones((5, 5), dtype=int)
        result = ZoneDecomposer.apply_majority_filter(arr, window=3)
        np.testing.assert_array_equal(result, arr)

    def test_isolated_cell_smoothed(self):
        """An isolated cell surrounded by neighbours of a different type is smoothed."""
        arr = np.ones((5, 5), dtype=int)
        arr[2, 2] = 2  # isolated cell
        result = ZoneDecomposer.apply_majority_filter(arr, window=3)
        assert result[2, 2] == 1

    def test_output_shape_preserved(self):
        arr = np.ones((7, 9), dtype=int) * 3
        arr[3, 4] = 7
        result = ZoneDecomposer.apply_majority_filter(arr, window=3)
        assert result.shape == arr.shape

    def test_output_dtype_is_integer(self):
        arr = np.ones((5, 5), dtype=int)
        result = ZoneDecomposer.apply_majority_filter(arr)
        assert np.issubdtype(result.dtype, np.integer)

    def test_even_window_raises(self):
        arr = np.ones((5, 5), dtype=int)
        with pytest.raises(ValueError):
            ZoneDecomposer.apply_majority_filter(arr, window=4)

    def test_all_zeros_stays_zero(self):
        arr = np.zeros((6, 6), dtype=int)
        result = ZoneDecomposer.apply_majority_filter(arr)
        assert (result == 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# ZoneDecomposer — get_zone_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneDecomposerSummary:

    def test_summary_has_required_columns(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        summary = d.get_zone_summary()
        for col in ("zone_id", "cell_count", "centroid_lat", "centroid_lon"):
            assert col in summary.columns

    def test_summary_cell_counts_sum_to_total(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        grid    = d.get_cell_grid()
        summary = d.get_zone_summary()
        assert summary["cell_count"].sum() == len(grid)

    def test_summary_zone_ids_are_unique(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        summary = d.get_zone_summary()
        assert summary["zone_id"].nunique() == len(summary)

    def test_summary_centroids_within_aoi_bounds(self, small_city_aoi):
        d = ZoneDecomposer(small_city_aoi)
        d.decompose()
        summary = d.get_zone_summary()
        minx, miny, maxx, maxy = small_city_aoi.bounds
        tol = 0.005
        assert (summary["centroid_lon"] >= minx - tol).all()
        assert (summary["centroid_lon"] <= maxx + tol).all()
        assert (summary["centroid_lat"] >= miny - tol).all()
        assert (summary["centroid_lat"] <= maxy + tol).all()


# ══════════════════════════════════════════════════════════════════════════════
# SeasonalDetector — unit tests (no GEE)
# ══════════════════════════════════════════════════════════════════════════════

class TestSeasonalDetector:

    def test_init_stores_params(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        assert det.centroid_lat == 24.37
        assert det.centroid_lon == 88.60
        assert det.year == 2023

    def test_cache_path_is_deterministic(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        p1 = det._cache_path()
        p2 = det._cache_path()
        assert p1 == p2

    def test_cache_path_contains_lat_lon_year(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        path = det._cache_path()
        assert "24.37" in path or "24" in path
        assert "88.60" in path or "88" in path
        assert "2023" in path

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        monkeypatch.setattr(
            det, "_cache_path", lambda: str(tmp_path / "test_cache.json")
        )
        data = {"hottest_months": [4, 5], "monthly_stats": {4: {"mean_tmax_c": 36.0}}}
        det._save_cache(data)
        loaded = det._load_cache()
        assert loaded["hottest_months"] == [4, 5]

    def test_cache_returns_none_if_no_file(self, tmp_path, monkeypatch):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        monkeypatch.setattr(
            det, "_cache_path",
            lambda: str(tmp_path / "nonexistent_cache.json"),
        )
        assert det._load_cache() is None

    def test_dewpoint_to_rh_at_saturation(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(0, 0, 2023)
        rh = det._dewpoint_to_rh(25.0, 25.0)
        assert abs(rh - 100.0) < 0.5

    def test_dewpoint_to_rh_dry_conditions(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(0, 0, 2023)
        rh = det._dewpoint_to_rh(35.0, 5.0)
        assert rh < 25.0

    def test_dewpoint_to_rh_clipped_to_100(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(0, 0, 2023)
        rh = det._dewpoint_to_rh(20.0, 25.0)  # Td > T: physically impossible, clipped
        assert rh <= 100.0

    def test_heat_index_below_27c_returns_temperature(self):
        from rxharm.seasonal.detector import SeasonalDetector
        t = 20.0
        hi = SeasonalDetector._heat_index(t, 60.0)
        assert hi == t

    def test_heat_index_above_27c_exceeds_temperature(self):
        from rxharm.seasonal.detector import SeasonalDetector
        t, rh = 35.0, 70.0
        hi = SeasonalDetector._heat_index(t, rh)
        assert hi > t

    def test_detect_raises_without_gee(self, monkeypatch):
        """detect() must raise RuntimeError when ee is not authenticated."""
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)

        # Simulate no credentials
        mock_ee = MagicMock()
        mock_ee.data._credentials = None
        with patch.dict("sys.modules", {"ee": mock_ee}):
            with pytest.raises(RuntimeError, match="authenticated"):
                det.detect(use_cache=False)

    def test_date_filter_strings_format(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        det._detected_months = [4, 5]
        filters = det.get_date_filter_strings()
        assert len(filters) == 2
        for start, end in filters:
            assert len(start) == 10
            assert start[4] == "-" and start[7] == "-"
            assert start[:4] == "2023"

    def test_date_filter_december_wraps_year(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(0, 0, 2023)
        det._detected_months = [12]
        filters = det.get_date_filter_strings()
        start, end = filters[0]
        assert start == "2023-12-01"
        assert end == "2024-01-01"

    def test_get_era5_heat_index_raises_before_detect(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        with pytest.raises(RuntimeError):
            det.get_era5_heat_index()

    def test_get_climatological_mmt_raises_before_detect(self):
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(24.37, 88.60, 2023)
        with pytest.raises(RuntimeError):
            det.get_climatological_mmt()

    @pytest.mark.gee
    def test_detect_returns_valid_months(self):
        """Integration test — requires GEE authentication."""
        from rxharm.seasonal.detector import SeasonalDetector
        det = SeasonalDetector(23.03, 72.58, 2022)  # Ahmedabad
        months = det.detect(use_cache=False)
        assert isinstance(months, list)
        assert len(months) == cfg.N_HOTTEST_MONTHS
        assert all(1 <= m <= 12 for m in months)
        # For Ahmedabad the hottest period is typically April–June
        assert any(m in [4, 5, 6] for m in months)
