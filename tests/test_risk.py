"""
tests/test_risk.py
==================
Tests for rxharm/risk/gfs_fetcher.py (no network required).
External API tests marked @pytest.mark.gee.
"""

from __future__ import annotations
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Heat Index formula
# ══════════════════════════════════════════════════════════════════════════════

class TestHeatIndexFormula:

    @pytest.fixture
    def gfs(self):
        from rxharm.risk.gfs_fetcher import GFSFetcher
        return GFSFetcher(23.03, 72.58)

    def test_hi_equals_temperature_below_threshold(self, gfs):
        T = np.array([20.0, 25.0, 26.9])
        RH = np.array([50.0, 60.0, 70.0])
        HI = gfs.compute_heat_index(T, RH)
        np.testing.assert_array_almost_equal(HI, T, decimal=5)

    def test_hi_greater_than_temperature_above_threshold(self, gfs):
        T = np.array([32.0, 38.0, 42.0])
        RH = np.array([70.0, 80.0, 75.0])
        HI = gfs.compute_heat_index(T, RH)
        assert np.all(HI > T)

    def test_hi_increases_with_humidity(self, gfs):
        T  = np.array([35.0, 35.0, 35.0])
        RH = np.array([40.0, 60.0, 80.0])
        HI = gfs.compute_heat_index(T, RH)
        assert HI[0] <= HI[1] <= HI[2]

    def test_hi_valid_range(self, gfs):
        T  = np.random.uniform(10, 50, 200)
        RH = np.random.uniform(10, 100, 200)
        HI = gfs.compute_heat_index(T, RH)
        assert np.all(HI > -50)
        assert np.all(HI < 100)

    def test_hi_output_same_shape_as_input(self, gfs):
        T  = np.ones((5, 10)) * 35.0
        RH = np.ones((5, 10)) * 70.0
        HI = gfs.compute_heat_index(T, RH)
        assert HI.shape == (5, 10)


# ══════════════════════════════════════════════════════════════════════════════
# Heatwave detection
# ══════════════════════════════════════════════════════════════════════════════

class TestHeatwaveDetection:
    """Uses synthetic forecast injected into _forecast_data."""

    @pytest.fixture
    def gfs_with_heatwave(self):
        import pandas as pd
        from datetime import datetime, timedelta
        from rxharm.risk.gfs_fetcher import GFSFetcher
        gfs = GFSFetcher(23.03, 72.58)
        base = datetime(2023, 5, 1)
        rows = []
        for fh in range(0, 169, 3):
            t  = base + timedelta(hours=fh)
            hi = 40.0 if (24 <= fh <= 96) else 30.0  # 3 days at 40°C
            rows.append({"valid_time": t, "T2m_C": hi - 2, "RH_pct": 70,
                         "HeatIndex_C": hi, "lat": 23.03, "lon": 72.58})
        gfs._forecast_data = pd.DataFrame(rows)
        return gfs

    @pytest.fixture
    def gfs_no_heatwave(self):
        import pandas as pd
        from datetime import datetime, timedelta
        from rxharm.risk.gfs_fetcher import GFSFetcher
        gfs = GFSFetcher(23.03, 72.58)
        base = datetime(2023, 5, 1)
        rows = [{"valid_time": base + timedelta(hours=h), "T2m_C": 30, "RH_pct": 50,
                 "HeatIndex_C": 31, "lat": 23.03, "lon": 72.58}
                for h in range(0, 169, 3)]
        gfs._forecast_data = pd.DataFrame(rows)
        return gfs

    def test_heatwave_detected_above_threshold(self, gfs_with_heatwave):
        result = gfs_with_heatwave.detect_heatwave(mmt=35.0, threshold_above_mmt=3.0)
        assert result["heatwave_detected"] is True

    def test_no_heatwave_below_threshold(self, gfs_no_heatwave):
        result = gfs_no_heatwave.detect_heatwave(mmt=35.0, threshold_above_mmt=3.0)
        assert result["heatwave_detected"] is False

    def test_event_days_counted_correctly(self, gfs_with_heatwave):
        result = gfs_with_heatwave.detect_heatwave(mmt=35.0, threshold_above_mmt=3.0)
        assert result["event_days"] >= 3

    def test_no_heatwave_returns_zero_days(self, gfs_no_heatwave):
        result = gfs_no_heatwave.detect_heatwave(mmt=35.0, threshold_above_mmt=3.0)
        assert result["event_days"] == 0

    def test_result_keys_present(self, gfs_with_heatwave):
        result = gfs_with_heatwave.detect_heatwave(mmt=35.0)
        for k in ("heatwave_detected", "start_date", "end_date",
                  "event_days", "max_HI", "mean_HI_excess"):
            assert k in result


# ══════════════════════════════════════════════════════════════════════════════
# GFS URL construction
# ══════════════════════════════════════════════════════════════════════════════

class TestGFSURLConstruction:

    @pytest.fixture
    def gfs(self):
        from rxharm.risk.gfs_fetcher import GFSFetcher
        return GFSFetcher(23.03, 72.58)

    def test_url_contains_nomads_domain(self, gfs):
        url = gfs._build_nomads_url("20231201", "00", 24)
        assert "nomads.ncep.noaa.gov" in url

    def test_url_contains_required_variables(self, gfs):
        url = gfs._build_nomads_url("20231201", "00", 24)
        assert "TMP" in url
        assert "DPT" in url

    def test_url_contains_bounding_box(self, gfs):
        url = gfs._build_nomads_url("20231201", "00", 24)
        assert "toplat" in url or "leftlon" in url

    def test_url_uses_given_run_date(self, gfs):
        url = gfs._build_nomads_url("20230601", "12", 0)
        assert "20230601" in url

    def test_valid_run_hours(self, gfs):
        for hour in ["00", "06", "12", "18"]:
            url = gfs._build_nomads_url("20231201", hour, 0)
            assert hour in url

    def test_forecast_hour_zero_padded(self, gfs):
        url = gfs._build_nomads_url("20231201", "00", 6)
        assert "006" in url


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic forecast
# ══════════════════════════════════════════════════════════════════════════════

class TestSyntheticForecast:

    def test_synthetic_has_correct_columns(self):
        from rxharm.risk.gfs_fetcher import GFSFetcher
        gfs = GFSFetcher(23.03, 72.58)
        df  = gfs._synthetic_forecast(48)
        for col in ("valid_time", "T2m_C", "RH_pct", "HeatIndex_C"):
            assert col in df.columns

    def test_synthetic_no_nan(self):
        from rxharm.risk.gfs_fetcher import GFSFetcher
        gfs = GFSFetcher(23.03, 72.58)
        df  = gfs._synthetic_forecast(48)
        assert not df[["T2m_C", "RH_pct", "HeatIndex_C"]].isna().any().any()

    def test_hri_scalar_returns_float(self):
        from rxharm.risk.gfs_fetcher import GFSFetcher
        gfs = GFSFetcher(23.03, 72.58)
        gfs._synthetic_forecast(48)
        scalar = gfs.get_hri_update_scalar(
            {"H_a_context": {"mean_HI_C": 38.0, "event_days": 3}}
        )
        assert isinstance(scalar, float)
        assert scalar >= 0
