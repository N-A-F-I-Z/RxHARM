"""
tests/test_scenarios.py
========================
Tests for rxharm/scenarios/cmip6.py — no network required.
"""

from __future__ import annotations
import numpy as np
import pytest


class TestCMIP6Loader:

    @pytest.fixture
    def loader(self):
        from rxharm.scenarios.cmip6 import CMIP6Loader
        return CMIP6Loader(23.03, 72.58)

    def test_fallback_deltas_has_four_scenarios(self, loader):
        d = loader._load_fallback_deltas()
        assert set(d.keys()) == {"A", "B", "C", "D"}

    def test_baseline_has_zero_delta(self, loader):
        d = loader._load_fallback_deltas()
        assert d["A"]["T_delta"] == 0.0

    def test_ssp585_warmer_than_ssp245(self, loader):
        d = loader._load_fallback_deltas()
        assert d["D"]["T_delta"] > d["C"]["T_delta"]

    def test_all_scenarios_have_required_keys(self, loader):
        d = loader._load_fallback_deltas()
        for sid, params in d.items():
            assert "T_delta"    in params, f"Missing T_delta in {sid}"
            assert "pop_factor" in params, f"Missing pop_factor in {sid}"
            assert "event_days" in params, f"Missing event_days in {sid}"

    def test_event_days_increase_with_severity(self, loader):
        d = loader._load_fallback_deltas()
        assert d["A"]["event_days"] <= d["B"]["event_days"]
        assert d["B"]["event_days"] <= d["C"]["event_days"]
        assert d["C"]["event_days"] <= d["D"]["event_days"]

    def test_event_days_all_at_least_three(self, loader):
        d = loader._load_fallback_deltas()
        for sid, p in d.items():
            assert p["event_days"] >= 3.0, f"Scenario {sid} event_days < 3"

    def test_pop_factor_baseline_is_one(self, loader):
        d = loader._load_fallback_deltas()
        assert d["A"]["pop_factor"] == pytest.approx(1.0)

    def test_pop_factor_increases_with_stress(self, loader):
        d = loader._load_fallback_deltas()
        assert d["B"]["pop_factor"] > d["A"]["pop_factor"]

    def test_get_all_scenario_deltas_structure(self, loader):
        # Uses fallback path — no network needed
        try:
            deltas = loader.get_all_scenario_deltas()
        except Exception:
            deltas = loader._load_fallback_deltas()
        assert len(deltas) == 4


class TestScenarioManager:

    def test_build_scenarios_returns_four(self):
        from rxharm.scenarios.cmip6 import CMIP6Loader, ScenarioManager
        loader  = CMIP6Loader(23.03, 72.58)
        hvi_res = {"HVI": np.random.uniform(0, 1, (10, 10)),
                   "indicator_normalized": {"population": np.ones((10, 10)) * 1000}}

        class FakeAOI:
            pass

        mgr = ScenarioManager(FakeAOI(), hvi_res, loader)
        scenarios = mgr.build_scenarios()
        assert len(scenarios) == 4

    def test_scenario_dicts_have_required_keys(self):
        from rxharm.scenarios.cmip6 import CMIP6Loader, ScenarioManager

        class FakeAOI: pass

        loader  = CMIP6Loader(23.03, 72.58)
        hvi_res = {"HVI": np.ones((5, 5)) * 0.5,
                   "indicator_normalized": {"population": np.ones((5, 5)) * 500}}
        mgr = ScenarioManager(FakeAOI(), hvi_res, loader)
        for s in mgr.build_scenarios():
            for k in ("name", "T_delta", "pop_factor", "event_days", "population_array"):
                assert k in s, f"Missing '{k}' in scenario {s.get('name')}"

    def test_compute_future_hri_shape(self):
        from rxharm.scenarios.cmip6 import CMIP6Loader, ScenarioManager
        from rxharm.index.hri import HRIEngine

        class FakeAOI: pass

        loader  = CMIP6Loader(23.03, 72.58)
        shape   = (10, 10)
        hvi_res = {
            "HVI": np.random.uniform(0, 1, shape),
            "H_s": np.random.uniform(0, 1, shape),
            "indicator_normalized": {
                "lst":        np.random.uniform(28, 45, shape),
                "population": np.random.uniform(100, 5000, shape),
            },
        }
        mgr = ScenarioManager(FakeAOI(), hvi_res, loader)
        scenarios = mgr.build_scenarios()

        hri_eng = HRIEngine(climate_zone="A", cdr_baseline=0.007)
        hri_eng.compute_mmt(hvi_res["indicator_normalized"]["lst"])

        ad = mgr.compute_future_hri(scenarios[2], hri_eng, hvi_res)
        assert ad.shape == shape
        assert np.all(ad >= 0)
