"""
tests/test_objectives.py
=========================
Tests for rxharm/optimize/objectives.py, constraints.py, and prescriber.py.
All pass without GEE or network access.
"""

from __future__ import annotations
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Objective functions
# ══════════════════════════════════════════════════════════════════════════════

class TestF1MortalityReduction:

    def test_zero_reduction_returns_one(self):
        from rxharm.optimize.objectives import f1_mortality_reduction
        ad = np.array([1.0, 2.0, 3.0])
        assert f1_mortality_reduction(ad, ad) == pytest.approx(1.0, abs=0.01)

    def test_full_prevention_returns_zero(self):
        from rxharm.optimize.objectives import f1_mortality_reduction
        ad_base = np.array([1.0, 2.0, 3.0])
        ad_post = np.zeros(3)
        assert f1_mortality_reduction(ad_post, ad_base) == pytest.approx(0.0, abs=0.01)

    def test_thirty_percent_prevention(self):
        from rxharm.optimize.objectives import f1_mortality_reduction
        ad_base = np.random.uniform(0.1, 2.0, 50)
        ad_post = ad_base * 0.70
        f1 = f1_mortality_reduction(ad_post, ad_base)
        assert 0.0 < f1 < 1.0
        assert f1 == pytest.approx(0.70, abs=0.05)

    def test_range_is_zero_to_one(self):
        from rxharm.optimize.objectives import f1_mortality_reduction
        ad_base = np.random.uniform(0.1, 1.0, 100)
        ad_post = np.random.uniform(0.0, 0.1, 100)
        f1 = f1_mortality_reduction(ad_post, ad_base)
        assert 0.0 <= f1 <= 1.0

    def test_zero_baseline_returns_zero(self):
        from rxharm.optimize.objectives import f1_mortality_reduction
        assert f1_mortality_reduction(np.zeros(5), np.zeros(5)) == 0.0


class TestF3EquityGini:

    def test_perfect_equality_returns_zero(self):
        from rxharm.optimize.objectives import f3_equity_gini
        ad  = np.full(50, 1.0)
        pop = np.full(50, 1000.0)
        g   = f3_equity_gini(ad, pop)
        assert g == pytest.approx(0.0, abs=0.01)

    def test_maximum_concentration_near_one(self):
        from rxharm.optimize.objectives import f3_equity_gini
        # All risk concentrated in one cell
        ad  = np.zeros(50)
        ad[0] = 100.0
        pop = np.full(50, 1000.0)
        g   = f3_equity_gini(ad, pop)
        assert g > 0.85

    def test_range_zero_to_one(self):
        from rxharm.optimize.objectives import f3_equity_gini
        rng = np.random.default_rng(5)
        ad  = rng.uniform(0, 5, 100)
        pop = rng.uniform(100, 5000, 100)
        g   = f3_equity_gini(ad, pop)
        assert 0.0 <= g <= 1.0

    def test_two_cells_too_few_returns_zero(self):
        from rxharm.optimize.objectives import f3_equity_gini
        assert f3_equity_gini(np.array([1.0]), np.array([100.0])) == 0.0


class TestF4CobenefitEfficiency:

    def test_returns_nonpositive(self):
        from rxharm.optimize.objectives import f4_cobenefit_efficiency
        from rxharm.interventions.library import InterventionLibrary
        lib = InterventionLibrary()
        x   = np.ones((5, 5)) * 2.0
        f4  = f4_cobenefit_efficiency(x, lib, budget=1_000_000)
        assert f4 <= 0.0

    def test_more_green_interventions_more_cobenefit(self):
        from rxharm.optimize.objectives import f4_cobenefit_efficiency
        from rxharm.interventions.library import InterventionLibrary
        lib  = InterventionLibrary()
        x_lo = np.ones((5, 5)) * 0.1
        x_hi = np.ones((5, 5)) * 10.0
        f4_lo = f4_cobenefit_efficiency(x_lo, lib, budget=1_000_000)
        f4_hi = f4_cobenefit_efficiency(x_hi, lib, budget=1_000_000)
        assert f4_hi <= f4_lo  # more negative = more cobenefit


# ══════════════════════════════════════════════════════════════════════════════
# Constraints
# ══════════════════════════════════════════════════════════════════════════════

class TestConstraints:

    @pytest.fixture
    def lib(self):
        from rxharm.interventions.library import InterventionLibrary
        return InterventionLibrary()

    def test_c1_satisfied_when_under_budget(self, lib):
        from rxharm.optimize.constraints import c1_budget
        x = np.zeros((5, 5))  # zero allocations → zero cost
        assert c1_budget(x, lib, budget=1_000_000) <= 0.0

    def test_c1_violated_when_over_budget(self, lib):
        from rxharm.optimize.constraints import c1_budget
        x = np.full((5, 5), 1e6)  # very large allocations
        assert c1_budget(x, lib, budget=1.0) > 0.0

    def test_c2_satisfied_when_within_bounds(self, lib):
        from rxharm.optimize.constraints import c2_area_feasibility
        x       = np.ones((5, 5))
        max_q   = np.full(5, 10.0)
        assert c2_area_feasibility(x, max_q) == pytest.approx(0.0, abs=1e-9)

    def test_c2_violated_when_exceeds_max(self, lib):
        from rxharm.optimize.constraints import c2_area_feasibility
        x     = np.full((5, 5), 20.0)
        max_q = np.full(5, 5.0)
        assert c2_area_feasibility(x, max_q) > 0.0

    def test_c3_satisfied_when_no_overlap(self, lib):
        from rxharm.optimize.constraints import c3_mutual_exclusivity
        x = np.zeros((5, 5))
        assert c3_mutual_exclusivity(x, lib) == pytest.approx(0.0)

    def test_c4_satisfied_when_all_zero(self, lib):
        from rxharm.optimize.constraints import c4_minimum_viable_unit
        x = np.zeros((5, 5))
        assert c4_minimum_viable_unit(x, [1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_c4_violated_with_subviable(self, lib):
        from rxharm.optimize.constraints import c4_minimum_viable_unit
        x = np.zeros((5, 5))
        x[0, 0] = 0.3  # non-zero but below min=1.0
        viol = c4_minimum_viable_unit(x, [1.0, 1.0, 1.0, 1.0, 1.0])
        assert viol > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Prescriber
# ══════════════════════════════════════════════════════════════════════════════

class TestPrescriber:

    @pytest.fixture
    def prescriber(self):
        from rxharm.spatial.prescriber import Prescriber, INTERVENTION_CODES
        n_cells = 500
        n_zones = 20
        zone_assignments = np.repeat(np.arange(n_zones), n_cells // n_zones)
        hri    = np.random.uniform(0, 1, n_cells)
        masks  = {k: np.ones(n_cells, dtype=bool)
                  for k in INTERVENTION_CODES if k != "none"}
        return Prescriber(
            zone_structure={"zone_assignments": zone_assignments},
            hri_array=hri,
            feasibility_masks=masks,
        )

    def test_disaggregate_output_shape(self, prescriber):
        from rxharm.spatial.prescriber import INTERVENTION_CODES
        x = np.random.uniform(0, 5, (20, 5))
        names = [k for k in INTERVENTION_CODES if k != "none"][:5]
        out = prescriber.disaggregate(x, names)
        assert out.shape == (500,)

    def test_output_codes_in_valid_range(self, prescriber):
        from rxharm.spatial.prescriber import INTERVENTION_CODES
        x = np.random.uniform(0, 5, (20, 5))
        names = [k for k in INTERVENTION_CODES if k != "none"][:5]
        out = prescriber.disaggregate(x, names)
        assert out.min() >= 0
        assert out.max() <= 10

    def test_zero_allocation_gives_no_prescription(self, prescriber):
        from rxharm.spatial.prescriber import INTERVENTION_CODES
        x = np.zeros((20, 5))
        names = [k for k in INTERVENTION_CODES if k != "none"][:5]
        out = prescriber.disaggregate(x, names)
        assert np.all(out == 0)

    def test_majority_filter_1d(self):
        from rxharm.spatial.prescriber import Prescriber
        arr = np.zeros(36, dtype=int)
        arr[18] = 5  # single isolated cell
        smoothed = Prescriber.apply_majority_filter(arr)
        assert len(smoothed) == 36

    def test_majority_filter_preserves_uniform(self):
        from rxharm.spatial.prescriber import Prescriber
        arr = np.full((10, 10), 3, dtype=int)
        smoothed = Prescriber.apply_majority_filter(arr)
        assert np.all(smoothed == 3)

    def test_to_prescription_map_keys(self, prescriber):
        from rxharm.spatial.prescriber import INTERVENTION_CODES
        x = np.random.uniform(0, 5, (20, 5))
        names = [k for k in INTERVENTION_CODES if k != "none"][:5]
        out = prescriber.disaggregate(x, names)
        pm = prescriber.to_prescription_map(out, transform=None)
        assert "array" in pm
        assert "code_map" in pm
        assert "statistics" in pm


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — Objective math fixes (Group 2)
# ══════════════════════════════════════════════════════════════════════════════

class TestObjectiveMathFixes:
    """FIX v0.1.0: Tests for corrected objective function math."""

    def test_f5_never_negative(self):
        """f5 must always be >= 0 regardless of AD values."""
        import numpy as np
        from rxharm.optimize.objectives import f5_scenario_robustness

        class MockHRI:
            cdr = 0.007; mmt = 35.0; beta = 0.004
            def compute_attributable_fraction(self, T, hvi):
                dT = np.maximum(0, T - self.mmt)
                rr = np.exp(self.beta * dT) * (1 + 0.5 * hvi)
                return (rr - 1) / rr

        class MockLib:
            lr = {}
            def compute_post_intervention_state(self, x, states, eff):
                return {**states, "hvi_post": states.get("HVI", np.ones(x.shape[0]) * 0.5),
                        "lst_post": states.get("H_s", np.ones(x.shape[0]) * 0.5)}

        n = 10
        hvi_r = {
            "HVI": np.random.default_rng(7).uniform(0.1, 0.9, n),
            "H_s": np.random.default_rng(7).uniform(0.2, 0.8, n),
            "indicator_normalized": {"population": np.ones(n) * 500},
        }
        scenarios = [
            {"T_delta": 0.0, "pop_factor": 1.0, "event_days": 3.0},
            {"T_delta": 2.0, "pop_factor": 1.1, "event_days": 4.5},
        ]
        result = f5_scenario_robustness(
            np.zeros((n, 1)), scenarios, MockHRI(), hvi_r, MockLib(), {}
        )
        assert result >= 0, f"f5 must be >= 0, got {result}"
        assert np.isfinite(result), f"f5 must be finite, got {result}"

    def test_f5_identical_scenarios_returns_zero(self):
        """When all scenarios produce identical AD, f5 should be 0."""
        import numpy as np
        from rxharm.optimize.objectives import f5_scenario_robustness

        class MockHRI:
            cdr = 0.007; mmt = 35.0; beta = 0.004
            def compute_attributable_fraction(self, T, hvi):
                return np.full_like(T, 0.05)   # constant AF

        class MockLib:
            lr = {}
            def compute_post_intervention_state(self, x, states, eff):
                return {**states, "hvi_post": states.get("HVI", np.ones(x.shape[0]) * 0.5),
                        "lst_post": states.get("H_s", np.ones(x.shape[0]) * 0.5)}

        n = 10
        hvi_r = {
            "HVI": np.ones(n) * 0.5,
            "H_s": np.ones(n) * 0.5,
            "indicator_normalized": {"population": np.ones(n) * 1000},
        }
        scenarios = [
            {"T_delta": 1.0, "pop_factor": 1.0, "event_days": 3.0},
            {"T_delta": 1.0, "pop_factor": 1.0, "event_days": 3.0},
        ]
        result = f5_scenario_robustness(
            np.zeros((n, 1)), scenarios, MockHRI(), hvi_r, MockLib(), {}
        )
        assert abs(result) < 1e-6, f"Expected ~0, got {result}"

    def test_f5_old_denominator_would_have_failed(self):
        """Demonstrate that f5 no longer fails when ad_min is near zero."""
        import numpy as np
        from rxharm.optimize.objectives import f5_scenario_robustness

        class MockHRI:
            cdr = 0.007; mmt = 35.0; beta = 0.001
            def compute_attributable_fraction(self, T, hvi):
                dT = np.maximum(0, T - self.mmt)
                rr = np.exp(self.beta * dT)
                return (rr - 1) / rr   # near-zero AF → near-zero AD

        class MockLib:
            lr = {}
            def compute_post_intervention_state(self, x, states, eff):
                return {**states, "hvi_post": states.get("HVI", np.ones(x.shape[0]) * 0.01),
                        "lst_post": states.get("H_s", np.ones(x.shape[0]) * 0.01)}

        n = 5
        # Simulate scenario A (baseline) with near-zero AD — old code: 0/0 → NaN
        hvi_r = {
            "HVI": np.ones(n) * 0.01,
            "H_s": np.ones(n) * 30.5,   # just above MMT — tiny AF
            "indicator_normalized": {"population": np.ones(n) * 1},   # 1 person
        }
        scenarios = [
            {"T_delta": 0.0, "pop_factor": 1.0, "event_days": 3.0},
            {"T_delta": 5.0, "pop_factor": 1.2, "event_days": 5.0},
        ]
        result = f5_scenario_robustness(
            np.zeros((n, 1)), scenarios, MockHRI(), hvi_r, MockLib(), {}
        )
        # Old code would return NaN or negative; new code must return finite >= 0
        assert np.isfinite(result), f"f5 must be finite even near-zero AD, got {result}"
        assert result >= 0, f"f5 must be non-negative, got {result}"

    def test_f4_no_nan_at_zero_allocation(self):
        """f4 must not produce NaN when x is all zeros (start of optimization)."""
        import numpy as np
        from rxharm.optimize.objectives import f4_cobenefit_efficiency

        class MockLib:
            lr = {
                "LR1": {"group_D_cobenefits": {
                    "carbon_tCO2_per_unit_per_year": [0.8, 1.5, 2.5],
                    "stormwater_m3_per_unit_per_year": [200, 500, 900],
                }},
            }

        x_zeros = np.zeros((20, 1))
        result = f4_cobenefit_efficiency(x_zeros, MockLib(), budget=1_000_000)
        assert np.isfinite(result), f"f4 must be finite at x=0, got {result}"
        assert result <= 0, f"f4 should be <= 0 (no co-benefit → 0), got {result}"

    def test_f4_returns_zero_not_nan_when_both_zero(self):
        """f4 must return 0 (not NaN) when there are genuinely no co-benefits."""
        import numpy as np
        from rxharm.optimize.objectives import f4_cobenefit_efficiency

        class MockLib:
            lr = {
                "LR1": {"group_D_cobenefits": {
                    "carbon_tCO2_per_unit_per_year": [0, 0, 0],
                    "stormwater_m3_per_unit_per_year": [0, 0, 0],
                }},
            }

        x = np.ones((5, 1)) * 10.0
        result = f4_cobenefit_efficiency(x, MockLib(), budget=500_000)
        assert result == 0.0, f"Expected 0.0, got {result}"
        assert np.isfinite(result)


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — Pareto export (Group 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestParetoExport:
    """FIX v0.1.0: pareto_to_dataframe and save_pareto_to_csv."""

    def _mock_result(self, n_sol=10, n_obj=5):
        import numpy as np
        rng = np.random.default_rng(42)

        class MockResult:
            pass

        r = MockResult()
        r.F = rng.uniform(0, 1, (n_sol, n_obj))
        r.X = rng.uniform(0, 5, (n_sol, 8))
        return r

    def test_returns_dataframe(self):
        import pandas as pd
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result())
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result(n_sol=15))
        assert len(df) == 15

    def test_required_columns_present(self):
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result())
        for col in ("solution_id", "f1_mortality_fraction", "f2_cost_fraction",
                    "mortality_reduction_pct", "is_health_focused",
                    "is_budget_focused", "is_balanced"):
            assert col in df.columns, f"Missing column: {col}"

    def test_exactly_one_of_each_strategic_flag(self):
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result(n_sol=20))
        assert df["is_health_focused"].sum() == 1
        assert df["is_budget_focused"].sum()  == 1
        assert df["is_balanced"].sum()        == 1

    def test_mortality_reduction_pct_consistent(self):
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result())
        # mortality_reduction_pct = (1 - f1) * 100
        import numpy as np
        diff = abs(df["mortality_reduction_pct"] - (1 - df["f1_mortality_fraction"]) * 100)
        assert np.all(diff < 1e-9)

    def test_empty_result_returns_empty_df(self):
        import pandas as pd
        from rxharm.optimize.runner import pareto_to_dataframe

        class EmptyResult:
            X = None; F = None

        df = pareto_to_dataframe(EmptyResult())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_3obj_default_names(self):
        from rxharm.optimize.runner import pareto_to_dataframe
        df = pareto_to_dataframe(self._mock_result(n_obj=3))
        assert "f1_coverage_gap" in df.columns
        assert "f3_inequity_gini" in df.columns

    def test_save_pareto_to_csv(self, tmp_path):
        from rxharm.optimize.runner import save_pareto_to_csv
        import os
        path = str(tmp_path / "pareto.csv")
        returned = save_pareto_to_csv(self._mock_result(), path)
        assert returned == path
        assert os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# FIX v0.1.0 tests — Prescription GeoDataFrame (Group 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestPrescriptionGeoDataFrame:
    """FIX v0.1.0: Prescriber.to_geodataframe and save_prescription."""

    def _make_prescriber(self):
        import numpy as np
        from rxharm.spatial.prescriber import Prescriber, INTERVENTION_CODES
        n = 20
        return Prescriber(
            zone_structure={"zone_assignments": np.zeros(n, dtype=int)},
            hri_array=np.random.default_rng(0).uniform(0.3, 0.9, n),
            feasibility_masks={k: np.ones(n, dtype=bool)
                               for k in INTERVENTION_CODES if k != "none"},
        )

    def test_to_geodataframe_returns_gdf(self):
        import numpy as np
        try:
            import geopandas as gpd
        except ImportError:
            import pytest; pytest.skip("geopandas not installed")
        from rasterio.transform import from_bounds
        p = self._make_prescriber()
        prescription = np.random.randint(0, 5, (4, 5))
        transform = from_bounds(72.0, 22.9, 72.1, 23.0, 5, 4)
        gdf = p.to_geodataframe(prescription, transform)
        assert hasattr(gdf, "crs")
        assert "intervention_name" in gdf.columns
        assert "geometry" in gdf.columns
        assert "hri_value" in gdf.columns
        assert len(gdf) == 20

    def test_to_geodataframe_codes_in_range(self):
        import numpy as np
        try:
            import geopandas as gpd
        except ImportError:
            import pytest; pytest.skip("geopandas not installed")
        from rasterio.transform import from_bounds
        p = self._make_prescriber()
        prescription = np.random.randint(0, 11, (4, 5))
        transform = from_bounds(0, 0, 1, 1, 5, 4)
        gdf = p.to_geodataframe(prescription, transform)
        assert gdf["intervention_code"].between(0, 10).all()

    def test_to_geodataframe_no_hri(self):
        import numpy as np
        try:
            import geopandas as gpd
        except ImportError:
            import pytest; pytest.skip("geopandas not installed")
        from rasterio.transform import from_bounds
        p = self._make_prescriber()
        prescription = np.zeros((4, 5), dtype=int)
        transform = from_bounds(0, 0, 1, 1, 5, 4)
        gdf = p.to_geodataframe(prescription, transform, include_hri=False)
        assert "hri_value" not in gdf.columns
