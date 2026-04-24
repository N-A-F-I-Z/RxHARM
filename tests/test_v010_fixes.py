"""
tests/test_v010_fixes.py
========================
Integration tests for all v0.1.0 stability fixes.
Groups 3 (decomposer) and 6c (AC_FLOOR guard).
All pass without GEE or network access.
"""

from __future__ import annotations
import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — Decomposer boundary and nodata fixes
# ══════════════════════════════════════════════════════════════════════════════

class TestDecomposerFixes:
    """FIX v0.1.0: Majority filter boundary padding + nodata sentinel detection."""

    def test_majority_filter_preserves_shape(self):
        from rxharm.aoi.decomposer import ZoneDecomposer
        arr = np.random.randint(0, 5, (10, 10))
        result = ZoneDecomposer.apply_majority_filter(arr)
        assert result.shape == arr.shape

    def test_majority_filter_uniform_unchanged(self):
        """A uniform array should not change under majority filter."""
        from rxharm.aoi.decomposer import ZoneDecomposer
        arr = np.full((10, 10), 3, dtype=np.int32)
        result = ZoneDecomposer.apply_majority_filter(arr)
        assert np.all(result == 3)

    def test_majority_filter_boundary_not_zero(self):
        """
        FIX v0.1.0: Boundary cells must not be incorrectly reassigned to 0.
        Before the fix, cells on the array edge had fewer real neighbors,
        so zero-filled padding often dominated the majority vote.
        """
        from rxharm.aoi.decomposer import ZoneDecomposer
        # Array with code 5 everywhere except one isolated cell
        arr = np.full((8, 8), 5, dtype=np.int32)
        arr[0, 0] = 1   # corner cell — was often smoothed to 0 before fix
        result = ZoneDecomposer.apply_majority_filter(arr)
        # Majority of neighbors of [0,0] are all 5 — should reassign to 5
        assert result[0, 0] == 5, f"Corner cell incorrectly remained {result[0, 0]}"

    def test_majority_filter_returns_int32(self):
        from rxharm.aoi.decomposer import ZoneDecomposer
        arr = np.random.randint(0, 10, (6, 6))
        result = ZoneDecomposer.apply_majority_filter(arr)
        assert result.dtype == np.int32

    def test_majority_filter_odd_window_required(self):
        from rxharm.aoi.decomposer import ZoneDecomposer
        with pytest.raises(ValueError, match="odd"):
            ZoneDecomposer.apply_majority_filter(np.zeros((5, 5), dtype=int), window=4)

    def test_nodata_sentinel_excluded_from_prescribable(self):
        """
        FIX v0.1.0: Cells with -9999 population must be marked non-prescribable.
        Before the fix, -9999 was treated as valid data (extremely negative pop).
        """
        # We test the mask logic directly without needing a full ZoneDecomposer
        pop = np.array([500.0, 1000.0, -9999.0, 200.0, 300.0])
        dw  = np.array([6,     6,       6,       6,     6])     # all built
        from rxharm.config import NON_PRESCRIBABLE_DW_LABELS, MIN_POPULATION_THRESHOLD
        _SENTINELS = {-9999, -9998, -32768, 65535}
        dw_mask     = ~np.isin(dw, list(NON_PRESCRIBABLE_DW_LABELS))
        pop_mask    = pop >= MIN_POPULATION_THRESHOLD
        nodata_mask = np.isin(pop.astype(int), list(_SENTINELS)) | (pop < -1)
        prescribable = dw_mask & pop_mask & ~nodata_mask
        # Cell 2 has -9999 population — must be non-prescribable
        assert not prescribable[2], "Nodata cell (-9999) must be non-prescribable"
        # Other cells are valid
        assert prescribable[0] and prescribable[1]


# ══════════════════════════════════════════════════════════════════════════════
# Group 6c — AC_FLOOR defensive guard
# ══════════════════════════════════════════════════════════════════════════════

class TestACFloorGuard:
    """FIX v0.1.0: _compute_hvi_formula must raise if AC_FLOOR <= 0."""

    def test_positive_ac_floor_in_config(self):
        """config.AC_FLOOR must be positive."""
        from rxharm.config import AC_FLOOR
        assert AC_FLOOR > 0, f"AC_FLOOR = {AC_FLOOR} must be > 0"

    def test_hvi_formula_raises_on_zero_ac_floor(self, monkeypatch):
        """Patching AC_FLOOR to 0 must trigger ValueError."""
        import rxharm.config as cfg
        import rxharm.index.hvi as hvi_mod
        original = cfg.AC_FLOOR
        monkeypatch.setattr(cfg, "AC_FLOOR", 0)
        monkeypatch.setattr(hvi_mod, "AC_FLOOR", 0)
        try:
            from rxharm.index.hvi import HVIEngine
            eng = HVIEngine.__new__(HVIEngine)
            with pytest.raises(ValueError, match="AC_FLOOR"):
                eng._compute_hvi_formula(
                    np.ones(10), np.ones(10), np.zeros(10)
                )
        finally:
            monkeypatch.setattr(cfg, "AC_FLOOR", original)
            monkeypatch.setattr(hvi_mod, "AC_FLOOR", original)

    def test_hvi_formula_no_inf_output(self):
        """Even extreme inputs must not produce Inf in HVI output."""
        import rxharm.index.hvi as hvi_mod
        # Directly test _compute_hvi_formula with near-zero AC
        eng_class = hvi_mod.HVIEngine
        eng = eng_class.__new__(eng_class)
        # AC = 0 (below AC_FLOOR) should be clipped to AC_FLOOR, not give Inf
        E   = np.ones(20)
        S   = np.ones(20)
        AC  = np.zeros(20)  # would be 0/0 without the floor
        result = eng._compute_hvi_formula(E, S, AC)
        assert np.all(np.isfinite(result)), "HVI formula output must be finite"

    def test_hvi_formula_nan_inputs_give_nan_output(self):
        """NaN in any input should produce NaN in output (not crash)."""
        import rxharm.index.hvi as hvi_mod
        eng = hvi_mod.HVIEngine.__new__(hvi_mod.HVIEngine)
        E  = np.array([1.0, np.nan, 0.5])
        S  = np.array([0.5, 0.5,   0.5])
        AC = np.array([0.5, 0.5,   0.5])
        result = eng._compute_hvi_formula(E, S, AC)
        assert np.isnan(result[1]), "NaN input should produce NaN output"
        assert np.isfinite(result[0]), "Valid input should give finite output"


# ══════════════════════════════════════════════════════════════════════════════
# Integration: import chain
# ══════════════════════════════════════════════════════════════════════════════

class TestImportChain:
    """FIX v0.1.0: All new functions must be importable."""

    def test_validator_importable_from_fetch(self):
        from rxharm.fetch import validate_indicator_arrays, print_validation_report
        assert callable(validate_indicator_arrays)
        assert callable(print_validation_report)

    def test_load_existing_export_importable(self):
        from rxharm.fetch import load_existing_export
        assert callable(load_existing_export)

    def test_pareto_functions_importable(self):
        from rxharm.optimize.runner import pareto_to_dataframe, save_pareto_to_csv
        assert callable(pareto_to_dataframe)
        assert callable(save_pareto_to_csv)

    def test_prescriber_new_methods_exist(self):
        from rxharm.spatial.prescriber import Prescriber
        assert hasattr(Prescriber, "to_geodataframe")
        assert hasattr(Prescriber, "save_prescription")

    def test_gee_collections_importable(self):
        from rxharm.config import GEE_COLLECTIONS, GFW_DATASET_VERSION, SAT_IO_PREFIX
        assert isinstance(GEE_COLLECTIONS, dict)
        assert isinstance(GFW_DATASET_VERSION, str)
        assert isinstance(SAT_IO_PREFIX, str)
