"""
tests/test_interventions.py
============================
Tests for rxharm/interventions/library.py and feasibility.py.
All pass without GEE or network access.
"""

from __future__ import annotations
import numpy as np
import pytest


SHAPE = (20, 20)


def _dummy_arrays():
    rng = np.random.default_rng(7)
    return {
        "population":    rng.uniform(100, 3000, SHAPE),
        "built_frac":    rng.uniform(0.3, 0.9,  SHAPE),
        "ndvi":          rng.uniform(0.1, 0.6,  SHAPE),
        "canopy_height": rng.uniform(0, 15,     SHAPE),
        "elderly_frac":  rng.uniform(0.05, 0.30, SHAPE),
        "HRI":           rng.uniform(0.1, 0.9,  SHAPE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# InterventionLibrary
# ══════════════════════════════════════════════════════════════════════════════

class TestInterventionLibrary:

    @pytest.fixture
    def lib(self):
        from rxharm.interventions.library import InterventionLibrary
        return InterventionLibrary()

    def test_loads_five_sr_interventions(self, lib):
        assert len(lib.sr) == 5

    def test_loads_five_lr_interventions(self, lib):
        assert len(lib.lr) == 8

    def test_get_lr_interventions_returns_dict(self, lib):
        lr = lib.get_lr_interventions()
        assert isinstance(lr, dict)

    def test_effectiveness_sampling_all_keys_present(self, lib):
        samples = lib.sample_effectiveness(rng=np.random.default_rng(42))
        # At least SR + LR keys exist
        all_keys = set(lib.sr.keys()) | set(lib.lr.keys())
        for k in all_keys:
            assert k in samples

    def test_effectiveness_samples_finite(self, lib):
        samples = lib.sample_effectiveness(rng=np.random.default_rng(0))
        for key, params in samples.items():
            for pname, val in params.items():
                assert np.isfinite(val), f"{key}.{pname} is not finite"

    def test_sample_effectiveness_reproducible(self, lib):
        s1 = lib.sample_effectiveness(rng=np.random.default_rng(99))
        s2 = lib.sample_effectiveness(rng=np.random.default_rng(99))
        for k in s1:
            for p in s1[k]:
                assert abs(s1[k][p] - s2[k][p]) < 1e-10

    def test_post_intervention_state_keys(self, lib):
        rng    = np.random.default_rng(1)
        n_z, n_i = 5, 5
        x      = rng.uniform(0, 2, (n_z, n_i))
        arrays = {k: v.ravel()[:n_z] for k, v in _dummy_arrays().items()}
        eff    = lib.sample_effectiveness(rng=rng)
        state  = lib.compute_post_intervention_state(x, arrays, eff)
        assert "hvi_post" in state
        assert "ac_post"  in state

    def test_post_intervention_hvi_in_range(self, lib):
        rng = np.random.default_rng(2)
        n_z = 10
        x   = rng.uniform(0, 1, (n_z, 5))
        arrays = {k: v.ravel()[:n_z] for k, v in _dummy_arrays().items()}
        arrays["E"]   = np.random.uniform(0.2, 0.8, n_z)
        arrays["S"]   = np.random.uniform(0.2, 0.8, n_z)
        arrays["AC"]  = np.random.uniform(0.2, 0.8, n_z)
        arrays["HVI"] = np.random.uniform(0.1, 0.9, n_z)
        eff   = lib.sample_effectiveness(rng=rng)
        state = lib.compute_post_intervention_state(x, arrays, eff)
        hvi_post = state["hvi_post"]
        assert np.all(hvi_post >= -1e-9)
        assert np.all(hvi_post <= 1 + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# FeasibilityEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestFeasibilityEngine:

    @pytest.fixture
    def engine(self):
        from rxharm.interventions.feasibility import FeasibilityEngine
        arrays = _dummy_arrays()
        return FeasibilityEngine(arrays, aoi_handler=None)

    def test_sr_masks_have_five_keys(self, engine):
        masks = engine.compute_sr_masks()
        assert len(masks) == 5

    def test_lr_masks_have_five_keys(self, engine):
        masks = engine.compute_lr_masks()
        assert len(masks) == 8

    def test_all_masks_have_ten_keys(self, engine):
        masks = engine.compute_all_masks()
        assert len(masks) == 13

    def test_masks_are_boolean_arrays(self, engine):
        masks = engine.compute_all_masks()
        for k, m in masks.items():
            assert m.dtype == bool, f"{k} is not bool"

    def test_masks_correct_shape(self, engine):
        masks = engine.compute_all_masks()
        for k, m in masks.items():
            assert m.shape == SHAPE, f"{k} shape mismatch"

    def test_cool_roof_requires_high_built_frac(self):
        from rxharm.interventions.feasibility import FeasibilityEngine
        arrays = _dummy_arrays()
        arrays["built_frac"] = np.full(SHAPE, 0.01)  # very low
        engine = FeasibilityEngine(arrays, aoi_handler=None)
        mask   = engine.mask_cool_roof()
        assert mask.sum() == 0 or mask.mean() < 0.1

    def test_tree_planting_requires_open_space(self):
        from rxharm.interventions.feasibility import FeasibilityEngine
        arrays = _dummy_arrays()
        arrays["built_frac"] = np.full(SHAPE, 0.99)  # fully built
        engine = FeasibilityEngine(arrays, aoi_handler=None)
        mask   = engine.mask_tree_planting()
        assert mask.sum() == 0

    def test_masks_not_all_false(self, engine):
        masks = engine.compute_all_masks()
        for k, m in masks.items():
            assert m.sum() > 0, f"{k} is entirely False"
