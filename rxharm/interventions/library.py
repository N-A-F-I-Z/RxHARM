"""
rxharm/interventions/library.py
================================
Loads and manages the intervention library from intervention_library.json.

Provides:
    - Monte Carlo sampling of effectiveness and cost parameters
    - Post-intervention state computation for objective functions
    - Spillover cooling computation for spatially extensive interventions

Dependencies: json, numpy, scipy
"""

from __future__ import annotations

import json
import numpy as np
from scipy.stats import triang
from typing import Any, Dict, Optional

from rxharm.config import (
    ADAPTIVE_CAPACITY_WEIGHTS,
    INTERVENTION_LIBRARY_PATH,
)


class InterventionLibrary:
    """
    Loads and exposes the RxHARM intervention library.

    Parameters
    ----------
    path : str, optional
        Override default path to intervention_library.json.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        fpath = path or INTERVENTION_LIBRARY_PATH
        with open(fpath) as f:
            self._raw = json.load(f)
        self.sr     = self._raw["short_run"]   # 5 SR interventions
        self.lr     = self._raw["long_run"]    # 5 LR interventions
        self.budget: Optional[float] = None    # set by optimizer

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_lr_interventions(self) -> dict:
        return self.lr

    def get_sr_interventions(self) -> dict:
        return self.sr

    # ── Monte Carlo sampling ───────────────────────────────────────────────────

    def sample_effectiveness(self, rng: Optional[np.random.Generator] = None) -> dict:
        """
        Sample all effectiveness parameters from triangular distributions.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        dict
            ``{intervention_key: {param_name: sampled_float}}``
        """
        if rng is None:
            rng = np.random.default_rng()

        samples: Dict[str, dict] = {}
        for scope in ["short_run", "long_run"]:
            interventions = self.sr if scope == "short_run" else self.lr
            for key, interv in interventions.items():
                samples[key] = {}
                grp_a = interv.get("group_A_spatial_effects", {})
                for param, val in grp_a.items():
                    if isinstance(val, list) and len(val) == 3:
                        lo, mode, hi = val
                        if hi > lo:
                            c = (mode - lo) / (hi - lo)
                            samples[key][param] = float(
                                triang.rvs(c, loc=lo, scale=(hi - lo), random_state=rng)
                            )
                        else:
                            samples[key][param] = float(mode)
        return samples

    def sample_costs(self, rng: Optional[np.random.Generator] = None) -> dict:
        """
        Sample cost_per_unit_usd from triangular distributions.

        Returns
        -------
        dict
            ``{intervention_key: sampled_cost_usd}``
        """
        if rng is None:
            rng = np.random.default_rng()

        costs = {}
        for scope in ["short_run", "long_run"]:
            interventions = self.sr if scope == "short_run" else self.lr
            for key, interv in interventions.items():
                grp_c = interv.get("group_C_cost", {})
                val   = grp_c.get("cost_per_unit_usd", [0, 0, 0])
                if isinstance(val, list) and len(val) == 3 and val[2] > val[0]:
                    lo, mode, hi = val
                    c = (mode - lo) / (hi - lo)
                    costs[key] = float(triang.rvs(c, loc=lo, scale=hi - lo, random_state=rng))
                else:
                    costs[key] = float(val[1] if isinstance(val, list) else val)
        return costs

    # ── Post-intervention state ────────────────────────────────────────────────

    def compute_post_intervention_state(
        self,
        x: np.ndarray,
        cell_states: dict,
        eff_sample: dict,
    ) -> dict:
        """
        Apply LR interventions to indicator arrays and return updated state.

        Parameters
        ----------
        x : np.ndarray
            Decision variable array, shape ``(n_zones, n_lr_interventions)``.
        cell_states : dict
            Current indicator arrays (normalised). Must include ``'lst'``,
            ``'ndvi'``, ``'tree_cover'``, ``'ndwi'``, ``'AC'``, ``'HVI'``.
        eff_sample : dict
            From sample_effectiveness().

        Returns
        -------
        dict
            Updated arrays: ``'lst_post'``, ``'ndvi_post'``, ``'tree_cover_post'``,
            ``'ndwi_post'``, ``'ac_post'``, ``'hvi_post'``
        """
        state = {k: v.copy() for k, v in cell_states.items()}
        interv_list = list(self.lr.keys())  # LR1…LR5 in order

        for k, interv_key in enumerate(interv_list):
            if k >= x.shape[1]:
                break
            qty = x[:, k]   # shape (n_zones,)
            eff = eff_sample.get(interv_key, {})
            params = self.lr[interv_key].get("group_A_spatial_effects", {})

            # LST effect
            delta_lst_tri = params.get("delta_LST_per_unit", [0, 0, 0])
            if isinstance(delta_lst_tri, list):
                delta_per_unit = eff.get("delta_LST_per_unit", delta_lst_tri[1])
            else:
                delta_per_unit = float(delta_lst_tri)

            if "lst" in state:
                state["lst_post"] = state.get("lst_post", state["lst"]) + qty * delta_per_unit

            # NDVI effect
            delta_ndvi_tri = params.get("delta_NDVI_per_unit", [0, 0, 0])
            if isinstance(delta_ndvi_tri, list):
                delta_ndvi = eff.get("delta_NDVI_per_unit", delta_ndvi_tri[1])
            else:
                delta_ndvi = float(delta_ndvi_tri)
            if "ndvi" in state and delta_ndvi:
                state["ndvi_post"] = np.clip(
                    state.get("ndvi_post", state["ndvi"]) + qty * delta_ndvi, 0, 1
                )

            # Tree cover effect
            delta_tc_tri = params.get("delta_tree_cover_per_unit", [0, 0, 0])
            if isinstance(delta_tc_tri, list):
                delta_tc = eff.get("delta_tree_cover_per_unit", delta_tc_tri[1])
            else:
                delta_tc = float(delta_tc_tri)
            if "tree_cover" in state and delta_tc:
                state["tree_cover_post"] = np.clip(
                    state.get("tree_cover_post", state["tree_cover"]) + qty * delta_tc, 0, 100
                )

        # Recompute AC sub-index proxy
        state["ac_post"] = self._recompute_ac(state)

        # Recompute HVI proxy
        E  = state.get("E",  np.ones_like(state.get("ac_post", np.ones(1))))
        S  = state.get("S",  np.ones_like(E))
        AC = state.get("ac_post", state.get("AC", np.ones_like(E)))
        from rxharm.config import AC_FLOOR
        AC_safe = np.maximum(np.where(np.isfinite(AC), AC, AC_FLOOR), AC_FLOOR)
        hvi_raw = E * S / AC_safe
        lo, hi  = np.nanmin(hvi_raw), np.nanmax(hvi_raw)
        state["hvi_post"] = (
            (hvi_raw - lo) / (hi - lo + 1e-12) if hi > lo else np.full_like(hvi_raw, 0.5)
        )
        return state

    def _recompute_ac(self, state: dict) -> np.ndarray:
        """Recompute a simple AC proxy from updated indicator values."""
        weights = ADAPTIVE_CAPACITY_WEIGHTS
        components = []
        total_w = 0.0
        for ind, w in weights.items():
            key = ind + "_post" if ind + "_post" in state else ind
            if key in state:
                components.append(state[key] * w)
                total_w += w
        if not components or total_w < 1e-9:
            return state.get("AC", np.ones(1))
        return sum(components) / total_w

    def _apply_spillover(self, state, k, qty, eff, params, radius):
        """Simplified spillover: apply same delta to neighbouring zones."""
        return state  # full spatial spillover implemented in prescriber
