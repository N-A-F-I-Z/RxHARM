"""
rxharm/index/hvi.py
====================
Heat Vulnerability Index (HVI) computation engine for Project RxHARM.

Formula:
    HVI = E * S / max(AC, AC_FLOOR)
    (all sub-indices normalised [0,1]; AC denominator reduces HVI for
    high-coping-capacity areas)

All sub-indices are weighted sums of their normalised indicators.
The final HVI is re-normalised to [0,1] for spatial comparability.

Dependencies: numpy, pandas, rxharm.index.normalizer, rxharm.index.weighter
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from rxharm.config import (
    AC_FLOOR,
    ADAPTIVE_CAPACITY_WEIGHTS,
    EXPOSURE_WEIGHTS,
    HAZARD_WEIGHTS,
    SENSITIVITY_WEIGHTS,
)
from rxharm.index.normalizer import NormalizerEngine
from rxharm.index.weighter import WeighterEngine

# Sub-index → indicator keys mapping (must match indicator_registry.json)
_SUB_INDEX_INDICATORS: Dict[str, list] = {
    "hazard":            list(HAZARD_WEIGHTS.keys()),
    "exposure":          list(EXPOSURE_WEIGHTS.keys()),
    "sensitivity":       list(SENSITIVITY_WEIGHTS.keys()),
    "adaptive_capacity": list(ADAPTIVE_CAPACITY_WEIGHTS.keys()),
}


class HVIEngine:
    """
    Computes HVI and all sub-indices from raw indicator arrays.

    Parameters
    ----------
    weighting_method : str
        One of ``'equal'``, ``'pca'``, ``'entropy'``, ``'critic'``, ``'manual'``.
    user_weights : dict, optional
        Required when ``weighting_method='manual'``.
        ``{sub_index: {indicator: weight}}``
    """

    def __init__(
        self,
        weighting_method: str = "equal",
        user_weights: Optional[Dict] = None,
    ) -> None:
        self.weighting_method = weighting_method
        self.user_weights = user_weights
        self.normalizer = NormalizerEngine()
        self.weighter   = WeighterEngine(weighting_method)

    # ── Public entry point ─────────────────────────────────────────────────────

    def compute_all(self, indicator_arrays: Dict[str, np.ndarray]) -> dict:
        """
        Normalise all 14 indicators, compute sub-indices, and compute HVI.

        Parameters
        ----------
        indicator_arrays : dict
            ``{indicator_name: raw_ndarray}`` — raw physical-unit arrays.
            Must contain (at minimum) all indicators in HAZARD_WEIGHTS,
            EXPOSURE_WEIGHTS, SENSITIVITY_WEIGHTS, ADAPTIVE_CAPACITY_WEIGHTS.

        Returns
        -------
        dict
            Keys:
                ``'H_s'``, ``'E'``, ``'S'``, ``'AC'``  — sub-indices [0,1]
                ``'HVI'``                                — composite [0,1]
                ``'HVI_raw'``                            — before final normalisation
                ``'indicator_normalized'``               — dict of 14 normalised arrays
                ``'weights_used'``                       — dict of per-sub-index weights
                ``'stats'``                              — DataFrame from normalizer
        """
        # FIX v0.1.0: Validate band statistics before any computation.
        # This catches silent GEE failures that produce plausible-looking but
        # wrong HVI maps (e.g. all-zero LST due to failed collection fetch).
        try:
            from rxharm.fetch.validator import validate_indicator_arrays
            validate_indicator_arrays(indicator_arrays, aoi_name="current AOI")
        except ImportError:
            pass  # FIX v0.1.0: validator not required at import time (lazy import)

        # Step 1: normalise all indicators
        normed = self.normalizer.normalize_batch(indicator_arrays)

        # Step 2: build sub-indices
        weights_used = {}
        sub_indices   = {}
        for si_name in ("hazard", "exposure", "sensitivity", "adaptive_capacity"):
            si_keys = [k for k in _SUB_INDEX_INDICATORS[si_name] if k in normed]
            si_arrays = {k: normed[k] for k in si_keys}
            w = self.weighter.compute_weights(si_arrays, si_name)
            weights_used[si_name] = w
            sub_indices[si_name]  = self.compute_subindex(si_arrays, si_name)

        H_s = sub_indices["hazard"]
        E   = sub_indices["exposure"]
        S   = sub_indices["sensitivity"]
        AC  = sub_indices["adaptive_capacity"]

        # Step 3: HVI formula
        hvi_raw = self._compute_hvi_formula(E, S, AC)

        # Step 4: re-normalise HVI to [0,1]
        hvi_norm = self._minmax_norm(hvi_raw)

        return {
            "H_s":                  H_s,
            "E":                    E,
            "S":                    S,
            "AC":                   AC,
            "HVI":                  hvi_norm,
            "HVI_raw":              hvi_raw,
            "indicator_normalized": normed,
            "weights_used":         weights_used,
            "stats":                self.normalizer.get_stats(),
        }

    def compute_subindex(
        self,
        normalized_arrays: Dict[str, np.ndarray],
        sub_index: str,
    ) -> np.ndarray:
        """
        Compute a single sub-index as a weighted sum of normalised indicators.

        Parameters
        ----------
        normalized_arrays : dict
            ``{indicator_name: normalised_ndarray}`` for this sub-index only.
        sub_index : str
            Sub-index name passed to WeighterEngine.

        Returns
        -------
        np.ndarray
            Weighted composite in [0, 1].
        """
        if not normalized_arrays:
            # Empty — return zeros
            return np.zeros(1)

        weights = self.weighter.compute_weights(normalized_arrays, sub_index)
        # REASON: Use nansum so that a single NaN indicator does not erase the
        # entire cell; weight is renormalised per-pixel over valid indicators.
        shape    = next(iter(normalized_arrays.values())).shape
        weighted = np.zeros(shape)
        w_sum    = np.zeros(shape)

        for name, arr in normalized_arrays.items():
            w = weights.get(name, 0.0)
            valid = np.isfinite(arr)
            weighted = np.where(valid, weighted + w * arr, weighted)
            w_sum    = np.where(valid, w_sum + w, w_sum)

        # Normalise by actual weight sum (handles missing pixels)
        safe_wsum = np.where(w_sum > 1e-12, w_sum, np.nan)
        return np.where(np.isfinite(safe_wsum), weighted / safe_wsum, np.nan)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_hvi_formula(
        self,
        E: np.ndarray,
        S: np.ndarray,
        AC: np.ndarray,
    ) -> np.ndarray:
        """
        Apply HVI = E * S / max(AC, AC_FLOOR).

        REASON: Multiplicative form captures compound risk correctly.
        If E=0 (no population), HVI=0 regardless of S and AC.
        If AC is maximal, S/AC is small, reducing risk — correct behaviour.
        AC floor prevents division by zero and represents the minimum
        coping capacity even in the most deprived areas.

        FIX v0.1.0: Added explicit AC_FLOOR > 0 guard. A misconfigured
        config.py with AC_FLOOR = 0 would cause division by zero in all
        cells where AC normalises to exactly 0 (e.g., unpopulated water cells).
        Also added np.where Inf/NaN guard for extreme floating-point edge cases.
        """
        # FIX v0.1.0: Defensive check — AC_FLOOR must be strictly positive
        if AC_FLOOR <= 0:
            raise ValueError(
                f"config.AC_FLOOR = {AC_FLOOR} is invalid (must be > 0). "
                "Set AC_FLOOR = 0.01 in rxharm/config.py."
            )

        AC_safe = np.maximum(np.where(np.isfinite(AC), AC, 0.0), AC_FLOOR)
        hvi_raw = E * S / AC_safe

        # FIX v0.1.0: Guard against residual Inf/NaN from extreme edge cases
        # (e.g., E=1.0, S=1.0, AC rounds to exactly AC_FLOOR due to floating point)
        hvi_raw = np.where(np.isfinite(hvi_raw), hvi_raw, 0.0)

        # Propagate NaN where any input is NaN (original behaviour preserved)
        nan_mask = ~(np.isfinite(E) & np.isfinite(S) & np.isfinite(AC))
        return np.where(nan_mask, np.nan, hvi_raw)

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0,1], preserving NaN."""
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return arr
        lo, hi = valid.min(), valid.max()
        if hi - lo < 1e-12:
            return np.where(np.isfinite(arr), 0.5, np.nan)
        return np.where(np.isfinite(arr), (arr - lo) / (hi - lo), np.nan)

    # ── Sensitivity analysis ───────────────────────────────────────────────────

    def sensitivity_test(
        self,
        indicator_arrays: Dict[str, np.ndarray],
        perturbation: float = 0.20,
    ) -> pd.DataFrame:
        """
        Perturb each indicator weight by ±perturbation and measure rank-stability.

        Parameters
        ----------
        indicator_arrays : dict
            Raw indicator arrays.
        perturbation : float
            Fractional weight perturbation (default 0.20 = ±20%).

        Returns
        -------
        pd.DataFrame
            Columns: indicator, weight_perturb_direction, spearman_r
        """
        from scipy.stats import spearmanr

        nominal_hvi = self.compute_all(indicator_arrays)["HVI"].ravel()
        valid_mask  = np.isfinite(nominal_hvi)

        rows = []
        all_weights = {
            "hazard":            HAZARD_WEIGHTS.copy(),
            "exposure":          EXPOSURE_WEIGHTS.copy(),
            "sensitivity":       SENSITIVITY_WEIGHTS.copy(),
            "adaptive_capacity": ADAPTIVE_CAPACITY_WEIGHTS.copy(),
        }
        for si_name, weights in all_weights.items():
            for ind in weights:
                for direction, sign in [("+", 1), ("-", -1)]:
                    perturbed = weights.copy()
                    perturbed[ind] = max(0.01, weights[ind] + sign * perturbation)
                    total = sum(perturbed.values())
                    perturbed = {k: v / total for k, v in perturbed.items()}
                    # Inject temporary override
                    self.weighter._manual_override = {si_name: perturbed}
                    try:
                        perturbed_hvi = self.compute_all(indicator_arrays)["HVI"].ravel()
                        both_valid = valid_mask & np.isfinite(perturbed_hvi)
                        if both_valid.sum() > 10:
                            r, _ = spearmanr(nominal_hvi[both_valid], perturbed_hvi[both_valid])
                        else:
                            r = np.nan
                    finally:
                        self.weighter._manual_override = None
                    rows.append({
                        "indicator":              ind,
                        "sub_index":              si_name,
                        "weight_perturb_direction": direction,
                        "spearman_r":             float(r) if np.isfinite(r) else np.nan,
                    })
        return pd.DataFrame(rows)
