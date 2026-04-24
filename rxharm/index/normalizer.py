"""
rxharm/index/normalizer.py
==========================
Indicator normalisation engine for Project RxHARM.

Transforms raw indicator arrays (in physical units) to [0, 1] for use
in the HVI sub-index formulae.

Algorithm per indicator:
    1. Clip to [p2, p98] percentile range (outlier suppression).
    2. Min-max scale to [0, 1].
    3. Preserve NaN (no-data) pixels unchanged.

Direction convention — ALL indicators are normalised with direction='positive'
(high normalised value = high quantity). The HVI formula HVI = E*S/AC handles
the AC denominator naturally: high AC lowers HVI without pre-inversion.

Dependencies: numpy, pandas
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ── Direction table ────────────────────────────────────────────────────────────
# All positive: normalised value represents magnitude of that quantity.
# High H/E/S = more vulnerable. High AC = more coping capacity (handled by formula).
INDICATOR_DIRECTIONS: Dict[str, str] = {
    # Hazard — high value = more heat stress
    "lst": "positive", "albedo": "positive", "uhi": "positive",
    # Exposure — high value = more people/infrastructure exposed
    "population": "positive", "built_frac": "positive",
    # Sensitivity — high value = more physiologically vulnerable
    "elderly_frac": "positive", "child_frac": "positive",
    "impervious": "positive", "cropland": "positive",
    # Adaptive Capacity — high value = more coping capacity
    # IMPORTANT: direction='positive' means high raw value → high normalised AC.
    # The HVI denominator (AC) naturally inverts this effect: high AC → lower HVI.
    "ndvi": "positive", "ndwi": "positive", "tree_cover": "positive",
    "canopy_height": "positive",
    # FIX 0.1.1: viirs_dnb = nighttime light → proxy for infrastructure/services.
    # High light = more developed area = more cooling infrastructure available.
    # Direction stays 'positive': high VIIRS → high AC → lower vulnerability.
    # Previous comment was correct; direction unchanged but reasoning clarified.
    "viirs_dnb": "positive",
}


class NormalizerEngine:
    """
    Normalises indicator arrays to [0, 1] with percentile-clipping.

    Parameters
    ----------
    clip_percentiles : tuple of (float, float)
        Lower and upper percentile bounds for clipping before scaling.
        Default (2, 98) trims 2 % of extreme values from each tail.
    """

    def __init__(self, clip_percentiles: Tuple[float, float] = (2, 98)) -> None:
        self.clip_low  = clip_percentiles[0]
        self.clip_high = clip_percentiles[1]
        self._stats: Dict[str, dict] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(
        self,
        arr: np.ndarray,
        indicator_name: str,
        direction: str = "positive",
    ) -> np.ndarray:
        """
        Normalise a single indicator array to [0, 1].

        Parameters
        ----------
        arr : np.ndarray
            Raw indicator values (any shape). NaN = no-data.
        indicator_name : str
            Used to cache statistics for reproducibility reporting.
        direction : str
            ``'positive'``: high raw → high normalised (default).
            ``'negative'``: high raw → low normalised (e.g. albedo as
            direct vulnerability — NOT used here; kept for compatibility).

        Returns
        -------
        np.ndarray
            Normalised array in [0, 1]. NaN pixels remain NaN.
        """
        valid_mask = ~np.isnan(arr)
        if valid_mask.sum() == 0:
            return np.full_like(arr, np.nan, dtype=float)

        valid = arr[valid_mask].astype(float)
        p_lo  = float(np.percentile(valid, self.clip_low))
        p_hi  = float(np.percentile(valid, self.clip_high))
        mean  = float(np.nanmean(arr))

        self._stats[indicator_name] = {
            "min": float(valid.min()),
            "max": float(valid.max()),
            f"p{int(self.clip_low)}": p_lo,
            f"p{int(self.clip_high)}": p_hi,
            "mean": mean,
        }

        # Clip then min-max scale (vectorised)
        clipped = np.clip(arr.astype(float), p_lo, p_hi)
        span = p_hi - p_lo
        if span < 1e-12:
            # Constant field — map to 0.5
            normed = np.where(valid_mask, 0.5, np.nan)
        else:
            normed = (clipped - p_lo) / span
            normed = np.where(valid_mask, normed, np.nan)

        if direction == "negative":
            # REASON: 'negative' direction means high raw = low vulnerability.
            # Invert so normalised value still means "more of this quantity".
            normed = np.where(valid_mask, 1.0 - normed, np.nan)

        return normed.astype(float)

    def normalize_batch(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalise a batch of named indicator arrays using the INDICATOR_DIRECTIONS table.

        Parameters
        ----------
        arrays : dict
            ``{indicator_name: raw_ndarray}``

        Returns
        -------
        dict
            ``{indicator_name: normalised_ndarray}``
        """
        out = {}
        for name, arr in arrays.items():
            direction = INDICATOR_DIRECTIONS.get(name, "positive")
            out[name] = self.normalize(arr, name, direction)
        return out

    def get_stats(self) -> pd.DataFrame:
        """
        Return per-indicator normalisation statistics as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: indicator, min, max, p2, p98, mean.
        """
        rows = []
        for name, s in self._stats.items():
            row = {"indicator": name}
            row.update(s)
            rows.append(row)
        return pd.DataFrame(rows)
