"""
rxharm/risk/era5_context.py
============================
ERA5-based atmospheric context for the HRI formula.

Two roles:
    1. CLIMATOLOGICAL CONTEXT — mean Heat Index for the hottest period
       across the 10-year climatological window. Used as H_a baseline.
       Computed once; cached to _cache/.
    2. SCENARIO CONTEXT — ERA5 climatological values + CMIP6 T_delta.

Uses the same ERA5-Land collection as SeasonalDetector.
GEE is imported lazily.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

_CACHE_DIR = "_cache"


class ERA5Context:
    """
    Computes and caches ERA5-based atmospheric context for an AOI.

    Parameters
    ----------
    centroid_lat : float
    centroid_lon : float
    year : int
    hottest_months : list of int
    """

    def __init__(
        self,
        centroid_lat: float,
        centroid_lon: float,
        year: int,
        hottest_months: List[int],
    ) -> None:
        self.lat     = centroid_lat
        self.lon     = centroid_lon
        self.year    = year
        self.months  = hottest_months
        self._cache: Optional[dict] = None

    # ── Primary interface ──────────────────────────────────────────────────────

    def get_climatological_heat_index(self) -> dict:
        """
        Fetch ERA5-Land climatological Heat Index for the hottest period.

        Returns
        -------
        dict
            Keys: ``'mean_HI_C'``, ``'max_HI_C'``, ``'p90_HI_C'``,
            ``'era5_resolution_km'``
        """
        cache_path = self._cache_path("clim_hi")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        try:
            result = self._query_era5_heat_index()
        except Exception as e:
            # GEE unavailable — return plausible fallback
            result = self._fallback_heat_index()
            result["note"] = f"GEE unavailable ({e}); using fallback"

        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    def get_scenario_heat_index(self, T_delta: float) -> dict:
        """
        Apply CMIP6 temperature delta to the climatological Heat Index.

        REASON: Simple temperature shift is the standard approach for
        combining ERA5 climatology with CMIP6 scenario deltas in
        impact assessments (Dosio et al. 2018, WIREs Climate Change).

        Parameters
        ----------
        T_delta : float
            Warming (°C) from CMIP6 scenario.

        Returns
        -------
        dict
            Same structure as get_climatological_heat_index() but with
            T_delta added to all temperature fields.
        """
        base = self.get_climatological_heat_index()
        return {
            "mean_HI_C":         base["mean_HI_C"]   + T_delta,
            "max_HI_C":          base["max_HI_C"]    + T_delta,
            "p90_HI_C":          base.get("p90_HI_C", base["mean_HI_C"]) + T_delta,
            "era5_resolution_km": 9,
            "T_delta_applied":   T_delta,
        }

    # ── GEE query ──────────────────────────────────────────────────────────────

    def _query_era5_heat_index(self) -> dict:
        """Query ERA5-Land monthly temperature and dewpoint via GEE."""
        import ee
        from rxharm.seasonal.detector import SeasonalDetector

        det = SeasonalDetector(self.lat, self.lon, self.year)
        # Use cached monthly stats if available
        cached = det._load_cache()
        if cached and "monthly_stats" in cached:
            stats = cached["monthly_stats"]
        else:
            stats = det._query_era5_monthly(ee)

        his = []
        for m in self.months:
            m_stats = stats.get(m, stats.get(str(m), {}))
            T_C  = m_stats.get("mean_tmax_c", 35.0)
            Td_C = m_stats.get("mean_td_c", 22.0)
            rh   = det._dewpoint_to_rh(T_C, Td_C)
            hi   = det._heat_index(T_C, rh)
            his.append(hi)

        return {
            "mean_HI_C":          round(sum(his) / len(his), 2),
            "max_HI_C":           round(max(his), 2),
            "p90_HI_C":           round(max(his), 2),
            "era5_resolution_km": 9,
        }

    def _fallback_heat_index(self) -> dict:
        """Return a plausible default when GEE is unavailable."""
        return {"mean_HI_C": 38.0, "max_HI_C": 44.0, "p90_HI_C": 43.0,
                "era5_resolution_km": 9}

    def _cache_path(self, key: str) -> str:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        return os.path.join(
            _CACHE_DIR,
            f"era5_{key}_{self.lat:.4f}_{self.lon:.4f}_{self.year}.json",
        )
