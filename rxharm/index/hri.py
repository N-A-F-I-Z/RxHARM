"""
rxharm/index/hri.py
====================
Heat Risk Index (HRI) computation engine for Project RxHARM.

Formula:
    HRI = H_s * HVI

Also computes attributable deaths (AD) per cell:
    AF  = (RR_adj - 1) / RR_adj
    RR_adj = exp(beta * max(0, T - MMT)) * (1 + lambda * HVI_norm)
    AD_i = Pop_i * CDR * AF_i * event_days

Literature basis:
    - Gasparrini et al. (2015), Lancet: base beta coefficients
    - Khatana et al. (2022), Circulation: HVI spatial vulnerability modifier

Dependencies: numpy, rxharm.config
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from rxharm.config import (
    BETA_BY_CLIMATE_ZONE,
    LAMBDA_HVI_DEFAULT,
    MMT_PERCENTILE,
)


class HRIEngine:
    """
    Computes HRI and attributable deaths from HVI results.

    Parameters
    ----------
    climate_zone : str
        Köppen zone code ``'A'``–``'E'``, from ``AOIHandler.get_koppen_zone()``.
    cdr_baseline : float or str
        Crude death rate (deaths / person / year).
        - Pass a float directly: e.g. ``0.007``
        - Pass an ISO3 country code (e.g. ``'IND'``) to look up from
          ``cdr_lookup.csv`` automatically.
        - Pass ``None`` to use the global-average fallback (0.0074).

    FIX 0.1.1: cdr_baseline now accepts an ISO3 country code string.
    Previously the caller was required to look up the CDR manually,
    which frequently resulted in the wrong value being passed or a
    hard-coded float being used for all countries.
    """

    # Global average fallback (WHO 2022 all-cause crude death rate)
    _CDR_GLOBAL_FALLBACK = 0.0074

    @staticmethod
    def cdr_from_iso3(iso3: str) -> float:
        """
        FIX 0.1.1: Look up crude death rate from cdr_lookup.csv by ISO3 code.

        Parameters
        ----------
        iso3 : str
            ISO 3166-1 alpha-3 country code (e.g., 'IND', 'BGD').

        Returns
        -------
        float
            CDR in deaths / person / year. Falls back to global average
            if the country is not found.
        """
        import csv
        from rxharm.config import CDR_LOOKUP_PATH

        try:
            with open(CDR_LOOKUP_PATH, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("country_code", "").strip().upper() == iso3.upper():
                        return float(row["crude_death_rate_per_1000"]) / 1000.0
        except Exception:
            pass

        print(
            f"  WARNING: CDR not found for '{iso3}'. "
            f"Using global average fallback ({HRIEngine._CDR_GLOBAL_FALLBACK}/person/yr)."
        )
        return HRIEngine._CDR_GLOBAL_FALLBACK

    def __init__(self, climate_zone: str, cdr_baseline) -> None:
        if climate_zone not in BETA_BY_CLIMATE_ZONE:
            raise ValueError(
                f"Unknown climate zone '{climate_zone}'. "
                f"Valid zones: {list(BETA_BY_CLIMATE_ZONE.keys())}"
            )
        self.beta = BETA_BY_CLIMATE_ZONE[climate_zone]

        # FIX 0.1.1: Accept ISO3 string, float, or None
        if cdr_baseline is None:
            self.cdr = self._CDR_GLOBAL_FALLBACK
            print(f"  HRIEngine: using global CDR fallback ({self.cdr})")
        elif isinstance(cdr_baseline, str):
            self.cdr = self.cdr_from_iso3(cdr_baseline)
            print(f"  HRIEngine: CDR for '{cdr_baseline}' = {self.cdr:.6f}/person/yr")
        else:
            self.cdr = float(cdr_baseline)

        self.mmt: Optional[float] = None
        self.ha_context: Optional[dict] = None

    # ── MMT and atmospheric context ────────────────────────────────────────────

    def compute_mmt(self, lst_array: np.ndarray) -> float:
        """
        Estimate the Minimum Mortality Temperature from the LST distribution.

        MMT = percentile(LST_raw, MMT_PERCENTILE) over all valid pixels.

        REASON: MMT is location-adaptive; hotter cities have higher MMT
        reflecting physiological acclimatisation (Gasparrini et al. 2015).

        Parameters
        ----------
        lst_array : np.ndarray
            Raw LST values (°C), any shape. NaN = no-data.

        Returns
        -------
        float
            MMT estimate in degrees Celsius. Stored as ``self.mmt``.
        """
        valid = lst_array[np.isfinite(lst_array)]
        if len(valid) == 0:
            self.mmt = 30.0  # fallback for all-NaN input
        else:
            self.mmt = float(np.percentile(valid, MMT_PERCENTILE))
        return self.mmt

    def set_atmospheric_context(
        self,
        era5_heat_index: float,
        event_days: int = 3,
    ) -> None:
        """
        Store ERA5-based atmospheric context for the hottest period.

        Parameters
        ----------
        era5_heat_index : float
            Mean Heat Index (°C) from SeasonalDetector.get_era5_heat_index().
        event_days : int
            Number of heatwave days for AD calculation.
        """
        self.ha_context = {
            "mean_HI_C":   era5_heat_index,
            "event_days":  event_days,
        }

    # ── Core computation ───────────────────────────────────────────────────────

    def compute_hri(
        self,
        H_s: np.ndarray,
        hvi: np.ndarray,
    ) -> np.ndarray:
        """
        Compute HRI = H_s * HVI, then normalise to [0, 1].

        Parameters
        ----------
        H_s : np.ndarray
            Hazard sub-index [0, 1].
        hvi : np.ndarray
            HVI [0, 1] from HVIEngine.

        Returns
        -------
        np.ndarray
            HRI in [0, 1].
        """
        raw = H_s * hvi
        valid = np.isfinite(raw)
        lo, hi = np.nanmin(raw), np.nanmax(raw)
        if hi - lo < 1e-12:
            return np.where(valid, 0.5, np.nan)
        return np.where(valid, (raw - lo) / (hi - lo), np.nan)

    def compute_attributable_fraction(
        self,
        lst_array: np.ndarray,
        hvi_norm: np.ndarray,
        lambda_hvi: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute the spatially explicit attributable fraction of mortality.

        Justification:
            Base AF from Gasparrini et al. (2015), Lancet, 384 study locations.
            HVI modifier from Khatana et al. (2022), Circulation — spatial
            vulnerability modifies the temperature–mortality slope.

        Formula:
            RR_adj = exp(beta * max(0, T - MMT)) * (1 + lambda * HVI_norm)
            AF = (RR_adj - 1) / RR_adj

        Parameters
        ----------
        lst_array : np.ndarray
            LST in °C (normalised or raw; must be in same units as self.mmt).
        hvi_norm : np.ndarray
            HVI [0, 1].
        lambda_hvi : float, optional
            HVI vulnerability modifier. Defaults to LAMBDA_HVI_DEFAULT.

        Returns
        -------
        np.ndarray
            AF in [0, 1], clipped to non-negative.
        """
        if self.mmt is None:
            raise RuntimeError("Call compute_mmt() before compute_attributable_fraction().")

        if lambda_hvi is None:
            lambda_hvi = LAMBDA_HVI_DEFAULT

        delta_T = np.maximum(0.0, lst_array - self.mmt)
        rr_base = np.exp(self.beta * delta_T)
        rr_adj  = rr_base * (1.0 + lambda_hvi * hvi_norm)
        af      = (rr_adj - 1.0) / rr_adj
        return np.clip(af, 0.0, 1.0)

    def compute_attributable_deaths(
        self,
        population: np.ndarray,
        lst_array: np.ndarray,
        hvi_norm: np.ndarray,
        event_days: int = 3,
    ) -> np.ndarray:
        """
        Compute baseline attributable deaths per cell.

        Formula: AD_i = Pop_i * CDR * AF_i * event_days

        Parameters
        ----------
        population : np.ndarray
            WorldPop population per 100 m cell.
        lst_array : np.ndarray
            LST (°C) per cell.
        hvi_norm : np.ndarray
            HVI [0, 1] per cell.
        event_days : int
            Heatwave duration in days.

        Returns
        -------
        np.ndarray
            AD per cell (non-negative float).
        """
        af = self.compute_attributable_fraction(lst_array, hvi_norm)
        ad = population * self.cdr * af * event_days
        return np.maximum(ad, 0.0)

    def compute_all(
        self,
        hvi_results: dict,
        event_days: int = 3,
    ) -> dict:
        """
        Convenience wrapper: compute HRI, AF, and AD from HVIEngine output.

        Parameters
        ----------
        hvi_results : dict
            Output of ``HVIEngine.compute_all()``.
        event_days : int
            Heatwave duration for AD calculation.

        Returns
        -------
        dict
            Keys: ``'HRI'``, ``'AF'``, ``'AD_baseline'``, ``'MMT'``,
            ``'H_a_context'``
        """
        H_s = hvi_results["H_s"]
        hvi = hvi_results["HVI"]

        # Extract population and LST from normalised arrays
        ind_norm = hvi_results.get("indicator_normalized", {})
        population = ind_norm.get(
            "population",
            np.ones_like(H_s),
        )
        lst = ind_norm.get("lst", H_s)  # fallback to H_s if LST unavailable

        hri = self.compute_hri(H_s, hvi)
        af  = self.compute_attributable_fraction(lst, hvi)
        ad  = self.compute_attributable_deaths(population, lst, hvi, event_days)

        return {
            "HRI":         hri,
            "AF":          af,
            "AD_baseline": ad,
            "MMT":         self.mmt,
            "H_a_context": self.ha_context,
        }
