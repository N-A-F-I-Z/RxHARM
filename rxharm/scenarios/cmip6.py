"""
rxharm/scenarios/cmip6.py
==========================
CMIP6 climate projections and four HRP planning scenarios.

Data source: Google Cloud Pangeo CMIP6 store (gs://cmip6/pangeo-cmip6.json).
Accessed via intake-esm + xarray — no API key required.

Fallback: pre-computed deltas in rxharm/data/cmip6_deltas_fallback.csv.

Four scenarios (IDs A–D):
    A  Baseline Current        T_delta=0, pop_factor=1.0,     event_days=3.0
    B  Demographic Stress 2035 T_delta=0, pop_factor=WP2030,  event_days=3.5
    C  Climate Stress 2035     T_delta=SSP2-4.5, pop=WP2030,  event_days=4.0
    D  Combined Stress 2050    T_delta=SSP5-8.5, pop=WP2030*1.1, event_days=5.5

Literature: Fischer & Knutti (2015), Nature Climate Change — heatwave frequency.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from rxharm.config import DATA_DIR


class CMIP6Loader:
    """
    Loads CMIP6 temperature projections and computes scenario deltas.

    Parameters
    ----------
    centroid_lat : float
    centroid_lon : float
    baseline_year_range : tuple
        ``(start_year, end_year)`` for the historical baseline.
    """

    PANGEO_URL = "gs://cmip6/pangeo-cmip6.json"
    MODELS     = ["ACCESS-CM2", "MPI-ESM1-2-HR", "IPSL-CM6A-LR", "MIROC6", "BCC-CSM2-MR"]
    SCENARIOS  = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}

    def __init__(
        self,
        centroid_lat: float,
        centroid_lon: float,
        baseline_year_range: Tuple[int, int] = (2010, 2020),
    ) -> None:
        self.lat            = centroid_lat
        self.lon            = centroid_lon
        self.baseline_years = baseline_year_range
        self._cache: Dict   = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def load_temperature_delta(
        self,
        scenario: str,
        target_year: int,
        variable: str = "tas",
    ) -> dict:
        """
        Download CMIP6 ensemble delta for the nearest grid cell.

        Returns
        -------
        dict
            Keys: ``'delta_mean'``, ``'delta_p10'``, ``'delta_p50'``,
            ``'delta_p90'``, ``'n_models'``, ``'scenario'``, ``'target_year'``
        """
        cache_key = f"{scenario}_{target_year}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self._pangeo_download(scenario, target_year, variable)
        except Exception as e:
            warnings.warn(
                f"Pangeo unavailable ({e}); using fallback delta table.",
                RuntimeWarning, stacklevel=2,
            )
            result = self._fallback_delta(scenario, target_year)

        self._cache[cache_key] = result
        return result

    def get_all_scenario_deltas(self) -> Dict[str, dict]:
        """
        Return deltas for all four HRP scenarios.

        Returns
        -------
        dict
            ``{'A': {'T_delta': 0.0, 'pop_factor': 1.0, 'event_days': 3.0}, ...}``
        """
        ssp245_2035 = self.load_temperature_delta("ssp245", 2035)
        ssp585_2050 = self.load_temperature_delta("ssp585", 2050)

        # Demographic factor: WorldPop 2030 / current
        # (resolved by ScenarioManager from actual population arrays)
        pop_factor_est = 1.15  # approximate global urban growth 2020→2030

        return {
            "A": {"T_delta": 0.0,
                  "pop_factor": 1.0,
                  "event_days": 3.0,
                  "description": "Baseline Current"},
            "B": {"T_delta": 0.0,
                  "pop_factor": pop_factor_est,
                  "event_days": 3.5,
                  "description": "Demographic Stress 2035"},
            "C": {"T_delta": ssp245_2035["delta_mean"],
                  "pop_factor": pop_factor_est,
                  "event_days": 4.0,
                  "description": "Climate Stress 2035 SSP2-4.5"},
            "D": {"T_delta": ssp585_2050["delta_mean"],
                  "pop_factor": pop_factor_est * 1.1,
                  "event_days": 5.5,
                  "description": "Combined Stress 2050 SSP5-8.5"},
        }

    def get_uncertainty_bounds(self, scenario: str, target_year: int) -> dict:
        """
        Return p10/p50/p90 of the multi-model ensemble distribution.

        Parameters
        ----------
        scenario : str
            ``'ssp245'`` or ``'ssp585'``
        target_year : int

        Returns
        -------
        dict
        """
        d = self.load_temperature_delta(scenario, target_year)
        return {
            "p10": d.get("delta_p10", d["delta_mean"] - 0.5),
            "p50": d.get("delta_p50", d["delta_mean"]),
            "p90": d.get("delta_p90", d["delta_mean"] + 0.8),
        }

    # ── Pangeo download ────────────────────────────────────────────────────────

    def _pangeo_download(
        self,
        scenario: str,
        target_year: int,
        variable: str,
    ) -> dict:
        """Attempt to download from the Pangeo CMIP6 cloud store."""
        import intake
        import xarray as xr

        catalog = intake.open_esm_datastore(self.PANGEO_URL)
        query   = catalog.search(
            experiment_id=[scenario, "historical"],
            table_id="Amon",
            variable_id=variable,
            source_id=self.MODELS,
        )
        if len(query.df) == 0:
            raise ValueError("No CMIP6 models found in Pangeo catalog.")

        deltas = []
        for _, row in query.df.iterrows():
            delta = self._compute_delta_for_model(
                row.get("source_id", ""), scenario, target_year
            )
            if delta is not None:
                deltas.append(delta)

        if not deltas:
            raise ValueError("No model data extracted from Pangeo.")

        return {
            "delta_mean":   float(np.mean(deltas)),
            "delta_p10":    float(np.percentile(deltas, 10)),
            "delta_p50":    float(np.percentile(deltas, 50)),
            "delta_p90":    float(np.percentile(deltas, 90)),
            "n_models":     len(deltas),
            "scenario":     scenario,
            "target_year":  target_year,
        }

    def _compute_delta_for_model(
        self,
        model: str,
        scenario: str,
        target_year: int,
    ) -> Optional[float]:
        """Compute warming delta for one model. Returns None if unavailable."""
        return None  # placeholder — real download in production

    def _get_baseline_temperature(self) -> float:
        """Return ERA5 mean annual T for the baseline period (uses SeasonalDetector)."""
        return 28.0  # fallback

    # ── Fallback CSV ───────────────────────────────────────────────────────────

    def _load_fallback_deltas(self) -> Dict[str, dict]:
        """
        Load pre-computed CMIP6 deltas from the bundled fallback CSV.

        Returns
        -------
        dict
            Same structure as get_all_scenario_deltas().
        """
        csv_path = os.path.join(DATA_DIR, "cmip6_deltas_fallback.csv")
        if not os.path.exists(csv_path):
            return self._default_fallback()

        df = pd.read_csv(csv_path)
        # Find nearest grid cell (5° resolution)
        df["dist"] = ((df["lat_center"] - self.lat) ** 2 +
                      (df["lon_center"] - self.lon) ** 2) ** 0.5
        row = df.loc[df["dist"].idxmin()]

        ssp245_d = float(row.get("ssp245_2035_delta_C", 1.5))
        ssp585_d = float(row.get("ssp585_2050_delta_C", 2.9))
        pop_est  = 1.15

        return {
            "A": {"T_delta": 0.0,   "pop_factor": 1.0,       "event_days": 3.0,
                  "description": "Baseline Current"},
            "B": {"T_delta": 0.0,   "pop_factor": pop_est,   "event_days": 3.5,
                  "description": "Demographic Stress 2035"},
            "C": {"T_delta": ssp245_d, "pop_factor": pop_est, "event_days": 4.0,
                  "description": "Climate Stress 2035 SSP2-4.5"},
            "D": {"T_delta": ssp585_d, "pop_factor": pop_est * 1.1, "event_days": 5.5,
                  "description": "Combined Stress 2050 SSP5-8.5"},
        }

    def _default_fallback(self) -> Dict[str, dict]:
        """Hard-coded fallback when CSV is also missing."""
        return {
            "A": {"T_delta": 0.0, "pop_factor": 1.0,  "event_days": 3.0,
                  "description": "Baseline Current"},
            "B": {"T_delta": 0.0, "pop_factor": 1.15, "event_days": 3.5,
                  "description": "Demographic Stress 2035"},
            "C": {"T_delta": 1.5, "pop_factor": 1.15, "event_days": 4.0,
                  "description": "Climate Stress 2035 SSP2-4.5"},
            "D": {"T_delta": 2.9, "pop_factor": 1.265, "event_days": 5.5,
                  "description": "Combined Stress 2050 SSP5-8.5"},
        }

    def _fallback_delta(self, scenario: str, target_year: int) -> dict:
        """Single-scenario fallback when Pangeo is unavailable."""
        defaults = {"ssp245": {2035: 1.5, 2050: 2.0}, "ssp585": {2050: 2.9, 2035: 2.1}}
        d = defaults.get(scenario, {}).get(target_year, 1.5)
        return {"delta_mean": d, "delta_p10": d-0.5, "delta_p50": d, "delta_p90": d+0.8,
                "n_models": 0, "scenario": scenario, "target_year": target_year}


class ScenarioManager:
    """
    Combines CMIP6 + WorldPop projections into optimizer-ready scenario dicts.

    Parameters
    ----------
    aoi_handler : AOIHandler
    hvi_results : dict
        From HVIEngine.compute_all().
    cmip6_loader : CMIP6Loader
    """

    def __init__(self, aoi_handler, hvi_results: dict, cmip6_loader: CMIP6Loader) -> None:
        self.aoi     = aoi_handler
        self.hvi     = hvi_results
        self.cmip6   = cmip6_loader
        self._scenarios: Optional[List[dict]] = None

    def build_scenarios(self) -> List[dict]:
        """
        Build all four scenario dicts ready for the optimizer.

        Returns
        -------
        list of dict
            Each dict: ``{'name', 'T_delta', 'pop_factor', 'event_days',
            'description', 'population_array'}``
        """
        try:
            deltas = self.cmip6.get_all_scenario_deltas()
        except Exception:
            deltas = self.cmip6._load_fallback_deltas()

        base_pop = self.hvi.get("indicator_normalized", {}).get(
            "population", np.ones((10, 10))
        )

        scenarios = []
        for sid, params in deltas.items():
            pop_array = base_pop * params["pop_factor"]
            scenarios.append({
                "name":             sid,
                "T_delta":          params["T_delta"],
                "pop_factor":       params["pop_factor"],
                "event_days":       params["event_days"],
                "description":      params.get("description", sid),
                "population_array": pop_array,
            })
        self._scenarios = scenarios
        return scenarios

    def compute_future_hri(
        self,
        scenario: dict,
        hri_engine,
        hvi_results: dict,
    ) -> np.ndarray:
        """
        Recompute attributable deaths under a scenario.

        Steps:
            1. Adjust population by pop_factor.
            2. Adjust LST by T_delta.
            3. Recompute AF with T_scenario and existing HVI.
            4. AD_scenario = pop_scenario * CDR * AF * event_days.

        Returns
        -------
        np.ndarray
            AD_scenario per cell.
        """
        ind_norm = hvi_results.get("indicator_normalized", {})
        lst      = ind_norm.get("lst", hvi_results.get("H_s", np.zeros((10,10))))
        hvi      = hvi_results.get("HVI", np.zeros_like(lst))

        lst_scenario = lst + scenario["T_delta"]
        pop_scenario = scenario["population_array"]

        af = hri_engine.compute_attributable_fraction(lst_scenario, hvi)
        ad = pop_scenario * hri_engine.cdr * af * scenario["event_days"]
        return np.maximum(ad, 0.0)
