"""
rxharm/seasonal/detector.py
============================
Hottest-period detector for Project RxHARM.

Queries ERA5-Land monthly climatology via the GEE Python API to identify
the N_HOTTEST_MONTHS calendar months that form the analysis composite window
for all Landsat and Sentinel-2 fetches.

GEE is imported lazily inside ``detect()`` so the rest of the AOI module
(handler, decomposer) works without GEE authentication.

ERA5-Land collection details:
    GEE ID  : ``ECMWF/ERA5_LAND/MONTHLY_AGGR``
    Bands   : ``temperature_2m`` (K), ``dewpoint_temperature_2m`` (K)
    Res.    : ~9 km  (0.1°)
    Period  : 1950 – present
    Usage   : last 10 years ending at analysis year (climatological period)

Caching:
    Results are saved to ``_cache/seasonal_{lat}_{lon}_{year}.json``
    to avoid repeated GEE calls during iterative development.

Dependencies (all in requirements.txt):
    numpy, earthengine-api (lazy import)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
import os
from typing import Dict, List, Optional, Tuple

# ── Third-party (non-GEE) ─────────────────────────────────────────────────────
import numpy as np

# ── Internal ──────────────────────────────────────────────────────────────────
from rxharm.config import MMT_PERCENTILE, N_HOTTEST_MONTHS

# ── Constants ─────────────────────────────────────────────────────────────────
_CACHE_DIR = "_cache"


class SeasonalDetector:
    """
    Identifies the hottest calendar months for a given location and year.

    The primary output — ``hottest_months`` — is a sorted list of month
    integers (e.g. ``[4, 5]``) used by all GEE fetchers as the composite
    date window. The detector also computes the atmospheric context (Heat
    Index) and the climatological Minimum Mortality Temperature (MMT).

    Parameters
    ----------
    centroid_lat : float
        Latitude of the AOI centroid in decimal degrees (−90 to 90).
    centroid_lon : float
        Longitude of the AOI centroid in decimal degrees (−180 to 180).
    year : int
        Analysis year. The climatological baseline uses the 10-year period
        ``[year-10, year-1]`` (inclusive).

    Attributes
    ----------
    centroid_lat : float
    centroid_lon : float
    year : int
    _detected_months : list of int or None
        Populated after ``detect()`` is called.
    _monthly_stats : dict or None
        Per-month temperature statistics from ERA5, populated after ``detect()``.
    """

    def __init__(
        self,
        centroid_lat: float,
        centroid_lon: float,
        year: int,
    ) -> None:
        self.centroid_lat  = float(centroid_lat)
        self.centroid_lon  = float(centroid_lon)
        self.year          = int(year)

        self._detected_months: Optional[List[int]] = None
        self._monthly_stats: Optional[Dict[int, Dict[str, float]]] = None

    # ── Primary interface ─────────────────────────────────────────────────────

    def detect(self, use_cache: bool = True) -> List[int]:
        """
        Detect the hottest calendar months for this location and year.

        FIX 0.1.1: Added adaptive window expansion. If fewer than
        MIN_LANDSAT_SCENES valid scenes exist in the top N_HOTTEST_MONTHS,
        the window expands by adding the next hottest month until
        MAX_WINDOW_MONTHS is reached. This prevents silent failures and
        flat HVI maps in cloud-prone or data-sparse regions.

        Parameters
        ----------
        use_cache : bool
            If True (default), load from the JSON cache if it exists.

        Returns
        -------
        list of int
            Month integers (1–12), sorted ascending.
            Example: ``[4, 5]`` for April–May.

        Raises
        ------
        ImportError
            If ``earthengine-api`` is not installed.
        """
        # ── Try cache first ───────────────────────────────────────────────────
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._detected_months = cached["hottest_months"]
                self._monthly_stats   = cached.get("monthly_stats")
                self._mmt             = cached.get("mmt")
                print(f"  Loaded from cache: months {self._detected_months}")
                return self._detected_months

        # ── Lazy GEE import ───────────────────────────────────────────────────
        try:
            import ee
        except ImportError as exc:
            raise ImportError(
                "earthengine-api is required for SeasonalDetector.detect(). "
                "Install with: pip install earthengine-api"
            ) from exc

        # ── Step 1: Find all 12 months sorted hottest→coolest from ERA5 ──────
        all_ranked = self._get_hottest_months_from_era5()
        monthly_stats = self._query_era5_monthly(ee)

        # FIX 0.1.1: Adaptive window expansion
        from rxharm.config import MIN_LANDSAT_SCENES, N_HOTTEST_MONTHS, MAX_WINDOW_MONTHS
        ee_geom = ee.Geometry.Point([self.centroid_lon, self.centroid_lat]).buffer(10000)
        window_months = sorted(all_ranked[:N_HOTTEST_MONTHS])
        final_months = None

        for attempt in range(MAX_WINDOW_MONTHS - N_HOTTEST_MONTHS + 1):
            n_scenes = self._count_landsat_scenes(ee, ee_geom, window_months)
            print(f"  Window {window_months}: {n_scenes} valid Landsat scenes")

            if n_scenes >= MIN_LANDSAT_SCENES:
                final_months = window_months
                break

            # Expand: add the next hottest month not already in the window
            expanded = False
            for candidate in all_ranked:
                if candidate not in window_months:
                    window_months = sorted(window_months + [candidate])
                    expanded = True
                    break

            if not expanded:
                # All 12 months tried — use what we have
                print(f"  WARNING: Only {n_scenes} scenes available across all months.")
                print(f"  Proceeding with {window_months} — results may have data gaps.")
                final_months = window_months
                break

        if final_months is None:
            final_months = window_months

        if len(final_months) > N_HOTTEST_MONTHS:
            print(
                f"  NOTE: Window expanded from {N_HOTTEST_MONTHS} to {len(final_months)}"
                f" months to reach minimum scene count ({MIN_LANDSAT_SCENES})."
            )

        self._detected_months = final_months
        self._monthly_stats   = monthly_stats

        # ── Cache result ──────────────────────────────────────────────────────
        self._save_cache({
            "hottest_months": final_months,
            "monthly_stats":  monthly_stats,
            "mmt":            self._compute_era5_mmt() if hasattr(self, '_compute_era5_mmt') else None,
        })

        return final_months

    def _count_landsat_scenes(self, ee: Any, ee_geom: Any, months: List[int]) -> int:
        """
        FIX 0.1.1: Count valid (low-cloud) Landsat 8+9 scenes in given months.
        Uses CLOUD_COVER < 70 to count available scenes before compositing.
        """
        from rxharm.config import GEE_COLLECTIONS
        year = self.year
        start  = f"{year}-{min(months):02d}-01"
        end_m  = max(months) + 1
        end_yr = year if end_m <= 12 else year + 1
        end_m  = end_m if end_m <= 12 else 1
        end    = f"{end_yr}-{end_m:02d}-01"

        def count_col(cid: str) -> int:
            try:
                return (ee.ImageCollection(cid)
                        .filterBounds(ee_geom)
                        .filterDate(start, end)
                        .filter(ee.Filter.lt("CLOUD_COVER", 70))
                        .size().getInfo())
            except Exception:
                return 0

        try:
            return count_col(GEE_COLLECTIONS["landsat8"]) + count_col(GEE_COLLECTIONS["landsat9"])
        except Exception:
            return 0

    def _get_hottest_months_from_era5(self) -> List[int]:
        """
        FIX 0.1.1: Returns all 12 months sorted from hottest to coolest
        using a 10-year ERA5 climatology. Used for adaptive window expansion.
        """
        try:
            import ee
            from rxharm.config import GEE_COLLECTIONS
            era5  = ee.ImageCollection(GEE_COLLECTIONS["era5_land"])
            point = ee.Geometry.Point([self.centroid_lon, self.centroid_lat])
            clim_start = str(max(1991, self.year - 10))
            clim_end   = str(self.year)
            monthly_temps: List[tuple] = []
            for month in range(1, 13):
                monthly = (era5
                           .filterDate(clim_start, clim_end)
                           .filter(ee.Filter.calendarRange(month, month, "month"))
                           .select("temperature_2m")
                           .mean())
                try:
                    val = (monthly
                           .reduceRegion(reducer=ee.Reducer.mean(),
                                         geometry=point.buffer(5000),
                                         scale=11132)
                           .get("temperature_2m").getInfo())
                    monthly_temps.append((month, val if val else 0))
                except Exception:
                    monthly_temps.append((month, 0))
            monthly_temps.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in monthly_temps]
        except Exception:
            # Fallback: Northern Hemisphere calendar ordering
            return [6, 7, 8, 5, 9, 4, 10, 3, 11, 2, 12, 1]

    # ── GEE query ─────────────────────────────────────────────────────────────

    def _query_era5_monthly(self, ee: Any) -> Dict[int, Dict[str, float]]:
        """
        Query ERA5-Land monthly temperature and dewpoint over the baseline period.

        Parameters
        ----------
        ee : module
            The authenticated Google Earth Engine Python module.

        Returns
        -------
        dict
            ``{month_int: {'mean_tmax_c': float, 'mean_td_c': float}}``
            for months 1–12.
        """
        start_year = self.year - 10
        end_year   = self.year - 1  # exclusive of analysis year for clean baseline

        start_date = f"{start_year}-01-01"
        end_date   = f"{end_year + 1}-01-01"

        point = ee.Geometry.Point([self.centroid_lon, self.centroid_lat])

        collection = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(start_date, end_date)
            .select(["temperature_2m", "dewpoint_temperature_2m"])
        )

        monthly_stats: Dict[int, Dict[str, float]] = {}

        for month in range(1, 13):
            monthly_col = collection.filter(
                ee.Filter.calendarRange(month, month, "month")
            )
            # Compute mean image across all years for this month
            mean_img = monthly_col.mean()
            values = mean_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=11_132,   # ~0.1 degree in metres at equator
                bestEffort=True,
            ).getInfo()

            # Convert Kelvin → Celsius (ERA5 bands are in Kelvin)
            tmax_c = values.get("temperature_2m", 273.15) - 273.15
            td_c   = values.get("dewpoint_temperature_2m", 273.15) - 273.15

            monthly_stats[month] = {
                "mean_tmax_c": round(tmax_c, 2),
                "mean_td_c":   round(td_c, 2),
            }

        return monthly_stats

    # ── Heat Index computation ────────────────────────────────────────────────

    def get_era5_heat_index(self) -> Dict[str, float]:
        """
        Compute the mean Heat Index for the detected hottest period.

        Uses the simplified Rothfusz equation for Heat Index, valid where
        T > 27 °C. Below 27 °C the Heat Index equals air temperature.

        Returns
        -------
        dict
            Keys: ``'mean_HI_C'``, ``'max_HI_C'``, ``'era5_resolution_km'``

        Raises
        ------
        RuntimeError
            If ``detect()`` has not been called yet.
        """
        if self._detected_months is None or self._monthly_stats is None:
            raise RuntimeError(
                "Call detect() before get_era5_heat_index()."
            )

        hi_values = []
        for month in self._detected_months:
            stats = self._monthly_stats[month]
            t_c  = stats["mean_tmax_c"]
            td_c = stats["mean_td_c"]
            rh   = self._dewpoint_to_rh(t_c, td_c)
            hi   = self._heat_index(t_c, rh)
            hi_values.append(hi)

        return {
            "mean_HI_C":          round(float(np.mean(hi_values)), 2),
            "max_HI_C":           round(float(np.max(hi_values)), 2),
            "era5_resolution_km": 9.0,
        }

    def get_climatological_mmt(self) -> float:
        """
        Estimate the Minimum Mortality Temperature from ERA5 climatology.

        MMT is estimated as the ``MMT_PERCENTILE`` (75th by default, from
        ``config.py``) of the distribution of monthly maximum temperatures
        across the 10-year baseline, restricted to the hottest months.

        Returns
        -------
        float
            MMT estimate in degrees Celsius.

        Raises
        ------
        RuntimeError
            If ``detect()`` has not been called yet.
        """
        if self._detected_months is None or self._monthly_stats is None:
            raise RuntimeError(
                "Call detect() before get_climatological_mmt()."
            )

        temps = [
            self._monthly_stats[m]["mean_tmax_c"]
            for m in self._detected_months
        ]
        return float(np.percentile(temps, MMT_PERCENTILE))

    # ── Date filter strings ───────────────────────────────────────────────────

    def get_date_filter_strings(self) -> List[Tuple[str, str]]:
        """
        Return GEE date filter tuples for the detected hottest months.

        Returns
        -------
        list of (str, str)
            Each tuple is ``(start_date, end_date)`` in ``'YYYY-MM-DD'`` format
            for one calendar month in ``self.year``.
            Example: ``[('2023-04-01', '2023-05-01'), ('2023-05-01', '2023-06-01')]``

        Raises
        ------
        RuntimeError
            If ``detect()`` has not been called yet.
        """
        if self._detected_months is None:
            raise RuntimeError("Call detect() before get_date_filter_strings().")

        result = []
        for month in self._detected_months:
            start = f"{self.year}-{month:02d}-01"
            # End is the first day of the next month
            if month == 12:
                end = f"{self.year + 1}-01-01"
            else:
                end = f"{self.year}-{month + 1:02d}-01"
            result.append((start, end))
        return result

    # ── Caching ───────────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        """
        FIX 0.1.1:
        1. Cache stored on Google Drive when mounted — persists across Colab
           session resets. Falls back to /tmp/ when Drive is not mounted.
        2. Cache filename now includes N_HOTTEST_MONTHS so changing the config
           value automatically invalidates the old cache.
        """
        import os
        from rxharm.config import N_HOTTEST_MONTHS

        # Prefer Google Drive cache (survives Colab session resets)
        drive_cache = "/content/drive/MyDrive/RxHARM_outputs/.seasonal_cache"
        local_cache = "/tmp/rxharm_seasonal_cache"

        cache_dir = drive_cache if os.path.exists("/content/drive/MyDrive") else local_cache
        os.makedirs(cache_dir, exist_ok=True)

        filename = (
            f"seasonal_{self.centroid_lat:.3f}_{self.centroid_lon:.3f}"
            f"_{self.year}_n{N_HOTTEST_MONTHS}.json"
        )
        return os.path.join(cache_dir, filename)

    def _save_cache(self, data: dict) -> None:
        """
        Write detection results to the JSON cache file.

        Parameters
        ----------
        data : dict
            Data to serialise. Must be JSON-serialisable.
        """
        # REASON: Convert any numpy scalar keys to plain Python ints
        # so json.dump does not raise a TypeError.
        serialisable = {
            "hottest_months": [int(m) for m in data.get("hottest_months", [])],
            "monthly_stats":  {
                int(k): v for k, v in data.get("monthly_stats", {}).items()
            },
        }
        path = self._cache_path()
        with open(path, "w") as fh:
            json.dump(serialisable, fh, indent=2)

    def _load_cache(self) -> Optional[dict]:
        """
        Load detection results from the JSON cache if it exists.

        Returns
        -------
        dict or None
            Cached data dict, or None if no cache file exists.
        """
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            data = json.load(fh)
        # Convert string keys back to int (JSON always serialises keys as str)
        if "monthly_stats" in data:
            data["monthly_stats"] = {
                int(k): v for k, v in data["monthly_stats"].items()
            }
        return data

    # ── Static helper methods ─────────────────────────────────────────────────

    @staticmethod
    def _dewpoint_to_rh(T_C: float, Td_C: float) -> float:
        """
        Compute relative humidity from air temperature and dewpoint temperature.

        Uses the August-Roche-Magnus approximation.

        Parameters
        ----------
        T_C : float
            Air temperature in degrees Celsius.
        Td_C : float
            Dewpoint temperature in degrees Celsius.

        Returns
        -------
        float
            Relative humidity in percent (0–100).
        """
        # REASON: This approximation is accurate to within ±0.4% RH for
        # temperatures between −40 °C and +60 °C. Source: Alduchov & Eskridge (1996).
        num_d = 17.625 * Td_C
        den_d = 243.04 + Td_C
        num_t = 17.625 * T_C
        den_t = 243.04 + T_C
        rh = 100.0 * np.exp(num_d / den_d) / np.exp(num_t / den_t)
        return float(np.clip(rh, 0.0, 100.0))

    @staticmethod
    def _heat_index(T_C: float, RH: float) -> float:
        """
        Compute the Heat Index (apparent temperature) using the Rothfusz equation.

        Valid for T > 27 °C. Below 27 °C returns T_C unchanged.

        Parameters
        ----------
        T_C : float
            Air temperature in degrees Celsius.
        RH : float
            Relative humidity in percent (0–100).

        Returns
        -------
        float
            Heat Index in degrees Celsius.
        """
        if T_C <= 27.0:
            return T_C

        T_F = T_C * 9.0 / 5.0 + 32.0
        HI_F = (
            -42.379
            + 2.04901523 * T_F
            + 10.14333127 * RH
            - 0.22475541 * T_F * RH
            - 0.00683783 * T_F ** 2
            - 0.05481717 * RH ** 2
            + 0.00122874 * T_F ** 2 * RH
            + 0.00085282 * T_F * RH ** 2
            - 0.00000199 * T_F ** 2 * RH ** 2
        )
        return (HI_F - 32.0) * 5.0 / 9.0

    def __repr__(self) -> str:
        months = self._detected_months or "not yet detected"
        return (
            f"SeasonalDetector(lat={self.centroid_lat}, "
            f"lon={self.centroid_lon}, year={self.year}, "
            f"hottest_months={months})"
        )
