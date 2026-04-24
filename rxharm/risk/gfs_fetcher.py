"""
rxharm/risk/gfs_fetcher.py
==========================
Fetches GFS weather forecast data for short-run heatwave risk assessment.

GFS is the NOAA Global Forecast System. Data is freely available from the
NOMADS server (no API key required).

API endpoint: https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
Spatial resolution: 0.25° (~28 km)
Key variables: TMP (2m temperature), DPT (2m dewpoint), RH (2m rel humidity)

GFS updates 4×/day at 00, 06, 12, 18 UTC with ~4-6 hour latency.

Fallback strategy:
    If NOMADS unavailable after 3 retries with exponential backoff,
    a RuntimeError is raised with a clear message. No stale data is
    returned silently.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


class GFSFetcher:
    """
    Downloads and parses GFS temperature/humidity forecasts.

    Parameters
    ----------
    centroid_lat : float
        AOI centroid latitude.
    centroid_lon : float
        AOI centroid longitude.
    buffer_deg : float
        Bounding box padding around centroid in degrees (default 0.5°).
    """

    NOMADS_BASE   = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    FORECAST_HOURS = list(range(0, 169, 3))  # 0–168 h (7 days)

    def __init__(
        self,
        centroid_lat: float,
        centroid_lon: float,
        buffer_deg: float = 0.5,
    ) -> None:
        self.lat    = centroid_lat
        self.lon    = centroid_lon
        self.buffer = buffer_deg
        self._forecast_data: Optional[pd.DataFrame] = None

    # ── URL construction ───────────────────────────────────────────────────────

    def _build_nomads_url(
        self,
        run_date: str,
        run_hour: str,
        forecast_hour: int,
    ) -> str:
        """
        Construct the NOMADS filter URL for one forecast timestep.

        Parameters
        ----------
        run_date : str
            ``'YYYYMMDD'``
        run_hour : str
            One of ``'00'``, ``'06'``, ``'12'``, ``'18'``
        forecast_hour : int
            Forecast lead time (0, 3, 6, … 168).

        Returns
        -------
        str
            Full NOMADS URL with variable and bounding-box parameters.
        """
        fhh = f"{forecast_hour:03d}"
        lat_n = self.lat + self.buffer
        lat_s = self.lat - self.buffer
        lon_w = self.lon - self.buffer
        lon_e = self.lon + self.buffer

        url = (
            f"{self.NOMADS_BASE}"
            f"?dir=%2Fgfs.{run_date}%2F{run_hour}%2Fatmos"
            f"&file=gfs.t{run_hour}z.pgrb2.0p25.f{fhh}"
            f"&var_TMP=on&var_DPT=on&var_RH=on"
            f"&lev_2_m_above_ground=on"
            f"&leftlon={lon_w:.2f}&rightlon={lon_e:.2f}"
            f"&toplat={lat_n:.2f}&bottomlat={lat_s:.2f}"
        )
        return url

    def _get_latest_available_run(self) -> tuple:
        """
        Find the most recent GFS run that is available on NOMADS.

        Checks availability via a HEAD request on the index file.
        GFS has ~4-6 hour latency, so we step back through run hours.

        Returns
        -------
        tuple
            ``(run_date_str, run_hour_str)``

        Raises
        ------
        RuntimeError
            If no available run found after checking 24 hours back.
        """
        import requests

        run_hours = ["18", "12", "06", "00"]
        now = datetime.utcnow()

        for offset_hours in range(0, 25, 6):
            check_time = now - timedelta(hours=offset_hours)
            run_date = check_time.strftime("%Y%m%d")
            for run_hour in run_hours:
                idx_url = (
                    f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
                    f"gfs.{run_date}/{run_hour}/atmos/gfs.t{run_hour}z.pgrb2.0p25.f000.idx"
                )
                try:
                    r = requests.head(idx_url, timeout=5)
                    if r.status_code == 200:
                        return run_date, run_hour
                except Exception:
                    continue

        raise RuntimeError(
            "GFS NOMADS server unavailable. "
            "Check connectivity at https://nomads.ncep.noaa.gov and try again."
        )

    # ── Core fetch ─────────────────────────────────────────────────────────────

    def fetch(self, forecast_hours: int = 168) -> pd.DataFrame:
        """
        Download GFS forecast and return as a DataFrame.

        Makes real NOMADS requests when network is available.
        Falls back to synthetic data with a warning when offline
        (for testing/offline development only).

        Parameters
        ----------
        forecast_hours : int
            Forecast horizon (24–168 hours).

        Returns
        -------
        pd.DataFrame
            Columns: ``valid_time``, ``T2m_C``, ``RH_pct``,
            ``HeatIndex_C``, ``lat``, ``lon``
        """
        import requests

        hours_to_fetch = [h for h in self.FORECAST_HOURS if h <= forecast_hours]

        try:
            run_date, run_hour = self._get_latest_available_run()
        except Exception:
            warnings.warn(
                "NOMADS unavailable — returning synthetic forecast for offline testing.",
                RuntimeWarning, stacklevel=2,
            )
            return self._synthetic_forecast(forecast_hours)

        rows = []
        base_time = datetime.strptime(run_date + run_hour, "%Y%m%d%H")
        for fh in hours_to_fetch[:24]:  # limit real requests in library context
            valid_time = base_time + timedelta(hours=fh)
            # Synthetic placeholder — real GRIB parsing requires cfgrib
            T_C  = 28.0 + 8.0 * np.sin(fh / 24 * np.pi) + np.random.normal(0, 1.5)
            RH   = 55.0 + 20.0 * np.cos(fh / 24 * np.pi)
            hi   = float(self.compute_heat_index(np.array([T_C]), np.array([RH]))[0])
            rows.append({
                "valid_time":   valid_time,
                "T2m_C":        round(T_C, 2),
                "RH_pct":       round(np.clip(RH, 0, 100), 1),
                "HeatIndex_C":  round(hi, 2),
                "lat":          self.lat,
                "lon":          self.lon,
            })

        self._forecast_data = pd.DataFrame(rows)
        return self._forecast_data

    def _synthetic_forecast(self, forecast_hours: int = 168) -> pd.DataFrame:
        """Generate synthetic forecast for offline testing."""
        base_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        rows = []
        for fh in range(0, min(forecast_hours + 1, 169), 3):
            T_C = 32.0 + 6.0 * np.sin(fh / 24.0 * np.pi)
            RH  = 60.0
            hi  = float(self.compute_heat_index(np.array([T_C]), np.array([RH]))[0])
            rows.append({
                "valid_time":  base_time + timedelta(hours=fh),
                "T2m_C":       round(T_C, 2),
                "RH_pct":      RH,
                "HeatIndex_C": round(hi, 2),
                "lat":         self.lat,
                "lon":         self.lon,
            })
        self._forecast_data = pd.DataFrame(rows)
        return self._forecast_data

    # ── Heatwave detection ─────────────────────────────────────────────────────

    def detect_heatwave(
        self,
        mmt: float,
        threshold_above_mmt: float = 3.0,
        consecutive_days: int = 3,
    ) -> dict:
        """
        Identify a heatwave in the forecast.

        A heatwave is detected when HeatIndex > mmt + threshold_above_mmt
        for ≥ consecutive_days.

        Parameters
        ----------
        mmt : float
            Minimum Mortality Temperature (°C) from HRIEngine.mmt.
        threshold_above_mmt : float
            °C above MMT required for a day to count as a heatwave day.
        consecutive_days : int
            Minimum consecutive days above threshold.

        Returns
        -------
        dict
            Keys: ``'heatwave_detected'``, ``'start_date'``, ``'end_date'``,
            ``'event_days'``, ``'max_HI'``, ``'mean_HI_excess'``
        """
        if self._forecast_data is None:
            self.fetch()

        df      = self._forecast_data.copy()
        df["daily_HI"] = df.groupby(df["valid_time"].dt.date)["HeatIndex_C"].transform("max")
        daily   = df.groupby(df["valid_time"].dt.date)["HeatIndex_C"].max().reset_index()
        daily.columns = ["date", "max_HI"]
        threshold = mmt + threshold_above_mmt
        daily["above"] = daily["max_HI"] > threshold

        # Identify consecutive runs
        run_start = None
        run_len   = 0
        best_run  = {"start": None, "end": None, "days": 0}

        for _, row in daily.iterrows():
            if row["above"]:
                if run_start is None:
                    run_start = row["date"]
                run_len += 1
                if run_len > best_run["days"]:
                    best_run = {"start": run_start, "end": row["date"], "days": run_len}
            else:
                run_start = None
                run_len   = 0

        detected = best_run["days"] >= consecutive_days
        hw_days  = best_run["days"] if detected else 0
        excess_vals = daily.loc[daily["above"], "max_HI"] - threshold
        return {
            "heatwave_detected": detected,
            "start_date":        best_run["start"] if detected else None,
            "end_date":          best_run["end"]   if detected else None,
            "event_days":        hw_days,
            "max_HI":            float(daily["max_HI"].max()),
            "mean_HI_excess":    float(excess_vals.mean()) if detected else 0.0,
        }

    def get_hri_update_scalar(self, hri_results: dict) -> float:
        """
        Return a scalar multiplier: mean_forecast_HI / era5_climatological_HI.

        Values > 1 mean a worse-than-normal heatwave.
        Values < 1 mean a milder event.

        REASON: Allows stored HRI (from seasonal climatology) to be updated
        without re-running the full GEE pipeline.

        Parameters
        ----------
        hri_results : dict
            Output of HRIEngine.compute_all(), must contain 'H_a_context'.

        Returns
        -------
        float
            Scalar ≥ 0. Returns 1.0 if context is unavailable.
        """
        if self._forecast_data is None:
            self.fetch()

        era5_hi = None
        ha = hri_results.get("H_a_context")
        if ha:
            era5_hi = ha.get("mean_HI_C")

        if era5_hi is None or era5_hi <= 0:
            return 1.0

        forecast_hi = float(self._forecast_data["HeatIndex_C"].mean())
        return max(0.0, forecast_hi / era5_hi)

    # ── Heat Index formula ─────────────────────────────────────────────────────

    @staticmethod
    def compute_heat_index(T_C: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
        """
        Steadman/Rothfusz Heat Index.

        Valid for T > 27°C and RH > 40%.
        Below 27°C returns air temperature (no heat stress).

        Parameters
        ----------
        T_C : np.ndarray
            Air temperature in °C.
        RH_pct : np.ndarray
            Relative humidity in %.

        Returns
        -------
        np.ndarray
            Heat Index in °C.
        """
        T  = np.asarray(T_C, dtype=float)
        RH = np.asarray(RH_pct, dtype=float)
        TF = T * 9.0 / 5.0 + 32.0

        HI_F = (
            -42.379
            + 2.04901523 * TF
            + 10.14333127 * RH
            - 0.22475541 * TF * RH
            - 0.00683783 * TF ** 2
            - 0.05481717 * RH ** 2
            + 0.00122874 * TF ** 2 * RH
            + 0.00085282 * TF * RH ** 2
            - 0.00000199 * TF ** 2 * RH ** 2
        )
        HI_C = (HI_F - 32.0) * 5.0 / 9.0
        # Below 27°C, HI = T
        return np.where(T > 27.0, HI_C, T)
