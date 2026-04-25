"""
rxharm/fetch/worldpop_fetcher.py
=================================
Downloads WorldPop Global2 (2015-2030) population data using direct HTTP
requests to the WorldPop FTP/web server, with wpgpDownloadPy as a secondary
option.  Downloads ONLY the age-sex bands required for HVI computation.

FIX 0.1.1: WorldPop Global2 is NOT available in the GEE catalog.
The previous code tried 'projects/sat-io/open-datasets/WorldPop/Global2'
which does not exist. This module replaces that with direct downloads.

BANDS DOWNLOADED:
  t_0   : age  0   (< 1 year, both sexes) — child fraction
  t_1   : age  1   (1–4 years)            — child fraction
  t_65  : age 65–69, both sexes            — elderly fraction
  t_70  : age 70–74                        — elderly fraction
  t_75  : age 75–79                        — elderly fraction
  t_80  : age 80+                          — elderly fraction
  + all remaining age bands to compute total population

SOURCE: WorldPop Global2 R2025A
  https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/

FALLBACK: If all download methods fail, returns (None, None, None) and the
caller uses GEE WorldPop 2020 with a printed warning.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Age band definitions ────────────────────────────────────────────────────
CHILD_AGE_BANDS   = [0, 1]            # under-5: 0 = <1yr, 1 = 1–4yr
ELDERLY_AGE_BANDS = [60, 65, 70, 75, 80, 85, 90]  # 60+
# All bands needed to reconstruct total population by summation
ALL_AGE_BANDS: List[int] = sorted(set([0, 1] + list(range(5, 95, 5))))


class WorldPopFetcher:
    """
    Downloads and processes WorldPop Global2 rasters for HVI computation.

    Parameters
    ----------
    iso3 : str
        ISO 3166-1 alpha-3 country code (e.g., 'IND', 'BGD', 'NGA').
    year : int
        Analysis year (2015–2030).
    cache_dir : str
        Directory to store downloaded files.
        Default: /content/worldpop_cache (Colab) or /tmp/worldpop_cache.
    """

    def __init__(
        self,
        iso3: str,
        year: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.iso3      = iso3.upper()
        self.year      = year
        if cache_dir is None:
            # Prefer persistent Google Drive location in Colab
            drive_path = "/content/drive/MyDrive/RxHARM_outputs/worldpop_cache"
            cache_dir  = drive_path if os.path.exists("/content/drive/MyDrive") else "/tmp/worldpop_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── URL helpers ───────────────────────────────────────────────────────────

    def _build_url(self, sex: str, age: int) -> str:
        """
        Construct the direct-download URL for one age-sex band.
        Uses 2000-2020 dataset for years < 2015, and 2015-2030 (Global2 R2025A) for years >= 2015.
        """
        iso_lower = self.iso3.lower()
        if self.year < 2015:
            # 2000-2020 dataset format
            filename = f"{iso_lower}_{sex}_{age}_{self.year}.tif"
            return f"https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/{self.year}/{self.iso3}/{filename}"
        else:
            # 2015-2030 dataset format (Global2 R2025A)
            # Use 100m resolution as required by the user
            filename = f"{iso_lower}_{sex}_{age:02d}_{self.year}_CN_100m_R2025A_v1.tif"
            return f"https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/{self.year}/{self.iso3}/v1/100m/constrained/{filename}"

    def _local_path(self, sex: str, age: int) -> Path:
        return self.cache_dir / f"{self.iso3}_{sex}_{age}_{self.year}.tif"

    # ── Download methods ─────────────────────────────────────────────────────

    def _download_band(self, sex: str, age: int) -> Optional[str]:
        """
        Download one age-sex band via direct HTTP request.
        Returns local file path on success, None on failure.
        """
        import requests

        local = self._local_path(sex, age)
        if local.exists():
            return str(local)

        url = self._build_url(sex, age)
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 404:
                return None   # band/year not available for this country
            resp.raise_for_status()
            with open(local, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return str(local)
        except Exception as e:
            print(f"    Download failed ({url}): {e}")
            return None

    def _try_wpgp_package(self, sex: str, age: int) -> Optional[str]:
        """
        Alternative download using wpgpDownloadPy if installed.
        Returns local file path or None.
        """
        try:
            from wpgpDownloadPy import wpgpGetCountryDatasets, wpgpDownloadCountryDatasets
        except ImportError:
            return None

        try:
            covariate = f"pop_age_sex_{sex}_{age}"
            df = wpgpGetCountryDatasets(self.iso3, self.year)
            row = df[df["covariate"] == covariate]
            if row.empty:
                return None
            wpgpDownloadCountryDatasets(self.iso3, self.year, str(self.cache_dir), [covariate])
            expected = self.cache_dir / f"{self.iso3}_{covariate}_{self.year}.tif"
            return str(expected) if expected.exists() else None
        except Exception:
            return None

    # ── Array reading ─────────────────────────────────────────────────────────

    def _read_band_array(
        self, filepath: str, bounds: Optional[Tuple] = None
    ) -> Optional[np.ndarray]:
        """
        Read a GeoTIFF band, optionally clipped to AOI bounds.

        Parameters
        ----------
        bounds : tuple (min_lon, min_lat, max_lon, max_lat), optional
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds as rasterio_from_bounds

            with rasterio.open(filepath) as src:
                if bounds is not None:
                    win  = rasterio_from_bounds(*bounds, src.transform)
                    data = src.read(1, window=win)
                else:
                    data = src.read(1)

                nodata = src.nodata
                arr    = data.astype(float)
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                arr = np.clip(arr, 0, None)  # population cannot be negative
                return arr
        except Exception as e:
            print(f"    Could not read {filepath}: {e}")
            return None

    # ── Main interface ────────────────────────────────────────────────────────

    def download_required_bands(
        self, bounds: Optional[Tuple] = None
    ) -> Dict[str, np.ndarray]:
        """
        Download all age-sex bands needed for HVI computation in parallel.

        Returns
        -------
        dict
            Keys: 'child_0', 'child_1', 'elderly_65'…'elderly_80', 'total'
        """
        import concurrent.futures
        
        print(f"  Fetching WorldPop Global2: {self.iso3} year {self.year}")

        arrays: Dict[str, np.ndarray] = {}

        # Define bands needed beyond total
        needed = {
            "child_0":    ("t", 0),
            "child_1":    ("t", 1),
            "elderly_60": ("t", 60),
            "elderly_65": ("t", 65),
            "elderly_70": ("t", 70),
            "elderly_75": ("t", 75),
            "elderly_80": ("t", 80),
            "elderly_85": ("t", 85),
            "elderly_90": ("t", 90),
        }

        # Gather all unique download tasks
        all_tasks = {}
        for key, (sex, age) in needed.items():
            task_key = f"{sex}_{age}"
            all_tasks[task_key] = (sex, age)
            
        for age in ALL_AGE_BANDS:
            task_key = f"t_{age}"
            if task_key not in all_tasks:
                all_tasks[task_key] = ("t", age)

        print(f"    Fetching {len(all_tasks)} bands (trying direct VFS read first)...")
        fetched_arrays = {}

        def _fetch_task(task_args):
            sex, age = task_args
            url = self._build_url(sex, age)
            
            # Strategy 1: Direct VFS read via Rasterio HTTP Range Requests (sub-second)
            if bounds is not None:
                try:
                    import rasterio
                    # Use a short timeout so we don't hang if the server is slow to stream
                    with rasterio.Env(GDAL_HTTP_TIMEOUT=15, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                        arr = self._read_band_array(url, bounds)
                    if arr is not None:
                        return f"{sex}_{age}", arr
                except Exception:
                    pass  # silently fall back to full download
                    
            # Strategy 2: Download full file and read locally (fallback)
            fp = self._download_band(sex, age)
            if fp is None:
                fp = self._try_wpgp_package(sex, age)
                
            if fp is not None:
                return f"{sex}_{age}", self._read_band_array(fp, bounds)
            return f"{sex}_{age}", None

        # Execute in parallel with up to 10 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {executor.submit(_fetch_task, args): args for args in all_tasks.values()}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    task_key, arr = future.result()
                    if arr is not None:
                        fetched_arrays[task_key] = arr
                    else:
                        print(f"    FAILED to fetch: {task_key}")
                except Exception as e:
                    print(f"    Task exception: {e}")

        # Map fetched arrays to output dictionary
        for key, (sex, age) in needed.items():
            task_key = f"{sex}_{age}"
            if task_key in fetched_arrays:
                arrays[key] = fetched_arrays[task_key]

        # Build total population from all age bands
        print("    Building total population from all age bands...")
        total: Optional[np.ndarray] = None
        for age in ALL_AGE_BANDS:
            task_key = f"t_{age}"
            if task_key in fetched_arrays:
                arr = fetched_arrays[task_key]
                total = arr if total is None else total + np.nan_to_num(arr)

        if total is not None:
            arrays["total"] = total

        return arrays

    def compute_hvi_inputs(
        self, bounds: Optional[Tuple] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (population, elderly_frac, child_frac) as NumPy arrays.

        These are the three population inputs needed for HVI computation.
        Returns (None, None, None) if download fails entirely.
        """
        bands = self.download_required_bands(bounds)

        if not bands or "total" not in bands:
            print(f"  WorldPop download failed for {self.iso3}/{self.year}.")
            print("  Falling back to GEE WorldPop 2020 (see sensitivity.py).")
            return None, None, None

        total      = bands["total"]
        safe_total = np.maximum(total, 1e-6)

        elderly_bands = [v for k, v in bands.items() if k.startswith("elderly_")]
        elderly_sum   = sum(np.nan_to_num(b) for b in elderly_bands)
        elderly_frac  = np.clip(elderly_sum / safe_total, 0, 0.7)

        child_bands = [v for k, v in bands.items() if k.startswith("child_")]
        child_sum   = sum(np.nan_to_num(b) for b in child_bands)
        child_frac  = np.clip(child_sum / safe_total, 0, 0.5)

        return total, elderly_frac, child_frac


# ── Standalone helper ────────────────────────────────────────────────────────

def get_iso3_from_centroid(lat: float, lon: float) -> str:
    """
    Reverse geocode a lat/lon centroid to an ISO3 country code.

    Tries the ``reverse_geocoder`` package first (lightweight, offline).
    Falls back to a hard-coded bounding-box lookup for the most common
    case-study countries.  Returns 'UNKNOWN' if both fail.
    """
    # Strategy 1: reverse_geocoder package
    try:
        import reverse_geocoder as rg
        import pycountry

        result = rg.search([(lat, lon)], mode=1)
        cc2    = result[0]["cc"]

        # FIX 0.1.1: Validate that cc2 is a real 2-char string (not a mock)
        if not isinstance(cc2, str) or len(cc2) != 2:
            raise ValueError(f"Invalid country code from reverse_geocoder: {cc2!r}")

        country = pycountry.countries.get(alpha_2=cc2)
        if country is None or not isinstance(getattr(country, "alpha_3", None), str):
            raise ValueError(f"pycountry could not resolve '{cc2}'")
        return country.alpha_3

    except ImportError:
        pass
    except Exception:
        pass  # fall through to bbox lookup

    # Strategy 2: manual bounding-box lookup (covers most case-study cities)
    BBOX_LOOKUP: Dict[Tuple, str] = {
        (6,  35,  68,  87): "IND",   # India (lon < 88 avoids overlap with BGD)
        (20, 27,  88,  92): "BGD",   # Bangladesh
        (24, 37,  62,  77): "PAK",   # Pakistan
        (-5, 15, -18,  52): "NGA",   # Nigeria (approximate)
        (36, 42,  28,  45): "TUR",   # Turkey
        (10, 24,  36,  44): "ETH",   # Ethiopia
        (20, 40, 100, 125): "PHL",   # Philippines
        (-8,  6, 102, 120): "IDN",   # Indonesia (rough)
    }
    for (lat_min, lat_max, lon_min, lon_max), iso3 in BBOX_LOOKUP.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return iso3

    print(
        "WARNING: Could not determine ISO3 country code from centroid. "
        "Using GEE WorldPop 2020 fallback."
    )
    return "UNKNOWN"
