"""
rxharm/fetch/adaptive_capacity.py
==================================
Fetches the five Adaptive Capacity sub-index indicators from GEE.

Indicators produced:
    ndvi          - Sentinel-2 NDVI (NIR-Red)/(NIR+Red), seasonal median.
                    10 m native → 100 m output.
    ndwi          - Sentinel-2 VEGETATION MOISTURE NDWI (NIR-SWIR1)/(NIR+SWIR1).
                    NOT the open-water NDWI (Green-NIR). Uses B8 + B11.
                    10 m native → 100 m output.
    tree_cover    - Hansen GFW effective tree cover: 2000 baseline minus
                    cumulative annual loss through the analysis year. 30 m → 100 m.
    canopy_height - Potapov/GLAD global canopy height 2019 (static Tier 3).
                    30 m → 100 m.
    viirs_dnb_raw - VIIRS DNB annual nighttime light, raw 500 m composite.
                    Suffixed '_raw' to signal it must be processed by
                    VIIRSDownscaler before entering the HVI formula.

GEE is imported lazily — importing this module does not require authentication.

Usage:
    fetcher = AdaptiveCapacityFetcher(ee_geometry, year, hottest_months)
    all_bands = fetcher.fetch_all()
    # Note: viirs_dnb_raw must be downscaled before HVI computation.
"""

from __future__ import annotations

from typing import List

from rxharm.config import (
    GEE_SCALE,
    OUTPUT_CRS,
    S2_MAX_CLOUD_PCT,
    S2_SCL_MASK_VALUES,
)
from rxharm.fetch.hazard import _check_collection_size

# GEE collection IDs
# FIX v0.1.0: Collection IDs sourced from config.GEE_COLLECTIONS.
from rxharm.config import GEE_COLLECTIONS as _GC
_S2_HARMONIZED  = _GC["sentinel2"]    # was: "COPERNICUS/S2_SR_HARMONIZED"
_HANSEN_2023    = _GC["hansen_gfw"]   # was: "UMD/hansen/global_forest_change_2023_v1_11"
_POTAPOV_HEIGHT = _GC["potapov_ch"]   # was: "projects/sat-io/open-datasets/GLAD/GEDI_V27"
_VIIRS_ANNUAL   = _GC["viirs_dnb"]    # was: "NOAA/VIIRS/DNB/ANNUAL_V21"

# Hansen dataset encodes loss year as years-since-2000 (max 23 for 2023 version)
_HANSEN_LOSS_YEAR_MAX = 23


class AdaptiveCapacityFetcher:
    """
    Fetches Adaptive Capacity sub-index indicators from GEE.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        AOI geometry from ``AOIHandler.to_ee_geometry()``.
    year : int
        Analysis year (2016–present for Sentinel-2).
    hottest_months : list of int
        Calendar months for Sentinel-2 composite window.
    """

    def __init__(
        self,
        ee_geometry: object,
        year: int,
        hottest_months: List[int],
    ) -> None:
        self.ee_geometry    = ee_geometry
        self.year           = year
        self.hottest_months = hottest_months

        first = min(hottest_months)
        last  = max(hottest_months)
        self._start_date = f"{year}-{first:02d}-01"
        self._end_date   = (
            f"{year + 1}-01-01" if last == 12 else f"{year}-{last + 1:02d}-01"
        )

    # ── Sentinel-2 shared helpers ─────────────────────────────────────────────

    def _build_s2_collection(self) -> object:
        """
        Build a cloud-masked Sentinel-2 SR Harmonized collection.

        FIX 0.1.1: Caches result on self._s2_col to avoid rebuilding
        for each of NDVI and NDWI (saves 1 GEE metadata call per run).

        Filters by bounds, date, and CLOUDY_PIXEL_PERCENTAGE, then masks
        cloud and snow pixels using the SCL band (S2_SCL_MASK_VALUES from config).

        Returns
        -------
        ee.ImageCollection
            Cloud-masked collection filtered to the AOI and date window.
        """
        # FIX 0.1.1: Return cached collection if already built
        if getattr(self, "_s2_col", None) is not None:
            return self._s2_col

        import ee

        def _scl_mask(img: object) -> object:
            """Mask pixels whose SCL class is in S2_SCL_MASK_VALUES."""
            scl = img.select("SCL")
            mask = scl.neq(S2_SCL_MASK_VALUES[0])
            for val in S2_SCL_MASK_VALUES[1:]:
                mask = mask.And(scl.neq(val))
            return img.updateMask(mask)

        self._s2_col = (
            ee.ImageCollection(_S2_HARMONIZED)
            .filterBounds(self.ee_geometry)
            .filterDate(self._start_date, self._end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_CLOUD_PCT))
            .map(_scl_mask)
        )
        return self._s2_col

    # ── Sentinel-2 indices ────────────────────────────────────────────────────

    def get_ndvi(self) -> object:
        """
        Compute Sentinel-2 NDVI as the median composite over the hottest period.

        Formula: NDVI = (B8 - B4) / (B8 + B4)

        Returns
        -------
        ee.Image
            Single-band image ``'ndvi'`` (−1 to 1) at 100 m.
        """
        import ee
        col = self._build_s2_collection()
        _check_collection_size(col, "ndvi")

        def _ndvi(img: object) -> object:
            return img.normalizedDifference(["B8", "B4"]).rename("ndvi")

        return (
            col.map(_ndvi)
            .median()
            .clamp(-1, 1)
            .rename("ndvi")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    def get_ndwi(self) -> object:
        """
        Compute Sentinel-2 VEGETATION MOISTURE NDWI over the hottest period.

        # VEGETATION MOISTURE NDWI (Gao 1996) — NOT open water NDWI
        Formula: NDWI_veg = (B8 - B11) / (B8 + B11)
            B8  = NIR (10 m)
            B11 = SWIR1 (20 m native; resampled to 10 m in Harmonized collection)

        This differs from the open-water NDWI = (Green-NIR)/(Green+NIR) (McFeeters).
        The vegetation moisture variant captures plant water stress during the
        pre-monsoon dry period, when NDVI may still appear green but ET cooling
        is suppressed due to soil water deficit.

        Returns
        -------
        ee.Image
            Single-band image ``'ndwi'`` (−1 to 1) at 100 m.
        """
        import ee
        col = self._build_s2_collection()
        _check_collection_size(col, "ndwi")

        def _ndwi(img: object) -> object:
            # VEGETATION MOISTURE NDWI (Gao 1996) — NOT open water NDWI
            return img.normalizedDifference(["B8", "B11"]).rename("ndwi")

        return (
            col.map(_ndwi)
            .median()
            .clamp(-1, 1)
            .rename("ndwi")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    # ── Tree cover (Hansen GFW) ───────────────────────────────────────────────

    def get_tree_cover(self) -> object:
        """
        Compute Hansen GFW effective tree cover accounting for cumulative loss.

        Steps:
            1. Load baseline tree cover in year 2000 (0–100 %).
            2. Load lossyear band (0 = no loss, 1–23 = year of loss since 2000).
            3. Create loss mask: pixels with loss in any year up to self.year.
            4. Set masked pixels to 0 (tree removed by analysis year).

        Returns
        -------
        ee.Image
            Single-band image ``'tree_cover'`` (0–100, percent canopy) at 100 m.
        """
        import ee
        hansen = ee.Image(_HANSEN_2023)
        tc2000    = hansen.select("treecover2000")
        loss_year = hansen.select("lossyear")

        years_since_2000 = min(self.year - 2000, _HANSEN_LOSS_YEAR_MAX)
        loss_occurred = (
            loss_year.gt(0).And(loss_year.lte(years_since_2000))
        )
        effective_cover = tc2000.where(loss_occurred, 0)

        return (
            effective_cover
            .rename("tree_cover")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    # ── Canopy height (Potapov GLAD) ──────────────────────────────────────────

    def get_canopy_height(self) -> object:
        """
        Load the Potapov/GLAD global forest canopy height mosaic (2019 static).

        Values are in metres (0–30+; saturates above 30 m). Known limitation:
        may confuse tall buildings with tree canopy in dense urban cores.

        Returns
        -------
        ee.Image
            Single-band image ``'canopy_height'`` (metres, 0–50) at 100 m.
        """
        import ee
        height = (
            ee.ImageCollection(_POTAPOV_HEIGHT)
            .filterBounds(self.ee_geometry)
            .mosaic()
            .select("b1")
            .clamp(0, 50)
            .rename("canopy_height")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return height

    # ── VIIRS nighttime light (raw 500 m) ─────────────────────────────────────

    def get_viirs_dnb_raw(self) -> object:
        """
        Fetch VIIRS DNB annual nighttime light composite for the analysis year.

        FIX 0.1.1: Added .toFloat() before clamp to prevent integer overflow
        when the raw band is stored as uint16. Also handles both 'maximum'
        and 'avg_rad' band names across different versions of the collection.

        Returns the raw 500 m product reprojected to 100 m with bilinear
        resampling. The ``_raw`` suffix signals that downstream code MUST
        pass this band through ``VIIRSDownscaler`` before HVI computation.

        Returns
        -------
        ee.Image
            Single-band image ``'viirs_dnb_raw'`` (nW/cm²/sr, 0–200) at 100 m.
        """
        import ee

        col = (
            ee.ImageCollection(_VIIRS_ANNUAL)
            .filterBounds(self.ee_geometry)
            .filter(ee.Filter.calendarRange(self.year, self.year, "year"))
        )
        size = col.size().getInfo()
        if size == 0:
            latest = (
                ee.ImageCollection(_VIIRS_ANNUAL)
                .filterBounds(self.ee_geometry)
                .sort("system:time_start", False)
                .first()
            )
            available_year = (
                ee.Date(latest.get("system:time_start")).get("year").getInfo()
            )
            print(
                f"  VIIRS DNB: year {self.year} unavailable, "
                f"using {available_year} as proxy."
            )
            img = latest
        else:
            img = col.first()

        # FIX 0.1.1: VIIRS ANNUAL_V21 uses 'maximum' band; older versions used
        # 'avg_rad'. Try 'maximum' first, fall back to 'avg_rad'.
        band_names = img.bandNames().getInfo()
        viirs_band = "maximum" if "maximum" in band_names else "avg_rad"

        return (
            img
            .select(viirs_band)
            .toFloat()            # FIX 0.1.1: cast to float before clamp
            .clamp(0, 200)
            .rename("viirs_dnb_raw")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    # ── Combined fetch ────────────────────────────────────────────────────────

    def fetch_all(self) -> object:
        """
        Fetch all five adaptive capacity indicators as a single image.

        Returns
        -------
        ee.Image
            Five-band image: ``'ndvi'``, ``'ndwi'``, ``'tree_cover'``,
            ``'canopy_height'``, ``'viirs_dnb_raw'``.
        """
        import ee
        return ee.Image.cat([
            self.get_ndvi(),
            self.get_ndwi(),
            self.get_tree_cover(),
            self.get_canopy_height(),
            self.get_viirs_dnb_raw(),
        ])
