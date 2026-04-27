"""
rxharm/fetch/hazard.py
======================
Fetches the three Hazard sub-index indicators from Google Earth Engine.

Indicators produced:
    lst    - Landsat 8+9 land surface temperature (°C), 90th-percentile
             composite over the hottest analysis period. 30m native → 100m.
    albedo - Landsat broadband albedo (dimensionless 0-1), mean composite.
             Computed from OLI bands using Liang (2000) coefficients.
             30m native → 100m.
    uhi    - Urban Heat Island intensity (°C), derived from lst composite
             minus the rural background LST within a 5 km focal window.
             30m native → 100m.

GEE is imported lazily — importing this module does not require authentication.

Usage:
    fetcher = HazardFetcher(ee_geometry, year, hottest_months)
    lst_image    = fetcher.get_lst()
    albedo_image = fetcher.get_albedo()
    uhi_image    = fetcher.get_uhi(lst_image)
    all_bands    = fetcher.fetch_all()   # returns merged ee.Image
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from typing import List, Tuple

# ── Internal ──────────────────────────────────────────────────────────────────
from rxharm.config import (
    ALBEDO_COEFFS,
    GEE_SCALE,
    LANDSAT_SR_OFFSET,
    LANDSAT_SR_SCALE,
    MIN_LANDSAT_SCENES,
    OUTPUT_CRS,
)

# ── GEE collections ───────────────────────────────────────────────────────────
_LC08 = None  # FIX v0.1.0: see _get_collections() below
_LC09 = None
_DW   = None


def _get_collections():
    """FIX v0.1.0: Lazy load from GEE_COLLECTIONS so IDs are centralised in config.py."""
    from rxharm.config import GEE_COLLECTIONS
    return (
        GEE_COLLECTIONS["landsat8"],       # was: "LANDSAT/LC08/C02/T1_L2"
        GEE_COLLECTIONS["landsat9"],       # was: "LANDSAT/LC09/C02/T1_L2"
        GEE_COLLECTIONS["dynamic_world"],  # was: "GOOGLE/DYNAMICWORLD/V1"
    )


def _lc08_id():
    from rxharm.config import GEE_COLLECTIONS
    return GEE_COLLECTIONS["landsat8"]


def _lc09_id():
    from rxharm.config import GEE_COLLECTIONS
    return GEE_COLLECTIONS["landsat9"]


def _dw_id():
    from rxharm.config import GEE_COLLECTIONS
    return GEE_COLLECTIONS["dynamic_world"]


def _check_collection_size(collection: object, indicator_name: str, min_size: int = 1) -> int:
    """
    Verify a GEE ImageCollection has enough images before compositing.

    Parameters
    ----------
    collection : ee.ImageCollection
        The filtered collection to check.
    indicator_name : str
        Human-readable indicator name for the error message.
    min_size : int
        Minimum acceptable scene count.

    Returns
    -------
    int
        Actual collection size.

    Raises
    ------
    RuntimeError
        If the collection has fewer than min_size images.
    """
    size = collection.size().getInfo()
    if size < min_size:
        raise RuntimeError(
            f"Insufficient GEE data for indicator '{indicator_name}': "
            f"found {size} scenes (minimum {min_size}). "
            f"Try extending the date window in config.py (N_HOTTEST_MONTHS) "
            f"or choosing a different year."
        )
    return size


class HazardFetcher:
    """
    Fetches and composites all three Hazard sub-index indicators from GEE.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        AOI geometry from ``AOIHandler.to_ee_geometry()`` wrapped in
        ``ee.Geometry(dict)``.
    year : int
        Analysis year (2013–present, Landsat 8 C2 availability).
    hottest_months : list of int
        Calendar months for the composite window, e.g. ``[4, 5]``.
        From ``SeasonalDetector.detect()``.
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

        # Build date and bounds filters once; reuse across all three indicators
        start, end = self._month_to_date_range(year, hottest_months)
        self._start_date = start
        self._end_date   = end

    # ── Static / class helpers ────────────────────────────────────────────────

    @staticmethod
    def _month_to_date_range(year: int, months: List[int]) -> Tuple[str, str]:
        """
        Convert a year and list of month integers to a GEE date range string pair.
        Expands the search window by +/- 1 month to increase clear-sky scene availability.
        """
        first = min(months) - 1
        last  = max(months) + 1
        
        # Handle year wrapping for the expanded window
        start_year = year
        if first < 1:
            first = 12
            start_year -= 1
            
        end_year = year
        if last > 12:
            last -= 12
            end_year += 1
            
        start = f"{start_year}-{first:02d}-01"
        
        if last == 12:
            end = f"{end_year + 1}-01-01"
        else:
            end = f"{end_year}-{last + 1:02d}-01"
        return start, end

    @staticmethod
    def _mask_landsat_clouds(image: object) -> object:
        """
        Apply a cloud and cloud-shadow mask using the QA_PIXEL band.
        """
        import ee
        qa = image.select("QA_PIXEL")
        cloud_mask = (
            qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
            .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
        )
        return image.updateMask(cloud_mask)

    @staticmethod
    def _scale_landsat_sr(image: object) -> object:
        """
        Apply Collection 2 Level-2 scale factors to convert DN to physical units.
        """
        import ee
        sr_bands = ["SR_B2", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
        optical = (
            image.select(sr_bands)
            .multiply(LANDSAT_SR_SCALE)
            .add(LANDSAT_SR_OFFSET)
        )
        lst_celsius = (
            image.select("ST_B10")
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename("LST_C")
        )
        return optical.addBands(lst_celsius)

    # ── Indicator methods ─────────────────────────────────────────────────────

    def _build_landsat_collection(self) -> object:
        """
        Build a merged, cloud-masked, scaled Landsat 8+9 collection.
        Filters to images with less than 15% cloud cover for better accuracy.
        """
        import ee

        def process(col_id: str) -> object:
            return (
                ee.ImageCollection(col_id)
                .filterBounds(self.ee_geometry)
                .filterDate(self._start_date, self._end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", 15))  # strict cloud threshold
                .map(self._mask_landsat_clouds)
                .map(self._scale_landsat_sr)
            )

        return process(_lc08_id()).merge(process(_lc09_id()))

    def get_lst(self) -> object:
        """
        Fetch Landsat LST as the 90th-percentile composite (hottest period).

        FIX 0.1.1: Added scene count check before reducing. If fewer than
        MIN_LANDSAT_SCENES scenes are available, the cloud mask is relaxed
        (bit 3 = confirmed cloud only) to include cloud-adjacent pixels.
        Prevents silent export failures in cloud-prone regions.

        Returns
        -------
        ee.Image
            Single-band image ``'lst'`` in degrees Celsius at 100 m.
        """
        import ee
        collection = self._build_landsat_collection()

        # FIX 0.1.1: Check scene count BEFORE reducing.
        n_scenes = collection.size().getInfo()
        if n_scenes < MIN_LANDSAT_SCENES:
            print(f"  WARNING: Only {n_scenes} Landsat scenes in composite window.")
            print("  Relaxing cloud mask (keeping cloud-adjacent pixels).")
            print("  LST quality may be reduced. Consider extending YEAR or window.")

            # Relaxed mask: mask ONLY confirmed cloud pixels (QA bit 3)
            def mask_clouds_relaxed(img: object) -> object:
                qa = img.select("QA_PIXEL")
                cloud_only = qa.bitwiseAnd(1 << 3).eq(0)   # bit 3 = cloud
                return img.updateMask(cloud_only)

            # Rebuild collection with relaxed mask
            def process_relaxed(col_id: str) -> object:
                return (ee.ImageCollection(col_id)
                        .filterBounds(self.ee_geometry)
                        .filterDate(self._start_date, self._end_date)
                        .filter(ee.Filter.lt("CLOUD_COVER", 15))
                        .map(mask_clouds_relaxed)
                        .map(self._scale_landsat_sr))

            collection = process_relaxed(_lc08_id()).merge(process_relaxed(_lc09_id()))

        lst = (
            collection
            .select("LST_C")
            .reduce(ee.Reducer.percentile([90]))
            .rename("lst")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return lst

    def get_albedo(self) -> object:
        """
        Derive broadband albedo using Liang (2000) coefficients.

        Applies: α = 0.356*B2 + 0.130*B4 + 0.373*B5 + 0.085*B6 + 0.072*B7 - 0.0018
        to mean-composited scaled surface reflectance.

        Returns
        -------
        ee.Image
            Single-band image ``'albedo'`` (dimensionless, 0–1) at 100 m.
        """
        import ee
        collection = self._build_landsat_collection()

        def _compute_albedo(img: object) -> object:
            """Per-image albedo from scaled OLI reflectance."""
            alb = (
                img.select("SR_B2").multiply(ALBEDO_COEFFS["B2"])
                .add(img.select("SR_B4").multiply(ALBEDO_COEFFS["B4"]))
                .add(img.select("SR_B5").multiply(ALBEDO_COEFFS["B5"]))
                .add(img.select("SR_B6").multiply(ALBEDO_COEFFS["B6"]))
                .add(img.select("SR_B7").multiply(ALBEDO_COEFFS["B7"]))
                .add(ALBEDO_COEFFS["intercept"])
            )
            return alb.rename("albedo").copyProperties(img, ["system:time_start"])

        albedo = (
            collection.map(_compute_albedo)
            .mean()
            .clamp(0, 1)
            .rename("albedo")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return albedo

    def get_uhi(self, lst_image: object, dw_built_mask: object = None) -> object:
        """
        Compute Urban Heat Island intensity from an existing LST composite.

        FIX 0.1.1: Accept a pre-computed Dynamic World built mask as an optional
        parameter to avoid fetching DW twice (once here, once in sensitivity.py).
        If ``dw_built_mask`` is None, fetches DW internally as before.

        Parameters
        ----------
        lst_image : ee.Image
            LST composite from ``get_lst()``.
        dw_built_mask : ee.Image, optional
            Pre-computed DW built probability or label image. If provided,
            avoids a second GEE fetch. Pass the 'built' band from sensitivity.

        Returns
        -------
        ee.Image
            Single-band image ``'uhi'`` (°C, clipped to [-5, 20]) at 100 m.
        """
        import ee
        if dw_built_mask is None:
            # FIX 0.1.1: Fetch DW only if not provided externally
            dw_label = (
                ee.ImageCollection(_dw_id())
                .filterBounds(self.ee_geometry)
                .filterDate(self._start_date, self._end_date)
                .select("label")
                .reduce(ee.Reducer.mode())
            )
            rural_mask = dw_label.neq(6)   # not built
        else:
            # Use the provided mask: invert so 0=built → rural = where mask != 1
            rural_mask = dw_built_mask.neq(1)

        # Rural LST: apply mask
        rural_lst = lst_image.updateMask(rural_mask)
        
        # FIX: Instead of a 5km local focal mean (which creates nodata gaps in the center
        # of large cities >5km from rural pixels), we compute a single scalar mean
        # representing the rural baseline temperature for the entire Area of Interest.
        rural_mean_dict = rural_lst.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.ee_geometry,
            scale=GEE_SCALE,
            maxPixels=1e10
        )
        
        # If the AOI is 100% urban (no rural pixels), fallback to the AOI's overall mean
        fallback_mean_dict = lst_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.ee_geometry,
            scale=GEE_SCALE,
            maxPixels=1e10
        )
        
        # Extract the scalar value and handle the fallback
        rural_mean = rural_mean_dict.get("lst")
        fallback_mean = fallback_mean_dict.get("lst")
        final_rural_bg = ee.Algorithms.If(
            ee.Algorithms.IsEqual(rural_mean, None),
            ee.Number(fallback_mean),
            ee.Number(rural_mean)
        )
        
        uhi = (
            lst_image
            .subtract(ee.Image.constant(final_rural_bg))
            .clamp(-5, 20)
            .rename("uhi")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return uhi

    def fetch_all(self) -> object:
        """
        Fetch all three hazard indicators and return a single multi-band image.

        Returns
        -------
        ee.Image
            Three-band image with bands ``'lst'``, ``'albedo'``, ``'uhi'``.
        """
        import ee
        print("Fetching hazard indicators: LST...", end=" ", flush=True)
        lst = self.get_lst()
        print("Albedo...", end=" ", flush=True)
        albedo = self.get_albedo()
        print("UHI...", flush=True)
        uhi = self.get_uhi(lst)
        return ee.Image.cat([lst, albedo, uhi])
