"""
rxharm/fetch/sensitivity.py
============================
Fetches the four Sensitivity sub-index indicators from Google Earth Engine.

Indicators produced:
    elderly_frac - Fraction of population aged 65+ (WorldPop constrained
                   age-sex disaggregated data). Ratio of elderly bands to
                   total, 100 m.
    child_frac   - Fraction of population aged <5 (WorldPop age-sex).
                   100 m.
    impervious   - Built surface fraction from Dynamic World annual median
                   probability composite. Proxy for impervious surface.
                   10 m native → 100 m output.
    cropland     - Crop surface fraction from Dynamic World annual median
                   probability composite. Proxy for mandatory outdoor
                   agricultural worker exposure. 10 m → 100 m.

GEE is imported lazily — importing this module does not require authentication.

Usage:
    fetcher = SensitivityFetcher(ee_geometry, year, hottest_months)
    fracs = fetcher.get_age_fractions()   # elderly_frac, child_frac
    dw    = fetcher.get_dynamic_world_fractions()  # impervious, cropland
    all_bands = fetcher.fetch_all()
"""

from __future__ import annotations

from typing import List

from rxharm.config import (
    GEE_SCALE,
    MIN_POPULATION_THRESHOLD,
    OUTPUT_CRS,
    S2_MAX_CLOUD_PCT,
)
from rxharm.fetch.hazard import _check_collection_size

# GEE collection IDs
# FIX v0.1.0: Collection IDs sourced from config.GEE_COLLECTIONS.
from rxharm.config import GEE_COLLECTIONS as _GC
_WORLDPOP_AGE  = _GC["worldpop_agesex"]   # was: "WorldPop/GP/100m/pop_age_sex_cons_unadj"
_DW            = _GC["dynamic_world"]      # was: "GOOGLE/DYNAMICWORLD/V1"

# WorldPop age-sex band names for elderly (65+) and child (<5)
_ELDERLY_BANDS = ["M_65", "M_70", "M_75", "M_80", "F_65", "F_70", "F_75", "F_80"]
_CHILD_BANDS   = ["M_0", "F_0"]   # age group 0–4 encoded as 0 in WorldPop


class SensitivityFetcher:
    """
    Fetches Sensitivity sub-index indicators from GEE.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        AOI geometry from ``AOIHandler.to_ee_geometry()``.
    year : int
        Analysis year.
    hottest_months : list of int
        Calendar months for Dynamic World composite window.
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

    # ── Age fractions ─────────────────────────────────────────────────────────

    def get_age_fractions(self) -> object:
        """
        FIX 0.1.1: Compute elderly (65+) and child (<5) population fractions.

        PRIMARY path: WorldPop Global2 R2025A — downloaded directly from
        worldpop.org using WorldPopFetcher for years 2015–2030.
        Local arrays are stored in self._worldpop_local for post-processing.

        FALLBACK path: GEE WorldPop 2020 age-sex dataset when download fails
        or when year is outside 2015–2030 range.

        Returns
        -------
        ee.Image
            Two-band image: ``'elderly_frac'``, ``'child_frac'`` (both 0–1).
            Returns a placeholder constant image if WorldPop local data is used.
        """
        import ee
        from rxharm.config import GEE_COLLECTIONS, GEE_SCALE, OUTPUT_CRS

        # FIX 0.1.1: Try WorldPop Global2 for years 2015-2030
        self._worldpop_local = None
        use_gee_fallback = False

        if 2015 <= self.year <= 2030:
            try:
                from rxharm.fetch.worldpop_fetcher import WorldPopFetcher, get_iso3_from_centroid
                # Get centroid coordinates from the GEE geometry
                centroid = self.geom.centroid().coordinates().getInfo()
                lon, lat = centroid[0], centroid[1]
                iso3 = get_iso3_from_centroid(lat, lon)

                if iso3 != "UNKNOWN":
                    # Get AOI bounding box for spatial clip during download
                    bounds_info = self.geom.bounds().getInfo()["coordinates"][0]
                    xcoords = [c[0] for c in bounds_info]
                    ycoords = [c[1] for c in bounds_info]
                    aoi_bounds = (min(xcoords), min(ycoords), max(xcoords), max(ycoords))

                    wpf = WorldPopFetcher(iso3, self.year)
                    pop, elderly_frac_arr, child_frac_arr = wpf.compute_hvi_inputs(aoi_bounds)

                    if pop is not None:
                        # Store arrays — they bypass GEE and are merged in post-processing
                        self._worldpop_local = {
                            "population":   pop,
                            "elderly_frac": elderly_frac_arr,
                            "child_frac":   child_frac_arr,
                            "source":       f"WorldPop Global2 R2025A {iso3} {self.year}",
                        }
                        print(f"  WorldPop Global2: downloaded for {iso3} {self.year} — local processing")
                        # Return constant placeholder — actual values come from local arrays
                        return self._build_worldpop_placeholder()
                    else:
                        use_gee_fallback = True
                else:
                    use_gee_fallback = True

            except Exception as e:
                print(f"  WorldPop Global2 download failed: {e}")
                use_gee_fallback = True
        else:
            use_gee_fallback = True

        # GEE FALLBACK: WorldPop 2020 native collection
        if use_gee_fallback:
            print(f"  Falling back to GEE WorldPop 2020 (year={self.year} requested)")
            if self.year != 2020:
                print("  WARNING: Population is 2020 data, not the requested year.")
                print("  This affects both count and age fractions.")
        return self._get_age_fractions_gee_fallback()

    def _build_worldpop_placeholder(self) -> object:
        """
        FIX 0.1.1: Returns a GEE constant image as a placeholder.
        The actual WorldPop data is stored in self._worldpop_local and
        merged into the final numpy arrays during post-export processing.
        """
        import ee
        return (
            ee.Image.constant([-1, -1])
            .rename(["elderly_frac", "child_frac"])
            .toFloat()
            .reproject(crs="EPSG:4326", scale=100)
        )

    def _get_age_fractions_gee_fallback(self) -> object:
        """GEE WorldPop 2020 fallback — original implementation preserved."""
        import ee
        from rxharm.config import GEE_COLLECTIONS, GEE_SCALE, OUTPUT_CRS
        from rxharm.config import MIN_POPULATION_THRESHOLD

        col = (
            ee.ImageCollection(GEE_COLLECTIONS["worldpop_agesex"])
            .filterBounds(self.ee_geometry)
            .filter(ee.Filter.eq("year", 2020))
        )
        img = col.mosaic()

        elderly_bands = ["M_65", "M_70", "M_75", "M_80", "F_65", "F_70", "F_75", "F_80"]
        child_bands   = ["M_0", "F_0"]

        elderly_sum  = img.select(elderly_bands).reduce(ee.Reducer.sum())
        child_sum    = img.select(child_bands).reduce(ee.Reducer.sum())
        total        = img.select("population")

        eps          = 1e-6
        elderly_frac = elderly_sum.divide(total.add(eps)).clamp(0, 1)
        child_frac   = child_sum.divide(total.add(eps)).clamp(0, 1)

        pop_mask     = total.gte(MIN_POPULATION_THRESHOLD)
        elderly_frac = elderly_frac.updateMask(pop_mask).unmask(0)
        child_frac   = child_frac.updateMask(pop_mask).unmask(0)

        return (
            elderly_frac.rename("elderly_frac")
            .addBands(child_frac.rename("child_frac"))
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    # ── Dynamic World fractions ───────────────────────────────────────────────

    def get_dynamic_world_fractions(self) -> object:
        """
        Fetch impervious and cropland probability fractions from Dynamic World.

        Uses the same date window as the Landsat composite for temporal
        consistency. Probability bands (built, crops) are continuous [0, 1]
        and composited with a median reducer.

        Returns
        -------
        ee.Image
            Two-band image: ``'impervious'``, ``'cropland'`` (both 0–1).
        """
        import ee

        col = (
            ee.ImageCollection(_DW)
            .filterBounds(self.ee_geometry)
            .filterDate(self._start_date, self._end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_CLOUD_PCT))
            .select(["built", "crops"])
        )
        _check_collection_size(col, "impervious/cropland")

        dw = (
            col
            .reduce(ee.Reducer.median())
            .rename(["impervious", "cropland"])
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return dw

    # ── Combined fetch ────────────────────────────────────────────────────────

    def fetch_all(self) -> object:
        """
        Fetch all four sensitivity indicators as a single multi-band image.

        FIX 0.1.1: Caches the Dynamic World label mosaic on ``self._dw_label_mosaic``
        so that ``fetch_all_indicators()`` can pass it to ``HazardFetcher.get_uhi()``
        without a second GEE fetch of the same dataset.

        Returns
        -------
        ee.Image
            Four-band image: ``'elderly_frac'``, ``'child_frac'``,
            ``'impervious'``, ``'cropland'``.
        """
        import ee
        age_img = self.get_age_fractions()
        dw_img  = self.get_dynamic_world_fractions()
        # FIX 0.1.1: expose the DW built band for UHI reuse
        # The DW label mosaic is built inside get_dynamic_world_fractions();
        # we approximate the built mask as the 'built_dw' band if available,
        # otherwise use the impervious band as a proxy.
        try:
            self._dw_label_mosaic = dw_img.select("impervious").gt(0.5)
        except Exception:
            self._dw_label_mosaic = None
        return ee.Image.cat([age_img, dw_img])
