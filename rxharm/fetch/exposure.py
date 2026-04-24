"""
rxharm/fetch/exposure.py
========================
Fetches the two Exposure sub-index indicators from Google Earth Engine.

Indicators produced:
    population - WorldPop 100 m annual population count for the analysis year.
                 Uses GEE native collection (2000–2020) or community catalog
                 Global2 (2021–2030), with automatic year-based routing.
    built_frac - GHS-BUILT-S built-up surface fraction (dimensionless 0–1)
                 at 100 m. Nearest 5-year epoch to the analysis year is
                 selected automatically. Values stored in m² per cell are
                 divided by 10 000 (100 m × 100 m) to obtain fraction.

GEE is imported lazily — importing this module does not require authentication.

Usage:
    fetcher = ExposureFetcher(ee_geometry, year)
    pop_img   = fetcher.get_population()
    built_img = fetcher.get_built_fraction()
    all_bands = fetcher.fetch_all()
"""

from __future__ import annotations

from typing import List

from rxharm.config import GEE_SCALE, GHS_BUILT_EPOCHS, OUTPUT_CRS

# GEE collection IDs
# FIX v0.1.0: Collection IDs sourced from config.GEE_COLLECTIONS.
# Previously hardcoded here — now centralised for easier version management.
from rxharm.config import GEE_COLLECTIONS as _GC
_WORLDPOP_NATIVE  = _GC["worldpop_pop"]       # was: "WorldPop/GP/100m/pop"
_WORLDPOP_GLOBAL2 = _GC["worldpop_global2"]   # was: "projects/sat-io/open-datasets/WorldPop/Global2"
_GHS_BUILT        = _GC["ghs_built_s"]         # was: "projects/sat-io/open-datasets/GHS/GHS_BUILT_S"

# GHS-BUILT-S stores built area in m² per 100 m cell; cell area = 100*100 m²
_GHS_CELL_AREA_M2 = 10_000.0


class ExposureFetcher:
    """
    Fetches Exposure sub-index indicators from GEE.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        AOI geometry from ``AOIHandler.to_ee_geometry()`` wrapped in
        ``ee.Geometry(dict)``.
    year : int
        Analysis year. Determines which WorldPop collection is used.
    """

    def __init__(self, ee_geometry: object, year: int) -> None:
        self.ee_geometry = ee_geometry
        self.year = year

    # ── Population ────────────────────────────────────────────────────────────

    def get_population(self) -> object:
        """
        Fetch WorldPop annual population count for the analysis year.

        Routing:
            year ≤ 2020 → ``WorldPop/GP/100m/pop`` (GEE native)
            year > 2020 → ``projects/sat-io/open-datasets/WorldPop/Global2``
                         Falls back to 2020 with a warning if unavailable.

        Returns
        -------
        ee.Image
            Single-band image ``'population'`` (persons per 100 m cell).
            Negative values clipped to 0.
        """
        import ee

        if self.year <= 2020:
            pop = self._get_worldpop_native()
        else:
            pop = self._get_worldpop_global2()

        return (
            pop
            .clamp(0, 1e9)
            .rename("population")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )

    def _get_worldpop_native(self) -> object:
        """
        Load WorldPop from the GEE native catalog for years 2000–2020.

        Returns
        -------
        ee.Image
            Single-band population image for self.year.
        """
        import ee
        col = (
            ee.ImageCollection(_WORLDPOP_NATIVE)
            .filterBounds(self.ee_geometry)
            .filter(ee.Filter.eq("year", self.year))
        )
        # REASON: Some years have multiple tiles; mosaic handles spatial seams.
        img = col.mosaic()
        # WorldPop native band may be 'population' or 'b1' depending on year
        band_names = col.first().bandNames().getInfo()
        band = "population" if "population" in band_names else band_names[0]
        return img.select(band)

    def _get_worldpop_global2(self) -> object:
        """
        Load WorldPop from the community catalog Global2 for years 2021+.

        Falls back to 2020 native collection if Global2 is inaccessible.

        Returns
        -------
        ee.Image
            Single-band population image.
        """
        import ee
        try:
            col = (
                ee.ImageCollection(_WORLDPOP_GLOBAL2)
                .filterBounds(self.ee_geometry)
                .filter(ee.Filter.eq("year", self.year))
            )
            size = col.size().getInfo()
            if size == 0:
                raise ValueError("No images found")
            band_names = col.first().bandNames().getInfo()
            band = "population" if "population" in band_names else band_names[0]
            return col.mosaic().select(band)
        except Exception:
            # REASON: Community catalog access can fail for various GEE account tiers.
            print(
                f"WorldPop Global2 unavailable, using 2020 as proxy for {self.year}"
            )
            col2020 = (
                ee.ImageCollection(_WORLDPOP_NATIVE)
                .filterBounds(self.ee_geometry)
                .filter(ee.Filter.eq("year", 2020))
            )
            return col2020.mosaic().select("population")

    # ── Built fraction ────────────────────────────────────────────────────────

    def get_built_fraction(self) -> object:
        """
        FIX 0.1.1: Corrected to use JRC/GHSL/P2023A/GHS_BUILT_S native catalog.

        Each epoch is accessed as a SEPARATE ee.Image, NOT an ImageCollection.
        The nearest available epoch to self.year is selected automatically.

        Values are in m² of built surface per 100m cell.
        Dividing by 10,000 (100m × 100m) gives the fraction [0, 1].

        Source: JRC GHSL P2023A release, Pesaresi et al. (2023).

        Returns
        -------
        ee.Image
            Single-band image ``'built_frac'`` (dimensionless 0–1) at 100 m.
        """
        import ee
        from rxharm.config import GEE_COLLECTIONS, GHS_BUILT_EPOCHS, GEE_SCALE, OUTPUT_CRS

        # Find nearest available epoch
        nearest_epoch = min(GHS_BUILT_EPOCHS, key=lambda e: abs(e - self.year))

        # FIX 0.1.1: Access as individual Image, not ImageCollection
        collection_prefix = GEE_COLLECTIONS["ghs_built_s"]
        image_id = f"{collection_prefix}/{nearest_epoch}"

        try:
            built_img = ee.Image(image_id).select("built_surface")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GHS-BUILT-S for epoch {nearest_epoch}: {e}\n"
                f"Tried: {image_id}\n"
                "Check that your GEE project has access to JRC/GHSL/P2023A assets."
            )

        if nearest_epoch != self.year:
            print(f"  GHS-BUILT-S: using {nearest_epoch} epoch (nearest to {self.year})")

        # FIX 0.1.1: Convert m² per cell → fraction [0, 1]
        # 100m × 100m cell = 10,000 m² total area
        cell_area_m2 = 10_000.0
        built_frac = (
            built_img
            .divide(cell_area_m2)
            .clamp(0, 1)
            .rename("built_frac")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return built_frac

    def get_building_height(self) -> object:
        """
        FIX 0.1.1: New method for GHS-BUILT-H building height (metres).

        Only the 2018 epoch is available from JRC GHSL P2023A.
        Used by FeasibilityEngine for cool-roof and BGI height thresholds.

        Returns
        -------
        ee.Image
            Single-band image ``'built_h'`` (metres) at 100 m.
        """
        import ee
        from rxharm.config import GEE_COLLECTIONS, GHS_BUILT_H_EPOCH, GEE_SCALE, OUTPUT_CRS

        image_id = f"{GEE_COLLECTIONS['ghs_built_h']}/{GHS_BUILT_H_EPOCH}"
        built_h = (
            ee.Image(image_id)
            .select("built_height")
            .rename("built_h")
            .reproject(crs=OUTPUT_CRS, scale=GEE_SCALE)
            .clip(self.ee_geometry)
        )
        return built_h

    # ── Combined fetch ────────────────────────────────────────────────────────

    def fetch_all(self) -> object:
        """
        Fetch both exposure indicators and return a two-band image.

        Returns
        -------
        ee.Image
            Two-band image with bands ``'population'`` and ``'built_frac'``.
        """
        import ee
        pop   = self.get_population()
        built = self.get_built_fraction()
        return ee.Image.cat([pop, built])
