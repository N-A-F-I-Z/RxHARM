"""
rxharm/aoi/handler.py
=====================
AOI (Area of Interest) input handler for Project RxHARM.

Accepts three input types and normalises them into a single-row GeoDataFrame
in EPSG:4326, ready for downstream indicator fetching and optimisation.

Supported input types:
    1. City name string   →  geocoded via Nominatim (free, no API key)
    2. Shapefile / GeoJSON / GPKG path  →  loaded with geopandas
    3. (lat, lon, radius_km) tuple      →  circular buffer via UTM projection

No Google Earth Engine authentication is required to use this module.
GEE geometry is produced lazily as a plain GeoJSON dict via to_ee_geometry().

Dependencies (all in requirements.txt):
    geopandas, shapely, pyproj, geopy, numpy
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import datetime
import json
import math
import os
from typing import Dict, Optional, Tuple, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping, Point

# ── Internal ──────────────────────────────────────────────────────────────────
from rxharm.config import (
    CELL_SIZE_M,
    MAX_CELLS_DIRECT,
    MAX_CELLS_MESO,
    MOORE_MAX_CELLS,
    OUTPUT_CRS,
)

# ── Module-level constants ─────────────────────────────────────────────────────
# Human-readable runtime estimates per processing mode
_RUNTIME_MAP: Dict[str, str] = {
    "moore":        "~5-10 minutes",
    "direct":       "~10-20 minutes",
    "meso":         "~25-45 minutes",
    "hierarchical": "~60+ minutes (consider selecting a smaller area)",
}


class AOIHandler:
    """
    Parses and validates user-supplied Area of Interest inputs.

    After initialisation the handler exposes all attributes needed by
    downstream modules without any network calls beyond the optional
    Nominatim geocoding step for city-name inputs.

    Parameters
    ----------
    source : str or tuple
        One of:
        - City name string, e.g. ``"Ahmedabad, India"``
        - Path to a vector file (.shp / .geojson / .gpkg)
        - ``(lat, lon, radius_km)`` tuple of floats
    year : int
        Analysis year. Must satisfy 2016 ≤ year ≤ current year.

    Attributes
    ----------
    gdf : GeoDataFrame
        Single-row GeoDataFrame in EPSG:4326 containing the AOI polygon.
    year : int
    n_cells : int
        Estimated number of 100 m × 100 m cells covering the AOI.
    mode : str
        Processing mode — ``'moore'`` | ``'direct'`` | ``'meso'`` | ``'hierarchical'``
    bounds : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326 degrees.
    centroid_ll : tuple
        ``(lat, lon)`` of the AOI centroid in decimal degrees.
    """

    def __init__(
        self,
        source: Union[str, Tuple[float, float, float]],
        year: int,
    ) -> None:
        self.source = source
        self.year = year

        # Populated by _resolve_source
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.bounds: Optional[Tuple[float, float, float, float]] = None
        self.centroid_ll: Optional[Tuple[float, float]] = None

        # Populated by _estimate_cell_count and _classify_size
        self.n_cells: Optional[int] = None
        self.mode: Optional[str] = None

        # GEE geometry cached after first to_ee_geometry() call
        self._ee_geom_cache: Optional[dict] = None

        self._resolve_source()
        self._compute_metadata()

    # ── Source resolution ─────────────────────────────────────────────────────

    def _resolve_source(self) -> None:
        """
        Dispatch to the correct parser based on source type.

        Sets self.gdf to a single-row GeoDataFrame in EPSG:4326.

        Raises
        ------
        ValueError
            If source type is unrecognised.
        FileNotFoundError
            If a file path is given but the file does not exist.
        RuntimeError
            If Nominatim returns no result for a city name.
        """
        if isinstance(self.source, (tuple, list)) and len(self.source) == 3:
            self.gdf = self._from_tuple(self.source)
        elif isinstance(self.source, str) and any(
            self.source.lower().endswith(ext)
            for ext in (".shp", ".geojson", ".gpkg", ".json")
        ):
            self.gdf = self._from_file(self.source)
        elif isinstance(self.source, str):
            self.gdf = self._from_city_name(self.source)
        else:
            raise ValueError(
                f"Unrecognised source type: {type(self.source)}. "
                "Expected a city name string, a file path (.shp/.geojson/.gpkg), "
                "or a (lat, lon, radius_km) tuple."
            )

    def _from_tuple(
        self,
        source: Tuple[float, float, float],
    ) -> gpd.GeoDataFrame:
        """
        Build a circular buffer polygon from (lat, lon, radius_km).

        Procedure:
            1. Create a GeoDataFrame with the centre point in EPSG:4326.
            2. Reproject to the local UTM zone (metric CRS) for accurate buffering.
            3. Buffer by ``radius_km × 1000`` metres.
            4. Reproject back to EPSG:4326.

        Parameters
        ----------
        source : tuple
            ``(lat, lon, radius_km)`` — all floats.

        Returns
        -------
        GeoDataFrame
            Single-row GeoDataFrame in EPSG:4326.
        """
        lat, lon, radius_km = float(source[0]), float(source[1]), float(source[2])
        if radius_km <= 0:
            raise ValueError(f"radius_km must be positive, got {radius_km}.")

        # REASON: UTM projection is needed to buffer in metres (not degrees).
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        utm_crs = (
            f"+proj=utm +zone={utm_zone} +{hemisphere} "
            "+datum=WGS84 +units=m +no_defs"
        )

        centre_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )
        centre_utm = centre_gdf.to_crs(utm_crs)
        buffered_utm = centre_utm.copy()
        buffered_utm["geometry"] = centre_utm.geometry.buffer(radius_km * 1000.0)
        result = buffered_utm.to_crs(OUTPUT_CRS).reset_index(drop=True)
        return result[["geometry"]]

    def _from_file(self, path: str) -> gpd.GeoDataFrame:
        """
        Load a vector file and dissolve to a single polygon in EPSG:4326.

        Parameters
        ----------
        path : str
            Absolute or relative path to a .shp, .geojson, or .gpkg file.

        Returns
        -------
        GeoDataFrame
            Single-row GeoDataFrame in EPSG:4326.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the given path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Vector file not found: '{path}'. "
                "Please check the path and try again."
            )
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(OUTPUT_CRS)
        if len(gdf) > 1:
            # REASON: Pipeline expects a single geometry; dissolve merges features.
            gdf = gdf.dissolve().reset_index(drop=True)
        return gdf[["geometry"]].copy()

    def _from_city_name(self, city_name: str) -> gpd.GeoDataFrame:
        """
        Geocode a city name via Nominatim and return its bounding-box polygon.

        Uses a 10-second timeout. If Nominatim returns a bounding box it is
        used directly; otherwise a 0.25° buffer around the point is used.

        Parameters
        ----------
        city_name : str
            Human-readable location string, e.g. ``"Ahmedabad, India"``.

        Returns
        -------
        GeoDataFrame
            Single-row GeoDataFrame in EPSG:4326 (bounding box polygon).

        Raises
        ------
        ImportError
            If geopy is not installed.
        RuntimeError
            If Nominatim returns no result for the given name.
        """
        try:
            from geopy.geocoders import Nominatim
        except ImportError as exc:
            raise ImportError(
                "geopy is required for city-name geocoding. "
                "Install with: pip install geopy"
            ) from exc

        geolocator = Nominatim(user_agent="rxharm_tool_v1", timeout=10)
        location = geolocator.geocode(city_name, exactly_one=True)
        if location is None:
            raise RuntimeError(
                f"Nominatim returned no result for '{city_name}'. "
                "Try a more specific query such as 'Ahmedabad, Gujarat, India', "
                "or supply coordinates directly: "
                "AOIHandler((lat, lon, radius_km), year)."
            )

        raw = location.raw
        if "boundingbox" in raw:
            # Nominatim bbox format: [south, north, west, east]
            south, north, west, east = (float(v) for v in raw["boundingbox"])
        else:
            # Fallback: 0.25° box around the geocoded point
            clat, clon = location.latitude, location.longitude
            south = clat - 0.25
            north = clat + 0.25
            west  = clon - 0.25
            east  = clon + 0.25

        polygon = box(west, south, east, north)
        return gpd.GeoDataFrame(geometry=[polygon], crs=OUTPUT_CRS)

    # ── Metadata computation ──────────────────────────────────────────────────

    def _compute_metadata(self) -> None:
        """
        Derive bounds, centroid, cell count, and mode from self.gdf.

        Called once after _resolve_source(). Sets:
            self.bounds, self.centroid_ll, self.n_cells, self.mode
        """
        geom = self.gdf.geometry.iloc[0]
        self.bounds = geom.bounds  # (minx, miny, maxx, maxy)

        centroid = geom.centroid
        self.centroid_ll = (centroid.y, centroid.x)  # (lat, lon)

        self.n_cells = self._estimate_cell_count()
        self.mode    = self._classify_size()

    def _estimate_cell_count(self) -> int:
        """
        Estimate the number of 100 m × 100 m cells covering the AOI.

        Reprojects to EPSG:3857 (metric) and divides area by CELL_SIZE_M².

        Returns
        -------
        int
            Estimated cell count (minimum 1).
        """
        gdf_metric = self.gdf.to_crs("EPSG:3857")
        area_m2 = gdf_metric.geometry.iloc[0].area
        return max(1, int(math.ceil(area_m2 / (CELL_SIZE_M ** 2))))

    def _classify_size(self) -> str:
        """
        Map estimated cell count to a processing mode string.

        Thresholds are read from config.py.

        Returns
        -------
        str
            ``'moore'`` | ``'direct'`` | ``'meso'`` | ``'hierarchical'``
        """
        n = self.n_cells
        if n <= MOORE_MAX_CELLS:
            return "moore"
        elif n <= MAX_CELLS_DIRECT:
            return "direct"
        elif n <= MAX_CELLS_MESO:
            return "meso"
        else:
            return "hierarchical"

    # ── Public interface ──────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Assert that this AOI is suitable for the RxHARM pipeline.

        Validation rules:
            1. Geometry is non-empty with positive area.
            2. Year is in the range [2016, current year].
            3. AOI area > 1 000 000 km² triggers a loud warning (not an error).

        Raises
        ------
        ValueError
            If the geometry is empty, area is zero, or year is out of range.
        """
        # ── Geometry ──────────────────────────────────────────────────────────
        if self.gdf is None or self.gdf.geometry.iloc[0].is_empty:
            raise ValueError("AOI geometry is empty. Please supply a valid location.")

        gdf_metric = self.gdf.to_crs("EPSG:3857")
        area_km2 = gdf_metric.geometry.iloc[0].area / 1e6
        if area_km2 <= 0:
            raise ValueError(
                "AOI has zero area. The supplied geometry may be a point "
                "rather than a polygon."
            )

        # ── Year ──────────────────────────────────────────────────────────────
        current_year = datetime.datetime.now().year
        if not (2016 <= self.year <= current_year):
            raise ValueError(
                f"year={self.year} is outside the supported range "
                f"[2016, {current_year}]. "
                "The full Tier-1 indicator stack requires Sentinel-2 data (2016+)."
            )

        # ── Size warning (non-fatal) ───────────────────────────────────────────
        if area_km2 > 1_000_000:
            print(
                f"\n⚠  WARNING: AOI area is {area_km2:,.0f} km². "
                "This is larger than most country-level analyses. "
                "Runtime will be very long; consider a smaller AOI.\n"
            )
        elif self.mode == "hierarchical":
            print(
                f"\n⚠  WARNING: AOI has ~{self.n_cells:,} cells "
                f"({area_km2:,.1f} km²). Hierarchical processing mode engaged. "
                f"Estimated runtime: {_RUNTIME_MAP['hierarchical']}\n"
            )

    def estimate_runtime(self) -> str:
        """
        Return a human-readable estimated runtime string for this AOI.

        Returns
        -------
        str
            E.g. ``'~25-45 minutes'``.
        """
        return _RUNTIME_MAP.get(self.mode, "runtime unknown")

    def display_summary(self) -> None:
        """
        Print a formatted AOI summary to stdout.

        Does not require GEE authentication. Safe to call at any stage.
        """
        minx, miny, maxx, maxy = self.bounds
        lat, lon = self.centroid_ll
        area_km2 = self.gdf.to_crs("EPSG:3857").geometry.iloc[0].area / 1e6

        print(
            f"\n{'─' * 52}\n"
            f"  Project RxHARM — AOI Summary\n"
            f"{'─' * 52}\n"
            f"  Source       : {self.source}\n"
            f"  Year         : {self.year}\n"
            f"  Centroid     : {lat:+.4f}°,  {lon:+.4f}°\n"
            f"  Bounds       : W={minx:.4f} S={miny:.4f} "
            f"E={maxx:.4f} N={maxy:.4f}\n"
            f"  Area         : {area_km2:,.1f} km²\n"
            f"  Est. cells   : {self.n_cells:,}  (100 m × 100 m)\n"
            f"  Mode         : {self.mode}\n"
            f"  Köppen zone  : {self.get_koppen_zone()}  "
            f"(latitude-based approximation)\n"
            f"  Est. runtime : {self.estimate_runtime()}\n"
            f"{'─' * 52}"
        )

    def to_ee_geometry(self) -> dict:
        """
        Return the AOI polygon as a GeoJSON geometry dict for GEE.

        This method does **not** require GEE to be initialised. It
        serialises the shapely geometry to a GeoJSON dict which can
        later be passed to ``ee.Geometry(aoi.to_ee_geometry())``.

        Returns
        -------
        dict
            GeoJSON geometry with ``'type'`` and ``'coordinates'`` keys.
        """
        if self._ee_geom_cache is None:
            # REASON: shapely's mapping() produces the GeoJSON geometry
            # format accepted by the GEE Python API.
            geom = self.gdf.geometry.iloc[0]
            self._ee_geom_cache = dict(mapping(geom))
        return self._ee_geom_cache

    def to_geojson(self) -> dict:
        """
        Return the AOI as a GeoJSON FeatureCollection dict.

        Returns
        -------
        dict
            GeoJSON FeatureCollection with one Feature.
        """
        return json.loads(self.gdf.to_json())

    def get_koppen_zone(self) -> str:
        """
        Return an approximate Köppen-Geiger climate zone for the AOI centroid.

        This is a latitude-only heuristic — it does not account for
        continentality, elevation, or precipitation patterns. It is
        sufficient for selecting the beta mortality coefficient from
        ``config.BETA_BY_CLIMATE_ZONE``. Override with a proper gridded
        Köppen map in Step III when GEE is available.

        Rules (applied to |centroid latitude|):
            < 25°  → ``'A'``  tropical
            25–35° → ``'B'``  arid / semi-arid (approximate)
            35–55° → ``'C'``  temperate
            55–70° → ``'D'``  continental
            > 70°  → ``'E'``  polar

        Returns
        -------
        str
            One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``.
        """
        lat_abs = abs(self.centroid_ll[0])
        if lat_abs < 25:
            return "A"
        elif lat_abs < 35:
            return "B"
        elif lat_abs < 55:
            return "C"
        elif lat_abs < 70:
            return "D"
        else:
            return "E"

    def __repr__(self) -> str:
        return (
            f"AOIHandler(source={self.source!r}, year={self.year}, "
            f"mode={self.mode!r}, n_cells={self.n_cells})"
        )
