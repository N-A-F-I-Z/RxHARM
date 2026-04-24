"""
rxharm/aoi/decomposer.py
========================
Spatial zone decomposition for Project RxHARM.
Modes: moore | direct | meso | hierarchical.
Grid generation uses vectorised NumPy (no Python loops).
Dependencies: geopandas, shapely, numpy, scipy, scikit-learn
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import math
from typing import Any, Dict, List, Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

# ── Internal ──────────────────────────────────────────────────────────────────
from rxharm.config import (
    CELL_SIZE_M,
    MAX_CELLS_DIRECT,
    MAX_CELLS_MESO,
    MOORE_MAX_CELLS,
    MOORE_WEIGHT_DIAGONAL,
    MOORE_WEIGHT_FACE,
    MOORE_WEIGHT_PRIMARY,
    NON_PRESCRIBABLE_DW_LABELS,
    MIN_POPULATION_THRESHOLD,
    OUTPUT_CRS,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Metres per degree of latitude (constant). Used with cosine correction for lon.
_M_PER_DEG = 111_320.0


class ZoneDecomposer:
    """
    Decomposes an AOI into optimisable zone structures.

    Parameters
    ----------
    aoi_handler : AOIHandler
        A validated AOIHandler instance (``validate()`` already called).

    Attributes
    ----------
    aoi : AOIHandler
    mode : str
        Mirrors ``aoi_handler.mode``.
    _cell_grid : GeoDataFrame or None
        Cached result of ``get_cell_grid()``.
    _zone_result : dict or None
        Cached result of ``decompose()``.
    """

    def __init__(self, aoi_handler: "AOIHandler") -> None:
        self.aoi  = aoi_handler
        self.mode = aoi_handler.mode
        self._cell_grid: Optional[gpd.GeoDataFrame] = None
        self._zone_result: Optional[Dict[str, Any]] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def decompose(
        self,
        n_meso_zones: int = 15,
        n_macro_zones: int = 30,
    ) -> Dict[str, Any]:
        """
        Run the mode-appropriate decomposition and return the zone structure.

        Parameters
        ----------
        n_meso_zones : int
            Target number of zones for meso and hierarchical-inner modes.
        n_macro_zones : int
            Target number of macro-zones for hierarchical mode.

        Returns
        -------
        dict
            Keys: ``'mode'``, ``'n_zones'``, ``'cells'``, ``'zones'``,
            ``'zone_assignments'``
        """
        if self._zone_result is not None:
            return self._zone_result

        if self.mode == "moore":
            result = self._decompose_moore()
        elif self.mode == "direct":
            result = self._decompose_direct()
        elif self.mode == "meso":
            result = self._decompose_meso(n_meso_zones)
        else:
            result = self._decompose_hierarchical(n_macro_zones, n_meso_zones)

        self._zone_result = result
        return result

    # ── Moore neighborhood ────────────────────────────────────────────────────

    def _decompose_moore(self) -> Dict[str, Any]:
        """
        Build a Moore's-neighborhood structure (≤ 9 cells with weights).

        The user's AOI centroid becomes the primary cell. All 8 adjacent
        100 m cells (N, NE, E, SE, S, SW, W, NW) are generated. Face-
        adjacent cells (N/S/E/W) get weight 0.65; diagonal cells get 0.45.

        Returns
        -------
        dict
            Moore-mode zone structure with per-cell weight and type fields.
        """
        lat, lon = self.aoi.centroid_ll

        # Degree offsets for one 100 m step
        dlat = CELL_SIZE_M / _M_PER_DEG
        dlon = CELL_SIZE_M / (_M_PER_DEG * math.cos(math.radians(lat)))

        # Build 3×3 grid of (row_offset, col_offset) → (dlat_mult, dlon_mult)
        offsets = [
            (0,  0,  "primary",  MOORE_WEIGHT_PRIMARY),   # centre
            (1,  0,  "neighbor", MOORE_WEIGHT_FACE),       # N
            (-1, 0,  "neighbor", MOORE_WEIGHT_FACE),       # S
            (0,  1,  "neighbor", MOORE_WEIGHT_FACE),       # E
            (0,  -1, "neighbor", MOORE_WEIGHT_FACE),       # W
            (1,  1,  "neighbor", MOORE_WEIGHT_DIAGONAL),   # NE
            (1,  -1, "neighbor", MOORE_WEIGHT_DIAGONAL),   # NW
            (-1, 1,  "neighbor", MOORE_WEIGHT_DIAGONAL),   # SE
            (-1, -1, "neighbor", MOORE_WEIGHT_DIAGONAL),   # SW
        ]

        aoi_geom = self.aoi.gdf.geometry.iloc[0]
        cells = []
        for row_off, col_off, cell_type, weight in offsets:
            clat = lat + row_off * dlat
            clon = lon + col_off * dlon
            pt   = Point(clon, clat)
            cells.append({
                "cell_id":       len(cells),
                "geometry":      pt,
                "type":          cell_type,
                "weight":        weight,
                "zone_id":       len(cells),
                "prescribable":  True,
            })

        # Build cell GeoDataFrame and cache it
        self._cell_grid = gpd.GeoDataFrame(cells, crs=OUTPUT_CRS)

        return {
            "mode":             "moore",
            "n_zones":          len(cells),
            "cells":            cells,
            "zones":            [{"zone_id": c["cell_id"]} for c in cells],
            "zone_assignments": list(range(len(cells))),
        }

    # ── Direct mode (flat grid) ───────────────────────────────────────────────

    def _decompose_direct(self) -> Dict[str, Any]:
        """
        Generate all 100 m cell centroids within the AOI polygon.

        Uses vectorised NumPy meshgrid operations (no Python loops).

        Returns
        -------
        dict
            Direct-mode zone structure; each cell is its own zone.
        """
        grid = self._build_cell_grid()
        n = len(grid)

        return {
            "mode":             "direct",
            "n_zones":          n,
            "cells":            grid.to_dict(orient="records"),
            "zones":            [{"zone_id": i} for i in range(n)],
            "zone_assignments": list(range(n)),
        }

    # ── Meso mode (KMeans clustering) ─────────────────────────────────────────

    def _decompose_meso(self, n_zones: int) -> Dict[str, Any]:
        """
        Cluster 100 m cells into spatially coherent meso-zones via KMeans.

        Parameters
        ----------
        n_zones : int
            Target number of meso-zones.

        Returns
        -------
        dict
            Meso-mode zone structure.
        """
        from sklearn.cluster import KMeans

        grid = self._build_cell_grid()
        n_zones = min(n_zones, len(grid))

        coords = np.column_stack([
            grid.geometry.x.values,
            grid.geometry.y.values,
        ])
        # REASON: Normalise coordinates so KMeans weights lat and lon equally.
        coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)

        km = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        labels = km.fit_predict(coords_norm)
        grid = grid.copy()
        grid["zone_id"] = labels

        return {
            "mode":             "meso",
            "n_zones":          n_zones,
            "cells":            grid.to_dict(orient="records"),
            "zones":            [{"zone_id": z} for z in range(n_zones)],
            "zone_assignments": labels.tolist(),
        }

    # ── Hierarchical mode (two-level) ─────────────────────────────────────────

    def _decompose_hierarchical(
        self,
        n_macro: int,
        n_meso_per_macro: int,
    ) -> Dict[str, Any]:
        """
        Two-level hierarchical decomposition for very large AOIs.

        Level 1: Aggregate 100 m cells to 1 km macro-cells, cluster into
                 n_macro macro-zones.
        Level 2: Within each macro-zone, cluster 100 m cells into
                 n_meso_per_macro meso-zones.

        Parameters
        ----------
        n_macro : int
            Number of level-1 macro-zones (default 30).
        n_meso_per_macro : int
            Number of level-2 meso-zones per macro-zone (default 15).

        Returns
        -------
        dict
            Hierarchical zone structure.
        """
        from sklearn.cluster import KMeans

        grid = self._build_cell_grid()
        coords = np.column_stack([grid.geometry.x.values, grid.geometry.y.values])
        coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)

        # Level 1: macro clustering
        n_macro = min(n_macro, len(grid))
        km_macro = KMeans(n_clusters=n_macro, random_state=42, n_init=10)
        macro_labels = km_macro.fit_predict(coords_norm)

        # Level 2: meso clustering within each macro zone
        meso_labels = np.zeros(len(grid), dtype=int)
        global_zone_id = 0
        for macro_id in range(n_macro):
            mask = macro_labels == macro_id
            sub_coords = coords_norm[mask]
            n_sub = min(n_meso_per_macro, len(sub_coords))
            if n_sub <= 1:
                meso_labels[mask] = global_zone_id
                global_zone_id += 1
                continue
            km_meso = KMeans(n_clusters=n_sub, random_state=42, n_init=5)
            sub_labels = km_meso.fit_predict(sub_coords)
            meso_labels[mask] = sub_labels + global_zone_id
            global_zone_id += n_sub

        grid = grid.copy()
        grid["macro_zone_id"] = macro_labels
        grid["zone_id"]       = meso_labels
        total_zones = int(meso_labels.max()) + 1

        return {
            "mode":             "hierarchical",
            "n_zones":          total_zones,
            "n_macro_zones":    n_macro,
            "cells":            grid.to_dict(orient="records"),
            "zones":            [{"zone_id": z} for z in range(total_zones)],
            "zone_assignments": meso_labels.tolist(),
        }

    # ── Cell grid generation (vectorised) ─────────────────────────────────────

    def _build_cell_grid(self) -> gpd.GeoDataFrame:
        """
        Generate a regular 100 m grid of cell centroids clipped to the AOI.

        Uses vectorised NumPy meshgrid operations. Latitude correction is
        applied to the longitude step so cells are approximately square.

        Returns
        -------
        GeoDataFrame
            Columns: ``cell_id``, ``geometry`` (Point centroids), ``zone_id``,
            ``prescribable``.
        """
        if self._cell_grid is not None:
            return self._cell_grid

        minx, miny, maxx, maxy = self.aoi.bounds
        lat_mid = (miny + maxy) / 2.0

        # Degree step sizes — longitude step uses cosine correction for latitude
        step_lat = CELL_SIZE_M / _M_PER_DEG
        step_lon = CELL_SIZE_M / (_M_PER_DEG * math.cos(math.radians(lat_mid)))

        # Build coordinate arrays (vectorised)
        lons = np.arange(minx + step_lon / 2, maxx, step_lon)
        lats = np.arange(miny + step_lat / 2, maxy, step_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        all_lons = lon_grid.ravel()
        all_lats = lat_grid.ravel()

        # Clip to AOI polygon using geopandas spatial join
        points_gdf = gpd.GeoDataFrame(
            {"geometry": gpd.points_from_xy(all_lons, all_lats)},
            crs=OUTPUT_CRS,
        )
        aoi_poly_gdf = self.aoi.gdf[["geometry"]].copy()

        # REASON: sjoin is vectorised and much faster than iterating points.
        clipped = gpd.sjoin(
            points_gdf, aoi_poly_gdf, how="inner", predicate="within"
        ).reset_index(drop=True)
        clipped = clipped[["geometry"]].copy()
        clipped["cell_id"]     = np.arange(len(clipped))
        clipped["zone_id"]     = np.arange(len(clipped))  # overwritten later
        clipped["prescribable"] = True

        self._cell_grid = clipped
        return self._cell_grid

    # ── Public interface ──────────────────────────────────────────────────────

    def get_cell_grid(self) -> gpd.GeoDataFrame:
        """
        Return the GeoDataFrame of all 100 m cell centroids within the AOI.

        Triggers grid generation if not already done.

        Returns
        -------
        GeoDataFrame
            Columns: ``cell_id``, ``geometry``, ``zone_id``, ``prescribable``.
            CRS: EPSG:4326.
        """
        if self._cell_grid is None:
            if self.mode == "moore":
                self.decompose()           # Moore sets self._cell_grid
            else:
                self._build_cell_grid()
        return self._cell_grid.copy()

    def filter_non_prescribable(
        self,
        dw_label_array: np.ndarray,
        population_array: np.ndarray,
    ) -> None:
        """
        Update the ``prescribable`` flag based on Dynamic World labels and population.

        Called after GEE indicator data is fetched in Step III. Mutates
        the internal ``_cell_grid`` in-place.

        Parameters
        ----------
        dw_label_array : np.ndarray
            1-D integer array of Dynamic World modal class labels,
            one value per cell in ``get_cell_grid()`` order.
            Labels 0 (water), 3 (flooded), 8 (snow/ice) are non-prescribable
            as defined in ``config.NON_PRESCRIBABLE_DW_LABELS``.
        population_array : np.ndarray
            1-D float array of WorldPop population per cell.
            Cells with population < MIN_POPULATION_THRESHOLD are non-prescribable.

        Raises
        ------
        ValueError
            If array lengths do not match the number of cells in the grid.
        """
        grid = self.get_cell_grid()
        n = len(grid)

        if len(dw_label_array) != n or len(population_array) != n:
            raise ValueError(
                f"dw_label_array length ({len(dw_label_array)}) and "
                f"population_array length ({len(population_array)}) "
                f"must both equal the number of cells ({n})."
            )

        # Build prescribability mask
        dw_mask  = ~np.isin(dw_label_array, list(NON_PRESCRIBABLE_DW_LABELS))
        pop_mask = population_array >= MIN_POPULATION_THRESHOLD

        # FIX v0.1.0: Nodata sentinel detection.
        # When the AOI boundary extends beyond the GeoTIFF extent (prevented
        # by the export buffer fix in fetch/__init__.py, but defensive check
        # here too), rasterio fills cells with -9999 or similar sentinel values.
        # These cells must be marked non-prescribable or they appear as highly
        # vulnerable (large negative population) and corrupt the optimizer.
        _NODATA_SENTINELS = {-9999, -9998, -32768, 65535}
        nodata_mask = (
            np.isin(dw_label_array, list(_NODATA_SENTINELS))
            | np.isin(population_array.astype(int), list(_NODATA_SENTINELS))
            | (population_array < -1)   # any strongly negative population is nodata
        )

        prescribable = dw_mask & pop_mask & ~nodata_mask

        # FIX v0.1.0: Log how many nodata cells were found for user transparency
        n_nodata = int(np.sum(nodata_mask))
        if n_nodata > 0:
            print(
                f"  NOTE: {n_nodata} cells marked non-prescribable due to nodata values "
                f"(likely from export boundary). This is expected for edge cells."
            )

        self._cell_grid = self._cell_grid.copy()
        self._cell_grid["prescribable"] = prescribable

        # Also update the cached zone_result cells list if it exists
        if self._zone_result is not None:
            for i, cell in enumerate(self._zone_result["cells"]):
                cell["prescribable"] = bool(prescribable[i])

    def get_zone_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising each zone.

        Triggers ``decompose()`` if not already called.

        Returns
        -------
        DataFrame
            Columns: ``zone_id``, ``cell_count``, ``centroid_lat``,
            ``centroid_lon``.
        """
        result = self.decompose()
        grid   = self.get_cell_grid()

        rows = []
        for zone_id in sorted(grid["zone_id"].unique()):
            zone_cells = grid[grid["zone_id"] == zone_id]
            centroid_lon = zone_cells.geometry.x.mean()
            centroid_lat = zone_cells.geometry.y.mean()
            rows.append({
                "zone_id":      int(zone_id),
                "cell_count":   len(zone_cells),
                "centroid_lat": round(float(centroid_lat), 6),
                "centroid_lon": round(float(centroid_lon), 6),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def apply_majority_filter(
        prescription_array: np.ndarray,
        window: int = 3,
    ) -> np.ndarray:
        """
        Smooth a 2-D prescription raster using a majority (modal) filter.

        For each cell, if its assigned intervention code differs from the
        majority of its Moore neighbors, reassign it to the most common
        neighbor code. This removes spatially isolated "salt-and-pepper"
        assignments that would be impractical to implement.

        Parameters
        ----------
        prescription_array : np.ndarray
            2-D integer array where each value is an intervention type code
            (0 = no intervention, 1..10 = intervention IDs).
            Shape: (n_rows, n_cols).
        window : int
            Neighbourhood window size (must be odd). Default 3 (3×3 = 9 cells).

        Returns
        -------
        np.ndarray
            Smoothed prescription array, same shape and dtype as input.

        Notes
        -----
        FIX v0.1.0: Added boundary padding to prevent edge artifacts.
        Cells at the array boundary have fewer than 8 neighbors; padding with
        ``mode='edge'`` replicates border values rather than introducing zeros,
        which is the correct behavior for a spatial prescription map.
        Previously, boundary cells could be incorrectly reassigned to 'no
        intervention' (code 0) because zero-filled neighbors dominated the vote.
        """
        from scipy.ndimage import generic_filter

        if window % 2 == 0:
            raise ValueError(f"window must be odd, got {window}.")

        arr = prescription_array.astype(np.int32)

        # FIX v0.1.0: Pad by 1 cell using edge-replication before filtering,
        # then crop back. This ensures boundary cells always have 8 valid
        # neighbors drawn from real data rather than zero fill.
        pad_width = window // 2
        padded = np.pad(arr.astype(float), pad_width=pad_width, mode="edge")

        def _majority(values: np.ndarray) -> float:
            """Return the modal value in a flat neighborhood array."""
            vals, counts = np.unique(values.astype(np.int32), return_counts=True)
            return float(vals[np.argmax(counts)])

        smoothed_padded = generic_filter(
            padded,
            _majority,
            size=window,
            mode="nearest",
        )

        # FIX v0.1.0: Crop back to original size after padding
        if arr.ndim == 2:
            smoothed = smoothed_padded[pad_width:-pad_width, pad_width:-pad_width]
        else:
            smoothed = smoothed_padded[pad_width:-pad_width]

        return smoothed.astype(np.int32)
