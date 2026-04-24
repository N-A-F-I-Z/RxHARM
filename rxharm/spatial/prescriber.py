"""
rxharm/spatial/prescriber.py
=============================
Disaggregates zone-level Pareto solutions to 100m cell prescriptions
and applies spatial coherency filtering.

Pipeline for a selected solution (single x vector):
    1. Identify which zones received non-zero quantities of which intervention
    2. Within each zone, rank cells by HRI_i * feasibility_i
    3. Assign intervention to top-ranked cells until zone quantity is exhausted
    4. Apply 3×3 majority filter for spatial coherency
    5. Return 2-D array of intervention codes (0=none, 1–10=intervention)

Intervention codes mirror INTERVENTION_CODES dict (SR1=1…SR5=5, LR1=6…LR5=10).

Dependencies: numpy, scipy
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from scipy.ndimage import generic_filter

from rxharm.config import MIN_POPULATION_THRESHOLD, NON_PRESCRIBABLE_DW_LABELS

# ── Intervention code map ──────────────────────────────────────────────────────
INTERVENTION_CODES: Dict[str, int] = {
    "none": 0,
    "SR1":  1, "SR2": 2, "SR3": 3, "SR4": 4, "SR5": 5,
    "LR1":  6, "LR2": 7, "LR3": 8, "LR4": 9, "LR5": 10,
}


class Prescriber:
    """
    Translates zone-level optimizer output to cell-level prescription maps.

    Parameters
    ----------
    zone_structure : dict
        From ZoneDecomposer.decompose() — must contain ``'zone_assignments'``.
    hri_array : np.ndarray
        1-D HRI values per cell (flattened from 2-D raster).
    feasibility_masks : dict
        ``{intervention_key: bool_ndarray}`` from FeasibilityEngine.
    """

    def __init__(
        self,
        zone_structure: dict,
        hri_array: np.ndarray,
        feasibility_masks: Dict[str, np.ndarray],
    ) -> None:
        self.zones = zone_structure
        self.hri   = np.asarray(hri_array).ravel()
        self.masks = feasibility_masks

    # ── Core disaggregation ────────────────────────────────────────────────────

    def disaggregate(
        self,
        x: np.ndarray,
        intervention_names: List[str],
        mode: str = "long",
    ) -> np.ndarray:
        """
        Disaggregate zone-level decisions to cell-level prescription codes.

        Parameters
        ----------
        x : np.ndarray
            Zone × intervention decision matrix (n_zones, n_interventions).
        intervention_names : list of str
            Intervention keys in column order (e.g. ['LR1_tree_planting', …]).
        mode : str
            ``'long'`` or ``'short'``

        Returns
        -------
        np.ndarray
            1-D integer array of intervention codes, length = n_cells.
        """
        zone_assignments = np.asarray(self.zones["zone_assignments"])
        n_cells          = len(zone_assignments)
        prescription     = np.zeros(n_cells, dtype=int)

        n_zones = x.shape[0]
        for zone_id in range(n_zones):
            zone_mask = (zone_assignments == zone_id)
            zone_hri  = self.hri[zone_mask]
            zone_idxs = np.where(zone_mask)[0]

            for k, interv_name in enumerate(intervention_names):
                if k >= x.shape[1]:
                    break
                qty = x[zone_id, k]
                if qty < 1e-6:
                    continue

                # Feasibility within this zone
                feas_key = interv_name
                feas     = self.masks.get(feas_key, np.ones(n_cells, dtype=bool))
                zone_feas = feas[zone_mask].astype(float)

                # Priority = HRI * feasibility
                priority          = zone_hri * zone_feas
                n_to_prescribe    = min(int(qty), int(np.sum(zone_feas)))
                if n_to_prescribe == 0:
                    continue

                # Rank highest-priority cells first
                top_local_idxs = np.argsort(priority)[-n_to_prescribe:]
                cell_idxs      = zone_idxs[top_local_idxs]

                # Assign intervention code
                code = INTERVENTION_CODES.get(
                    interv_name.split("_")[0],
                    INTERVENTION_CODES.get(interv_name[:3], 0)
                )
                prescription[cell_idxs] = code

        return self.apply_majority_filter(prescription)

    # ── Moore neighborhood special case ───────────────────────────────────────

    def moore_neighborhood_prescription(
        self,
        primary_cells: list,
        x_cell: np.ndarray,
    ) -> dict:
        """
        Handle Moore's neighborhood AOI mode (≤9 cells).

        Parameters
        ----------
        primary_cells : list
            Indices of primary (user-selected) cells.
        x_cell : np.ndarray
            Shape (9, n_interventions).

        Returns
        -------
        dict
            ``{cell_index: {'type', 'intervention', 'quantity', 'weight'}}``
        """
        result = {}
        interv_names = list(INTERVENTION_CODES.keys())[1:]  # skip 'none'
        for i in range(min(9, len(x_cell))):
            is_primary      = i in primary_cells
            best_interv_idx = int(np.argmax(x_cell[i]))
            best_qty        = float(x_cell[i, best_interv_idx])
            interv_name     = interv_names[best_interv_idx] if best_interv_idx < len(interv_names) else "none"
            result[i] = {
                "type":         "primary" if is_primary else "neighbor",
                "intervention": interv_name,
                "quantity":     best_qty,
                "weight":       1.0 if is_primary else 0.65,
            }
        return result

    # ── Spatial coherency filter ───────────────────────────────────────────────

    @staticmethod
    def apply_majority_filter(
        prescription: np.ndarray,
        window: int = 3,
    ) -> np.ndarray:
        """
        Smooth a prescription raster with a majority filter.

        For each cell: if its prescription differs from ≥6 of 8 Moore neighbors,
        reassign it to the modal neighborhood intervention.

        Parameters
        ----------
        prescription : np.ndarray
            1-D or 2-D integer prescription codes.
        window : int
            Neighborhood size (odd number, default 3).

        Returns
        -------
        np.ndarray
            Smoothed prescription array (same shape and dtype).
        """
        flat_input = prescription.ndim == 1
        if flat_input:
            side = int(np.ceil(np.sqrt(len(prescription))))
            arr  = np.zeros(side * side, dtype=np.int32)
            arr[:len(prescription)] = prescription
            arr  = arr.reshape(side, side)
        else:
            arr = prescription.astype(np.int32)

        def majority(values: np.ndarray) -> float:
            vals   = values.astype(int)
            center = vals[len(vals) // 2]
            neighbors = np.delete(vals, len(vals) // 2)
            if np.sum(neighbors == center) >= 6:
                return float(center)
            counts = np.bincount(neighbors[neighbors >= 0], minlength=11)
            return float(np.argmax(counts))

        smoothed = generic_filter(
            arr.astype(float), majority, size=window, mode="nearest"
        ).astype(np.int32)

        if flat_input:
            return smoothed.ravel()[:len(prescription)]
        return smoothed

    # ── Output formatting ──────────────────────────────────────────────────────

    def to_prescription_map(
        self,
        prescription: np.ndarray,
        transform,
        crs: str = "EPSG:4326",
    ) -> dict:
        """
        Convert prescription array to a GeoTIFF-ready dict with statistics.

        Parameters
        ----------
        prescription : np.ndarray
            1-D or 2-D integer prescription codes.
        transform : rasterio.Affine
        crs : str

        Returns
        -------
        dict
            Keys: ``'array'``, ``'transform'``, ``'crs'``,
            ``'code_map'``, ``'statistics'``
        """
        stats = {
            name: int(np.sum(prescription == code))
            for name, code in INTERVENTION_CODES.items()
        }
        return {
            "array":      prescription,
            "transform":  transform,
            "crs":        crs,
            "code_map":   INTERVENTION_CODES,
            "statistics": stats,
        }

    def to_geodataframe(
        self,
        prescription: np.ndarray,
        transform,
        crs: str = "EPSG:4326",
        include_hri: bool = True,
    ) -> "gpd.GeoDataFrame":
        """
        Convert a prescription array to a GeoDataFrame with one row per cell.

        FIX v0.1.0: Prescription output was previously only accessible as a
        NumPy array or GeoTIFF. This method returns a GeoDataFrame that is
        directly usable in QGIS, R (via st_read of GeoPackage export),
        ArcGIS Pro, and any geopandas-compatible workflow.

        Parameters
        ----------
        prescription : np.ndarray
            2-D array of intervention codes (output of disaggregate()).
        transform : affine.Affine
            Rasterio affine transform from the loaded GeoTIFF.
        crs : str
            Coordinate reference system (default EPSG:4326).
        include_hri : bool
            If True, include the HRI value as a column for joining.

        Returns
        -------
        gpd.GeoDataFrame
            Columns: ``cell_id``, ``geometry`` (Point centroid),
            ``intervention_code``, ``intervention_name``, ``zone_id``,
            ``hri_value`` (if include_hri=True).

        Example
        -------
        gdf = prescriber.to_geodataframe(prescription, transform)
        gdf.to_file('prescription.gpkg', driver='GPKG')  # QGIS/ArcGIS
        gdf.to_csv('prescription.csv')                   # Excel/R
        """
        import geopandas as gpd
        from shapely.geometry import Point

        CODE_TO_NAME = {v: k for k, v in INTERVENTION_CODES.items()}
        flat = prescription.flatten()
        n    = len(flat)

        zone_assignments = self.zones.get("zone_assignments", np.arange(n))
        hri_flat = self.hri.flatten() if self.hri is not None else None

        rows = []
        for i, code in enumerate(flat):
            # Convert flat index to row, col
            if prescription.ndim == 2:
                row_idx = i // prescription.shape[1]
                col_idx = i  % prescription.shape[1]
            else:
                row_idx, col_idx = i, 0

            # FIX v0.1.0: use rasterio transform to compute cell centroid
            try:
                x_coord = transform.c + (col_idx + 0.5) * transform.a
                y_coord = transform.f + (row_idx + 0.5) * transform.e
            except Exception:
                x_coord, y_coord = 0.0, 0.0

            row = {
                "cell_id":            i,
                "geometry":           Point(x_coord, y_coord),
                "intervention_code":  int(code),
                "intervention_name":  CODE_TO_NAME.get(int(code), "unknown"),
                "zone_id":            int(zone_assignments[i]) if i < len(zone_assignments) else -1,
            }

            if include_hri and hri_flat is not None:
                row["hri_value"] = float(hri_flat[i]) if i < len(hri_flat) else float("nan")

            rows.append(row)

        return gpd.GeoDataFrame(rows, crs=crs)

    def save_prescription(
        self,
        prescription: np.ndarray,
        transform,
        output_dir: str,
        city_name: str = "city",
        crs: str = "EPSG:4326",
    ) -> dict:
        """
        Save prescription outputs in multiple formats.

        FIX v0.1.0: Consolidated export function that saves:
          1. GeoPackage (.gpkg) — for QGIS, ArcGIS, R sf
          2. CSV with lat/lon  — for Excel, pandas, any tool
          3. Summary statistics CSV — counts by intervention type

        Parameters
        ----------
        prescription : np.ndarray
        transform : affine.Affine
        output_dir : str
        city_name : str
        crs : str

        Returns
        -------
        dict
            ``{format_name: filepath}``
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        gdf = self.to_geodataframe(prescription, transform, crs)
        outputs = {}

        # GeoPackage (spatial — for GIS tools)
        gpkg_path = os.path.join(output_dir, f"{city_name}_prescription.gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")
        outputs["geopackage"] = gpkg_path

        # CSV with coordinates (for non-GIS tools)
        csv_df = gdf.copy()
        csv_df["longitude"] = gdf.geometry.x
        csv_df["latitude"]  = gdf.geometry.y
        csv_df = csv_df.drop(columns="geometry")
        csv_path = os.path.join(output_dir, f"{city_name}_prescription.csv")
        csv_df.to_csv(csv_path, index=False)
        outputs["csv"] = csv_path

        # Summary statistics
        summary = (
            gdf.groupby("intervention_name")
            .size()
            .reset_index(name="n_cells")
        )
        if "hri_value" in gdf.columns:
            mean_hri = gdf.groupby("intervention_name")["hri_value"].mean().reset_index()
            mean_hri.columns = ["intervention_name", "mean_hri"]
            summary = summary.merge(mean_hri, on="intervention_name", how="left")
        summary_path = os.path.join(output_dir, f"{city_name}_prescription_summary.csv")
        summary.to_csv(summary_path, index=False)
        outputs["summary"] = summary_path

        print(f"Prescription saved to {output_dir}:")
        for fmt, path in outputs.items():
            print(f"  [{fmt}] {os.path.basename(path)}")

        return outputs
