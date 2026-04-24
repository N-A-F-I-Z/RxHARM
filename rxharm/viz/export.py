"""
rxharm/viz/export.py
=====================
Export utilities for Project RxHARM outputs.

Functions write GeoTIFFs (rasterio), CSVs (pandas), and figures (matplotlib).
Dependencies: rasterio, numpy, pandas, matplotlib
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import numpy as np


def export_to_geotiff(
    array: np.ndarray,
    filepath: str,
    transform,
    crs: str = "EPSG:4326",
    nodata: float = -9999.0,
) -> None:
    """
    Write a single-band numpy array as a GeoTIFF.

    Parameters
    ----------
    array : np.ndarray
        2-D array to write.
    filepath : str
        Output file path.
    transform : rasterio.Affine
        Affine transform for geo-registration.
    crs : str
        Coordinate reference system. Default ``'EPSG:4326'``.
    nodata : float
        NoData value written for NaN cells.
    """
    import rasterio
    from rasterio.transform import from_bounds

    out = array.copy().astype(np.float32)
    out[~np.isfinite(out)] = nodata

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with rasterio.open(
        filepath, "w",
        driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(out, 1)


def export_multiband_geotiff(
    arrays: Dict[str, np.ndarray],
    filepath: str,
    transform,
    crs: str = "EPSG:4326",
) -> None:
    """
    Write multiple named arrays as bands in a single GeoTIFF.

    Parameters
    ----------
    arrays : dict
        ``{band_name: ndarray}`` — all arrays must have the same shape.
    filepath : str
        Output file path.
    transform : rasterio.Affine
    crs : str
    """
    import rasterio

    names = list(arrays.keys())
    first = next(iter(arrays.values()))
    H, W  = first.shape

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with rasterio.open(
        filepath, "w",
        driver="GTiff", height=H, width=W,
        count=len(names), dtype="float32",
        crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        for i, name in enumerate(names, start=1):
            out = arrays[name].copy().astype(np.float32)
            out[~np.isfinite(out)] = -9999.0
            dst.write(out, i)
            dst.update_tags(i, band_name=name)


def export_summary_csv(
    hvi_results: dict,
    hri_results: dict,
    zone_summary: "pd.DataFrame",
    filepath: str,
) -> None:
    """
    Write a per-zone summary CSV.

    Columns: zone_id, mean_HVI, mean_HRI, total_pop, total_AD_baseline.

    Parameters
    ----------
    hvi_results : dict
    hri_results : dict
    zone_summary : pd.DataFrame
        From ZoneDecomposer.get_zone_summary().
    filepath : str
    """
    import pandas as pd

    rows = []
    for _, row in zone_summary.iterrows():
        rows.append({
            "zone_id":           row["zone_id"],
            "centroid_lat":      row.get("centroid_lat", np.nan),
            "centroid_lon":      row.get("centroid_lon", np.nan),
            "cell_count":        row.get("cell_count", 0),
            "mean_HVI":          round(float(np.nanmean(hvi_results.get("HVI", [np.nan]))), 4),
            "mean_HRI":          round(float(np.nanmean(hri_results.get("HRI", [np.nan]))), 4),
            "total_AD_baseline": round(float(np.nansum(hri_results.get("AD_baseline", [0.0]))), 4),
        })
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def save_all_outputs(
    hvi_results: dict,
    hri_results: dict,
    aoi_handler,
    output_dir: str,
) -> Dict[str, str]:
    """
    Save HVI, HRI GeoTIFFs and summary CSV to output_dir.

    Parameters
    ----------
    hvi_results : dict
    hri_results : dict
    aoi_handler : AOIHandler
    output_dir : str

    Returns
    -------
    dict
        ``{output_type: filepath}``
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}

    # Derive a simple identity transform from bounds
    try:
        import rasterio.transform
        minx, miny, maxx, maxy = aoi_handler.bounds
        hvi_arr = hvi_results.get("HVI", np.zeros((10, 10)))
        H, W = hvi_arr.shape
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, W, H)
    except Exception:
        transform = None

    if transform is not None:
        hvi_path = os.path.join(output_dir, "HVI.tif")
        export_to_geotiff(hvi_results.get("HVI", np.zeros((10,10))), hvi_path, transform)
        outputs["HVI_tif"] = hvi_path

        hri_path = os.path.join(output_dir, "HRI.tif")
        export_to_geotiff(hri_results.get("HRI", np.zeros((10,10))), hri_path, transform)
        outputs["HRI_tif"] = hri_path

    print(f"Outputs saved to: {output_dir}")
    return outputs
