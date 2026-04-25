"""
rxharm/fetch
============
Google Earth Engine data fetching for all 14 HVI indicators.

Modules:
    hazard            - LST, albedo, UHI intensity      (Hazard sub-index)
    exposure          - Population, built fraction       (Exposure sub-index)
    sensitivity       - Age fractions, impervious, cropland (Sensitivity)
    adaptive_capacity - NDVI, NDWI, tree cover, canopy height, VIIRS raw (AC)
    viirs_downscaler  - RFATPK 500m → 100m VIIRS downscaling (no GEE)

Public API:
    HazardFetcher
    ExposureFetcher
    SensitivityFetcher
    AdaptiveCapacityFetcher
    VIIRSDownscaler
    fetch_all_indicators     ← convenience function for the full pipeline
"""

from rxharm.fetch.hazard import HazardFetcher
from rxharm.fetch.exposure import ExposureFetcher
from rxharm.fetch.sensitivity import SensitivityFetcher
from rxharm.fetch.adaptive_capacity import AdaptiveCapacityFetcher
from rxharm.fetch.viirs_downscaler import VIIRSDownscaler

# FIX v0.1.0: Import validator so it is accessible via rxharm.fetch.validator
from rxharm.fetch.validator import validate_indicator_arrays, print_validation_report

__all__ = [
    "HazardFetcher",
    "ExposureFetcher",
    "SensitivityFetcher",
    "AdaptiveCapacityFetcher",
    "VIIRSDownscaler",
    "fetch_all_indicators",
    "validate_indicator_arrays",    # FIX v0.1.0: added
    "print_validation_report",      # FIX v0.1.0: added
    "load_existing_export",         # FIX v0.1.0: resume mode
    "merge_worldpop_local",         # FIX 0.1.1: WorldPop local merge
]


def fetch_all_indicators(
    aoi_handler: object,
    seasonal_detector: object,
    export_to_drive: bool = True,
    drive_folder: str = "RxHARM_outputs",
    show_progress: bool = True,
) -> dict:
    """
    Convenience function: fetch all 14 indicators and optionally export to Drive.

    FIX 0.1.1:
    - [1d] Replaced simple print() progress messages with a task-tracker
      that shows elapsed time per step and a ✓/✗ status for each indicator.
    - [6a] DW label mosaic is fetched ONCE and shared with both
      SensitivityFetcher (impervious/cropland) and HazardFetcher (UHI
      rural background), eliminating a duplicate GEE fetch.
    - [3b] WorldPop local arrays are merged into the GeoTIFF post-export.
    - Adds 'sensitivity_fetcher' to return dict so WorldPop local data
      is accessible downstream.

    Parameters
    ----------
    aoi_handler : AOIHandler
    seasonal_detector : SeasonalDetector
        ``detect()`` must have been called first.
    export_to_drive : bool
    drive_folder : str
    show_progress : bool

    Returns
    -------
    dict
        Keys: 'ee_image', 'export_task', 'band_names', 'fetch_metadata',
              'sensitivity_fetcher' (may contain _worldpop_local arrays)
    """
    import ee
    import datetime
    import time

    from rxharm.config import GEE_SCALE, OUTPUT_CRS, CELL_SIZE_M

    ee_geom = ee.Geometry(aoi_handler.to_ee_geometry())
    year    = aoi_handler.year
    months  = seasonal_detector._detected_months

    # FIX 0.1.1 [1d]: Task-level progress tracker
    _step_times = {}
    def _start(label: str) -> float:
        t = time.time()
        if show_progress:
            print(f"  ⏳ {label}...", end=" ", flush=True)
        return t

    def _done(t0: float, label: str = "") -> None:
        elapsed = time.time() - t0
        _step_times[label] = elapsed
        if show_progress:
            print(f"✓  ({elapsed:.1f}s)")

    def _fail(t0: float, msg: str = "") -> None:
        elapsed = time.time() - t0
        if show_progress:
            print(f"✗  ({elapsed:.1f}s) {msg}")

    # ── Step 1: Hazard ────────────────────────────────────────────────────────
    t = _start("[1/5] Hazard (LST, Albedo, UHI)")
    try:
        hazard     = HazardFetcher(ee_geom, year, months)
        lst_img    = hazard.get_lst()
        albedo_img = hazard.get_albedo()
        _done(t, "hazard_lst_albedo")
    except Exception as e:
        _fail(t, str(e))
        raise

    # ── Step 2: Exposure ─────────────────────────────────────────────────────
    t = _start("[2/5] Exposure (Population, Built fraction)")
    try:
        exposure     = ExposureFetcher(ee_geom, year)
        exposure_img = exposure.fetch_all()
        _done(t, "exposure")
    except Exception as e:
        _fail(t, str(e))
        raise

    # ── Step 3: Dynamic World (fetched ONCE — shared for UHI + Sensitivity) ──
    # FIX 0.1.1 [6a]: Fetch DW label mosaic here, pass to both downstream uses
    t = _start("[3/5] Sensitivity + Dynamic World (shared)")
    try:
        sensitivity     = SensitivityFetcher(ee_geom, year, months)
        sensitivity_img = sensitivity.fetch_all()

        # Build DW built mask from the already-fetched sensitivity result
        # SensitivityFetcher exposes _dw_label_mosaic after fetch_all()
        dw_built_mask = getattr(sensitivity, "_dw_label_mosaic", None)
        _done(t, "sensitivity")
    except Exception as e:
        _fail(t, str(e))
        raise

    # ── Step 4: UHI (uses shared DW mask) ────────────────────────────────────
    t = _start("[4/5] UHI (reusing DW mask)")
    try:
        # FIX 0.1.1 [6a]: Pass dw_built_mask to avoid second DW fetch
        uhi_img    = hazard.get_uhi(lst_img, dw_built_mask=dw_built_mask)
        hazard_img = ee.Image.cat([lst_img, albedo_img, uhi_img])
        _done(t, "uhi")
    except Exception as e:
        _fail(t, str(e))
        raise

    # ── Step 5: Adaptive Capacity ─────────────────────────────────────────────
    t = _start("[5/5] Adaptive Capacity (NDVI, NDWI, Trees, Canopy, VIIRS)")
    try:
        ac     = AdaptiveCapacityFetcher(ee_geom, year, months)
        ac_img = ac.fetch_all()
        _done(t, "adaptive_capacity")
    except Exception as e:
        _fail(t, str(e))
        raise

    # ── Merge all 14 bands ────────────────────────────────────────────────────
    combined = ee.Image.cat([hazard_img, exposure_img, sensitivity_img, ac_img]).toFloat()

    band_names = [
        "lst", "albedo", "uhi",
        "population", "built_frac",
        "elderly_frac", "child_frac",
        "impervious", "cropland",
        "ndvi", "ndwi",
        "tree_cover", "canopy_height", "viirs_dnb_raw",
    ]

    # ── Export to Drive (async) ───────────────────────────────────────────────
    export_task = None
    if export_to_drive:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename  = f"RxHARM_indicators_{year}_{timestamp}"

        buffer_m = CELL_SIZE_M * 5
        try:
            export_region = ee_geom.buffer(buffer_m).bounds()
        except Exception:
            export_region = ee_geom.bounds()

        export_task = ee.batch.Export.image.toDrive(
            image=combined,
            description=filename,
            folder=drive_folder,
            fileNamePrefix=filename,
            region=export_region,
            scale=GEE_SCALE,
            crs=OUTPUT_CRS,
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        export_task.start()

        if show_progress:
            total_t = sum(_step_times.values())
            print(f"\n  ✅ All indicators fetched in {total_t:.0f}s")
            print(f"  📤 Export started: '{filename}.tif' → Drive/{drive_folder}/")
            print("  🔗 Monitor: https://code.earthengine.google.com/tasks")
            # FIX 0.1.1 [3b]: Warn if WorldPop local data is being used
            if getattr(sensitivity, "_worldpop_local", None) is not None:
                print(
                    "\n  ⚠  WorldPop Global2 data was downloaded locally.\n"
                    "     The GeoTIFF bands 6–7 (elderly_frac, child_frac) will be\n"
                    "     PLACEHOLDER values (-1). After the export completes, call:\n"
                    "     data = load_existing_export(tiff_path)\n"
                    "     data = merge_worldpop_local(data, sensitivity_fetcher)\n"
                    "     before running HVIEngine."
                )

    metadata = {
        "year":              year,
        "hottest_months":    months,
        "aoi_centroid":      aoi_handler.centroid_ll,
        "n_cells_estimated": aoi_handler.n_cells,
        "timestamp":         datetime.datetime.now().isoformat(),
        "band_names":        band_names,
        "step_times_s":      _step_times,
    }

    return {
        "ee_image":            combined,
        "export_task":         export_task,
        "band_names":          band_names,
        "fetch_metadata":      metadata,
        "sensitivity_fetcher": sensitivity,   # FIX 0.1.1: exposes _worldpop_local
    }


def load_existing_export(
    tiff_path: str,
    band_names: list = None,
    run_validation: bool = True,
) -> dict:
    """
    Load a previously exported RxHARM GeoTIFF from Google Drive.

    FIX v0.1.0: Resume mode — allows users to load an existing GeoTIFF
    instead of re-running GEE when they want to explore different weighting
    methods, intervention budgets, or scenario parameters without the
    30-minute GEE wait. This is the single most impactful usability
    improvement for iterative analysis workflows.

    Parameters
    ----------
    tiff_path : str
        Path to the GeoTIFF file. Accepts glob patterns — will use the
        most recently modified matching file.
    band_names : list of str, optional
        If None, uses the standard 14-band order from fetch_all_indicators().
    run_validation : bool
        If True (recommended), runs validate_indicator_arrays() after loading.

    Returns
    -------
    dict
        ``'arrays'``   : {band_name: np.ndarray} for all bands
        ``'transform'``: rasterio affine transform
        ``'crs'``      : coordinate reference system string
        ``'meta'``     : rasterio metadata dict
        ``'tiff_path'``: resolved file path that was loaded

    Example
    -------
    # Instead of re-running GEE fetch (30 min):
    data = load_existing_export('/content/drive/MyDrive/RxHARM_outputs/RxHARM_indicators_2023*.tif')

    # Then compute HVI with different weights:
    from rxharm.index.hvi import HVIEngine
    results = HVIEngine(weighting_method='pca').compute_all(data['arrays'])
    """
    import rasterio
    import glob
    import os
    import numpy as np
    from rxharm.fetch.validator import validate_indicator_arrays

    # FIX v0.1.0: Resolve glob pattern to actual file (most recently modified)
    if "*" in tiff_path or "?" in tiff_path:
        matches = sorted(glob.glob(tiff_path), key=os.path.getmtime)
        if not matches:
            raise FileNotFoundError(
                f"No files matching '{tiff_path}'. "
                "Check your Google Drive mount and folder path."
            )
        resolved = matches[-1]
        if len(matches) > 1:
            print(
                f"  Found {len(matches)} matching files — loading most recent: "
                f"{os.path.basename(resolved)}"
            )
    else:
        resolved = tiff_path
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"File not found: {resolved}. "
                "Check your Google Drive mount at /content/drive/MyDrive/"
            )

    # Standard 14-band order (must match order in fetch_all_indicators)
    DEFAULT_BAND_NAMES = [
        "lst", "albedo", "uhi",
        "population", "built_frac",
        "elderly_frac", "child_frac", "impervious", "cropland",
        "ndvi", "ndwi", "tree_cover", "canopy_height", "viirs_dnb_raw",
    ]
    names = band_names or DEFAULT_BAND_NAMES

    with rasterio.open(resolved) as src:
        n_bands = src.count
        if n_bands != len(names):
            raise ValueError(
                f"GeoTIFF has {n_bands} bands but {len(names)} band names provided. "
                f"A valid RxHARM export should have {len(DEFAULT_BAND_NAMES)} bands."
            )

        arrays = {}
        for i, name in enumerate(names):
            band_data = src.read(i + 1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                # FIX v0.1.0: Convert fill value to NaN so downstream code
                # handles missing data correctly rather than treating -9999 as data.
                band_data[band_data == nodata] = np.nan
            arrays[name] = band_data

        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        meta = src.meta.copy()

    # FIX v0.1.0: Inform user if raw VIIRS band is present
    if "viirs_dnb_raw" in arrays and "viirs_dnb" not in arrays:
        print(
            "  NOTE: 'viirs_dnb_raw' loaded. If already downscaled in a previous run, "
            "rename: arrays['viirs_dnb'] = arrays.pop('viirs_dnb_raw')"
        )

    print(f"  Loaded: {os.path.basename(resolved)}")
    print(f"  Shape:  {list(arrays.values())[0].shape}")
    print(f"  Bands:  {len(arrays)}")

    # FIX v0.1.0: Run validation immediately so the user knows about problems
    # before spending time computing HVI on corrupted data.
    if run_validation:
        validation_arrays = {
            ("viirs_dnb" if k == "viirs_dnb_raw" else k): v
            for k, v in arrays.items()
        }
        print("  Running band validation...")
        try:
            validate_indicator_arrays(validation_arrays, aoi_name=os.path.basename(resolved))
            print("  Validation: PASSED ✓")
        except ValueError as e:
            print(f"  Validation: FAILED\n{e}")
            raise

    return {
        "arrays":    arrays,
        "transform": transform,
        "crs":       crs,
        "meta":      meta,
        "tiff_path": resolved,
    }


def merge_worldpop_local(data: dict, sensitivity_fetcher: object) -> dict:
    """
    FIX 0.1.1: Merge WorldPop Global2 locally-downloaded arrays into a
    load_existing_export() result dict, replacing the GEE placeholder
    bands (-1 values) with the real age-fraction data.

    Call this AFTER load_existing_export() when WorldPop was downloaded locally:

        data = load_existing_export('/content/drive/MyDrive/RxHARM_outputs/*.tif')
        data = merge_worldpop_local(data, sensitivity_fetcher)

    Parameters
    ----------
    data : dict
        Output of load_existing_export().
    sensitivity_fetcher : SensitivityFetcher
        The fetcher instance from fetch_all_indicators()['sensitivity_fetcher'].
        Must have a _worldpop_local attribute (not None).

    Returns
    -------
    dict
        Updated data dict with 'elderly_frac', 'child_frac', 'population'
        replaced by the locally downloaded WorldPop Global2 arrays.
    """
    import numpy as np

    wp = getattr(sensitivity_fetcher, "_worldpop_local", None)
    if wp is None:
        # No local WorldPop data — return unchanged
        return data

    arrays = dict(data["arrays"])  # shallow copy

    # Resize WorldPop arrays to match the GeoTIFF shape if needed
    target_shape = list(arrays.values())[0].shape

    def _resize_to(arr: np.ndarray, shape: tuple) -> np.ndarray:
        if arr.shape == shape:
            return arr
        from PIL import Image as _PIL
        img   = _PIL.fromarray(arr.astype(np.float32))
        img_r = img.resize((shape[1], shape[0]), _PIL.BILINEAR)
        return np.array(img_r).astype(float)

    if wp.get("population") is not None:
        arrays["population"]   = _resize_to(wp["population"],   target_shape)
    if wp.get("elderly_frac") is not None:
        arrays["elderly_frac"] = _resize_to(wp["elderly_frac"], target_shape)
    if wp.get("child_frac") is not None:
        arrays["child_frac"]   = _resize_to(wp["child_frac"],   target_shape)

    print(f"  WorldPop merged: source = {wp.get('source', 'local')}")
    return {**data, "arrays": arrays}
