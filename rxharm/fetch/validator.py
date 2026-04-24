"""
rxharm/fetch/validator.py
==========================
Post-export band statistics validator.
Called immediately after loading a GeoTIFF from Google Drive.
Raises ValueError with actionable message if any band is suspect.

FIX v0.1.0: Addresses silent wrong-result failure when GEE collection
returns empty or zero-filled images without raising Python exceptions.
"""

from __future__ import annotations
import numpy as np

# Expected physical ranges for each indicator band.
# Values outside these ranges indicate a failed GEE fetch.
# Ranges are deliberately wide to avoid false positives.
BAND_EXPECTED_RANGES = {
    "lst":           (-20.0,  80.0),   # °C — land surface temperature
    "albedo":        (0.01,   0.95),   # fraction
    "uhi":           (-8.0,   25.0),   # °C UHI intensity
    "population":    (0.0,    1e7),    # count per 100m cell
    "built_frac":    (0.0,    1.0),    # fraction
    "elderly_frac":  (0.0,    0.7),    # fraction (>70% is physically impossible at city scale)
    "child_frac":    (0.0,    0.5),    # fraction
    "impervious":    (0.0,    1.0),    # DW probability
    "cropland":      (0.0,    1.0),    # DW probability
    "ndvi":          (-0.15,  0.95),   # NDVI (slight negatives possible over water)
    "ndwi":          (-0.70,  0.70),   # vegetation moisture NDWI
    "tree_cover":    (0.0,    100.0),  # percent
    "canopy_height": (0.0,    60.0),   # metres (>60m not urban)
    "viirs_dnb":     (0.0,    200.0),  # nW/cm²/sr
}

# Minimum fraction of non-NaN pixels required.
# A mostly-masked band (heavy cloud cover) will trigger this check.
MIN_VALID_PIXEL_FRACTION = 0.30  # at least 30% of AOI must have valid data

# Bands that are non-critical (Tier 3 static — known gaps acceptable)
_NON_CRITICAL = {"canopy_height"}


def validate_indicator_arrays(arrays: dict, aoi_name: str = "AOI") -> dict:
    """
    Validate that all 14 indicator arrays have plausible band statistics.

    Parameters
    ----------
    arrays : dict
        ``{band_name: np.ndarray}`` — raw (not normalised) indicator values
        as loaded from the exported GeoTIFF.
    aoi_name : str
        Used in error messages to identify the AOI.

    Returns
    -------
    dict
        Validation report: ``{band_name: {'status', 'issue', 'stats'}}``

    Raises
    ------
    ValueError
        If any critical band fails validation.
        Non-critical bands (``canopy_height``) produce warnings only.

    Notes
    -----
    FIX v0.1.0: This function must be called before HVIEngine.compute_all()
    to catch silent GEE failures that produce plausible-looking wrong results.
    """
    report: dict = {}
    errors: list = []
    warnings_list: list = []

    for band_name, arr in arrays.items():
        if band_name not in BAND_EXPECTED_RANGES:
            # FIX v0.1.0: skip bands not in registry (e.g. viirs_dnb_raw before downscaling)
            continue

        arr_f = np.asarray(arr, dtype=float)
        valid = arr_f[~np.isnan(arr_f)]
        total = arr_f.size

        stats = {
            "n_total":    total,
            "n_valid":    len(valid),
            "valid_frac": len(valid) / max(total, 1),
            "min":        float(valid.min())  if len(valid) > 0 else None,
            "max":        float(valid.max())  if len(valid) > 0 else None,
            "mean":       float(valid.mean()) if len(valid) > 0 else None,
        }

        expected_lo, expected_hi = BAND_EXPECTED_RANGES[band_name]
        issue  = None
        status = "ok"

        # Check 1: Mostly NaN (empty collection from GEE)
        if stats["valid_frac"] < MIN_VALID_PIXEL_FRACTION:
            issue = (
                f"Only {stats['valid_frac']*100:.1f}% valid pixels "
                f"(minimum required: {MIN_VALID_PIXEL_FRACTION*100:.0f}%). "
                "GEE collection likely returned empty or heavily cloud-masked result."
            )
            status = "fail"

        # Check 2: All zeros (collection returned but filled with zeros)
        elif len(valid) > 0 and np.all(valid == 0):
            issue = (
                "All values are exactly zero — GEE collection likely failed silently. "
                "Check collection ID and date filter in the fetch module."
            )
            status = "fail"

        # Check 3: Values outside physical range (using percentiles to avoid isolated outliers)
        elif len(valid) > 0:
            p2  = float(np.nanpercentile(arr_f, 2))
            p98 = float(np.nanpercentile(arr_f, 98))
            if p98 < expected_lo or p2 > expected_hi:
                issue = (
                    f"Value range [{p2:.3f}, {p98:.3f}] (2nd–98th pct) is entirely "
                    f"outside expected physical range [{expected_lo}, {expected_hi}]. "
                    "Check unit conversion or scale factor in fetch module."
                )
                status = "fail"

        report[band_name] = {"status": status, "issue": issue, "stats": stats}

        if status == "fail":
            msg = f"BAND '{band_name}': {issue}"
            if band_name in _NON_CRITICAL:
                warnings_list.append(f"WARNING — {msg}")
            else:
                errors.append(f"ERROR   — {msg}")

    # Print warnings (non-critical)
    for w in warnings_list:
        print(w)

    # Raise on critical errors
    if errors:
        error_text = "\n".join(errors)
        raise ValueError(
            f"\n{'='*70}\n"
            f"RxHARM VALIDATION FAILED for {aoi_name}\n"
            f"{'='*70}\n"
            f"{error_text}\n"
            f"{'='*70}\n"
            "COMMON CAUSES:\n"
            "  1. GEE export did not complete — check Drive for the file\n"
            "  2. Wrong file loaded — ensure you loaded the correct timestamped file\n"
            "  3. GEE collection ID changed — check rxharm/data/indicator_registry.json\n"
            "  4. Year out of range for this collection — check temporal coverage\n"
            "  5. AOI is in a region with persistent cloud cover during hottest months\n"
            f"{'='*70}\n"
            "Run rxharm.fetch.validator.print_validation_report(arrays) for full stats."
        )

    return report


def print_validation_report(arrays: dict) -> None:
    """Print a human-readable validation report without raising on failure."""
    try:
        report = validate_indicator_arrays(arrays)
    except ValueError as e:
        print(str(e))
        return

    print(f"\n{'='*60}")
    print("RxHARM BAND VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"{'Band':<20} {'Status':<8} {'Valid%':<8} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print(f"{'-'*60}")
    for band, result in report.items():
        s = result["stats"]
        status_str = "✓ OK  " if result["status"] == "ok" else "✗ FAIL"
        if s["min"] is not None:
            print(
                f"{band:<20} {status_str:<8} {s['valid_frac']*100:>5.1f}%  "
                f"{s['min']:>10.3f} {s['max']:>10.3f} {s['mean']:>10.3f}"
            )
        else:
            print(f"{band:<20} {status_str:<8} {'0.0%':>6}  {'ALL NaN':>10}")
    print(f"{'='*60}\n")
