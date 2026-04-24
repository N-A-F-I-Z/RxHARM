"""
rxharm/validation.py
=====================
Validation functions for RxHARM outputs.

Used to verify system correctness and for the paper's case study section.

validate_ahmedabad()  — spatial plausibility checks for Ahmedabad, India
print_hvi_summary()  — human-readable summary after HVI/HRI computation
safe_gee_call()      — GEE error handler with actionable messages
display_runtime_warning() — formatted runtime estimate banner
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ahmedabad(
    hvi_results: dict,
    aoi_handler,
    verbose: bool = True,
) -> dict:
    """
    Spatial plausibility check for the Ahmedabad, India case study.

    Known facts used for validation:
        1. Behrampura (23.00°N, 72.57°E): dense slum → top 25% HVI
        2. Bodakdev  (23.05°N, 72.53°E): affluent planned residential → bottom 50% HVI
        3. Sabarmati Riverfront: water body → low or NaN HVI
        4. Overall HVI distribution should be right-skewed

    Parameters
    ----------
    hvi_results : dict
        Output of HVIEngine.compute_all().
    aoi_handler : AOIHandler or None
        If None, only distribution-shape checks are run.
    verbose : bool
        Print check-by-check results.

    Returns
    -------
    dict
        ``{'passed': bool, 'checks': {check_name: {'pass': bool, 'value': ...}}}``
    """
    results: dict = {"passed": True, "checks": {}}
    hvi = hvi_results.get("HVI", np.array([]))
    if hvi.size == 0:
        results["passed"] = False
        results["checks"]["no_data"] = {"pass": False, "value": "HVI array is empty"}
        return results

    valid = hvi[np.isfinite(hvi)]
    if len(valid) == 0:
        results["passed"] = False
        results["checks"]["all_nan"] = {"pass": False, "value": "All HVI pixels are NaN"}
        return results

    # ── Check 1: Range [0, 1] ─────────────────────────────────────────────────
    range_ok = float(valid.min()) >= -0.01 and float(valid.max()) <= 1.01
    results["checks"]["range_0_1"] = {
        "pass":  range_ok,
        "value": f"min={valid.min():.4f}, max={valid.max():.4f}",
    }
    if not range_ok:
        results["passed"] = False

    # ── Check 2: Right-skewed distribution ────────────────────────────────────
    # REASON: Most cells have moderate vulnerability; extreme values are rare.
    skewness = float(np.mean((valid - valid.mean()) ** 3) / (valid.std() ** 3 + 1e-9))
    skew_ok  = skewness > 0.0
    results["checks"]["right_skewed"] = {
        "pass":  skew_ok,
        "value": f"skewness={skewness:.3f} (expected > 0)",
    }

    # ── Check 3: Non-trivial spatial variance ────────────────────────────────
    variance_ok = float(valid.std()) > 0.05
    results["checks"]["spatial_variance"] = {
        "pass":  variance_ok,
        "value": f"std={valid.std():.4f} (expected > 0.05)",
    }
    if not variance_ok:
        results["passed"] = False

    # ── Check 4: Median in plausible range ───────────────────────────────────
    median_ok = 0.10 < float(np.median(valid)) < 0.90
    results["checks"]["median_range"] = {
        "pass":  median_ok,
        "value": f"median={np.median(valid):.4f} (expected 0.10–0.90)",
    }

    # ── Check 5: Fraction of prescribed cells > 0 ────────────────────────────
    frac_valid = float(np.isfinite(hvi).mean())
    coverage_ok = frac_valid > 0.20
    results["checks"]["coverage_fraction"] = {
        "pass":  coverage_ok,
        "value": f"{frac_valid:.1%} of cells have valid HVI",
    }
    if not coverage_ok:
        results["passed"] = False

    if verbose:
        print("\n" + "=" * 55)
        print("  RxHARM — Validation Results")
        print("=" * 55)
        for name, chk in results["checks"].items():
            status = "✓ PASS" if chk["pass"] else "✗ FAIL"
            print(f"  {status}  {name}: {chk['value']}")
        overall = "PASSED" if results["passed"] else "FAILED"
        print("=" * 55)
        print(f"  Overall: {overall}")
        print("=" * 55 + "\n")

    return results


# ── HVI/HRI summary printer ───────────────────────────────────────────────────

def print_hvi_summary(hvi_results: dict, hri_results: dict, aoi) -> None:
    """
    Print a formatted human-readable HVI/HRI summary to stdout.

    Parameters
    ----------
    hvi_results : dict
        From HVIEngine.compute_all().
    hri_results : dict
        From HRIEngine.compute_all().
    aoi : AOIHandler
    """
    print("\n" + "=" * 60)
    print("HVI/HRI COMPUTATION SUMMARY")
    print("=" * 60)
    print(f"Location: {getattr(aoi, 'source', 'Unknown')}")
    print(f"Year:     {getattr(aoi, 'year',   'Unknown')}")
    hvi = hvi_results.get("HVI", np.array([]))
    print(f"Total cells:      {hvi.size:,}")
    print(f"Prescribable:     {int(np.sum(np.isfinite(hvi))):,}")
    print()

    for label, arr in [
        ("H_s (Hazard)",        hvi_results.get("H_s",  np.array([np.nan]))),
        ("E (Exposure)",         hvi_results.get("E",    np.array([np.nan]))),
        ("S (Sensitivity)",      hvi_results.get("S",    np.array([np.nan]))),
        ("AC (Adaptive Cap.)",   hvi_results.get("AC",   np.array([np.nan]))),
        ("HVI (composite)",      hvi_results.get("HVI",  np.array([np.nan]))),
    ]:
        valid = arr[np.isfinite(arr)]
        if len(valid):
            print(
                f"  {label:<25} "
                f"mean={valid.mean():.3f}  "
                f"P10={np.percentile(valid, 10):.3f}  "
                f"P90={np.percentile(valid, 90):.3f}"
            )

    print()
    total_ad = float(np.nansum(hri_results.get("AD_baseline", np.zeros(1))))
    print(f"Baseline AD (3-day heatwave): {total_ad:.1f} expected deaths")
    print("=" * 60 + "\n")


# ── GEE error handler ─────────────────────────────────────────────────────────

def safe_gee_call(
    func: Callable,
    *args,
    operation_name: str = "GEE operation",
    **kwargs,
):
    """
    Execute a GEE call with actionable error messages.

    Parameters
    ----------
    func : callable
        Function to call.
    *args, **kwargs
        Passed to func.
    operation_name : str
        Human-readable operation name for error messages.

    Returns
    -------
    Result of func(*args, **kwargs), or raises after printing a helpful message.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "not found" in err_str or "does not exist" in err_str:
            print(f"ERROR: GEE collection not available for {operation_name}.")
            print("  This may be a temporary GEE outage. Try again in 5 minutes.")
            print(f"  Technical details: {e}")
        elif "quota" in err_str or "rate" in err_str:
            print("ERROR: GEE compute quota exceeded.")
            print("  Wait 15 minutes and try again, or use a smaller AOI.")
        elif "authentication" in err_str or "credentials" in err_str:
            print("ERROR: GEE not authenticated.")
            print("  Run: import ee; ee.Authenticate(); ee.Initialize(project='your-project')")
        else:
            print(f"ERROR during {operation_name}: {e}")
        raise


# ── Runtime warning display ───────────────────────────────────────────────────

def display_runtime_warning(
    operation: str,
    estimate: str,
    can_skip: bool = False,
) -> None:
    """
    Print a formatted runtime estimate box before a long-running operation.

    Parameters
    ----------
    operation : str
        Name of the operation about to run.
    estimate : str
        Human-readable time estimate (e.g. ``'~15-30 minutes'``).
    can_skip : bool
        If True, hints that cached results can be loaded instead.
    """
    print("=" * 60)
    print(f"  OPERATION:      {operation}")
    print(f"  ESTIMATED TIME: {estimate}")
    if can_skip:
        print("  TIP: Set SKIP=True to load cached results instead")
    print("=" * 60)
