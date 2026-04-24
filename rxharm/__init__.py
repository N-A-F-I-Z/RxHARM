"""
rxharm/__init__.py
==================
Project RxHARM — Prescriptive Optimization of Heat Associated Risk Mitigation

This is the public face of the rxharm package. It exposes:
    - __version__   : semantic version string
    - __author__    : project attribution
    - run()         : top-level pipeline convenience function (Step VI)
    - load_config() : returns the config module so users can inspect defaults

A welcome message is printed once on first import (guarded by _WELCOMED flag).
"""

from __future__ import annotations

# ── Version & authorship ───────────────────────────────────────────────────────
__version__ = "0.1.0"
__author__  = "Project RxHARM Contributors"
__license__ = "MIT"

# ── Lazy import guard — welcome message prints only once per session ───────────
# REASON: Repeated imports (e.g., in notebooks that re-run cells) should not
# spam the output with the welcome banner. The flag is stored on the module
# object itself so it persists for the duration of the Python session.
_WELCOMED: bool = False


def _print_welcome() -> None:
    """Print a one-time welcome banner when the package is first imported."""
    global _WELCOMED
    if not _WELCOMED:
        print(
            f"\n"
            f"  ██████╗ ██╗  ██╗██╗  ██╗ █████╗ ██████╗ ███╗   ███╗\n"
            f"  ██╔══██╗╚██╗██╔╝██║  ██║██╔══██╗██╔══██╗████╗ ████║\n"
            f"  ██████╔╝ ╚███╔╝ ███████║███████║██████╔╝██╔████╔██║\n"
            f"  ██╔══██╗ ██╔██╗ ██╔══██║██╔══██║██╔══██╗██║╚██╔╝██║\n"
            f"  ██║  ██║██╔╝ ██╗██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║\n"
            f"  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝\n"
            f"\n"
            f"  Project RxHARM v{__version__}\n"
            f"  Prescriptive Optimization of Heat Associated Risk Mitigation\n"
            f"  100m-resolution HVI · HRI · NSGA-III Intervention Prescriber\n"
            f"  https://github.com/YOUR_USERNAME/rxharm\n"
        )
        _WELCOMED = True


_print_welcome()

# ── Expose AOIHandler at the top level for the quick-start pattern ────────────
# REASON: Allows  `rxharm.AOIHandler("Ahmedabad", 2023)` without a sub-import.
# The import is deferred inside a try/except so that the package is still
# importable even if a stub raises NotImplementedError at class-body level.
try:
    from rxharm.aoi.handler import AOIHandler  # noqa: F401
except NotImplementedError:
    pass  # stub not yet implemented — that is expected in Step I


# ── Public API ────────────────────────────────────────────────────────────────

def run(source: object, year: int = None, **kwargs) -> dict:
    """
    High-level convenience function for the RxHARM pipeline.

    Full pipeline will be implemented progressively through Steps II–VI.

    Currently available (Step II):
        - AOI parsing and mode classification
        - Spatial zone decomposition
        - Seasonal hottest-period detection (requires GEE auth when called)

    Quick-start usage::

        import rxharm
        result = rxharm.run("Ahmedabad, India", year=2023)
        # or with coordinates:
        result = rxharm.run((23.03, 72.58, 3.0), year=2023)

    Parameters
    ----------
    source : str or tuple
        City name string, file path, or (lat, lon, radius_km) tuple.
        Passed directly to AOIHandler.
    year : int, optional
        Analysis year. Defaults to the previous calendar year if omitted.
    **kwargs
        Reserved for future pipeline control arguments.

    Returns
    -------
    dict
        ``{'aoi': AOIHandler, 'zones': dict}`` — populated after Step II.
        Additional keys (indicators, HVI, HRI, prescriptions) added in later steps.
    """
    import datetime
    from rxharm.aoi.handler import AOIHandler
    from rxharm.aoi.decomposer import ZoneDecomposer

    if year is None:
        year = datetime.datetime.now().year - 1

    print(f"RxHARM v{__version__} | Source: {source} | Year: {year}")

    aoi = AOIHandler(source, year)
    aoi.validate()
    aoi.display_summary()

    decomposer = ZoneDecomposer(aoi)
    zones = decomposer.decompose()

    print(
        f"\nZone structure ready: {zones['n_zones']} zones "
        f"in '{zones['mode']}' mode."
    )
    print(
        "Next: call SeasonalDetector (requires GEE auth), "
        "then run notebooks for full HVI/HRI computation."
    )

    return {"aoi": aoi, "zones": zones}


def load_config() -> object:
    """
    Return the rxharm.config module so users can inspect all defaults.

    Useful in notebooks for quick introspection::

        cfg = rxharm.load_config()
        print(cfg.CELL_SIZE_M)          # 100
        print(cfg.HAZARD_WEIGHTS)       # {'lst': 0.5, 'albedo': 0.25, 'uhi': 0.25}
        print(cfg.NSGA3_N_GEN_LR)       # 300

    Returns
    -------
    module
        The rxharm.config module object.
    """
    import rxharm.config as _cfg
    return _cfg
