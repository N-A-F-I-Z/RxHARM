"""
rxharm/index
============
HVI and HRI computation engine for Project RxHARM.

Public API:
    NormalizerEngine  — normalise indicator arrays to [0, 1]
    WeighterEngine    — compute per-sub-index indicator weights
    HVIEngine         — compute HVI and all sub-indices
    HRIEngine         — compute HRI, attributable fraction, attributable deaths
"""

from rxharm.index.normalizer import NormalizerEngine, INDICATOR_DIRECTIONS
from rxharm.index.weighter   import WeighterEngine
from rxharm.index.hvi        import HVIEngine
from rxharm.index.hri        import HRIEngine

__all__ = [
    "NormalizerEngine", "INDICATOR_DIRECTIONS",
    "WeighterEngine",
    "HVIEngine",
    "HRIEngine",
]
