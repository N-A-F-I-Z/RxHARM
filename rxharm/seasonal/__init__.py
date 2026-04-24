"""
rxharm/seasonal
===============
Automatic detection of the hottest analysis period for any global location.

Public API:
    SeasonalDetector — ERA5-based hottest month identification
"""

from rxharm.seasonal.detector import SeasonalDetector

__all__ = ["SeasonalDetector"]
