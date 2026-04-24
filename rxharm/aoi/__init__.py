"""
rxharm/aoi
==========
AOI input handling and spatial decomposition for Project RxHARM.

Public API:
    AOIHandler     — parse and validate user location input (city name,
                     shapefile path, or lat/lon/radius tuple)
    ZoneDecomposer — produce zone structure for the NSGA-III optimizer
"""

from rxharm.aoi.handler import AOIHandler
from rxharm.aoi.decomposer import ZoneDecomposer

__all__ = ["AOIHandler", "ZoneDecomposer"]
