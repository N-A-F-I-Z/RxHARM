"""
tests/test_decomposer.py
========================
Tests for rxharm/aoi/decomposer.py.

Verifies that ZoneDecomposer correctly partitions large AOIs into
spatially contiguous meso-zones of appropriate sizes.

These tests will FAIL until Step II is implemented.
"""

from __future__ import annotations

import pytest


class TestZoneDecomposer:
    """Tests for ZoneDecomposer spatial decomposition."""

    def test_decompose_returns_correct_number_of_zones(self):
        """Number of returned zones must equal n_zones parameter."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_all_cells_are_assigned_to_a_zone(self):
        """zone_assignments must cover every cell (no unassigned cells)."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_zone_assignments_are_contiguous(self):
        """Each zone must be spatially contiguous (no isolated cells)."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_auto_n_zones_minimum_three(self):
        """_auto_n_zones must return at least 3 zones."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_small_aoi_below_meso_threshold_not_decomposed(self):
        """AOIs with n_cells <= MAX_CELLS_DIRECT must not trigger decomposition."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_zone_gdfs_have_correct_crs(self):
        """All zone GeoDataFrames must be in OUTPUT_CRS (EPSG:4326)."""
        raise NotImplementedError(
            "Implemented in Step II."
        )

    def test_zone_sizes_are_balanced(self):
        """No single zone should contain more than 3x the average cells per zone."""
        raise NotImplementedError(
            "Implemented in Step II."
        )
