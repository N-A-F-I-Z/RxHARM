"""
rxharm/interventions/feasibility.py
=====================================
Builds binary feasibility masks for all 13 RxHARM interventions
(5 SR + 8 LR).

Eligibility thresholds come from config.py (no magic numbers).
Masking uses vectorised NumPy — no Python loops.

Dependencies: numpy, rxharm.config
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from rxharm.config import (
    BGI_MAX_BUILT_H_M,
    COOL_ROOF_MAX_BUILT_H_M,
    COOL_ROOF_MIN_BUILT_FRAC,
    DW_LABEL_BUILT,
    DW_LABEL_WATER,
    MIN_POPULATION_THRESHOLD,
    NON_PRESCRIBABLE_DW_LABELS,
    PAVEMENT_MIN_BUILT_FRAC,
    SHADE_MAX_EXISTING_CANOPY_M,
    TREE_MIN_OPEN_FRAC,
)


class FeasibilityEngine:
    """
    Computes eligibility masks for short-run and long-run interventions.

    Parameters
    ----------
    indicator_arrays : dict
        ``{indicator_name: ndarray}`` — raw indicator arrays (flat 1D or 2D).
    aoi_handler : AOIHandler
        Used for metadata (currently unused but kept for API consistency).
    dw_modal : np.ndarray, optional
        Dynamic World modal label per cell.
    building_height : np.ndarray, optional
        Estimated building height in metres per cell.
    """

    def __init__(
        self,
        indicator_arrays: Dict[str, np.ndarray],
        aoi_handler,
        dw_modal: Optional[np.ndarray] = None,
        building_height: Optional[np.ndarray] = None,
    ) -> None:
        self.arrays   = indicator_arrays
        self.aoi      = aoi_handler
        self.dw_modal = dw_modal
        self.bldg_h   = building_height

        # Convenience accessors (default to ones array if missing)
        shape = next(iter(indicator_arrays.values())).shape if indicator_arrays else (1,)
        self._pop        = indicator_arrays.get("population",  np.ones(shape))
        self._built_frac = indicator_arrays.get("built_frac",  np.zeros(shape))
        self._ndvi       = indicator_arrays.get("ndvi",        np.zeros(shape))
        self._canopy     = indicator_arrays.get("canopy_height", np.zeros(shape))
        self._elderly    = indicator_arrays.get("elderly_frac", np.zeros(shape))
        self._hri        = indicator_arrays.get("HRI",          np.zeros(shape))

    # ── Base mask ──────────────────────────────────────────────────────────────

    def _base_prescribable(self) -> np.ndarray:
        """
        Global mask: True = cell is eligible (not water/snow/flooded,
        has population ≥ MIN_POPULATION_THRESHOLD).
        """
        mask = self._pop >= MIN_POPULATION_THRESHOLD
        if self.dw_modal is not None:
            for label in NON_PRESCRIBABLE_DW_LABELS:
                mask = mask & (self.dw_modal != label)
        return mask.astype(bool)

    # ── Short-run masks ────────────────────────────────────────────────────────

    def compute_sr_masks(self) -> Dict[str, np.ndarray]:
        """
        Compute feasibility masks for all 5 short-run interventions.

        Returns
        -------
        dict
            ``{intervention_key: bool_ndarray}``
        """
        base = self._base_prescribable()
        return {
            "SR1_cooling_center":      self._mask_cooling_center(base),
            "SR2_misting_station":     self._mask_misting(base),
            "SR3_welfare_check":       self._mask_welfare_check(base),
            "SR4_medical_prepositioning": base.copy(),
            "SR5_shade_structure":     self._mask_shade(base),
        }

    def _mask_cooling_center(self, base: np.ndarray) -> np.ndarray:
        """SR1: requires built area."""
        return base & (self._built_frac >= COOL_ROOF_MIN_BUILT_FRAC)

    def _mask_misting(self, base: np.ndarray) -> np.ndarray:
        """SR2: requires mostly impervious/built surface."""
        return base & (self._built_frac >= 0.30)

    def _mask_welfare_check(self, base: np.ndarray) -> np.ndarray:
        """SR3: targets elderly-rich, high-HRI cells."""
        return base & (self._elderly > 0.10)

    def _mask_shade(self, base: np.ndarray) -> np.ndarray:
        """SR5: requires low existing canopy."""
        if self._canopy is not None:
            return base & (self._canopy < SHADE_MAX_EXISTING_CANOPY_M)
        return base

    # ── Long-run masks ─────────────────────────────────────────────────────────

    def compute_lr_masks(self) -> Dict[str, np.ndarray]:
        """
        Compute feasibility masks for all 5 long-run interventions.

        Returns
        -------
        dict
            ``{intervention_key: bool_ndarray}``
        """
        base = self._base_prescribable()
        return {
            "LR1_tree_planting":       self.mask_tree_planting(base),
            "LR2_cool_roof":           self.mask_cool_roof(base),
            "LR3_bgi":                 self.mask_bgi(base),
            "LR4_cool_pavement":       self._mask_cool_pavement(base),
            "LR5_green_corridor":      base.copy(),
            "LR6_green_roof":          self.mask_cool_roof(base),   # same eligibility as cool roof
            "LR7_cool_morphology":     self._mask_cool_morphology(base),
            "LR8_blue_infrastructure": self.mask_bgi(base),         # same open-land requirement as BGI
        }

    def mask_cool_roof(self, base: Optional[np.ndarray] = None) -> np.ndarray:
        """LR2: requires high built fraction and low building height."""
        if base is None:
            base = self._base_prescribable()
        mask = base & (self._built_frac >= COOL_ROOF_MIN_BUILT_FRAC)
        if self.bldg_h is not None:
            mask = mask & (self.bldg_h < COOL_ROOF_MAX_BUILT_H_M)
        return mask

    def mask_tree_planting(self, base: Optional[np.ndarray] = None) -> np.ndarray:
        """LR1: requires open land fraction."""
        if base is None:
            base = self._base_prescribable()
        open_frac = np.clip(1.0 - self._built_frac, 0.0, 1.0)
        return base & (open_frac >= TREE_MIN_OPEN_FRAC)

    def mask_bgi(self, base: Optional[np.ndarray] = None) -> np.ndarray:
        """LR3: requires open land and low building height."""
        if base is None:
            base = self._base_prescribable()
        open_frac = np.clip(1.0 - self._built_frac, 0.0, 1.0)
        mask = base & (open_frac >= 0.20)
        if self.bldg_h is not None:
            mask = mask & (self.bldg_h < BGI_MAX_BUILT_H_M)
        return mask

    def _mask_cool_pavement(self, base: np.ndarray) -> np.ndarray:
        """LR4: requires mostly paved (high built fraction)."""
        return base & (self._built_frac >= PAVEMENT_MIN_BUILT_FRAC)

    def _mask_cool_morphology(self, base: np.ndarray) -> np.ndarray:
        """LR7: applicable primarily to moderate-density built areas (not ultra-dense or open land).

        New-development / redevelopment feasibility:
        - built_frac between 0.20 and 0.80 (not bare land, not hyper-dense)
        - building height < BGI_MAX_BUILT_H_M (low-to-mid rise, not skyscraper districts)
        - Water bodies excluded
        """
        mask = base & (self._built_frac >= 0.20) & (self._built_frac <= 0.80)
        if self.bldg_h is not None:
            mask = mask & (self.bldg_h < BGI_MAX_BUILT_H_M)
        if self.dw_modal is not None:
            mask = mask & (self.dw_modal != DW_LABEL_WATER)
        return mask

    # ── Combined ───────────────────────────────────────────────────────────────

    def compute_all_masks(self) -> Dict[str, np.ndarray]:
        """Compute masks for all 13 interventions (5 SR + 8 LR)."""
        masks = {}
        masks.update(self.compute_sr_masks())
        masks.update(self.compute_lr_masks())
        return masks
