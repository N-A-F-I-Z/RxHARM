"""
rxharm/uncertainty/monte_carlo.py
===================================
Monte Carlo uncertainty propagation for HVI indicator measurements.

Sources of indicator uncertainty (literature values):
    LST:          RMSE ~0.4°C  (Landsat C2 thermal accuracy)
    NDVI:         σ ~0.03       (atmospheric correction residuals)
    WorldPop:     CV ~15%       (dasymetric model uncertainty)
    GHS-BUILT:    σ ~0.05       (classification accuracy ~87%)
    VIIRS_DNB:    CV ~10%       (sensor noise + downscaling)
    tree_cover:   σ ~5.0        (GFW loss detection accuracy)
    canopy_h:     RMSE ~6.6m   (Potapov 2021 validation)

For each indicator, uncertainty is additive Gaussian (zero mean, σ above)
or multiplicative lognormal (for count-based: population, VIIRS).

Dependencies: numpy, scipy
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm

# Per-indicator uncertainty specifications
INDICATOR_UNCERTAINTIES: Dict[str, dict] = {
    "lst":           {"type": "additive_gaussian",       "std":  0.40},
    "albedo":        {"type": "additive_gaussian",       "std":  0.02},
    "uhi":           {"type": "additive_gaussian",       "std":  0.50},
    "population":    {"type": "multiplicative_lognormal","cv":   0.15},
    "built_frac":    {"type": "additive_gaussian",       "std":  0.05},
    "elderly_frac":  {"type": "additive_gaussian",       "std":  0.02},
    "child_frac":    {"type": "additive_gaussian",       "std":  0.01},
    "impervious":    {"type": "additive_gaussian",       "std":  0.05},
    "cropland":      {"type": "additive_gaussian",       "std":  0.04},
    "ndvi":          {"type": "additive_gaussian",       "std":  0.03},
    "ndwi":          {"type": "additive_gaussian",       "std":  0.03},
    "tree_cover":    {"type": "additive_gaussian",       "std":  5.00},
    "canopy_height": {"type": "additive_gaussian",       "std":  6.60},
    "viirs_dnb":     {"type": "multiplicative_lognormal","cv":   0.10},
}


class MCUncertaintyEngine:
    """
    Propagates indicator measurement uncertainty through the HVI pipeline.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo realisations.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(self, n_samples: int = 500, random_seed: int = 42) -> None:
        self.n   = n_samples
        self.rng = np.random.default_rng(random_seed)

    # ── Perturbation ───────────────────────────────────────────────────────────

    def perturb_indicators(
        self,
        indicator_arrays: Dict[str, np.ndarray],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate n_samples perturbed copies of all indicator arrays.

        Parameters
        ----------
        indicator_arrays : dict
            ``{indicator_name: ndarray}`` — normalised or raw, any shape.

        Returns
        -------
        list of dict
            Length n_samples; each dict has same structure as input.
        """
        samples = []
        for _ in range(self.n):
            sample: Dict[str, np.ndarray] = {}
            for name, arr in indicator_arrays.items():
                unc = INDICATOR_UNCERTAINTIES.get(name)
                if unc is None:
                    sample[name] = arr.copy()
                elif unc["type"] == "additive_gaussian":
                    noise = self.rng.normal(0, unc["std"], arr.shape)
                    sample[name] = arr + noise
                elif unc["type"] == "multiplicative_lognormal":
                    sigma  = float(np.sqrt(np.log(1 + unc["cv"] ** 2)))
                    factor = self.rng.lognormal(0.0, sigma, arr.shape)
                    sample[name] = arr * factor
                else:
                    sample[name] = arr.copy()
            samples.append(sample)
        return samples

    # ── HVI distribution ───────────────────────────────────────────────────────

    def compute_hvi_distribution(
        self,
        indicator_arrays: Dict[str, np.ndarray],
        hvi_engine,
    ) -> dict:
        """
        Run HVI for all MC samples and return percentile bounds.

        Parameters
        ----------
        indicator_arrays : dict
            Raw indicator arrays.
        hvi_engine : HVIEngine

        Returns
        -------
        dict
            Keys: ``'p10'``, ``'p50'``, ``'p90'``, ``'mean'``, ``'std'``,
            ``'confidence_width'`` (p90 - p10)
        """
        perturbed = self.perturb_indicators(indicator_arrays)
        hvi_stack = []
        for sample in perturbed:
            try:
                result = hvi_engine.compute_all(sample)
                hvi_stack.append(result["HVI"])
            except Exception:
                continue

        if not hvi_stack:
            arr = np.zeros_like(next(iter(indicator_arrays.values())))
            return {k: arr.copy() for k in ("p10", "p50", "p90", "mean", "std",
                                             "confidence_width")}

        stack = np.stack(hvi_stack, axis=0)
        p10   = np.nanpercentile(stack, 10, axis=0)
        p50   = np.nanpercentile(stack, 50, axis=0)
        p90   = np.nanpercentile(stack, 90, axis=0)
        return {
            "p10":              p10,
            "p50":              p50,
            "p90":              p90,
            "mean":             np.nanmean(stack, axis=0),
            "std":              np.nanstd(stack, axis=0),
            "confidence_width": p90 - p10,
        }
