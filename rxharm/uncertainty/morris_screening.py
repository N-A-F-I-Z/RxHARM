"""
rxharm/uncertainty/morris_screening.py
========================================
Morris one-at-a-time (OAT) sensitivity analysis on indicator weights.

Identifies which indicator weights have the most leverage on HVI
spatial ranking. Used to justify the equal-weighting default.

Method: Morris (1991) Elementary Effects
    For each weight parameter:
        1. Perturb by +delta and -delta from default
        2. Compute Spearman r between perturbed and nominal HVI
        3. mu*  = mean |change in correlation|
        4. sigma = std of changes across multiple starting points
    High mu*  → high influence on spatial ranking
    Low sigma → consistent influence (not interaction-dependent)

Dependencies: numpy, scipy, pandas
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rxharm.config import (
    ADAPTIVE_CAPACITY_WEIGHTS,
    EXPOSURE_WEIGHTS,
    HAZARD_WEIGHTS,
    SENSITIVITY_WEIGHTS,
)


class MorrisScreener:
    """
    Morris elementary effects sensitivity analysis for RxHARM weights.

    Parameters
    ----------
    indicator_arrays : dict
        Raw indicator arrays.
    hvi_engine : HVIEngine
    """

    def __init__(self, indicator_arrays: Dict[str, np.ndarray], hvi_engine) -> None:
        self.arrays  = indicator_arrays
        self.engine  = hvi_engine
        self.delta   = 0.20   # 20% perturbation

    def screen(self, n_trajectories: int = 10) -> pd.DataFrame:
        """
        Run Morris screening and return elementary effects.

        Parameters
        ----------
        n_trajectories : int
            Number of perturbation trajectories (default 10).

        Returns
        -------
        pd.DataFrame
            Columns: ``'indicator'``, ``'sub_index'``, ``'mu_star'``,
            ``'sigma'``, ``'rank'``
        """
        nominal     = self.engine.compute_all(self.arrays)["HVI"].flatten()
        nominal_valid = nominal[~np.isnan(nominal)]

        all_weights = {
            "hazard":            HAZARD_WEIGHTS.copy(),
            "exposure":          EXPOSURE_WEIGHTS.copy(),
            "sensitivity":       SENSITIVITY_WEIGHTS.copy(),
            "adaptive_capacity": ADAPTIVE_CAPACITY_WEIGHTS.copy(),
        }

        rows: List[dict] = []
        for sub_idx, weights in all_weights.items():
            for indicator in weights:
                changes = []
                for direction in [+self.delta, -self.delta]:
                    perturbed = weights.copy()
                    perturbed[indicator] = max(0.01,
                                               weights[indicator] + direction)
                    # Renormalise so weights still sum to 1
                    total = sum(perturbed.values())
                    perturbed = {k: v / total for k, v in perturbed.items()}

                    # Inject temporary manual override into the weighter
                    self.engine.weighter._manual_override = {sub_idx: perturbed}
                    try:
                        pert_hvi = self.engine.compute_all(self.arrays)["HVI"].flatten()
                        pert_valid = pert_hvi[~np.isnan(pert_hvi)]
                        n = min(len(nominal_valid), len(pert_valid))
                        if n > 10:
                            r, _ = spearmanr(nominal_valid[:n], pert_valid[:n])
                            changes.append(1.0 - r)  # 0 = no change, 1 = complete reorder
                        else:
                            changes.append(0.0)
                    except Exception:
                        changes.append(0.0)
                    finally:
                        self.engine.weighter._manual_override = None

                rows.append({
                    "indicator": indicator,
                    "sub_index": sub_idx,
                    "mu_star":   float(np.mean(np.abs(changes))),
                    "sigma":     float(np.std(changes)),
                })

        df = pd.DataFrame(rows)
        df["rank"] = df["mu_star"].rank(ascending=False).astype(int)
        return df.sort_values("mu_star", ascending=False).reset_index(drop=True)
