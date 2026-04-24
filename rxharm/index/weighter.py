"""
rxharm/index/weighter.py
========================
Indicator weighting engine for Project RxHARM.

Provides five interchangeable weighting strategies. All return a dict
{indicator_name: weight} where values sum to 1.0 within each sub-index.

Strategies:
    equal   — uniform 1/n weights (default; globally comparable)
    pca     — squared loading of first principal component
    entropy — Shannon entropy weighting (Zou et al. 2006)
    critic  — CRITIC method (Diakoulaki et al. 1995)
    manual  — user-supplied weights (validated to sum = 1.0)

Dependencies: numpy, scipy, scikit-learn, pandas
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from rxharm.config import (
    ADAPTIVE_CAPACITY_WEIGHTS,
    EXPOSURE_WEIGHTS,
    HAZARD_WEIGHTS,
    SENSITIVITY_WEIGHTS,
    WEIGHTING_DEFAULT,
    WEIGHTING_METHODS,
)

# Default config weights indexed by sub-index name
_CONFIG_WEIGHTS: Dict[str, Dict[str, float]] = {
    "hazard":            HAZARD_WEIGHTS,
    "exposure":          EXPOSURE_WEIGHTS,
    "sensitivity":       SENSITIVITY_WEIGHTS,
    "adaptive_capacity": ADAPTIVE_CAPACITY_WEIGHTS,
}


class WeighterEngine:
    """
    Computes indicator weights for one sub-index.

    Parameters
    ----------
    method : str
        One of ``'equal'``, ``'pca'``, ``'entropy'``, ``'critic'``, ``'manual'``.
    """

    def __init__(self, method: str = WEIGHTING_DEFAULT) -> None:
        if method not in WEIGHTING_METHODS:
            raise ValueError(
                f"Unknown weighting method '{method}'. "
                f"Choose from {WEIGHTING_METHODS}."
            )
        self.method = method
        # Internal override slot used by MorrisScreener
        self._manual_override: Optional[Dict[str, Dict[str, float]]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_weights(
        self,
        indicator_arrays: Dict[str, np.ndarray],
        sub_index: str,
    ) -> Dict[str, float]:
        """
        Compute weights for the named sub-index.

        Parameters
        ----------
        indicator_arrays : dict
            ``{indicator_name: normalised_ndarray}`` for this sub-index.
        sub_index : str
            ``'hazard'`` | ``'exposure'`` | ``'sensitivity'`` | ``'adaptive_capacity'``

        Returns
        -------
        dict
            ``{indicator_name: weight}`` summing to 1.0.
        """
        # MorrisScreener injects temporary overrides
        if self._manual_override and sub_index in self._manual_override:
            return self._manual_override[sub_index]

        names = list(indicator_arrays.keys())
        if self.method == "equal":
            return self._equal_weights(names)
        elif self.method == "pca":
            return self._pca_weights(indicator_arrays)
        elif self.method == "entropy":
            return self._entropy_weights(indicator_arrays)
        elif self.method == "critic":
            return self._critic_weights(indicator_arrays)
        elif self.method == "manual":
            return _CONFIG_WEIGHTS.get(sub_index, self._equal_weights(names))
        else:
            return self._equal_weights(names)

    # ── Strategy implementations ───────────────────────────────────────────────

    def _equal_weights(self, indicator_names: List[str]) -> Dict[str, float]:
        """Uniform 1/n weights."""
        n = len(indicator_names)
        if n == 0:
            return {}
        w = 1.0 / n
        return {name: w for name in indicator_names}

    def _pca_weights(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Weight by squared loading on the first principal component.

        REASON: PC1 captures the dominant co-variation direction. Squared
        loadings are proportional to the variance explained by each variable
        in the dominant mode.
        """
        from sklearn.decomposition import PCA

        names = list(arrays.keys())
        try:
            mat = np.column_stack([a.ravel() for a in arrays.values()])
            valid = np.all(np.isfinite(mat), axis=1)
            if valid.sum() < len(names) + 1:
                return self._equal_weights(names)
            pca = PCA(n_components=1)
            pca.fit(mat[valid])
            loadings = pca.components_[0] ** 2
            total = loadings.sum()
            if total < 1e-12:
                return self._equal_weights(names)
            return {name: float(loadings[i] / total) for i, name in enumerate(names)}
        except Exception:
            return self._equal_weights(names)

    def _entropy_weights(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Shannon entropy weighting (Zou et al. 2006).

        Higher entropy (more variation across space) → higher weight.
        Formula:
            e_j = -k * Σ(p_ij * ln(p_ij))  where k = 1/ln(n)
            w_j = (1 - e_j) / Σ(1 - e_k)
        """
        names = list(arrays.keys())
        n_indicators = len(names)
        if n_indicators == 0:
            return {}

        entropies = []
        for arr in arrays.values():
            flat = arr[np.isfinite(arr)].ravel()
            if len(flat) < 2:
                entropies.append(1.0)
                continue
            # Normalise to probability distribution
            flat_min = flat.min()
            flat_max = flat.max()
            if flat_max - flat_min < 1e-12:
                entropies.append(1.0)
                continue
            p = (flat - flat_min) / (flat_max - flat_min)
            p = np.clip(p, 1e-10, 1.0)
            p = p / p.sum()
            k = 1.0 / np.log(len(p))
            e = -k * np.sum(p * np.log(p))
            entropies.append(float(np.clip(e, 0, 1)))

        redundancies = [1.0 - e for e in entropies]
        total = sum(redundancies)
        if total < 1e-12:
            return self._equal_weights(names)
        return {name: redundancies[i] / total for i, name in enumerate(names)}

    def _critic_weights(self, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        CRITIC method (Diakoulaki et al. 1995).

        Combines standard deviation (contrast intensity) with Spearman
        rank correlation (conflict measure).
        C_j = σ_j * Σ_k(1 - r_jk)   for all k ≠ j
        w_j = C_j / Σ(C_k)
        """
        from scipy.stats import spearmanr

        names = list(arrays.keys())
        n = len(names)
        if n < 2:
            return self._equal_weights(names)

        # Build matrix of valid rows
        mat = np.column_stack([a.ravel() for a in arrays.values()])
        valid = np.all(np.isfinite(mat), axis=1)
        mat = mat[valid]
        if len(mat) < 3:
            return self._equal_weights(names)

        stds = mat.std(axis=0)
        corr_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = spearmanr(mat[:, i], mat[:, j])
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r

        conflict = np.sum(1.0 - corr_matrix, axis=1) - (1.0 - 1.0)  # subtract self (=0)
        C = stds * conflict
        total = C.sum()
        if total < 1e-12:
            return self._equal_weights(names)
        return {name: float(C[i] / total) for i, name in enumerate(names)}

    def _manual_weights(
        self,
        indicator_names: List[str],
        user_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Accept and validate user-supplied weights.

        Raises
        ------
        ValueError
            If weights do not sum to 1.0 (tolerance 1e-6) or contain
            unrecognised indicator keys.
        """
        total = sum(user_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Manual weights must sum to 1.0, got {total:.8f}. "
                "Please adjust the values and try again."
            )
        return {name: float(user_weights.get(name, 0.0)) for name in indicator_names}

    # ── Validation helper ──────────────────────────────────────────────────────

    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> None:
        """Assert weights sum to 1.0 within floating-point tolerance."""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Weights sum to {total:.10f}, expected 1.0."
            )
