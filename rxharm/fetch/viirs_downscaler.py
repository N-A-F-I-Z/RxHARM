"""
rxharm/fetch/viirs_downscaler.py
=================================
Downscales VIIRS DNB nighttime light from ~500 m to 100 m resolution
using the RFATPK (Random Forest Area-to-Point regression Kriging) method.

Method summary:
    1. The Random Forest is trained to predict coarse-resolution VIIRS
       radiance from fine-resolution covariates (WorldPop population,
       Sentinel-2 NDVI, GHS-BUILT-S built fraction).
    2. RF predictions capture the large-scale spatial pattern.
    3. ATPK (Area-to-Point Kriging) downscales the residuals between
       the actual coarse VIIRS values and the RF-aggregated predictions,
       preserving spatial coherence at the coarse scale.
    4. Final output = RF predictions + interpolated residuals.

Simplified ATPK note:
    Full ATPK using variogram fitting is the theoretically correct method
    (Jeswani 2023). This implementation uses bilinear residual interpolation
    as a computationally tractable approximation for the research prototype.
    Full ATPK can be added as a future enhancement.

Reference:
    Jeswani et al. (2023), International Journal of Applied Earth
    Observation and Geoinformation, 122:103395.
    DOI: 10.1016/j.jag.2023.103395

Notes:
    - Operates on NumPy arrays after GEE export (no GEE dependency).
    - Input: coarse VIIRS array (500 m) + fine covariate arrays (100 m).
    - Output: downscaled VIIRS at 100 m.
    - For large AOIs (H_fine * W_fine > 250 000), tiled processing is used.

Usage:
    downscaler = VIIRSDownscaler(n_estimators=100, random_state=42)
    viirs_100m = downscaler.downscale(
        viirs_coarse=viirs_array,
        covariates_fine={
            'population': pop_array,
            'ndvi': ndvi_array,
            'built_frac': built_array,
        },
        coarse_resolution_m=500,
        fine_resolution_m=100,
    )
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, Optional

# ── Third-party (non-GEE; always available) ───────────────────────────────────
import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor

# ── Constants ─────────────────────────────────────────────────────────────────
_VIIRS_MAX_RADIANCE = 200.0   # nW/cm²/sr physical ceiling (clips gas flare outliers)
_TILE_PIXEL_LIMIT   = 250_000  # fine pixels above which tiled processing is used
_TILE_SIZE_FINE     = 500      # default tile side in fine pixels
_TILE_OVERLAP       = 50       # overlap pixels per edge for boundary artifact reduction


class VIIRSDownscaler:
    """
    Downscales VIIRS DNB nighttime light from 500 m to 100 m using RFATPK.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Random Forest. Default 100.
    random_state : int
        Random seed for RF reproducibility. Default 42.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        # REASON: Create the RF model at init so the same model object
        # persists and can be inspected after downscale() returns.
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def downscale(
        self,
        viirs_coarse: np.ndarray,
        covariates_fine: Dict[str, np.ndarray],
        coarse_resolution_m: int = 500,
        fine_resolution_m: int = 100,
    ) -> np.ndarray:
        """
        Run the full RFATPK pipeline and return downscaled radiance at 100 m.

        Parameters
        ----------
        viirs_coarse : np.ndarray
            2-D array of VIIRS radiance at coarse resolution.
            Shape: ``(H_coarse, W_coarse)``. May contain NaN for masked pixels.
        covariates_fine : dict
            Dict of 2-D covariate arrays at fine resolution.
            Required keys: ``'population'``, ``'ndvi'``, ``'built_frac'``.
            Shape: ``(H_fine, W_fine)`` — must be exactly
            ``(H_coarse * factor, W_coarse * factor)`` where
            ``factor = coarse_resolution_m // fine_resolution_m``.
        coarse_resolution_m : int
            Native VIIRS resolution in metres. Default 500.
        fine_resolution_m : int
            Target output resolution in metres. Default 100.

        Returns
        -------
        np.ndarray
            Downscaled VIIRS radiance at fine resolution.
            Shape: ``(H_fine, W_fine)``. No NaN values; clipped to [0, 200].
        """
        factor = coarse_resolution_m // fine_resolution_m

        H_fine = viirs_coarse.shape[0] * factor
        W_fine = viirs_coarse.shape[1] * factor

        # Route to tiled processor for large AOIs
        if H_fine * W_fine > _TILE_PIXEL_LIMIT:
            return self.downscale_tiled(
                viirs_coarse, covariates_fine,
                coarse_resolution_m, fine_resolution_m,
            )

        return self._downscale_block(
            viirs_coarse, covariates_fine, factor
        )

    def downscale_tiled(
        self,
        viirs_coarse: np.ndarray,
        covariates_fine: Dict[str, np.ndarray],
        coarse_resolution_m: int = 500,
        fine_resolution_m: int = 100,
        tile_size_fine: int = _TILE_SIZE_FINE,
    ) -> np.ndarray:
        """
        Process large arrays in overlapping tiles to manage memory.

        Parameters
        ----------
        viirs_coarse : np.ndarray
            Coarse VIIRS array.
        covariates_fine : dict
            Fine covariate arrays.
        coarse_resolution_m : int
            Native coarse resolution in metres.
        fine_resolution_m : int
            Target fine resolution in metres.
        tile_size_fine : int
            Side length of each tile at fine resolution. Default 500 px.

        Returns
        -------
        np.ndarray
            Full downscaled array, stitched from tiles.
        """
        factor    = coarse_resolution_m // fine_resolution_m
        H_fine    = viirs_coarse.shape[0] * factor
        W_fine    = viirs_coarse.shape[1] * factor
        output    = np.zeros((H_fine, W_fine), dtype=np.float32)
        weight    = np.zeros((H_fine, W_fine), dtype=np.float32)

        tile_c    = tile_size_fine // factor  # tile size in coarse pixels
        overlap_c = _TILE_OVERLAP // factor   # overlap in coarse pixels

        # Iterate over coarse tiles
        row_c = 0
        while row_c < viirs_coarse.shape[0]:
            col_c = 0
            while col_c < viirs_coarse.shape[1]:
                # Extract tile (with overlap)
                r0c = max(0, row_c - overlap_c)
                r1c = min(viirs_coarse.shape[0], row_c + tile_c + overlap_c)
                c0c = max(0, col_c - overlap_c)
                c1c = min(viirs_coarse.shape[1], col_c + tile_c + overlap_c)

                tile_coarse = viirs_coarse[r0c:r1c, c0c:c1c]
                tile_cov    = {k: v[r0c*factor:r1c*factor, c0c*factor:c1c*factor]
                               for k, v in covariates_fine.items()}

                tile_result = self._downscale_block(tile_coarse, tile_cov, factor)

                # Place in output (discard overlap edges for central tiles)
                r0f = r0c * factor
                r1f = r1c * factor
                c0f = c0c * factor
                c1f = c1c * factor
                output[r0f:r1f, c0f:c1f] += tile_result
                weight[r0f:r1f, c0f:c1f] += 1.0

                col_c += tile_c
            row_c += tile_c

        # Normalise by overlap counts
        mask = weight > 0
        output[mask] /= weight[mask]
        return np.clip(output, 0, _VIIRS_MAX_RADIANCE).astype(np.float32)

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _downscale_block(
        self,
        viirs_coarse: np.ndarray,
        covariates_fine: Dict[str, np.ndarray],
        factor: int,
    ) -> np.ndarray:
        """
        Apply the RFATPK pipeline to a single array block.

        Steps:
            1. Aggregate fine covariates to coarse resolution.
            2. Train RF on coarse data (VIIRS ~ coarse covariates).
            3. Predict at fine resolution using fine covariates.
            4. Compute coarse-scale residuals.
            5. Bilinear-interpolate residuals to fine scale.
            6. Combine and clip.

        Parameters
        ----------
        viirs_coarse : np.ndarray
            2-D coarse VIIRS array.
        covariates_fine : dict
            Fine-resolution covariate arrays.
        factor : int
            Downscaling factor (coarse / fine).

        Returns
        -------
        np.ndarray
            Downscaled radiance at fine resolution.
        """
        H_coarse, W_coarse = viirs_coarse.shape
        H_fine   = H_coarse * factor
        W_fine   = W_coarse * factor

        # Step 1: Aggregate fine covariates to coarse
        cov_coarse = {
            k: self._block_reduce(v, factor)
            for k, v in covariates_fine.items()
        }

        # Step 2: Train RF on coarse data
        X_coarse = self._stack_covariates(cov_coarse, H_coarse, W_coarse)
        y_coarse = viirs_coarse.ravel()

        # Mask NaN pixels in both X and y for training
        valid = np.isfinite(y_coarse) & np.all(np.isfinite(X_coarse), axis=1)
        if valid.sum() < 10:
            # Not enough valid pixels — fall back to bilinear zoom
            viirs_coarse_clean = np.where(np.isfinite(viirs_coarse), viirs_coarse, 0.0)
            return np.clip(
                zoom(viirs_coarse_clean, factor, order=1), 0, _VIIRS_MAX_RADIANCE
            ).astype(np.float32)

        # REASON: A fresh RF is fitted per block so tile boundaries don't bleed.
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_coarse[valid], y_coarse[valid])
        self.rf_model = rf  # expose last-fitted model for inspection

        # Step 3: Predict at fine resolution
        X_fine     = self._stack_covariates(covariates_fine, H_fine, W_fine)
        rf_pred_fine = rf.predict(X_fine).reshape(H_fine, W_fine)

        # Step 4: Compute coarse-scale residuals
        rf_pred_coarse = self._block_reduce(rf_pred_fine, factor)
        viirs_filled   = np.where(np.isfinite(viirs_coarse), viirs_coarse, rf_pred_coarse)
        residuals_c    = viirs_filled - rf_pred_coarse

        # Step 5: Bilinear interpolation of residuals to fine scale
        # REASON: Full variogram ATPK is the theoretically correct method
        # (Jeswani 2023) but bilinear zoom is a computationally tractable
        # approximation for this research prototype.
        residuals_fine = zoom(residuals_c, factor, order=1)

        # Step 6: Combine and clip
        result = rf_pred_fine + residuals_fine
        return np.clip(result, 0, _VIIRS_MAX_RADIANCE).astype(np.float32)

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _block_reduce(
        arr: np.ndarray,
        factor: int,
        func=np.nanmean,
    ) -> np.ndarray:
        """
        Aggregate a 2-D array by averaging non-overlapping blocks.

        Handles arrays whose dimensions are not divisible by factor by
        padding with NaN before reducing.

        Parameters
        ----------
        arr : np.ndarray
            2-D array to reduce.
        factor : int
            Aggregation factor (output will be ``arr.shape // factor``).
        func : callable
            Reduction function applied to each block. Default ``np.nanmean``.

        Returns
        -------
        np.ndarray
            Reduced array of shape ``(H // factor, W // factor)``.
        """
        H, W = arr.shape
        # Pad to make dimensions divisible by factor
        pad_H = (-H) % factor
        pad_W = (-W) % factor
        if pad_H > 0 or pad_W > 0:
            arr = np.pad(arr, ((0, pad_H), (0, pad_W)), constant_values=np.nan)
        H_p, W_p = arr.shape
        # REASON: Reshape into blocks then reduce — vectorised, no Python loops.
        reshaped = arr.reshape(H_p // factor, factor, W_p // factor, factor)
        reduced  = func(reshaped, axis=(1, 3))
        return reduced

    @staticmethod
    def _stack_covariates(
        covariates: Dict[str, np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Stack covariate arrays into a feature matrix for scikit-learn.

        FIX 0.1.1: Column order is FIXED by COVARIATE_ORDER regardless of
        dict key ordering. This prevents the RF from being trained on column
        order A and predicting on column order B (silent wrong-result bug).

        Parameters
        ----------
        covariates : dict
            Keys: ``'population'``, ``'ndvi'``, ``'built_frac'``.
            Values: 2-D NumPy arrays.
        H : int
            Expected row count.
        W : int
            Expected column count.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(H * W, n_features)``.
            Column order always: population, ndvi, built_frac.
        """
        # FIX 0.1.1: Canonical fixed column order
        COVARIATE_ORDER = ("population", "ndvi", "built_frac")
        arrays = []
        for key in COVARIATE_ORDER:
            arr = covariates.get(key, np.zeros((H, W)))
            # Ensure correct shape (resize if needed due to rounding)
            if arr.shape != (H, W):
                arr = zoom(arr, (H / arr.shape[0], W / arr.shape[1]), order=1)
            arrays.append(arr.ravel())
        return np.column_stack(arrays)
