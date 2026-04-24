"""
rxharm/viz/maps.py
==================
Spatial visualisation for HVI, HRI, and prescription maps.

All functions work in Google Colab. Interactive maps use folium;
static maps use matplotlib.

Dependencies: matplotlib, numpy, pandas (folium optional for interactive)
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def show_hvi_map(
    hvi_results: dict,
    aoi_handler,
    mode: str = "static",
) -> object:
    """
    Display HVI as a choropleth map.

    Parameters
    ----------
    hvi_results : dict
        Output of HVIEngine.compute_all().
    aoi_handler : AOIHandler
        For bounds and centroid.
    mode : str
        ``'interactive'`` (folium) or ``'static'`` (matplotlib).

    Returns
    -------
    matplotlib.figure.Figure or folium.Map
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    hvi = hvi_results.get("HVI", np.zeros((10, 10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(hvi, cmap="viridis", vmin=0, vmax=1, origin="upper")
    plt.colorbar(im, ax=ax, label="HVI [0–1]")
    ax.set_title("Heat Vulnerability Index (HVI)")
    ax.set_xlabel("Column (W→E)")
    ax.set_ylabel("Row (N→S)")
    plt.tight_layout()
    return fig


def show_subindex_comparison(hvi_results: dict, aoi_handler) -> object:
    """
    2×2 subplot showing H_s, E, S, AC sub-indices.

    Parameters
    ----------
    hvi_results : dict
        Output of HVIEngine.compute_all().
    aoi_handler : AOIHandler

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    keys   = ["H_s", "E", "S", "AC"]
    labels = ["Hazard (H_s)", "Exposure (E)", "Sensitivity (S)", "Adaptive Capacity (AC)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, key, label in zip(axes.ravel(), keys, labels):
        arr = hvi_results.get(key, np.zeros((10, 10)))
        im  = ax.imshow(arr, cmap="YlOrRd", vmin=0, vmax=1, origin="upper")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{label}\n[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
    fig.suptitle("HVI Sub-indices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def show_hri_map(hri_results: dict, aoi_handler, mode: str = "static") -> object:
    """
    Display HRI using a red-orange palette.

    Parameters
    ----------
    hri_results : dict
        Output of HRIEngine.compute_all().
    aoi_handler : AOIHandler
    mode : str
        ``'static'`` or ``'interactive'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    hri = hri_results.get("HRI", np.zeros((10, 10)))
    ad  = hri_results.get("AD_baseline", None)

    fig, axes = plt.subplots(1, 2 if ad is not None else 1, figsize=(12 if ad is not None else 7, 5))
    if ad is None:
        axes = [axes]

    im = axes[0].imshow(hri, cmap="hot_r", vmin=0, vmax=1, origin="upper")
    plt.colorbar(im, ax=axes[0], label="HRI [0–1]")
    axes[0].set_title("Heat Risk Index (HRI)")

    if ad is not None:
        im2 = axes[1].imshow(ad, cmap="Reds", origin="upper")
        plt.colorbar(im2, ax=axes[1], label="Expected deaths / 3-day event")
        axes[1].set_title(f"Attributable Deaths\n(Total: {np.nansum(ad):.2f})")

    plt.tight_layout()
    return fig


def show_prescription_map(
    pareto_result,
    cell_states: dict,
    zones: dict,
    strategy: str = "balanced",
) -> object:
    """
    Display a prescription map for one Pareto strategy.

    Parameters
    ----------
    pareto_result : pymoo Result
    cell_states : dict
    zones : dict
    strategy : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(f"Intervention Prescription — '{strategy}' strategy")
    ax.text(0.5, 0.5, "Run optimizer to generate prescriptions",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")
    plt.tight_layout()
    return fig
