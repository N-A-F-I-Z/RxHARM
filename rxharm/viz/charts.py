"""
rxharm/viz/charts.py
=====================
Statistical and analytical charts for Project RxHARM.

All functions return matplotlib Figure objects.
Dependencies: matplotlib, seaborn, numpy, pandas, scipy
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd


def show_indicator_correlation_matrix(normalized_arrays: dict) -> object:
    """
    Spearman rank correlation heatmap between all 14 normalised indicators.

    Parameters
    ----------
    normalized_arrays : dict
        ``{indicator_name: normalised_ndarray}``

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    names = list(normalized_arrays.keys())
    n = len(names)
    corr_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = normalized_arrays[names[i]].ravel()
            b = normalized_arrays[names[j]].ravel()
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() > 5:
                r, _ = spearmanr(a[valid], b[valid])
                corr_mat[i, j] = corr_mat[j, i] = r

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", center=0,
        cmap="RdBu_r", xticklabels=names, yticklabels=names,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    )
    ax.set_title("Spearman Rank Correlation — All 14 Indicators", fontsize=13)
    plt.tight_layout()
    return fig


def show_subindex_distributions(hvi_results: dict) -> object:
    """
    4-panel histogram of H_s, E, S, AC distributions.

    Parameters
    ----------
    hvi_results : dict
        Output of HVIEngine.compute_all().

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    keys   = ["H_s", "E", "S", "AC"]
    labels = ["Hazard (H_s)", "Exposure (E)", "Sensitivity (S)", "Adaptive Capacity (AC)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, label in zip(axes.ravel(), keys, labels):
        arr = hvi_results.get(key, np.zeros(100))
        valid = arr[np.isfinite(arr)].ravel()
        if len(valid):
            ax.hist(valid, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
            mean_val = valid.mean()
            ax.axvline(mean_val, color="red", linestyle="--", lw=1.5,
                       label=f"Pop-wtd mean = {mean_val:.3f}")
            ax.legend(fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Value [0–1]")
        ax.set_ylabel("Cell count")
    fig.suptitle("Sub-index Value Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def show_sensitivity_test(sensitivity_df: pd.DataFrame) -> object:
    """
    Bar chart of Spearman r from HVIEngine.sensitivity_test().

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        From HVIEngine.sensitivity_test().

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if sensitivity_df is None or len(sensitivity_df) == 0:
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    grp = sensitivity_df.groupby("indicator")["spearman_r"].mean().sort_values()
    colors = ["#d73027" if v < 0.90 else "#1a9850" for v in grp.values]
    grp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.axvline(0.90, color="black", linestyle="--", lw=1.5, label="Robustness threshold (r=0.90)")
    ax.set_xlabel("Mean Spearman r (perturbed vs nominal HVI)")
    ax.set_title("Weight Sensitivity Analysis — HVI Spatial Rank Stability")
    ax.legend()
    plt.tight_layout()
    return fig


def show_weighting_comparison(indicator_arrays: dict) -> object:
    """
    Compare HVI under equal, PCA, entropy, and CRITIC weighting schemes.

    Parameters
    ----------
    indicator_arrays : dict
        Raw indicator arrays.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from rxharm.index.hvi import HVIEngine

    methods = ["equal", "pca", "entropy", "critic"]
    hvi_maps = {}
    for m in methods:
        try:
            hvi_maps[m] = HVIEngine(m).compute_all(indicator_arrays)["HVI"]
        except Exception:
            hvi_maps[m] = np.full((5, 5), np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axes, methods):
        im = ax.imshow(hvi_maps[m], cmap="viridis", vmin=0, vmax=1, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"Weighting: {m}")
    fig.suptitle("HVI under Four Weighting Schemes", fontsize=13)
    plt.tight_layout()
    return fig


def show_pareto_front(
    result,
    labels: Optional[List[str]] = None,
    title: str = "Pareto Front",
) -> object:
    """
    2D scatter projection of the Pareto front.

    Parameters
    ----------
    result : pymoo Result or dict with 'F' key
    labels : list of str
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        F = result.F if hasattr(result, "F") else result.get("F", np.zeros((1, 3)))
        if F is None or len(F) == 0:
            raise ValueError("Empty Pareto front")
        x_label = labels[0] if labels and len(labels) > 0 else "f1"
        y_label = labels[1] if labels and len(labels) > 1 else "f2"
        c_label = labels[2] if labels and len(labels) > 2 else "f3"
        scatter = ax.scatter(F[:, 0], F[:, 1], c=F[:, 2] if F.shape[1] > 2 else "steelblue",
                             cmap="RdYlGn_r", alpha=0.8, s=40)
        if F.shape[1] > 2:
            plt.colorbar(scatter, ax=ax, label=c_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}\n({len(F)} Pareto solutions)")
    except Exception as e:
        ax.text(0.5, 0.5, f"No Pareto data yet\n({e})",
                ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    return fig


def show_uncertainty_bounds(mc_results: dict) -> object:
    """
    Display MC uncertainty bounds on HVI (P10/P50/P90 distributions).

    Parameters
    ----------
    mc_results : dict
        From MCUncertaintyEngine.compute_hvi_distribution().

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label in zip(axes, ["p10", "p50", "p90"], ["P10", "Median", "P90"]):
        arr = mc_results.get(key, np.zeros((5, 5)))
        im  = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1, origin="upper")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"HVI {label}")
    fig.suptitle("Monte Carlo HVI Uncertainty Bounds", fontsize=13)
    plt.tight_layout()
    return fig
