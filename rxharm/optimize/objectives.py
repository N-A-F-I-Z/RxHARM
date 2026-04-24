"""
rxharm/optimize/objectives.py
==============================
Objective functions for NSGA-III optimization in Project RxHARM.

All objectives are MINIMIZED (pymoo convention).

Short-run (3 objectives):
    f1_sr  minimize coverage gap for vulnerable population
    f2_sr  minimize cost fraction of SR budget
    f3_sr  minimize spatial inequity of coverage (Gini)

Long-run (5 objectives):
    f1  minimize fraction of AD not prevented
    f2  minimize cost fraction
    f3  minimize equity Gini of residual per-capita risk
    f4  minimize negative co-benefit efficiency (= maximize)
    f5  minimize scenario robustness coefficient of variation

References:
    Gasparrini et al. (2015), Lancet — base mortality model
    Hsu et al. (2021), Nature Communications — equity metric
    Quinn et al. (2017), Env Modelling & Software — MORDM robustness
"""

from __future__ import annotations
from typing import List
import numpy as np

from rxharm.config import CARBON_PRICE_USD_PER_TCO2, STORMWATER_VALUE_USD_PER_M3


# ── Long-run objectives ────────────────────────────────────────────────────────

def f1_mortality_reduction(
    ad_post: np.ndarray,
    ad_baseline: np.ndarray,
) -> float:
    """
    Fraction of total baseline AD not prevented.

    f1 = 1 - (sum(AD_baseline - AD_post) / sum(AD_baseline))
    [minimize; 0 = all deaths prevented, 1 = none prevented]

    Parameters
    ----------
    ad_post : np.ndarray   Expected AD after interventions.
    ad_baseline : np.ndarray   Baseline AD without interventions.

    Returns
    -------
    float in [0, 1]
    """
    total_baseline = float(np.nansum(ad_baseline))
    if total_baseline < 1e-9:
        return 0.0
    total_prevented = float(np.nansum(np.maximum(0.0, ad_baseline - ad_post)))
    f1 = 1.0 - total_prevented / total_baseline
    return float(np.clip(f1, 0.0, 1.0))


def f2_cost_fraction(
    x: np.ndarray,
    intervention_lib,
    budget: float,
) -> float:
    """
    Total cost as fraction of the planning budget.

    f2 = sum(x[:, k] * cost_mode_k) / budget
    [minimize; 0 = free, 1 = exactly at budget]

    Parameters
    ----------
    x : np.ndarray   Shape (n_zones, n_interventions).
    intervention_lib : InterventionLibrary
    budget : float   Planning budget in USD.

    Returns
    -------
    float ≥ 0
    """
    if budget < 1e-9:
        return 1.0
    total_cost = 0.0
    for k, key in enumerate(intervention_lib.lr.keys()):
        if k >= x.shape[1]:
            break
        cost_mode = intervention_lib.lr[key]["group_C_cost"]["cost_per_unit_usd"][1]
        total_cost += float(np.nansum(x[:, k] * cost_mode))
    return float(total_cost / budget)


def f3_equity_gini(
    ad_post: np.ndarray,
    population: np.ndarray,
) -> float:
    """
    Gini coefficient of population-normalised residual mortality risk.

    f3 = Gini(AD_post_i / Pop_i)  [minimize; 0 = perfectly equal]

    Justification: Hsu et al. (2021, Nature Communications) demonstrated
    spatial concentration of UHI exposure. Erreygers (2009) health equity.

    Parameters
    ----------
    ad_post : np.ndarray   Residual AD per cell after interventions.
    population : np.ndarray   Population per cell.

    Returns
    -------
    float in [0, 1]
    """
    pop_safe        = np.maximum(population, 1.0)
    risk_per_capita = ad_post / pop_safe
    valid           = risk_per_capita[~np.isnan(risk_per_capita)]
    if len(valid) < 2:
        return 0.0
    sorted_r = np.sort(valid)
    n        = len(sorted_r)
    index    = np.arange(1, n + 1)
    gini     = (2 * np.sum(index * sorted_r) / (n * np.sum(sorted_r))) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def f4_cobenefit_efficiency(
    x: np.ndarray,
    intervention_lib,
    budget: float,
    carbon_price: float = CARBON_PRICE_USD_PER_TCO2,
    stormwater_value: float = STORMWATER_VALUE_USD_PER_M3,
) -> float:
    """
    Negative co-benefit value per unit budget (minimize = maximize efficiency).

    f4 = -(TotalCobenefit / Budget)

    Co-benefit = carbon_tCO2 * carbon_price + stormwater_m3 * stormwater_value

    FIX v0.1.0:
    - Old code divided by ``total_cost`` which is near zero in early NSGA-III
      generations (x≈0), causing NaN that propagated through the entire Pareto
      calculation and corrupted the optimizer's archive.
    - Fixed: divide by ``budget`` (user-specified constant), not ``total_cost``.
      This makes f4 = 'co-benefit per unit of available budget' — a consistent,
      interpretable measure even when little or nothing has been allocated.
    - Added final NaN guard to prevent optimizer corruption in edge cases.

    Source: IPCC AR6 Ch.8 (2022); Elmqvist et al. (2015) Nature Communications.
    """
    total_cb = 0.0
    for k, key in enumerate(intervention_lib.lr.keys()):
        if k >= x.shape[1]:
            break
        d    = intervention_lib.lr[key].get("group_D_cobenefits", {})
        # FIX v0.1.0: use .get with default [0,0,0] to handle missing keys gracefully
        c_m  = d.get("carbon_tCO2_per_unit_per_year",   [0, 0, 0])
        sw_m = d.get("stormwater_m3_per_unit_per_year", [0, 0, 0])
        c_val  = c_m[1]  if isinstance(c_m,  list) else float(c_m)
        sw_val = sw_m[1] if isinstance(sw_m, list) else float(sw_m)
        cb = (c_val * carbon_price + sw_val * stormwater_value) * x[:, k]
        total_cb += float(np.nansum(cb))

    # FIX v0.1.0: divide by budget, not total_cost — prevents NaN in early generations
    result = -(total_cb / (budget + 1e-9))

    # FIX v0.1.0: final NaN guard — should never trigger but prevents optimizer corruption
    if not np.isfinite(result):
        return 0.0

    return float(result)


def f5_scenario_robustness(
    x: np.ndarray,
    scenarios: list,
    hri_engine,
    hvi_results: dict,
    intervention_lib,
    eff_sample: dict,
) -> float:
    """
    Coefficient of variation of total AD across all four scenarios.

    f5 = (max_s(AD_total_s) - min_s(AD_total_s)) / denominator
    [minimize; 0 = same outcome regardless of scenario = robust]

    FIX v0.1.0:
    - Old denominator was ``min_s(AD_total_s)`` which becomes zero or negative
      due to MC noise in low-population cells, causing NaN or negative f5
      that the optimizer exploited to find meaningless solutions.
    - Fixed denominator: ``max(|min_s(AD_total_s)|, 1.0)`` — always positive.
    - Added ``np.maximum(0, ad)`` before summing: negative AD is physically
      impossible and must not propagate through the objective.

    Source: Quinn et al. (2017), Env Modelling & Software (MORDM framework).
    """
    ad_totals = []
    for scenario in scenarios:
        state   = intervention_lib.compute_post_intervention_state(x, hvi_results, eff_sample)
        pop_adj = hvi_results.get("indicator_normalized", {}).get(
            "population", np.ones(x.shape[0])
        ) * scenario["pop_factor"]
        lst_adj = (
            state.get("lst_post", hvi_results.get("H_s", np.zeros(x.shape[0])))
            + scenario["T_delta"]
        )
        af = hri_engine.compute_attributable_fraction(lst_adj, state["hvi_post"])

        # FIX v0.1.0: clip AD to non-negative — prevents MC noise from producing
        # physically impossible negative deaths that corrupt the denominator.
        ad = np.maximum(0.0, pop_adj * hri_engine.cdr * af * scenario["event_days"])
        ad_totals.append(float(np.nansum(ad)))

    if len(ad_totals) < 2:
        return 0.0

    ad_max = max(ad_totals)
    ad_min = min(ad_totals)

    # FIX v0.1.0: use max(|min_AD|, 1.0) as denominator.
    # Old code: divided by min_AD which could be zero → NaN, or negative → negative f5.
    # New code: floor at 1.0 death so the ratio is always 'deaths of variation per
    # death of baseline' — physically interpretable and numerically stable.
    denominator = max(abs(ad_min), 1.0)
    return float((ad_max - ad_min) / denominator)
