"""
rxharm/optimize/constraints.py
================================
Constraint functions for pymoo NSGA-III.

All return values <= 0 when satisfied (pymoo convention).
Positive value = constraint violated (magnitude = degree of violation).

C1  budget          total cost <= budget
C2  area_feasibility quantity per zone <= feasible maximum
C3  mutual_exclusivity LR1 (trees) + LR3 (BGI) <= open land available
C4  minimum_viable_unit each zone: quantity is 0 OR >= min_viable
"""

from __future__ import annotations
import numpy as np


def c1_budget(
    x: np.ndarray,
    intervention_lib,
    budget: float,
) -> float:
    """
    C1: Total cost must not exceed budget.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_zones, n_lr_interventions).
    intervention_lib : InterventionLibrary
    budget : float

    Returns
    -------
    float
        total_cost - budget  (<=0 = satisfied)
    """
    cost = 0.0
    for k, key in enumerate(intervention_lib.lr.keys()):
        if k >= x.shape[1]:
            break
        cost_mode = intervention_lib.lr[key]["group_C_cost"]["cost_per_unit_usd"][1]
        cost += float(np.nansum(x[:, k] * cost_mode))
    return float(cost - budget)


def c2_area_feasibility(
    x: np.ndarray,
    max_quantities: np.ndarray,
) -> float:
    """
    C2: Quantity per zone must not exceed feasible maximum.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_zones, n_interventions).
    max_quantities : np.ndarray
        Shape (n_zones, n_interventions) or (n_interventions,).

    Returns
    -------
    float
        Maximum excess across all zones and interventions.
    """
    violations = []
    for k in range(x.shape[1]):
        if max_quantities.ndim == 1:
            max_q = max_quantities[k]
        else:
            max_q = max_quantities[:, k]
        excess = np.maximum(0.0, x[:, k] - max_q)
        violations.append(float(np.sum(excess)))
    return max(violations) if violations else 0.0


def c3_mutual_exclusivity(
    x: np.ndarray,
    intervention_lib,
) -> float:
    """
    C3: LR1 (trees) and LR3 (BGI) cannot exceed available open land together.

    Combined quantity should not exceed 120% of the larger allocation alone
    (20% tolerance for partial spatial overlap).

    Parameters
    ----------
    x : np.ndarray
    intervention_lib : InterventionLibrary

    Returns
    -------
    float
    """
    keys    = list(intervention_lib.lr.keys())
    lr1_idx = next((i for i, k in enumerate(keys) if "tree" in k.lower()), None)
    lr3_idx = next((i for i, k in enumerate(keys) if "bgi"  in k.lower()), None)

    if lr1_idx is None or lr3_idx is None:
        return 0.0
    if lr1_idx >= x.shape[1] or lr3_idx >= x.shape[1]:
        return 0.0

    combined  = x[:, lr1_idx] + x[:, lr3_idx]
    max_single = np.maximum(x[:, lr1_idx], x[:, lr3_idx])
    excess    = np.maximum(0.0, combined - max_single * 1.2)
    return float(np.sum(excess))


def c4_minimum_viable_unit(
    x: np.ndarray,
    min_viable_units: list,
) -> float:
    """
    C4: Each zone's allocation is either 0 or >= min_viable.

    Prevents allocations too small to be physically meaningful
    (e.g. 0.1 trees — round up or remove).

    Parameters
    ----------
    x : np.ndarray
    min_viable_units : list of float
        One minimum per intervention column.

    Returns
    -------
    float
        Total number of zone-intervention pairs with sub-viable quantities.
    """
    violations = 0.0
    for k, min_qty in enumerate(min_viable_units):
        if k >= x.shape[1]:
            break
        col         = x[:, k]
        problematic = (col > 0) & (col < min_qty)
        violations += float(np.sum(problematic))
    return violations
