"""
rxharm/optimize/problem.py
===========================
pymoo Problem subclasses for NSGA-III optimization.

ShortRunProblem  — 3 objectives (coverage gap, cost, inequity)
LongRunProblem   — 5 objectives (mortality, cost, equity, cobenefit, robustness)

Decision variables:
    Short-run: counts of each SR intervention type per zone
    Long-run:  quantities of each LR intervention type per zone

All objectives are minimized (pymoo convention).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from rxharm.config import (
    CARBON_PRICE_USD_PER_TCO2,
    COOLING_CENTER_RADIUS_M,
    LAMBDA_HVI_DEFAULT,
    STORMWATER_VALUE_USD_PER_M3,
)
from rxharm.optimize.objectives import (
    f1_mortality_reduction,
    f2_cost_fraction,
    f3_equity_gini,
    f4_cobenefit_efficiency,
    f5_scenario_robustness,
)
from rxharm.optimize.constraints import (
    c1_budget,
    c2_area_feasibility,
    c3_mutual_exclusivity,
)


# ── Short-run problem ──────────────────────────────────────────────────────────

class ShortRunProblem:
    """
    3-objective short-run heatwave intervention problem.

    Objectives:
        f1  minimize fraction of vulnerable pop without coverage
        f2  minimize cost fraction of SR budget
        f3  minimize spatial inequity of coverage (Gini)

    Decision variables (5 integers per zone):
        x[0]  cooling centers activated
        x[1]  misting units deployed
        x[2]  welfare check zones (binary per high-priority zone)
        x[3]  medical units repositioned
        x[4]  shade structures deployed

    Parameters
    ----------
    cell_states : dict
        Contains: 'HRI', 'elderly_frac', 'population', 'ad_baseline'
        and all indicator arrays.
    intervention_lib : InterventionLibrary
    budget : float
        Emergency budget in USD.
    hri_threshold_percentile : int
        Percentile of HRI above which welfare checks are targeted.
    k_medical_units : int
        Total medical units available for repositioning.
    """

    def __init__(
        self,
        cell_states: dict,
        intervention_lib,
        budget: float,
        hri_threshold_percentile: int = 75,
        k_medical_units: int = 3,
    ) -> None:
        self.cs           = cell_states
        self.lib          = intervention_lib
        self.budget       = budget
        self.hri_p75      = float(np.nanpercentile(
            cell_states.get("HRI", np.zeros(10)), hri_threshold_percentile
        ))
        self.k_med        = k_medical_units
        # Problem dimensions
        self.n_obj        = 3
        self.n_ieq_constr = 2  # C1 budget, C2 area

    def _evaluate(self, x_flat: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        Evaluate 3 objectives for a candidate short-run solution.

        x_flat : 1-D decision vector [n_sr_intervs * n_zones]
        """
        n_sr  = len(self.lib.sr)
        n_zones = max(1, len(x_flat) // n_sr)
        x     = x_flat[:n_sr * n_zones].reshape(n_zones, n_sr)

        # Compute coverage
        hri       = np.asarray(self.cs.get("HRI", np.zeros(n_zones)))
        elderly   = np.asarray(self.cs.get("elderly_frac", np.zeros(n_zones)))
        pop       = np.asarray(self.cs.get("population", np.ones(n_zones)))
        ad_base   = np.asarray(self.cs.get("ad_baseline", np.ones(n_zones) * 0.01))

        # Simple coverage proxy: cooling centres + shade reach cells within radius
        coverage = np.minimum(1.0, (x[:, 0] + x[:, 4]) * 0.3)

        # f1: coverage gap for vulnerable (elderly in high-HRI zones)
        priority = elderly * (hri >= self.hri_p75).astype(float)
        total_priority = max(float(np.sum(priority)), 1.0)
        covered_priority = float(np.sum(priority * coverage))
        f1 = 1.0 - covered_priority / total_priority

        # f2: cost fraction
        sr_keys = list(self.lib.sr.keys())
        total_cost = 0.0
        for k, key in enumerate(sr_keys):
            if k < x.shape[1]:
                cost_mode = self.lib.sr[key]["group_C_cost"]["cost_per_unit_usd"][1]
                total_cost += float(np.sum(x[:, k] * cost_mode))
        f2 = float(total_cost / max(self.budget, 1.0))

        # f3: Gini of (HRI * (1 - coverage)) — spatial inequity of residual risk
        residual = hri * (1.0 - coverage)
        pop_safe = np.maximum(pop, 1.0)
        rpc = residual / pop_safe
        valid = rpc[np.isfinite(rpc)]
        if len(valid) >= 2:
            s = np.sort(valid)
            n = len(s)
            idx = np.arange(1, n + 1)
            f3 = float(np.clip(
                (2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n, 0, 1
            ))
        else:
            f3 = 0.0

        out["F"] = [f1, f2, f3]

        # Constraints
        out["G"] = [
            float(total_cost - self.budget),   # C1: budget
            float(np.maximum(0, x[:, 3] - self.k_med).sum()),  # C3: med units
        ]


# ── Long-run problem ───────────────────────────────────────────────────────────

class LongRunProblem:
    """
    5-objective long-run heat intervention planning problem.

    Objectives: f1 mortality, f2 cost, f3 equity, f4 cobenefit, f5 robustness.
    Decision variables: x[zone, intervention] = quantity applied.

    Parameters
    ----------
    cell_states : dict
        Indicator arrays (1-D, zone-aggregated or cell-level).
    intervention_lib : InterventionLibrary
    scenarios : list
        From ScenarioManager.build_scenarios().
    hri_engine : HRIEngine
    budget : float
    max_quantities : np.ndarray
        Shape (n_zones, n_lr_interventions) — upper bounds.
    min_viable_units : list
        Minimum non-zero quantity per intervention.
    n_mc_samples : int
        Monte Carlo samples per evaluation (default 50 for speed).
    """

    def __init__(
        self,
        cell_states: dict,
        intervention_lib,
        scenarios: list,
        hri_engine,
        budget: float,
        max_quantities: np.ndarray,
        min_viable_units: Optional[List[float]] = None,
        n_mc_samples: int = 50,
    ) -> None:
        self.cs         = cell_states
        self.lib        = intervention_lib
        self.scenarios  = scenarios
        self.hri        = hri_engine
        self.budget     = budget
        self.max_q      = max_quantities
        self.min_viable = min_viable_units or [1.0] * len(intervention_lib.lr)
        self.n_mc       = n_mc_samples

        n_zones    = max_quantities.shape[0]
        n_interv   = len(intervention_lib.lr)
        self.n_zones   = n_zones
        self.n_interv  = n_interv
        self.n_var     = n_zones * n_interv
        self.n_obj     = 5
        self.n_ieq_constr = 3
        self.xl        = np.zeros(self.n_var)
        self.xu        = max_quantities.flatten()

    def _evaluate(self, x_flat: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate all 5 objectives for one candidate solution."""
        x   = x_flat.reshape(self.n_zones, self.n_interv)
        rng = np.random.default_rng()

        f1_vals, f3_vals, f5_vals = [], [], []

        for _ in range(self.n_mc):
            eff     = self.lib.sample_effectiveness(rng=rng)
            state   = self.lib.compute_post_intervention_state(x, self.cs, eff)
            pop     = self.cs.get("population", np.ones(self.n_zones))
            lst_post= state.get("lst_post", self.cs.get("lst", self.cs.get("H_s", np.zeros(self.n_zones))))
            hvi_post= state.get("hvi_post", self.cs.get("HVI", np.zeros(self.n_zones)))

            ad_post = self.hri.compute_attributable_deaths(
                pop, lst_post, hvi_post,
                event_days=self.scenarios[0]["event_days"] if self.scenarios else 3,
            )
            ad_base = self.cs.get("ad_baseline", np.ones(self.n_zones) * 0.01)

            f1_vals.append(f1_mortality_reduction(ad_post, ad_base))
            f3_vals.append(f3_equity_gini(ad_post, pop))
            f5_vals.append(f5_scenario_robustness(x, self.scenarios, self.hri,
                                                   self.cs, self.lib, eff))

        f1 = float(np.mean(f1_vals))
        f2 = f2_cost_fraction(x, self.lib, self.budget)
        f3 = float(np.mean(f3_vals))
        f4 = f4_cobenefit_efficiency(x, self.lib, self.budget,
                                     CARBON_PRICE_USD_PER_TCO2,
                                     STORMWATER_VALUE_USD_PER_M3)
        f5 = float(np.mean(f5_vals))

        out["F"] = [f1, f2, f3, f4, f5]
        out["G"] = [
            c1_budget(x, self.lib, self.budget),
            c2_area_feasibility(x, self.max_q),
            c3_mutual_exclusivity(x, self.lib),
        ]
