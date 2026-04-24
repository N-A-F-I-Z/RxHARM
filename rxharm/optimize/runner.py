"""
rxharm/optimize/runner.py
==========================
Configures and runs NSGA-III using pymoo for both SR (3-obj) and LR (5-obj).

Functions:
    run_nsga3_long    — 5-objective long-run optimizer
    run_nsga3_short   — 3-objective short-run optimizer
    run_multi_seed    — convergence check across multiple seeds
    extract_strategic_solutions — pick 3 representative Pareto points

All GEE and pymoo imports are at function level so the module is importable
even when pymoo is not installed (it degrades gracefully).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from rxharm.config import (
    EPSILON_LR,
    NSGA3_N_GEN_LR,
    NSGA3_N_GEN_SR,
    NSGA3_POP_SIZE_LR,
    NSGA3_POP_SIZE_SR,
    RANDOM_SEEDS,
)


def run_nsga3_long(
    problem,
    n_gen: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> object:
    """
    Run NSGA-III for the 5-objective long-run problem.

    Parameters
    ----------
    problem : LongRunProblem
    n_gen : int, optional
        Override NSGA3_N_GEN_LR.
    seed : int
    verbose : bool

    Returns
    -------
    pymoo Result
        Access Pareto front: result.F, result.X
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.problem import Problem

    n_gen    = n_gen or NSGA3_N_GEN_LR
    ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=6)
    algo     = NSGA3(pop_size=NSGA3_POP_SIZE_LR, ref_dirs=ref_dirs)
    term     = get_termination("n_gen", n_gen)

    # Wrap our problem in a pymoo-compatible Problem if needed
    pym_problem = _wrap_problem(problem, n_obj=5)
    return minimize(pym_problem, algo, term, seed=seed, verbose=verbose)


def run_nsga3_short(
    problem,
    n_gen: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> object:
    """
    Run NSGA-III for the 3-objective short-run problem.

    Parameters
    ----------
    problem : ShortRunProblem
    n_gen : int, optional
        Override NSGA3_N_GEN_SR.
    seed : int
    verbose : bool

    Returns
    -------
    pymoo Result
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    n_gen    = n_gen or NSGA3_N_GEN_SR
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)
    algo     = NSGA3(pop_size=NSGA3_POP_SIZE_SR, ref_dirs=ref_dirs)
    pym_problem = _wrap_problem(problem, n_obj=3)
    return minimize(pym_problem, algo, get_termination("n_gen", n_gen),
                    seed=seed, verbose=verbose)


def run_multi_seed(
    problem,
    n_seeds: int = 3,
    mode: str = "long",
) -> List[object]:
    """
    Run optimizer with multiple random seeds to check Pareto convergence.

    Parameters
    ----------
    problem : LongRunProblem or ShortRunProblem
    n_seeds : int
    mode : str
        ``'long'`` or ``'short'``

    Returns
    -------
    list of pymoo Result
    """
    runner = run_nsga3_long if mode == "long" else run_nsga3_short
    return [runner(problem, seed=s, verbose=False)
            for s in RANDOM_SEEDS[:n_seeds]]


def extract_strategic_solutions(
    result,
    n_strategies: int = 3,
) -> Dict[str, dict]:
    """
    Extract 3 representative solutions from the Pareto front.

    1. ``'health_focused'``  — minimum f1 (best mortality reduction)
    2. ``'budget_focused'``  — minimum f2 (cheapest)
    3. ``'balanced'``        — closest to utopia point (normalised distance)

    Parameters
    ----------
    result : pymoo Result

    Returns
    -------
    dict
        ``{strategy: {'x': ndarray, 'F': ndarray}}``
    """
    F = result.F
    X = result.X

    if F is None or len(F) == 0:
        return {}

    # Health focused: min f1
    health_idx = int(np.argmin(F[:, 0]))
    # Budget focused: min f2
    budget_idx = int(np.argmin(F[:, 1]))
    # Balanced: min normalised distance to (0,…,0) utopia
    F_norm   = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    dist     = np.linalg.norm(F_norm, axis=1)
    balanced_idx = int(np.argmin(dist))

    return {
        "health_focused": {"x": X[health_idx], "F": F[health_idx]},
        "budget_focused": {"x": X[budget_idx], "F": F[budget_idx]},
        "balanced":       {"x": X[balanced_idx], "F": F[balanced_idx]},
    }


# ── pymoo problem wrapper ──────────────────────────────────────────────────────

def _wrap_problem(problem, n_obj: int):
    """Wrap our custom problem class in a pymoo Problem subclass."""
    from pymoo.core.problem import Problem

    class _WrappedProblem(Problem):
        def __init__(self, inner, n_objectives):
            xl = getattr(inner, "xl", np.zeros(getattr(inner, "n_var", 10)))
            xu = getattr(inner, "xu", np.ones(getattr(inner, "n_var", 10)))
            n_var    = getattr(inner, "n_var", len(xl))
            n_constr = getattr(inner, "n_ieq_constr", 0)
            super().__init__(
                n_var=n_var, n_obj=n_objectives,
                n_ieq_constr=n_constr,
                xl=xl, xu=xu, vtype=float,
            )
            self._inner = inner

        def _evaluate(self, X, out, *args, **kwargs):
            # X shape: (pop_size, n_var) — evaluate each row
            Fout = []
            Gout = []
            for xi in X:
                tmp = {}
                self._inner._evaluate(xi, tmp)
                Fout.append(tmp.get("F", [0.0] * self.n_obj))
                if "G" in tmp:
                    Gout.append(tmp["G"])
            out["F"] = np.array(Fout)
            if Gout:
                out["G"] = np.array(Gout)

    return _WrappedProblem(problem, n_obj)


def pareto_to_dataframe(result, objective_names: list = None) -> "pd.DataFrame":
    """
    Convert a pymoo Pareto front result to a pandas DataFrame.

    FIX v0.1.0: Pareto front was previously only accessible as a pymoo
    Result object, requiring Python + pymoo to read. This function exports
    to a standard DataFrame that can be saved as CSV, opened in R or Excel,
    and cited in papers as 'Supplementary Table'.

    Parameters
    ----------
    result : pymoo Result
        Output of run_nsga3_long() or run_nsga3_short().
    objective_names : list of str, optional
        Human-readable names for objective columns. Defaults to standard names
        for 5-obj (long-run) or 3-obj (short-run) problems.

    Returns
    -------
    pd.DataFrame
        One row per Pareto solution. Columns:
        - ``solution_id`` — index on Pareto front
        - Objective columns (f1_mortality_fraction, etc.)
        - ``mortality_reduction_pct`` — convenience column derived from f1
        - ``x_NNNN`` — decision variable columns
        - ``is_health_focused``, ``is_budget_focused``, ``is_balanced`` flags

    Example
    -------
    result = run_nsga3_long(problem)
    df = pareto_to_dataframe(result)
    df.to_csv('pareto_front.csv', index=False)
    """
    import pandas as pd

    if result is None or result.X is None or result.F is None:
        return pd.DataFrame()

    n_obj = result.F.shape[1]

    # Default objective names by problem size
    if objective_names is None:
        if n_obj == 5:
            objective_names = [
                "f1_mortality_fraction",
                "f2_cost_fraction",
                "f3_equity_gini",
                "f4_cobenefit_neg_efficiency",
                "f5_scenario_robustness",
            ]
        elif n_obj == 3:
            objective_names = [
                "f1_coverage_gap",
                "f2_cost_fraction",
                "f3_inequity_gini",
            ]
        else:
            objective_names = [f"f{i+1}" for i in range(n_obj)]

    # Build DataFrame from objectives
    rows = []
    for i in range(result.F.shape[0]):
        row = {"solution_id": i}
        for j, name in enumerate(objective_names):
            row[name] = float(result.F[i, j])
        # FIX v0.1.0: Add convenience mortality-reduction column
        if "f1_mortality_fraction" in row:
            row["mortality_reduction_pct"] = (1.0 - row["f1_mortality_fraction"]) * 100.0
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add decision variable columns
    for v in range(result.X.shape[1]):
        df[f"x_{v:04d}"] = result.X[:, v]

    # FIX v0.1.0: Flag strategic solutions so the CSV can be filtered directly
    F_norm = (result.F - result.F.min(axis=0)) / (
        (result.F.max(axis=0) - result.F.min(axis=0)) + 1e-9
    )
    dist = np.linalg.norm(F_norm, axis=1)

    df["is_health_focused"] = False
    df["is_budget_focused"]  = False
    df["is_balanced"]        = False
    df.loc[int(result.F[:, 0].argmin()), "is_health_focused"] = True
    df.loc[int(result.F[:, 1].argmin()), "is_budget_focused"]  = True
    df.loc[int(dist.argmin()),            "is_balanced"]        = True

    return df


def save_pareto_to_csv(result, filepath: str, objective_names: list = None) -> str:
    """
    Save the Pareto front to a CSV file.

    FIX v0.1.0: Convenience wrapper around pareto_to_dataframe() that
    saves directly to file and returns the filepath for confirmation.

    Parameters
    ----------
    result : pymoo Result
    filepath : str
        Full path including filename.

    Returns
    -------
    str
        The filepath that was written.

    Example
    -------
    save_pareto_to_csv(result, '/content/drive/MyDrive/outputs/pareto.csv')
    """
    df = pareto_to_dataframe(result, objective_names)
    df.to_csv(filepath, index=False)
    print(f"Pareto front saved: {len(df)} solutions → {filepath}")
    return filepath
