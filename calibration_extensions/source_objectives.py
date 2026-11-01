"""Source losses, shared objectives, scalar baselines and the fixed select gate."""
from __future__ import annotations

import warnings

import numpy as np
import cvxpy as cp
from scipy.special import logsumexp

from .calibrators import Calibrator, IdentityCalibrator, ScalarTS
from .numerics import TOL, ProbSet, losses, source_deltas, surprisal

OBJECTIVES = ("MINIMAX", "MEAN_SAFE", "POOLED")


def beta_grid(lo: float = 0.25, hi: float = 4.0, n: int = 241) -> np.ndarray:
    g = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    return np.unique(np.append(g, 1.0))


def argmin_near_one(values: np.ndarray, betas: np.ndarray, tol: float = TOL) -> int:
    """Smallest value; numerical ties (1e-8) go to the beta closest to 1 in log."""
    best = values.min()
    cand = np.flatnonzero(values <= best + tol)
    return int(cand[np.argmin(np.abs(np.log(betas[cand])))])


def scalar_losses(ps: ProbSet, beta: float) -> np.ndarray:
    out = np.empty(ps.n)
    for g in ps.groups:
        U = beta * g.logP
        out[g.idx] = logsumexp(U, axis=1) - U[np.arange(len(g.idx)), g.y]
    return out


def scalar_delta_table(ps: ProbSet, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-source Delta_w(beta) for every grid beta, plus the pooled mean delta."""
    srcs = ps.sources()
    base = surprisal(ps)
    keys = np.array([str(s) for s in ps.source_id])
    masks = [keys == s for s in srcs]
    table = np.empty((len(grid), len(srcs)))
    pooled = np.empty(len(grid))
    for j, b in enumerate(grid):
        d = scalar_losses(ps, b) - base
        table[j] = [d[m].mean() for m in masks]
        pooled[j] = d.mean()
    return table, pooled, srcs


def fit_scalar(ps: ProbSet, grid: np.ndarray, rule: str) -> tuple[ScalarTS, dict]:
    """rule='pooled': pooled TS. rule='minimax': MRTS (min over beta of max source delta)."""
    table, pooled, srcs = scalar_delta_table(ps, grid)
    score = pooled if rule == "pooled" else table.max(axis=1)
    j = argmin_near_one(score, grid)
    return ScalarTS(grid[j]), {"rule": rule, "beta": float(grid[j]), "score": float(score[j]),
                               "train_deltas": dict(zip(srcs, table[j].tolist()))}


def calibrator_deltas(ps: ProbSet, cal: Calibrator) -> dict[str, float]:
    return source_deltas(ps, losses(ps, cal.log_q_groups(ps)))


def fixed_gate(ps_select: ProbSet, cal: Calibrator, eps: float = 0.0, tol: float = TOL) -> dict:
    """Accept iff every select source has Delta_w <= eps (+tol). Not a target guarantee."""
    deltas = calibrator_deltas(ps_select, cal)
    worst = max(deltas.values())
    return {"accept": bool(worst <= eps + tol), "max_delta": worst, "deltas": deltas,
            "epsilon_select": eps, "scope": "finite_source_select_check"}


def gated(cal: Calibrator, gate: dict) -> Calibrator:
    return cal if gate["accept"] else IdentityCalibrator()


# ---------------------------------------------------------------- convex programs

def beta_field_builder(Phi: np.ndarray):
    """u = beta(p) * log p with beta = Phi @ theta."""
    def build(g, theta):
        n, K = g.logP.shape
        beta = Phi[g.idx] @ theta
        U = cp.multiply(cp.reshape(beta, (n, 1), order="C") @ np.ones((1, K)), g.logP)
        uy = cp.multiply(beta, g.logP[np.arange(n), g.y])
        return U, uy
    return build


def cvx_source_losses(ps: ProbSet, theta, u_builder) -> tuple[list[str], object, np.ndarray, np.ndarray]:
    """Return (sources, L_w expression vector, L_w(identity), source weights pi_w = n_w/n)."""
    srcs = ps.sources()
    pos = {s: k for k, s in enumerate(srcs)}
    keys = np.array([pos[str(s)] for s in ps.source_id])
    counts = np.bincount(keys, minlength=len(srcs)).astype(float)
    L = 0
    for g in ps.groups:
        U, uy = u_builder(g, theta)
        loss_g = cp.log_sum_exp(U, axis=1) - uy
        A = np.zeros((len(srcs), len(g.idx)))
        A[keys[g.idx], np.arange(len(g.idx))] = 1.0 / counts[keys[g.idx]]
        L = L + A @ loss_g
    base = surprisal(ps)
    L0 = np.array([base[keys == k].mean() for k in range(len(srcs))])
    return srcs, L, L0, counts / counts.sum()


def solve_problem(prob: cp.Problem) -> str:
    for solver in ("CLARABEL", "SCS"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # inaccuracy is reported via status
                prob.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return f"{solver}:{prob.status}"
    return f"FAILED:{prob.status}"


def solve_source_objective(ps: ProbSet, n_params: int, u_builder, objective: str,
                           constraints_fn, penalty_fn, extra_constraints=None) -> dict:
    """MINIMAX: min t + pen s.t. Delta_w <= t. MEAN_SAFE: min sum pi L + pen s.t. Delta_w <= 0.
    POOLED: min sum pi L + pen (no per-source constraint)."""
    theta = cp.Variable(n_params)
    srcs, L, L0, pi = cvx_source_losses(ps, theta, u_builder)
    delta = L - L0
    cons = list(constraints_fn(theta))
    if extra_constraints is not None:
        cons += list(extra_constraints(theta))
    pen = penalty_fn(theta)
    if objective == "MINIMAX":
        t = cp.Variable()
        prob = cp.Problem(cp.Minimize(t + pen), cons + [delta <= t])
    elif objective == "MEAN_SAFE":
        prob = cp.Problem(cp.Minimize(pi @ delta + pen), cons + [delta <= 0])
    elif objective == "POOLED":
        prob = cp.Problem(cp.Minimize(pi @ delta + pen), cons)
    else:
        raise ValueError(objective)
    status = solve_problem(prob)
    ok = not status.startswith("FAILED") and theta.value is not None
    return {"theta": np.array(theta.value) if ok else None, "status": status, "ok": ok,
            "objective_value": float(prob.value) if ok else None, "sources": srcs}


def verify_objective(objective: str, train_deltas: dict[str, float], penalty: float,
                     tol: float = 1e-7) -> tuple[bool, str]:
    """Numerical acceptance of a solver result (spec 3.6, 3.7)."""
    worst = max(train_deltas.values())
    if objective == "MINIMAX" and worst + penalty > tol:
        return False, f"minimax objective {worst + penalty:.3g} > 0"
    if objective == "MEAN_SAFE" and worst > tol:
        return False, f"source constraint violated by {worst:.3g}"
    return True, "ok"
