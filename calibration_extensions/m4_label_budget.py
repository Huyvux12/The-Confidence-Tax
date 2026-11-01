"""M4 - minimax NLL change under a state budget of source label noise (spec section 6).

For a tempered calibrator delta_i(k) - delta_i(y~) = (1 - beta_i)(log p_ik - log p_iy~), so the
adversary's per-row gain is a max of affine functions of theta and Top-B keeps the problem convex.
The budget rho is an assumption reported as a sensitivity curve, never estimated from target.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cvxpy as cp
import numpy as np

from .calibrators import Calibrator, FieldBeta, FitResult, IdentityCalibrator, ScalarTS
from .features import beta_bound_slack, fit_feature_map, project_to_bounds
from .m1_functional import beta_constraints
from .numerics import TOL, ProbSet, losses, surprisal
from .source_objectives import argmin_near_one, beta_grid, cvx_source_losses, beta_field_builder, solve_problem


@dataclass
class M4Config:
    rho: float = 0.10
    function_class: str = "scalar"       # scalar | field
    grid_lo: float = 0.25
    grid_hi: float = 4.0
    grid_n: int = 241
    J: int = 4                            # field: take M1's selected (J, sigma, lam) when available
    sigma: float = 0.5
    lam: float = 0.01
    beta_min: float = 0.25
    beta_max: float = 4.0


def state_budget(n_states: int, rho: float) -> int:
    return min(n_states, math.ceil(rho * n_states - 1e-9))


def _source_state_index(ps: ProbSet):
    srcs = ps.sources()
    keys = np.array([str(s) for s in ps.source_id])
    out = {}
    for s in srcs:
        rows = np.flatnonzero(keys == s)
        cases = sorted({str(ps.case_id[i]) for i in rows})
        pos = {c: k for k, c in enumerate(cases)}
        out[s] = (rows, cases, np.array([pos[str(ps.case_id[i])] for i in rows]))
    return out


def adversary_row_gain(ps: ProbSet, mats: list[np.ndarray]) -> np.ndarray:
    """max_k delta_i(k) - delta_i(y~) >= 0 for every row."""
    gain = np.empty(ps.n)
    for g, M in zip(ps.groups, mats):
        delta = -M + g.logP
        rows = np.arange(len(g.idx))
        gain[g.idx] = delta.max(axis=1) - delta[rows, g.y]
    return gain


def robust_risk(ps: ProbSet, cal: Calibrator, rho: float, top_cases: int = 10) -> dict[str, dict]:
    """Exact worst case R_w^rho for the state-budget noise set (spec 6.3)."""
    mats = cal.log_q_groups(ps)
    obs = losses(ps, mats) - surprisal(ps)
    gain = adversary_row_gain(ps, mats)
    out = {}
    for s, (rows, cases, cidx) in _source_state_index(ps).items():
        A = np.bincount(cidx, weights=gain[rows], minlength=len(cases))
        B = state_budget(len(cases), rho)
        top = np.argsort(-A, kind="stable")[:B]
        r_obs = float(obs[rows].mean())
        inc = float(A[top].sum() / len(rows))
        out[s] = {"n_states": len(cases), "n_rows": len(rows), "rho": rho, "B": B, "R_obs": r_obs,
                  "adversarial_increase": inc, "R_rho": r_obs + inc,
                  "dominating_cases": [cases[k] for k in top[:top_cases]]}
    return out


def fit_scalar(ps: ProbSet, cfg: M4Config) -> tuple[ScalarTS, dict]:
    grid = beta_grid(cfg.grid_lo, cfg.grid_hi, cfg.grid_n)
    worst = np.array([max(v["R_rho"] for v in robust_risk(ps, ScalarTS(b), cfg.rho).values()) for b in grid])
    j = argmin_near_one(worst, grid)
    return ScalarTS(grid[j]), {"solver_status": "grid", "valid": True, "worst_R": float(worst[j])}


def fit_field(ps: ProbSet, cfg: M4Config) -> tuple[FieldBeta | None, dict]:
    fmap = fit_feature_map(ps, "rbf", cfg.J, cfg.sigma)
    Phi = fmap.phi(ps)
    theta = cp.Variable(fmap.dim)
    srcs, L, L0, _ = cvx_source_losses(ps, theta, beta_field_builder(Phi))
    index = _source_state_index(ps)
    row_pos = np.empty(ps.n, dtype=int)
    for g in ps.groups:
        row_pos[g.idx] = np.arange(len(g.idx))
    gains = []
    for g in ps.groups:
        n, K = g.logP.shape
        beta = Phi[g.idx] @ theta
        C = g.logP - g.logP[np.arange(n), g.y][:, None]
        gains.append(cp.max(cp.multiply(cp.reshape(1 - beta, (n, 1), order="C") @ np.ones((1, K)), C), axis=1))
    group_of = np.empty(ps.n, dtype=int)
    for k, g in enumerate(ps.groups):
        group_of[g.idx] = k
    t = cp.Variable()
    cons = list(beta_constraints(fmap, cfg.beta_min, cfg.beta_max)(theta))
    for w, s in enumerate(srcs):
        rows, cases, cidx = index[s]
        A = 0
        for k, g in enumerate(ps.groups):
            sel = group_of[rows] == k
            if not sel.any():
                continue
            M = np.zeros((len(cases), len(g.idx)))
            M[cidx[sel], row_pos[rows[sel]]] = 1.0
            A = A + M @ gains[k]
        B = state_budget(len(cases), cfg.rho)
        adv = cp.sum_largest(A, B) / len(rows) if B > 0 else 0
        cons.append(L[w] - L0[w] + adv <= t)
    pen = cfg.lam * cp.sum_squares(theta[1:]) if cfg.lam > 0 else 0
    prob = cp.Problem(cp.Minimize(t + pen), cons)
    status = solve_problem(prob)
    if status.startswith("FAILED") or theta.value is None:
        return None, {"solver_status": status, "valid": False, "reason": "SOLVER_FAILURE"}
    th = project_to_bounds(fmap, np.array(theta.value), cfg.beta_min, cfg.beta_max)
    cal = FieldBeta(fmap, th)
    worst = max(v["R_rho"] for v in robust_risk(ps, cal, cfg.rho).values())
    pen_v = cfg.lam * float(np.sum(th[1:] ** 2))
    ok = worst + pen_v <= 1e-7
    return (cal if ok else None), {"solver_status": status, "valid": ok, "worst_R": worst,
                                   "reason": "ok" if ok else f"robust objective {worst + pen_v:.3g} > 0",
                                   "bound_slack": beta_bound_slack(fmap, th, cfg.beta_min, cfg.beta_max)}


def robust_gate(ps_select: ProbSet, cal: Calibrator, rho: float, tol: float = TOL) -> dict:
    """Recompute R_w^rho on select with the budget implied by select's own state counts."""
    risk = robust_risk(ps_select, cal, rho)
    worst = max(v["R_rho"] for v in risk.values())
    return {"accept": bool(worst <= tol), "max_R_rho": worst, "risk": risk,
            "scope": "finite_source_select_check_under_assumed_state_budget"}


def fit_m4(source_train: ProbSet, source_select: ProbSet, cfg: M4Config = M4Config()) -> FitResult:
    method_id = f"M4_{cfg.function_class.upper()}_rho{cfg.rho:.2f}"
    fitted, info = (fit_scalar if cfg.function_class == "scalar" else fit_field)(source_train, cfg)
    hyper = {"rho_assumed": cfg.rho, "function_class": cfg.function_class}
    if cfg.function_class == "field":
        hyper |= {"J": cfg.J, "sigma": cfg.sigma, "lam": cfg.lam}
    diag = {"fit": info}
    if fitted is None:
        return FitResult(method_id, "source_only", IdentityCalibrator(), None, {}, hyper,
                         info["solver_status"], False, None, info["reason"],
                         "assumed_state_budget_noise_set", diagnostics=diag)
    diag["train_risk"] = robust_risk(source_train, fitted, cfg.rho)
    gate = robust_gate(source_select, fitted, cfg.rho)
    return FitResult(method_id, "source_only", fitted if gate["accept"] else IdentityCalibrator(), fitted,
                     fitted.params(), hyper, info["solver_status"], True, gate,
                     None if gate["accept"] else "SOURCE_GATE_REJECTED",
                     "assumed_state_budget_noise_set", diagnostics=diag)


def adversary_budget_rows(method_id: str, risk: dict[str, dict], split: str) -> list[dict]:
    return [{"method_id": method_id, "split": split, "source_id": s,
             **{k: v for k, v in r.items() if k != "dominating_cases"},
             "dominating_cases": ";".join(r["dominating_cases"])} for s, r in risk.items()]


def config_dict(cfg: M4Config) -> dict:
    return asdict(cfg)
