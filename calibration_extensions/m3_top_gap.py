"""M3 - separate the top-1/top-2 gap from the remaining answers (spec section 5).

u_k = a z_k + (b - a) d 1[k = k*], written as u = a Z' + b D with fixed Z', D from p.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import numpy as np
from scipy.special import logsumexp

from .calibrators import FitResult, IdentityCalibrator, TopGap, top_gap_parts
from .numerics import TOL, ProbSet
from .source_objectives import calibrator_deltas, fixed_gate, solve_source_objective, verify_objective
from .source_splits import loso_folds


@dataclass
class M3Config:
    box: tuple = (0.25, 4.0)
    lambdas: tuple = (0.0, 0.01, 0.1)
    objective: str = "MINIMAX"           # MINIMAX | MEAN_SAFE | POOLED
    epsilon_select: float = 0.0


def effective_dof(ps: ProbSet) -> int:
    """K = 2 only depends on b, so a is fixed to 1 there (spec 5.2)."""
    return 1 if np.all(ps.K == 2) else 2


def top_gap_builder(g, theta):
    Zp, D, _, _ = top_gap_parts(g.logP)
    rows = np.arange(len(g.idx))
    U = theta[0] * Zp + theta[1] * D
    uy = theta[0] * Zp[rows, g.y] + theta[1] * D[rows, g.y]
    return U, uy


def fit_head(ps: ProbSet, objective: str, lam: float, cfg: M3Config) -> tuple[TopGap | None, dict]:
    lo, hi = cfg.box
    dof = effective_dof(ps)

    def cons(th):
        c = [th >= lo, th <= hi]
        return c + ([th[0] == 1.0] if dof == 1 else [])

    pen = (lambda th: lam * cp.square(th[0] - th[1])) if lam > 0 else (lambda th: 0)
    res = solve_source_objective(ps, 2, top_gap_builder, objective, cons, pen)
    info = {"solver_status": res["status"], "objective": objective, "lambda": lam, "effective_dof": dof}
    if not res["ok"]:
        return None, info | {"valid": False, "reason": "SOLVER_FAILURE"}
    a, b = np.clip(res["theta"], lo, hi)
    if dof == 1:
        a = 1.0
    cal = TopGap(a, b)
    deltas = calibrator_deltas(ps, cal)
    ok, why = verify_objective(objective, deltas, lam * (a - b) ** 2)
    return (cal if ok else None), info | {"valid": ok, "reason": why, "train_deltas": deltas}


def fit_m3(source_train: ProbSet, source_select: ProbSet, cfg: M3Config = M3Config()) -> FitResult:
    method_id = f"M3_{cfg.objective}"
    scores = []
    for k, lam in enumerate(cfg.lambdas):
        d = []
        for held, train_fold, val in loso_folds(source_train):
            cal, _ = fit_head(train_fold, cfg.objective, lam, cfg)
            d.append(np.inf if cal is None else calibrator_deltas(val, cal)[held])
        scores.append({"lambda": lam, "order": k, "max_heldout_delta": float(np.max(d)),
                       "mean_heldout_delta": float(np.mean(d)), "heldout_deltas": d})
    pool = scores
    for key in ("max_heldout_delta", "mean_heldout_delta"):
        best = min(s[key] for s in pool)
        pool = [s for s in pool if s[key] <= best + TOL]
    chosen = min(pool, key=lambda s: (-s["lambda"], s["order"]))  # stronger tie toward TS is simpler
    fitted, info = fit_head(source_train, cfg.objective, chosen["lambda"], cfg)
    hyper = {"lambda": chosen["lambda"], "objective": cfg.objective, "box": list(cfg.box)}
    diagnostics = {"loso": scores, "fit": info}
    if fitted is None:
        return FitResult(method_id, "source_only", IdentityCalibrator(), None, {}, hyper,
                         info["solver_status"], False, None, info["reason"], "observed_source_losses",
                         diagnostics=diagnostics)
    gate = fixed_gate(source_select, fitted, cfg.epsilon_select)
    lo, hi = cfg.box
    params = fitted.params() | {"effective_dof": info["effective_dof"],
                                "at_bound": [bool(min(v - lo, hi - v) < 1e-6) for v in (fitted.a, fitted.b)]}
    return FitResult(method_id, "source_only", fitted if gate["accept"] else IdentityCalibrator(), fitted,
                     params, hyper, info["solver_status"], True, gate,
                     None if gate["accept"] else "SOURCE_GATE_REJECTED", "observed_source_losses",
                     diagnostics=diagnostics)


# ---------------------------------------------------------------- mechanism metrics

def nll_decomposition(ps: ProbSet, mats: list[np.ndarray]) -> dict:
    """-log q_y = top-vs-all BCE + (1-C) * (-log r_y)  (spec 5.3), k* taken from p."""
    bce = np.empty(ps.n)
    tail = np.empty(ps.n)
    correct = np.empty(ps.n, dtype=bool)
    nll = np.empty(ps.n)
    for g, M in zip(ps.groups, mats):
        rows = np.arange(len(g.idx))
        top = np.argmax(g.logP, axis=1)
        C = g.y == top
        lq_top = M[rows, top]
        others = M.copy()
        others[rows, top] = -np.inf
        log_rest = logsumexp(others, axis=1)          # log(1 - q_top), stable
        bce[g.idx] = np.where(C, -lq_top, -log_rest)
        tail[g.idx] = np.where(C, 0.0, -(M[rows, g.y] - log_rest))
        correct[g.idx] = C
        nll[g.idx] = -M[rows, g.y]
    wrong = ~correct
    return {"top_bce": float(bce.mean()), "tail_contribution": float(tail.mean()),
            "tail_nll_wrong_only": float(tail[wrong].mean()) if wrong.any() else None,
            "n_wrong": int(wrong.sum()), "nll": float(nll.mean()),
            "decomposition_error": float(abs(bce.mean() + tail.mean() - nll.mean()))}


def rank_change_rate(ps: ProbSet, mats: list[np.ndarray], tie_tol: float = 1e-12, tol: float = 1e-9) -> float:
    """Fraction of rows where some pair flips order, or an exact tie in p is broken in q."""
    changed = 0
    for g, M in zip(ps.groups, mats):
        dp = g.logP[:, :, None] - g.logP[:, None, :]
        dq = M[:, :, None] - M[:, None, :]
        flipped = (dp * dq < 0) & (np.abs(dq) > tol)
        broken = (np.abs(dp) <= tie_tol) & (np.abs(dq) > tol)
        changed += int(np.any(flipped | broken, axis=(1, 2)).sum())
    return changed / max(ps.n, 1)


def config_dict(cfg: M3Config) -> dict:
    return asdict(cfg)
