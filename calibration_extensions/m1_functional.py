"""M1 - functional calibration with a source-conflict diagnostic (spec section 3)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from .calibrators import FieldBeta, FitResult, IdentityCalibrator
from .features import FeatureMap, beta_bound_slack, fit_feature_map, project_to_bounds
from .numerics import TOL, ProbSet, losses, surprisal
from .source_objectives import (beta_field_builder, calibrator_deltas, fixed_gate,
                                solve_problem, solve_source_objective, verify_objective)
from .source_splits import loso_folds


@dataclass
class M1Config:
    centers: tuple = (0, 1, 2, 4, 8)
    bandwidth: tuple = (0.2, 0.5, 1.0)
    field_l2: tuple = (0.0, 0.01, 0.1)
    beta_min: float = 0.25
    beta_max: float = 4.0
    objective: str = "MINIMAX"           # MINIMAX | MEAN_SAFE | POOLED
    function_class: str = "rbf"          # rbf | entropy_affine | constant
    epsilon_select: float = 0.0
    seed: int = 0
    solver: str = "auto"                 # auto (SLSQP, CVXPY fallback) | scipy | cvxpy
    source_tolerance: float = 0.0        # MEAN_SAFE variant: Delta_w <= tolerance (0 = spec)


def beta_constraints(fmap: FeatureMap, bmin: float, bmax: float):
    """Global bounds: beta(p) in [bmin, bmax] for every input, not only train (spec 3.3)."""
    def cons(th):
        if fmap.kind == "constant":
            return [th[0] >= bmin, th[0] <= bmax]
        if fmap.kind == "entropy_affine":
            return [th[0] >= bmin, th[0] <= bmax, th[0] + th[1] >= bmin, th[0] + th[1] <= bmax]
        nv = cp.norm(th[1:], 2)
        return [nv <= th[0] - bmin, nv <= bmax - th[0]]
    return cons


def field_penalty(fmap: FeatureMap, lam: float):
    return lambda th: lam * cp.sum_squares(th[1:]) if fmap.dim > 1 and lam > 0 else 0


class SourceLossOracle:
    """Per-source mean NLL change and its gradient for beta = Phi @ theta (numpy, spec 3.7)."""

    def __init__(self, ps: ProbSet, Phi: np.ndarray):
        self.srcs = ps.sources()
        keys = np.array([str(s) for s in ps.source_id])
        self.W = np.stack([(keys == s) / max((keys == s).sum(), 1) for s in self.srcs])   # (S, n)
        self.pi = np.array([(keys == s).sum() for s in self.srcs], dtype=float) / ps.n
        self.parts = [(g.idx, g.logP, g.logP[np.arange(len(g.idx)), g.y], Phi[g.idx]) for g in ps.groups]
        self.base = self.W @ surprisal(ps)
        self.n = ps.n

    def __call__(self, theta):
        loss = np.empty(self.n)
        dl = np.empty(self.n)
        for idx, logP, lpy, Ph in self.parts:
            beta = Ph @ theta
            U = beta[:, None] * logP
            m = U.max(axis=1, keepdims=True)
            e = np.exp(U - m)
            s = e.sum(axis=1, keepdims=True)
            loss[idx] = (m + np.log(s))[:, 0] - beta * lpy
            dl[idx] = (e * logP).sum(axis=1) / s[:, 0] - lpy
        delta = self.W @ loss - self.base
        grad = np.zeros((len(self.srcs), len(theta)))
        for idx, _, _, Ph in self.parts:
            grad += (self.W[:, idx] * dl[idx]) @ Ph
        return delta, grad


def solve_scipy(ps: ProbSet, fmap: FeatureMap, objective: str, lam: float, cfg: M1Config) -> dict:
    """SLSQP on the same convex program; a smooth form of the norm bound ((b0-bmin)^2 >= |v|^2, b0 >= bmin)."""
    oracle = SourceLossOracle(ps, fmap.phi(ps))
    D, S = fmap.dim, len(oracle.srcs)
    lo, hi = cfg.beta_min, cfg.beta_max
    minimax = objective == "MINIMAX"
    x0 = np.r_[fmap.identity_theta(), [0.0] if minimax else []]
    cache = {}

    def ev(x):
        k = x.tobytes()
        if k not in cache:
            cache.clear()
            cache[k] = oracle(x[:D])
        return cache[k]

    def fun(x):
        th = x[:D]
        pen = lam * float(th[1:] @ th[1:])
        return (x[D] if minimax else float(oracle.pi @ ev(x)[0])) + pen

    def jac(x):
        th = x[:D]
        g = np.zeros_like(x)
        g[1:D] = 2 * lam * th[1:]
        if minimax:
            g[D] = 1.0
        else:
            g[:D] += oracle.pi @ ev(x)[1]
        return g

    cons = []
    if objective in ("MINIMAX", "MEAN_SAFE"):
        def cf(x):
            d = ev(x)[0]
            return (x[D] - d) if minimax else cfg.source_tolerance - d

        def cj(x):
            J = np.zeros((S, len(x)))
            J[:, :D] = -ev(x)[1]
            if minimax:
                J[:, D] = 1.0
            return J
        cons.append({"type": "ineq", "fun": cf, "jac": cj})
    bounds = [(None, None)] * len(x0)
    if fmap.kind == "constant":
        bounds[0] = (lo, hi)
    elif fmap.kind == "entropy_affine":
        A = np.zeros((4, len(x0)))
        A[0, 0], A[1, 0] = 1, -1
        A[2, :2], A[3, :2] = 1, -1
        b = np.array([-lo, hi, -lo, hi])
        cons.append({"type": "ineq", "fun": lambda x: A @ x + b, "jac": lambda x: A})
    else:
        bounds[0] = (lo, hi)

        def nf(x):
            v = x[1:D]
            return np.array([(x[0] - lo) ** 2 - v @ v, (hi - x[0]) ** 2 - v @ v])

        def nj(x):
            J = np.zeros((2, len(x)))
            J[0, 0], J[1, 0] = 2 * (x[0] - lo), -2 * (hi - x[0])
            J[:, 1:D] = -2 * x[1:D]
            return J
        cons.append({"type": "ineq", "fun": nf, "jac": nj})
    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    return {"theta": res.x[:D], "status": f"SLSQP:{res.status}:{res.message}", "ok": bool(res.success and np.all(np.isfinite(res.x)))}


def fit_head(ps: ProbSet, fmap: FeatureMap, objective: str, lam: float, cfg: M1Config) -> tuple[FieldBeta | None, dict]:
    """solver 'auto': SLSQP first; if its verified result fails, re-solve with CVXPY (Clarabel)."""
    if cfg.solver in ("auto", "scipy"):
        cal, info = _finish(ps, fmap, objective, lam, cfg, solve_scipy(ps, fmap, objective, lam, cfg))
        if cal is not None or cfg.solver == "scipy" or cfg.source_tolerance:
            return cal, info
    res = solve_source_objective(ps, fmap.dim, beta_field_builder(fmap.phi(ps)), objective,
                                 beta_constraints(fmap, cfg.beta_min, cfg.beta_max), field_penalty(fmap, lam))
    return _finish(ps, fmap, objective, lam, cfg, res)


def _finish(ps, fmap, objective, lam, cfg, res) -> tuple[FieldBeta | None, dict]:
    info = {"solver_status": res["status"], "objective": objective, "lambda": lam}
    if not res["ok"]:
        return None, info | {"valid": False, "reason": "SOLVER_FAILURE"}
    theta = project_to_bounds(fmap, res["theta"], cfg.beta_min, cfg.beta_max)
    cal = FieldBeta(fmap, theta)
    deltas = calibrator_deltas(ps, cal)
    pen = lam * float(np.sum(theta[1:] ** 2))
    ok, why = verify_objective(objective, {k: v - cfg.source_tolerance for k, v in deltas.items()} if objective == "MEAN_SAFE" else deltas, pen)
    info |= {"valid": ok, "reason": why, "train_deltas": deltas, "penalty": pen,
             "projection_shift": float(np.max(np.abs(theta - res["theta"]))),
             "bound_slack": float(beta_bound_slack(fmap, theta, cfg.beta_min, cfg.beta_max)),
             "field_norm": float(np.linalg.norm(theta[1:]))}
    return (cal if ok else None), info


# ---------------------------------------------------------------- diagnostic at identity

def simplex_qp(G: np.ndarray, maxiter: int = 1000) -> tuple[np.ndarray, dict]:
    """min 1/2 a'Ga over the simplex, uniform start (spec 3.7). No ridge is added."""
    """Solved as min |F a|^2 with G = F'F (rows scaled to unit max norm) by Clarabel, then polished
    by solving the KKT system exactly on the support. SLSQP alone stops early when a'Ga is tiny."""
    W = G.shape[0]
    lam_, V = np.linalg.eigh(G)
    F = (np.sqrt(np.clip(lam_, 0.0, None))[:, None] * V.T)
    scale = max(float(np.sqrt(np.max(np.diag(G)))), 1e-300)
    a_var = cp.Variable(W, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares((F / scale) @ a_var)), [cp.sum(a_var) == 1])
    status = solve_problem(prob)
    a = np.clip(np.asarray(a_var.value if a_var.value is not None else np.full(W, 1.0 / W)), 0.0, None)
    a /= a.sum()
    supp = np.flatnonzero(a > 1e-8 * a.max())
    K = np.block([[G[np.ix_(supp, supp)], np.ones((len(supp), 1))], [np.ones((1, len(supp))), np.zeros((1, 1))]])
    sol = np.linalg.lstsq(K, np.r_[np.zeros(len(supp)), 1.0], rcond=None)[0][:len(supp)]
    polished = False
    if np.all(sol >= 0):
        cand = np.zeros(W)
        cand[supp] = sol
        if cand @ G @ cand <= a @ G @ a:
            a, polished = cand, True
    lam = a @ G @ a
    Ga = G @ a
    kkt = max(float(np.max(np.maximum(lam - Ga, 0.0))),
              float(np.max(np.abs(Ga[a > 1e-10] - lam))) if np.any(a > 1e-10) else 0.0)
    return a, {"converged": not status.startswith("FAILED"), "solver": status, "polished": polished,
               "iterations": maxiter if status.startswith("FAILED") else 0, "kkt_error": kkt,
               "kkt_error_relative": kkt / max(float(np.max(np.diag(G))), 1e-300),
               "sum_alpha_error": float(abs(a.sum() - 1.0)), "min_alpha": float(a.min())}


def diagnose_at_identity(ps: ProbSet, fmap: FeatureMap, cfg: M1Config, seed: int = 0) -> dict:
    """Common-descent geometry of the observed source losses for this function class only."""
    srcs = ps.sources()
    keys = np.array([str(s) for s in ps.source_id])
    Phi = fmap.phi(ps)
    ent = np.empty(ps.n)
    for g in ps.groups:
        ent[g.idx] = -(np.exp(g.logP) * g.logP).sum(axis=1)
    r = surprisal(ps) - ent
    gw = np.stack([(r[keys == s][:, None] * Phi[keys == s]).mean(axis=0) for s in srcs])
    alpha, qp = simplex_qp(gw @ gw.T)
    gbar = alpha @ gw
    rho = float(np.linalg.norm(gbar))
    rho_tol = 1e-6 * max(1.0, float(np.max(np.linalg.norm(gw, axis=1))))
    theta0 = fmap.identity_theta()
    out = {"function_class": fmap.kind, "n_centers": fmap.n_centers, "sigma": fmap.sigma,
           "sources": srcs, "alpha": alpha.tolist(), "g_w": gw.tolist(), "rho": rho, "rho_tol": rho_tol,
           "qp": qp, "scope": "observed_source_loss_geometry_for_this_function_class"}

    # analytic gradient vs central finite difference along a random unit direction
    rng = np.random.default_rng(seed)
    e = rng.normal(size=fmap.dim)
    e /= np.linalg.norm(e)
    h = 1e-5
    lp = calibrator_deltas(ps, FieldBeta(fmap, theta0 + h * e))
    lm = calibrator_deltas(ps, FieldBeta(fmap, theta0 - h * e))
    fd = np.array([(lp[s] - lm[s]) / (2 * h) for s in srcs])
    out["gradient_check_max_abs_error"] = float(np.max(np.abs(fd - gw @ e)))

    if rho <= rho_tol:
        out |= {"status": "NO_VERIFIED_COMMON_DESCENT", "common_descent": False,
                "g_w_dot_d": None, "step_check": None}
        return out
    d = -gbar / rho
    gdd = gw @ d
    step = None
    for eta in (1e-2, 1e-3, 1e-4, 1e-5):
        th = theta0 + eta * d
        if beta_bound_slack(fmap, th, cfg.beta_min, cfg.beta_max) < 0:
            continue
        deltas = calibrator_deltas(ps, FieldBeta(fmap, th))
        if all(v < 0 for v in deltas.values()):
            step = {"eta": eta, "deltas": deltas}
            break
    verified = bool(np.all(gdd < -TOL)) and step is not None
    out |= {"d": d.tolist(), "g_w_dot_d": gdd.tolist(), "step_check": step,
            "common_descent": verified,
            "status": "COMMON_DESCENT_VERIFIED" if verified else "NUMERICAL_UNRESOLVED"}
    return out


# ---------------------------------------------------------------- selection and fit

def build_candidates(cfg: M1Config) -> list[dict]:
    if cfg.function_class == "constant":
        return [{"kind": "constant", "J": 0, "sigma": None, "lam": 0.0, "order": 0}]
    if cfg.function_class == "entropy_affine":
        return [{"kind": "entropy_affine", "J": 0, "sigma": None, "lam": l, "order": k}
                for k, l in enumerate(cfg.field_l2)]
    out = []
    for J in cfg.centers:
        if J == 0:
            out.append({"kind": "constant", "J": 0, "sigma": None, "lam": 0.0})
            continue
        for s in cfg.bandwidth:
            for l in cfg.field_l2:
                out.append({"kind": "rbf", "J": J, "sigma": s, "lam": l})
    for k, c in enumerate(out):
        c["order"] = k
    return out


def loso_score(ps: ProbSet, cand: dict, cfg: M1Config, objective: str) -> dict:
    folds = []
    for held, train_fold, val in loso_folds(ps):
        fmap = fit_feature_map(train_fold, cand["kind"], cand["J"], cand["sigma"])
        cal, info = fit_head(train_fold, fmap, objective, cand["lam"], cfg)
        if cal is None:
            folds.append({"heldout": held, "delta": np.inf, "field_norm": np.inf, "reason": info["reason"]})
            continue
        folds.append({"heldout": held, "delta": calibrator_deltas(val, cal)[held], "field_norm": info["field_norm"]})
    d = np.array([f["delta"] for f in folds])
    return cand | {"max_heldout_delta": float(d.max()), "mean_heldout_delta": float(d.mean()),
                   "mean_field_norm": float(np.mean([f["field_norm"] for f in folds])), "folds": folds}


def select_candidate(scores: list[dict], tol: float = TOL) -> dict:
    """max heldout delta, then mean, then fewer centers, smaller field norm, fixed order (ties at 1e-8)."""
    pool = scores
    for key in ("max_heldout_delta", "mean_heldout_delta"):
        best = min(s[key] for s in pool)
        pool = [s for s in pool if s[key] <= best + tol]
    return min(pool, key=lambda s: (s["J"], round(s["mean_field_norm"] / tol), s["order"]))


def fit_m1(source_train: ProbSet, source_select: ProbSet, cfg: M1Config = M1Config(),
           diagnose_classes: bool = True) -> FitResult:
    if len(source_train.sources()) < 2:
        raise ValueError("M1 needs at least two train sources for LOSO")
    objective = cfg.objective
    method_id = f"M1_{objective}" + ("" if cfg.function_class == "rbf" else f"_{cfg.function_class.upper()}")
    scores = [loso_score(source_train, c, cfg, objective) for c in build_candidates(cfg)]
    chosen = select_candidate(scores)
    fmap = fit_feature_map(source_train, chosen["kind"], chosen["J"], chosen["sigma"])
    diagnostics = {"selected": diagnose_at_identity(source_train, fmap, cfg, cfg.seed)}
    if diagnose_classes:
        for kind, J, s in (("constant", 0, None), ("entropy_affine", 0, None),
                           ("rbf", max(cfg.centers), cfg.bandwidth[len(cfg.bandwidth) // 2])):
            diagnostics[f"class_{kind}"] = diagnose_at_identity(
                source_train, fit_feature_map(source_train, kind, J, s), cfg, cfg.seed)
    fitted, info = fit_head(source_train, fmap, objective, chosen["lam"], cfg)
    diagnostics["fit"] = info
    diagnostics["loso"] = [{k: v for k, v in s.items() if k != "folds"} | {"folds": s["folds"]} for s in scores]
    hyper = {k: chosen[k] for k in ("kind", "J", "sigma", "lam")} | {"objective": objective}
    if fitted is None:
        return FitResult(method_id, "source_only", IdentityCalibrator(), None, {}, hyper,
                         info["solver_status"], False, None, info["reason"],
                         "observed_source_loss_geometry_for_this_function_class",
                         fmap.to_dict(), diagnostics)
    gate = fixed_gate(source_select, fitted, cfg.epsilon_select)
    return FitResult(method_id, "source_only", fitted if gate["accept"] else IdentityCalibrator(), fitted,
                     fitted.params() | {"theta": fitted.theta.tolist()}, hyper, info["solver_status"], True,
                     gate, None if gate["accept"] else "SOURCE_GATE_REJECTED",
                     "observed_source_loss_geometry_for_this_function_class", fmap.to_dict(), diagnostics)


def config_dict(cfg: M1Config) -> dict:
    return asdict(cfg)
