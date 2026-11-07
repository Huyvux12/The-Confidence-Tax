"""Evolutionary solver for RB-Cal (EvoLearn extension).

The regret-budgeted problem with the clustered-SE margin,

    min_theta  sum_w pi_w D_w(theta) + lam * Omega(theta)
    s.t.       D_w(theta) + z * SE_w(theta) <= eps   for every source w,

is non-convex once z > 0 (the SE term), and the default solver (SLSQP from the identity) is local. Here the same
problem is solved with CMA-ES (Hansen & Ostermeier 2001; pycma): beta bounds are kept by a repair operator
(projection onto the feasible beta field), budget violations are penalised, and an elitist archive keeps the best
feasible point ever evaluated (the identity is always feasible, so the method never returns an infeasible field).
Every evaluated point is also logged as (pooled regret, worst-source bound) to trace the empirical Pareto front of
mean gain against per-source harm.
"""
from __future__ import annotations

import numpy as np

from .features import project_to_bounds
from .rb_calibration import BOUNDS, Spec, build
from .source_objectives import calibrator_deltas


def solve_cma(ps, spec: Spec, eps: float | None, z: float = 0.0, seed: int = 1, maxfevals: int = 3000,
              sigma0: float = 0.1, penalty: float = 100.0, log_front: bool = False):
    import cma
    oracle, x0, kind, fixed_a, make, fmap = build(ps, spec)
    if kind == "top_gap":
        raise NotImplementedError("CMA-ES solver covers the beta-field classes")
    lo, hi = BOUNDS
    D = len(x0)
    pad = D == 1                                   # pycma needs >= 2 dims: add an inert coordinate

    def repair(x):
        x = np.asarray(x[:D], dtype=float)
        if kind == "constant":
            return np.clip(x, lo, hi)
        return project_to_bounds(fmap, x, lo, hi)

    def pen(x):
        return float(x[1:] @ x[1:])

    def evaluate(x):
        th = repair(x)
        delta, _ = oracle(th)
        f1 = float(oracle.pi @ delta) + spec.lam * pen(th)
        if eps is None:
            return th, f1, 0.0, -np.inf
        bound = oracle.ucb(th, z)[0] if z > 0 else delta
        viol = np.maximum(bound - eps, 0.0)
        return th, f1, float(viol.sum()), float(bound.max())

    best = {"x": x0.copy(), "f1": evaluate(x0)[1]}  # identity: feasible, f1 = 0
    front = []
    start = np.r_[x0, 0.0] if pad else x0.copy()
    es = cma.CMAEvolutionStrategy(start, sigma0, {"seed": seed, "maxfevals": maxfevals, "verbose": -9,
                                                  "tolfun": 1e-11, "tolx": 1e-11})
    n_eval = 0
    while not es.stop():
        X = es.ask()
        fit = []
        for x in X:
            th, f1, viol, worst = evaluate(x)
            n_eval += 1
            fit.append(f1 + penalty * viol)
            if viol <= 0 and f1 < best["f1"]:
                best = {"x": th.copy(), "f1": f1}
            if log_front:
                front.append((f1, worst))
        es.tell(X, fit)
    cal = make(best["x"])
    deltas = calibrator_deltas(ps, cal)
    th = best["x"]
    worst_bound = float(np.max(oracle.ucb(th, z)[0])) if (eps is not None and z > 0) else max(deltas.values())
    ok = bool(np.all(np.isfinite(th))) and (eps is None or worst_bound <= eps + 1e-7)
    info = {"status": f"CMA-ES:{n_eval} evals", "valid": ok, "train_deltas": deltas,
            "train_worst": max(deltas.values()),
            "train_mean": float(oracle.pi @ np.array([deltas[s] for s in oracle.srcs])),
            "objective": best["f1"], "worst_bound": worst_bound, "theta": th.tolist(), "seed": seed}
    if log_front:
        info["front"] = front
    return (cal if ok else None), info
