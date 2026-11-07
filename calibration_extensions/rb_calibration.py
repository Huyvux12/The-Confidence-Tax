"""RB-Cal: regret-budgeted multi-source calibration (see PREREGISTRATION.md).

    min_theta sum_w pi_w Delta_w(theta) + lam * Omega(theta)   s.t. Delta_w(theta) <= eps for all w

Every class here has logits linear in theta, u_i = sum_j theta_j B_ij, so NLL is convex and the
gradient is sum_k q_k B_k - B_y. `eps=None` drops the budget (pooled, same capacity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .calibrators import Calibrator, FieldBeta, FitResult, IdentityCalibrator, ScalarTS, TopGap, top_gap_parts
from .features import FeatureMap, beta_bound_slack, fit_feature_map, project_to_bounds
from .numerics import TOL, ProbSet, surprisal
from .source_objectives import beta_grid, calibrator_deltas, fit_scalar
from .source_splits import loso_folds

RBF_MENU = [("constant", 0, None, 0.0)] + [("rbf", J, s, l) for J in (1, 2, 4, 8)
                                           for s in (0.2, 0.5, 1.0) for l in (0.0, 0.01, 0.1)]
TOPGAP_MENU = [("top_gap", 0, None, l) for l in (0.0, 0.01, 0.1)]
# phase 8: same menu, RBF on z extended with answer-order consistency (features.py)
RBF_OC_MENU = [("constant", 0, None, 0.0)] + [("rbf_oc", J, s, l) for (_, J, s, l) in RBF_MENU[1:]]
RBF_OCT_MENU = [("constant", 0, None, 0.0)] + [("rbf_oct", J, s, l) for (_, J, s, l) in RBF_MENU[1:]]
RBF_OCP_MENU = [("constant", 0, None, 0.0)] + [("rbf_ocp", J, s, l) for (_, J, s, l) in RBF_MENU[1:]]
RBF_OCP1_MENU = [("constant", 0, None, 0.0)] + [("rbf_ocp1", J, s, l) for (_, J, s, l) in RBF_MENU[1:]]
CONSTANT_MENU = [("constant", 0, None, 0.0)]
BOUNDS = (0.25, 4.0)


class LinearLogitOracle:
    def __init__(self, ps: ProbSet, basis):
        self.srcs = ps.sources()
        keys = np.array([str(s) for s in ps.source_id])
        self.W = np.stack([(keys == s) / (keys == s).sum() for s in self.srcs])
        self.pi = np.array([(keys == s).sum() for s in self.srcs], dtype=float) / ps.n
        self.parts = []
        for g in ps.groups:
            B = basis(g)                                        # (n_g, D, K)
            self.parts.append((g.idx, B, B[np.arange(len(g.idx)), :, g.y]))
        self.surprisal = surprisal(ps)
        self.base = self.W @ self.surprisal
        self.n, self.D = ps.n, self.parts[0][1].shape[1]
        self.cases = []          # per source: rows, case index, rows per case, rows grouped by case, group starts
        for s in self.srcs:
            rows = np.flatnonzero(keys == s)
            _, cidx, counts = np.unique(np.array([str(c) for c in ps.case_id[rows]]),
                                        return_inverse=True, return_counts=True)
            order = np.argsort(cidx, kind="stable")              # keeps row order inside a case, as add.at
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            self.cases.append((rows, cidx, counts.astype(float), rows[order], starts))
        self._last = (None, None)

    def rows(self, theta):
        """Per-row loss and d loss / d theta; the last result is reused (objective and UCB share x)."""
        key = np.asarray(theta, dtype=float).tobytes()
        if self._last[0] != key:
            self._last = (key, self._rows(theta))
        return self._last[1]

    def _rows(self, theta):
        loss = np.empty(self.n)
        dl = np.empty((self.n, self.D))
        for idx, B, By in self.parts:
            U = np.einsum("ndk,d->nk", B, theta)
            m = U.max(axis=1, keepdims=True)
            e = np.exp(U - m)
            s = e.sum(axis=1, keepdims=True)
            q = e / s
            loss[idx] = (m + np.log(s))[:, 0] - By @ theta
            dl[idx] = np.einsum("nk,ndk->nd", q, B) - By
        return loss, dl

    def __call__(self, theta):
        loss, dl = self.rows(theta)
        return self.W @ loss - self.base, self.W @ dl

    def ucb(self, theta, z: float):
        """Delta_w + z * case-clustered SE_w and its gradient (SE smoothed by 1e-12 under the root)."""
        loss, dl = self.rows(theta)
        delta = loss - self.surprisal
        val = np.empty(len(self.srcs))
        jac = np.empty((len(self.srcs), self.D))
        for w, (rows, cidx, cnt, grouped, starts) in enumerate(self.cases):
            m, nw = len(cnt), len(rows)
            s = np.bincount(cidx, weights=delta[rows], minlength=m)
            ds = np.add.reduceat(dl[grouped], starts, axis=0)
            mean, dmean = s.sum() / nw, ds.sum(axis=0) / nw
            e = s - cnt * mean
            de = ds - cnt[:, None] * dmean
            c = m / max(m - 1, 1) / nw ** 2
            var = c * float(e @ e)
            se = np.sqrt(var + 1e-12)
            val[w] = mean + z * se
            jac[w] = dmean + z * (c * 2 * (e @ de)) / (2 * se)
        return val, jac


@dataclass
class Spec:
    kind: str                 # constant | entropy_affine | rbf | top_gap
    J: int
    sigma: float | None
    lam: float


def build(ps: ProbSet, spec: Spec):
    """Return (oracle, identity theta, constraint list builder, calibrator factory, penalty mask)."""
    lo, hi = BOUNDS
    if spec.kind == "top_gap":
        def basis(g):
            Zp, D, _, _ = top_gap_parts(g.logP)
            return np.stack([Zp, D], axis=1)
        fixed_a = bool(np.all(ps.K == 2))
        return (LinearLogitOracle(ps, basis), np.array([1.0, 1.0]), "top_gap", fixed_a,
                lambda th: TopGap(1.0 if fixed_a else th[0], th[1]), None)
    fmap = fit_feature_map(ps, spec.kind, spec.J, spec.sigma)
    Phi = fmap.phi(ps)
    basis = lambda g: Phi[g.idx][:, :, None] * g.logP[:, None, :]
    return (LinearLogitOracle(ps, basis), fmap.identity_theta(), fmap.kind, False,
            lambda th: FieldBeta(fmap, project_to_bounds(fmap, th, lo, hi)), fmap)


def solve(ps: ProbSet, spec: Spec, eps: float | None, z: float = 0.0) -> tuple[Calibrator | None, dict]:
    """z > 0 budgets the upper confidence bound Delta_w + z * SE_w instead of Delta_w."""
    oracle, x0, kind, fixed_a, make, fmap = build(ps, spec)
    lo, hi = BOUNDS
    D = len(x0)
    cache = {}

    def ev(x):
        k = x.tobytes()
        if k not in cache:
            cache.clear()
            cache[k] = oracle(x)
        return cache[k]

    def pen(x):
        return (x[0] - x[1]) ** 2 if kind == "top_gap" else float(x[1:] @ x[1:])

    def pen_grad(x):
        g = np.zeros(D)
        if kind == "top_gap":
            g[0], g[1] = 2 * (x[0] - x[1]), -2 * (x[0] - x[1])
        else:
            g[1:] = 2 * x[1:]
        return g

    fun = lambda x: float(oracle.pi @ ev(x)[0]) + spec.lam * pen(x)
    jac = lambda x: oracle.pi @ ev(x)[1] + spec.lam * pen_grad(x)
    cons, bounds = [], [(None, None)] * D
    if eps is not None and z > 0:
        ucache = {}

        def uv(x):
            k = x.tobytes()
            if k not in ucache:
                ucache.clear()
                ucache[k] = oracle.ucb(x, z)
            return ucache[k]
        cons.append({"type": "ineq", "fun": lambda x: eps - uv(x)[0], "jac": lambda x: -uv(x)[1]})
    elif eps is not None:
        cons.append({"type": "ineq", "fun": lambda x: eps - ev(x)[0], "jac": lambda x: -ev(x)[1]})
    if kind == "top_gap":
        bounds = [(1.0, 1.0) if fixed_a else (lo, hi), (lo, hi)]
    elif kind == "constant":
        bounds = [(lo, hi)]
    elif kind == "entropy_affine":
        A = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        b = np.array([-lo, hi, -lo, hi])
        cons.append({"type": "ineq", "fun": lambda x: A @ x + b, "jac": lambda x: A})
    else:
        bounds[0] = (lo, hi)
        cons.append({"type": "ineq",
                     "fun": lambda x: np.array([(x[0] - lo) ** 2 - x[1:] @ x[1:], (hi - x[0]) ** 2 - x[1:] @ x[1:]]),
                     "jac": lambda x: np.vstack([np.r_[2 * (x[0] - lo), -2 * x[1:]], np.r_[-2 * (hi - x[0]), -2 * x[1:]]])})
    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    cal = make(res.x)
    deltas = calibrator_deltas(ps, cal)
    worst = max(deltas.values())
    bound = worst if z == 0 else float(np.max(oracle.ucb(res.x, z)[0]))
    ok = bool(np.all(np.isfinite(res.x))) and (eps is None or bound <= eps + 1e-7) and \
        (fun(res.x) <= fun(x0) + 1e-9)
    info = {"status": f"SLSQP:{res.status}", "valid": ok, "train_deltas": deltas, "train_worst": worst,
            "train_mean": float(oracle.pi @ np.array([deltas[s] for s in oracle.srcs])), "theta": res.x.tolist()}
    if fmap is not None:
        info["bound_slack"] = float(beta_bound_slack(fmap, cal.theta, lo, hi))
    return (cal if ok else None), info


def complexity(spec: Spec) -> tuple:
    return (spec.J, -spec.lam)


def select_by_loso(ps: ProbSet, menu: list[Spec], eps: float | None, rank_eps: float,
                   z: float = 0.0) -> tuple[Spec, list[dict]]:
    """Rank by held-out violations (Delta > rank_eps), then mean held-out Delta, then simplicity."""
    scores = []
    for k, spec in enumerate(menu):
        d = []
        for held, trf, val in loso_folds(ps):
            cal, _ = solve(trf, spec, eps, z)
            d.append(np.inf if cal is None else calibrator_deltas(val, cal)[held])
        d = np.array(d)
        scores.append({"spec": spec, "order": k, "violations": int(np.sum(d > rank_eps + TOL)),
                       "mean_heldout": float(np.mean(d)), "max_heldout": float(np.max(d))})
    pool = [s for s in scores if s["violations"] == min(x["violations"] for x in scores)]
    best = min(s["mean_heldout"] for s in pool)
    pool = [s for s in pool if s["mean_heldout"] <= best + TOL]
    chosen = min(pool, key=lambda s: (complexity(s["spec"]), s["order"]))
    return chosen["spec"], [{k: (vars(v) if k == "spec" else v) for k, v in s.items()} for s in scores]


def fit_rbcal(train: ProbSet, select: ProbSet, menu, eps: float | None, method_id: str,
              rank_eps: float = 0.01, z: float = 0.0) -> FitResult:
    """eps=None: same class and selection rule, no budget (pooled control); the gate then uses rank_eps."""
    specs = [Spec(*m) for m in menu]
    spec = specs[0] if len(specs) == 1 else select_by_loso(train, specs, eps, rank_eps, z)[0]
    cal, info = solve(train, spec, eps, z)
    hyper = vars(spec) | {"epsilon": eps, "z": z}
    if cal is None:
        return FitResult(method_id, "source_only", IdentityCalibrator(), None, {}, hyper, info["status"], False,
                         None, "SOLVER_FAILURE", "observed_source_losses", diagnostics={"fit": info})
    gate_eps = rank_eps if eps is None else eps
    sel = calibrator_deltas(select, cal)
    accept = max(sel.values()) <= gate_eps + TOL
    gate = {"accept": accept, "max_delta": max(sel.values()), "deltas": sel, "epsilon": gate_eps}
    return FitResult(method_id, "source_only", cal if accept else IdentityCalibrator(), cal, cal.params(), hyper,
                     info["status"], True, gate, None if accept else "SOURCE_GATE_REJECTED",
                     "observed_source_losses", diagnostics={"fit": info})


def fit_shrinkage_loso(train: ProbSet, select: ProbSet, rank_eps: float = 0.01) -> FitResult:
    """beta = 1 + s (beta_pooled - 1), s on a fixed grid chosen by the RB-Cal LOSO rule."""
    grid = beta_grid()
    s_grid = np.round(np.linspace(0, 1, 11), 2)
    rows = []
    folds = [(held, fit_scalar(trf, grid, "pooled")[0].beta, val) for held, trf, val in loso_folds(train)]
    for s in s_grid:
        d = []
        for held, pooled_beta, val in folds:
            b = 1 + s * (pooled_beta - 1)
            d.append(calibrator_deltas(val, ScalarTS(b))[held])
        d = np.array(d)
        rows.append((int(np.sum(d > rank_eps + TOL)), float(d.mean()), s))
    v0 = min(r[0] for r in rows)
    pool = [r for r in rows if r[0] == v0]
    m0 = min(r[1] for r in pool)
    s = min(r[2] for r in pool if r[1] <= m0 + TOL)
    beta = 1 + s * (fit_scalar(train, grid, "pooled")[0].beta - 1)
    cal = ScalarTS(beta)
    sel = calibrator_deltas(select, cal)
    accept = max(sel.values()) <= rank_eps + TOL
    return FitResult("SHRINKAGE_LOSO", "source_only", cal if accept else IdentityCalibrator(), cal,
                     {"beta": beta, "s": float(s)}, {"s": float(s)}, "grid", True,
                     {"accept": accept, "max_delta": max(sel.values()), "deltas": sel},
                     None if accept else "SOURCE_GATE_REJECTED", "observed_source_losses")
