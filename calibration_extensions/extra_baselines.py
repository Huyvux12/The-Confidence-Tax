"""Probability-feature adaptations of two published LLM calibrators (secondary analysis, todo.md item 1).

Our data hold option-letter probabilities only, no hidden states, so both read the label-free features
z = [entropy, top-1, margin] through the RBF map of `features.py` (phi = [1, psi_1..psi_J]).

THERMO_P  (Thermometer, Shen et al. 2024): tau_i = softplus(theta . phi_i) per input; a task's temperature
          is the mean tau_i over that task's unlabelled inputs; trained on the source tasks with NLL averaged
          equally over tasks.
ATS_P     (Adaptive Temperature Scaling, Xie et al. 2024): per-input beta_i = exp(theta . phi_i), trained on
          the pooled sources with selective smoothing: NLL on rows the base model gets right, cross-entropy
          to the uniform distribution on rows it gets wrong, equal weights.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .calibrators import Calibrator
from .features import FeatureMap, fit_feature_map
from .numerics import ProbSet, temper

FEATURES = ("rbf", 8, 0.5)
L2 = 1e-3
SOFTPLUS_ONE = float(np.log(np.e - 1.0))      # softplus(x) = 1


def _softplus(a):
    return np.logaddexp(0.0, a)


def _sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def _row_terms(ps: ProbSet, beta: np.ndarray):
    """Per-row NLL, smoothing loss, their d/d beta, and base-model correctness."""
    nll, smooth = np.empty(ps.n), np.empty(ps.n)
    g_nll, g_smooth = np.empty(ps.n), np.empty(ps.n)
    correct = np.empty(ps.n, dtype=bool)
    for g in ps.groups:
        b = beta[g.idx]
        lq = temper(g.logP, b)
        q = np.exp(lq)
        r = np.arange(len(g.idx))
        eq = (q * g.logP).sum(axis=1)
        nll[g.idx] = -lq[r, g.y]
        g_nll[g.idx] = eq - g.logP[r, g.y]
        smooth[g.idx] = -lq.mean(axis=1)
        g_smooth[g.idx] = eq - g.logP.mean(axis=1)
        correct[g.idx] = np.argmax(g.logP, axis=1) == g.y
    return nll, g_nll, smooth, g_smooth, correct


def _batch_index(ps: ProbSet, fields) -> tuple[np.ndarray, int]:
    keys = np.array(["|".join(str(getattr(ps, f)[i]) for f in fields) for i in range(ps.n)])
    _, inv = np.unique(keys, return_inverse=True)
    return inv, int(inv.max()) + 1


class ThermometerP(Calibrator):
    kind = "thermometer_p"
    batch_fields = ("model_id", "prompt_id", "task_id")

    def __init__(self, fmap: FeatureMap, theta: np.ndarray):
        self.fmap, self.theta = fmap, np.asarray(theta, dtype=np.float64)

    def task_beta(self, ps: ProbSet) -> np.ndarray:
        tau = _softplus(self.fmap.phi(ps) @ self.theta)
        inv, nb = _batch_index(ps, self.batch_fields)
        T = np.bincount(inv, weights=tau, minlength=nb) / np.bincount(inv, minlength=nb)
        return 1.0 / T[inv]

    def log_q_groups(self, ps):
        beta = self.task_beta(ps)
        return [temper(g.logP, beta[g.idx]) for g in ps.groups]

    def params(self):
        return {"feature_map": self.fmap.to_dict(), "theta": self.theta.tolist()}


class ATSP(Calibrator):
    kind = "ats_p"

    def __init__(self, fmap: FeatureMap, theta: np.ndarray):
        self.fmap, self.theta = fmap, np.asarray(theta, dtype=np.float64)

    def log_q_groups(self, ps):
        beta = np.exp(self.fmap.phi(ps) @ self.theta)
        return [temper(g.logP, beta[g.idx]) for g in ps.groups]

    def params(self):
        return {"feature_map": self.fmap.to_dict(), "theta": self.theta.tolist()}


def thermometer_objective(theta, Phi, ps, src_inv, n_src):
    a = Phi @ theta
    tau = _softplus(a)
    cnt = np.bincount(src_inv, minlength=n_src).astype(float)
    T = np.bincount(src_inv, weights=tau, minlength=n_src) / cnt
    beta = 1.0 / T[src_inv]
    nll, g_nll, *_ = _row_terms(ps, beta)
    L_w = np.bincount(src_inv, weights=nll, minlength=n_src) / cnt
    dL_dbeta_w = np.bincount(src_inv, weights=g_nll, minlength=n_src) / cnt
    dL_dT_w = dL_dbeta_w * (-1.0 / T ** 2)
    # dT_w / d theta = mean_{i in w} sigmoid(a_i) phi_i
    w_row = (dL_dT_w[src_inv] / cnt[src_inv]) * _sigmoid(a)
    grad = Phi.T @ w_row / n_src
    reg = theta.copy()
    reg[0] = 0.0
    return float(L_w.mean() + L2 * reg @ reg), grad + 2 * L2 * reg


def ats_objective(theta, Phi, ps):
    beta = np.exp(Phi @ theta)
    nll, g_nll, smooth, g_smooth, correct = _row_terms(ps, beta)
    loss = np.where(correct, nll, smooth)
    dl = np.where(correct, g_nll, g_smooth)
    grad = Phi.T @ (dl * beta) / ps.n
    reg = theta.copy()
    reg[0] = 0.0
    return float(loss.mean() + L2 * reg @ reg), grad + 2 * L2 * reg


def fit_thermometer_p(train: ProbSet, kind: str = FEATURES[0]) -> tuple[ThermometerP, dict]:
    fmap = fit_feature_map(train, kind, *FEATURES[1:])
    Phi = fmap.phi(train)
    src_inv, n_src = _batch_index(train, ("source_id",))
    x0 = np.zeros(Phi.shape[1])
    x0[0] = SOFTPLUS_ONE
    res = minimize(thermometer_objective, x0, args=(Phi, train, src_inv, n_src), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500})
    return ThermometerP(fmap, res.x), {"status": f"LBFGS:{res.status}", "fun": float(res.fun)}


def fit_ats_p(train: ProbSet, kind: str = FEATURES[0]) -> tuple[ATSP, dict]:
    fmap = fit_feature_map(train, kind, *FEATURES[1:])
    Phi = fmap.phi(train)
    res = minimize(ats_objective, np.zeros(Phi.shape[1]), args=(Phi, train), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500})
    return ATSP(fmap, res.x), {"status": f"LBFGS:{res.status}", "fun": float(res.fun)}
