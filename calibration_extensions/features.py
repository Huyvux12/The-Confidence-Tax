"""Label-free features z = [h, c, m] and the finite function classes of M1 (spec 2.5, 3.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerics import ProbSet

KINDS = ("constant", "entropy_affine", "rbf")


def group_features(logP: np.ndarray) -> np.ndarray:
    K = logP.shape[1]
    p = np.exp(logP)
    h = np.clip(-(p * logP).sum(axis=1) / np.log(K), 0.0, 1.0)
    top2 = np.sort(p, axis=1)[:, -2:]
    return np.column_stack([h, top2[:, 1], top2[:, 1] - top2[:, 0]])


def prob_features(ps: ProbSet) -> np.ndarray:
    z = np.empty((ps.n, 3))
    for g in ps.groups:
        z[g.idx] = group_features(g.logP)
    return z


def select_centers(z: np.ndarray, sample_id, J: int) -> np.ndarray:
    """Point nearest the mean, then farthest-point traversal; ties broken by sample_id."""
    if J == 0:
        return np.zeros((0, z.shape[1]))
    order = np.argsort(np.asarray([str(s) for s in sample_id]), kind="stable")
    zs = z[order]
    first = int(np.argmin(np.sum((zs - zs.mean(axis=0)) ** 2, axis=1)))
    chosen = [first]
    dmin = np.sum((zs - zs[first]) ** 2, axis=1)
    while len(chosen) < J:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.sum((zs - zs[nxt]) ** 2, axis=1))
    return zs[chosen].copy()


def rbf_psi(z: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    J = centers.shape[0]
    d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma ** 2)) / np.sqrt(J)


@dataclass
class FeatureMap:
    """beta(p) = phi(p) @ theta with phi = [1, ...]; theta_0 = [1, 0, ...] is the identity."""
    kind: str
    centers: np.ndarray | None = None
    sigma: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(self.kind)
        if self.kind == "rbf" and (self.centers is None or self.centers.shape[0] == 0):
            self.kind, self.centers, self.sigma = "constant", None, None

    @property
    def dim(self) -> int:
        return {"constant": 1, "entropy_affine": 2}.get(self.kind, 1 + (0 if self.centers is None else self.centers.shape[0]))

    @property
    def n_centers(self) -> int:
        return self.dim - 1 if self.kind == "rbf" else 0

    def identity_theta(self) -> np.ndarray:
        t = np.zeros(self.dim)
        t[0] = 1.0
        return t

    def phi_from_z(self, z: np.ndarray) -> np.ndarray:
        ones = np.ones((z.shape[0], 1))
        if self.kind == "constant":
            return ones
        if self.kind == "entropy_affine":
            return np.column_stack([ones, z[:, 0]])
        return np.column_stack([ones, rbf_psi(z, self.centers, self.sigma)])

    def phi(self, ps: ProbSet) -> np.ndarray:
        return self.phi_from_z(prob_features(ps))

    def beta(self, ps: ProbSet, theta: np.ndarray) -> np.ndarray:
        return self.phi(ps) @ theta

    def to_dict(self) -> dict:
        return {"kind": self.kind, "sigma": self.sigma, "n_centers": self.n_centers}


def fit_feature_map(ps: ProbSet, kind: str, J: int = 0, sigma: float | None = None) -> FeatureMap:
    if kind != "rbf" or J == 0:
        return FeatureMap(kind if kind != "rbf" else "constant")
    return FeatureMap("rbf", select_centers(prob_features(ps), ps.sample_id, J), sigma)


def beta_bound_slack(fmap: FeatureMap, theta: np.ndarray, beta_min: float, beta_max: float) -> float:
    """min over the global constraints of (bound - value); >= 0 means beta(p) in range for every p."""
    b0 = theta[0]
    if fmap.kind == "constant":
        return min(b0 - beta_min, beta_max - b0)
    if fmap.kind == "entropy_affine":
        ends = (b0, b0 + theta[1])
        return min(min(e - beta_min, beta_max - e) for e in ends)
    nv = float(np.linalg.norm(theta[1:]))
    return min(b0 - beta_min - nv, beta_max - b0 - nv)


def project_to_bounds(fmap: FeatureMap, theta: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
    """Remove solver-tolerance violations of the global beta bounds (never a large move)."""
    t = theta.copy()
    if fmap.kind == "rbf":
        t[0] = np.clip(t[0], beta_min, beta_max)
        room = min(t[0] - beta_min, beta_max - t[0])
        nv = np.linalg.norm(t[1:])
        if nv > room:
            t[1:] *= room / nv
    elif fmap.kind == "constant":
        t[0] = np.clip(t[0], beta_min, beta_max)
    else:
        t[0] = np.clip(t[0], beta_min, beta_max)
        t[1] = np.clip(t[0] + t[1], beta_min, beta_max) - t[0]
    return t
