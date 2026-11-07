"""Label-free features z = [h, c, m] and the finite function classes of M1 (spec 2.5, 3.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerics import ProbSet

KINDS = ("constant", "entropy_affine", "rbf", "rbf_oc", "rbf_oct", "rbf_ocp", "rbf_ocp1")
# rbf_oc / rbf_oct (phase 8): the RBF field on z extended with label-free answer-order consistency,
#   tv  = total variation between the two answer orders of the same question (semantic options), per row
#   tvT = mean tv of the row's task batch (source task for sources, target task for targets; no labels)
# rbf_ocp / rbf_ocp1 (phase 10): rbf_oct plus peer consensus (row value and task mean) from the data field
#   peer_top (4 other models) or peer_top1 (one pre-chosen peer); see data_gen/make_aug.py
RBF_LIKE = ("rbf", "rbf_oc", "rbf_oct", "rbf_ocp", "rbf_ocp1")
PEER_FIELD = {"rbf_ocp": "peer_top", "rbf_ocp1": "peer_top1"}


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


def semantic_probs(ps: ProbSet, i: int) -> np.ndarray:
    """Probabilities re-indexed by the original option index (label_ids[i][pos] = option shown at pos)."""
    p = np.exp(ps.log_p[i])
    out = np.zeros(p.size)
    for pos, lab in enumerate(ps.label_ids[i]):
        out[int(lab)] = p[pos]
    return out


def order_tv(ps: ProbSet) -> tuple[np.ndarray, int]:
    """Per row: TV distance to the other answer order of the same question (same model and prompt).
    Rows without a partner get 0; returns (tv, number of unpaired rows). If the data carry the precomputed
    field order_tv_raw (derived datasets whose probs were averaged over orders), that value is used."""
    raw = ps.extra.get("order_tv_raw")
    if raw is not None and all(v is not None for v in raw):
        return np.asarray(raw, dtype=float), 0
    by_key: dict[tuple, list[int]] = {}
    for i in range(ps.n):
        by_key.setdefault((str(ps.case_id[i]), str(ps.model_id[i]), str(ps.prompt_id[i])), []).append(i)
    tv = np.zeros(ps.n)
    unpaired = 0
    for rows in by_key.values():
        if len(rows) < 2:
            unpaired += len(rows)
            continue
        sem = [semantic_probs(ps, i) for i in rows]
        for a, i in enumerate(rows):          # mean TV to the other orders (two orders in this data)
            tv[i] = np.mean([0.5 * np.abs(sem[a] - sem[b]).sum() for b in range(len(rows)) if b != a])
    return tv, unpaired


def batch_key(ps: ProbSet) -> np.ndarray:
    return np.array([str(s) if str(s) not in ("", "None") else str(t) for s, t in zip(ps.source_id, ps.task_id)])


def task_mean(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    for k in set(keys.tolist()):
        m = keys == k
        out[m] = values[m].mean()
    return out


def features_for(ps: ProbSet, kind: str) -> np.ndarray:
    z = prob_features(ps)
    if kind not in ("rbf_oc", "rbf_oct", "rbf_ocp", "rbf_ocp1"):
        return z
    tv, _ = order_tv(ps)
    keys = batch_key(ps)
    cols = [z, tv[:, None]]
    if kind != "rbf_oc":
        cols.append(task_mean(tv, keys)[:, None])
    if kind in PEER_FIELD:
        vals = ps.extra.get(PEER_FIELD[kind])
        if vals is None or any(v is None for v in vals):
            raise ValueError(f"{kind} needs the data field {PEER_FIELD[kind]} (data_gen/make_aug.py)")
        peer = np.asarray(vals, dtype=float)
        cols += [peer[:, None], task_mean(peer, keys)[:, None]]
    return np.hstack(cols)


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
        if self.kind in RBF_LIKE and (self.centers is None or self.centers.shape[0] == 0):
            self.kind, self.centers, self.sigma = "constant", None, None

    @property
    def dim(self) -> int:
        return {"constant": 1, "entropy_affine": 2}.get(self.kind, 1 + (0 if self.centers is None else self.centers.shape[0]))

    @property
    def n_centers(self) -> int:
        return self.dim - 1 if self.kind in RBF_LIKE else 0

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
        return self.phi_from_z(features_for(ps, self.kind))

    def beta(self, ps: ProbSet, theta: np.ndarray) -> np.ndarray:
        return self.phi(ps) @ theta

    def to_dict(self) -> dict:
        return {"kind": self.kind, "sigma": self.sigma, "n_centers": self.n_centers}


def fit_feature_map(ps: ProbSet, kind: str, J: int = 0, sigma: float | None = None) -> FeatureMap:
    if kind not in RBF_LIKE or J == 0:
        return FeatureMap(kind if kind not in RBF_LIKE else "constant")
    return FeatureMap(kind, select_centers(features_for(ps, kind), ps.sample_id, J), sigma)


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
    if fmap.kind in RBF_LIKE:
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
