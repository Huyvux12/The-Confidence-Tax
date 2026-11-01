"""Numerical primitives shared by every method (spec section 2.2).

Probabilities are stored ragged: one float64 log-probability vector per row.
For vectorised work rows are bucketed by K (`ProbSet.groups`); no padding cell
ever enters an entropy, a top-2 or a softmax denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.special import logsumexp

FLOOR = 1e-12
TOL = 1e-8
META_FIELDS = ("case_id", "model_id", "prompt_id", "source_id", "task_id")


def prepare_probs(probs) -> tuple[np.ndarray, bool]:
    """Validate, renormalise and floor one probability vector; return (log_p, floored)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("probs must be a 1-D vector with K >= 2")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probs must be finite and non-negative")
    total = p.sum()
    if not total > 0:
        raise ValueError("probs must have a positive sum")
    p = p / total
    floored = bool(np.any(p < FLOOR))
    p = np.maximum(p, FLOOR)
    p = p / p.sum()
    return np.log(p), floored


@dataclass
class KGroup:
    K: int
    idx: np.ndarray          # positions in the parent ProbSet
    logP: np.ndarray         # (n_g, K)
    y: np.ndarray | None     # (n_g,) or None when unlabelled


@dataclass
class ProbSet:
    sample_id: np.ndarray
    case_id: np.ndarray
    model_id: np.ndarray
    prompt_id: np.ndarray
    source_id: np.ndarray
    task_id: np.ndarray
    label_ids: list
    log_p: list
    y: np.ndarray | None = None
    floored: np.ndarray = field(default=None)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.floored is None:
            self.floored = np.zeros(len(self.log_p), dtype=bool)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=np.int64)
            for i, lp in enumerate(self.log_p):
                if not 0 <= self.y[i] < lp.size:
                    raise ValueError(f"label out of range for {self.sample_id[i]}")

    @property
    def n(self) -> int:
        return len(self.log_p)

    @property
    def labelled(self) -> bool:
        return self.y is not None

    @cached_property
    def K(self) -> np.ndarray:
        return np.array([lp.size for lp in self.log_p], dtype=np.int64)

    @cached_property
    def groups(self) -> list[KGroup]:
        out = []
        for K in np.unique(self.K):
            idx = np.flatnonzero(self.K == K)
            logP = np.stack([self.log_p[i] for i in idx]) if idx.size else np.zeros((0, K))
            y = self.y[idx] if self.y is not None else None
            out.append(KGroup(int(K), idx, logP, y))
        return out

    def subset(self, idx) -> "ProbSet":
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return ProbSet(
            sample_id=self.sample_id[idx], case_id=self.case_id[idx],
            model_id=self.model_id[idx], prompt_id=self.prompt_id[idx],
            source_id=self.source_id[idx], task_id=self.task_id[idx],
            label_ids=[self.label_ids[i] for i in idx],
            log_p=[self.log_p[i] for i in idx],
            y=None if self.y is None else self.y[idx],
            floored=self.floored[idx],
            extra={k: [v[i] for i in idx] for k, v in self.extra.items()},
        )

    def without_labels(self) -> "ProbSet":
        out = self.subset(np.arange(self.n))
        out.y = None
        return out

    def with_labels(self, y) -> "ProbSet":
        out = self.subset(np.arange(self.n))
        out.y = np.asarray(y, dtype=np.int64)
        out.__post_init__()
        return out

    def sources(self) -> list[str]:
        return sorted({str(s) for s in self.source_id})


def concat(sets: list[ProbSet]) -> ProbSet:
    sets = [s for s in sets if s.n]
    labelled = all(s.labelled for s in sets)
    keys = set().union(*(s.extra.keys() for s in sets)) if sets else set()
    return ProbSet(
        sample_id=np.concatenate([s.sample_id for s in sets]),
        case_id=np.concatenate([s.case_id for s in sets]),
        model_id=np.concatenate([s.model_id for s in sets]),
        prompt_id=np.concatenate([s.prompt_id for s in sets]),
        source_id=np.concatenate([s.source_id for s in sets]),
        task_id=np.concatenate([s.task_id for s in sets]),
        label_ids=[l for s in sets for l in s.label_ids],
        log_p=[lp for s in sets for lp in s.log_p],
        y=np.concatenate([s.y for s in sets]) if labelled else None,
        floored=np.concatenate([s.floored for s in sets]),
        extra={k: [v for s in sets for v in s.extra.get(k, [None] * s.n)] for k in keys},
    )


def normalise_rows(U: np.ndarray) -> np.ndarray:
    return U - logsumexp(U, axis=1, keepdims=True)


def temper(logP: np.ndarray, beta) -> np.ndarray:
    """log q = beta*log p - LSE(beta*log p); beta scalar or per-row."""
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim == 1:
        beta = beta[:, None]
    return normalise_rows(beta * logP)


def scatter(ps: ProbSet, mats: list[np.ndarray]) -> list[np.ndarray]:
    """Group matrices aligned with ps.groups -> ragged list aligned with rows."""
    out = [None] * ps.n
    for g, M in zip(ps.groups, mats):
        for j, i in enumerate(g.idx):
            out[i] = M[j]
    return out


def gather(ps: ProbSet, rows: list[np.ndarray]) -> list[np.ndarray]:
    return [np.stack([rows[i] for i in g.idx]) for g in ps.groups]


def losses(ps: ProbSet, mats: list[np.ndarray]) -> np.ndarray:
    """Per-row NLL -log q_y computed directly from log q."""
    if not ps.labelled:
        raise ValueError("losses need labels")
    out = np.empty(ps.n)
    for g, M in zip(ps.groups, mats):
        out[g.idx] = -M[np.arange(len(g.idx)), g.y]
    return out


def surprisal(ps: ProbSet) -> np.ndarray:
    """-log p_y under the reference distribution (the identity loss)."""
    return losses(ps, [g.logP for g in ps.groups])


def group_means(keys: np.ndarray, values: np.ndarray) -> dict[str, float]:
    keys = np.asarray([str(k) for k in keys])
    return {k: float(values[keys == k].mean()) for k in sorted(set(keys.tolist()))}


def source_deltas(ps: ProbSet, loss: np.ndarray) -> dict[str, float]:
    """Delta_w = L_w(theta) - L_w(theta_0) for every source."""
    return group_means(ps.source_id, loss - surprisal(ps))


def check_log_q(mats: list[np.ndarray], tol: float = 1e-9) -> dict:
    """Every q sums to one, no NaN, log q finite."""
    worst, finite = 0.0, True
    for M in mats:
        if M.size == 0:
            continue
        finite &= bool(np.all(np.isfinite(M)))
        worst = max(worst, float(np.max(np.abs(np.exp(logsumexp(M, axis=1)) - 1.0))))
    return {"finite": finite, "max_sum_error": worst, "ok": finite and worst <= tol}
