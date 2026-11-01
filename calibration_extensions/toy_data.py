"""Synthetic scenarios with a known calibration map (spec 3.10, 6.6, 7.6).

A case draws reported probabilities p; the true label distribution is
temper(p, beta_star(z(p))) so the ideal calibrator is known exactly. Every case
emits 1-3 rows that permute answer order (jittered), sharing one semantic label.
"""
from __future__ import annotations

import string

import numpy as np

from .calibrators import TopGap
from .features import group_features
from .numerics import ProbSet, concat, prepare_probs, temper

LETTERS = string.ascii_uppercase


def _beta_const(b):
    return lambda z: np.full(z.shape[0], float(b))


class TrueTopGap:
    """True labels follow the M3 transform of p (gap scaled by b, the rest by a)."""
    def __init__(self, a: float, b: float):
        self.a, self.b = a, b


def _single(lp: np.ndarray) -> ProbSet:
    o = np.array(["x"], dtype=object)
    return ProbSet(o, o, o, o, o, o, [list(range(lp.size))], [lp])


BETA_FNS = {
    "affine": lambda z: 0.3 + 2.0 * z[:, 0],
    "vshape": lambda z: 1.6 - 2.4 * np.abs(z[:, 0] - 0.5),
}


def make_rows(rng, *, n_cases: int, K: int, source_id: str, task_id: str, beta_star,
              scale: tuple[float, float], prefix: str, rows_per_case=(1, 3), jitter: float = 0.05,
              label_noise: float = 0.0, model_id: str = "toy_model", prompt_id: str = "toy_prompt") -> ProbSet:
    cols = {k: [] for k in ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id")}
    log_p, label_ids, ys, floored = [], [], [], []
    labels = list(LETTERS[:K]) if K <= 26 else [f"L{k}" for k in range(K)]
    for c in range(n_cases):
        case_id = f"{prefix}_c{c:05d}"
        logits = rng.normal(size=K) * rng.uniform(*scale)
        lp0 = logits - np.logaddexp.reduce(logits)
        if isinstance(beta_star, TrueTopGap):
            true = np.exp(TopGap(beta_star.a, beta_star.b).log_q_groups(_single(lp0))[0][0])
        else:
            true = np.exp(temper(lp0[None, :], beta_star(group_features(lp0[None, :])))[0])
        y_sem = int(rng.choice(K, p=true / true.sum()))
        if label_noise and rng.random() < label_noise:
            y_sem = int(rng.integers(K))
        for r in range(int(rng.integers(rows_per_case[0], rows_per_case[1] + 1))):
            perm = rng.permutation(K) if r else np.arange(K)
            jl = lp0[perm] + (rng.normal(size=K) * jitter if r else 0.0)
            lp, fl = prepare_probs(np.exp(jl - np.logaddexp.reduce(jl)))
            for k, v in zip(cols, (f"{case_id}_r{r}", case_id, model_id, prompt_id, source_id, task_id)):
                cols[k].append(v)
            log_p.append(lp)
            floored.append(fl)
            label_ids.append([labels[j] for j in perm])
            ys.append(int(np.flatnonzero(perm == y_sem)[0]))
    return ProbSet(**{k: np.array(v, dtype=object) for k, v in cols.items()}, label_ids=label_ids,
                   log_p=log_p, y=np.array(ys), floored=np.array(floored))


# name -> (sources [(source_id, beta_fn, scale, n_train_cases, label_noise)], target (beta_fn, scale))
SCENARIOS = {
    "a_shared_temperature": (
        [("src_A", _beta_const(0.6), (1.0, 3.0), 150, 0.0),
         ("src_B", _beta_const(0.6), (2.0, 5.0), 150, 0.0),
         ("src_C", _beta_const(0.6), (0.5, 2.0), 150, 0.0)],
        (_beta_const(0.6), (0.5, 5.0))),
    "b_entropy_conflict": (
        [("src_conf", BETA_FNS["affine"], (4.0, 7.0), 200, 0.0),
         ("src_unc", BETA_FNS["affine"], (0.8, 1.8), 200, 0.0),
         ("src_mix", BETA_FNS["affine"], (0.8, 7.0), 200, 0.0)],
        (BETA_FNS["affine"], (0.8, 7.0))),
    "c_nonlinear": (
        [("src_low", BETA_FNS["vshape"], (5.0, 8.0), 200, 0.0),
         ("src_mid", BETA_FNS["vshape"], (1.0, 2.0), 200, 0.0),
         ("src_high", BETA_FNS["vshape"], (0.1, 0.4), 200, 0.0)],
        (BETA_FNS["vshape"], (0.1, 8.0))),
    "d_unidentifiable": (
        [("src_hot", _beta_const(0.5), (0.5, 4.0), 200, 0.0),
         ("src_cold", _beta_const(2.0), (0.5, 4.0), 200, 0.0),
         ("src_hot2", _beta_const(0.5), (0.5, 4.0), 200, 0.0)],
        (_beta_const(1.0), (0.5, 4.0))),
    "e_small_noisy": (
        [("src_A", _beta_const(0.6), (1.0, 3.0), 150, 0.0),
         ("src_B", _beta_const(0.6), (1.0, 3.0), 150, 0.0),
         ("src_small", _beta_const(0.6), (1.0, 3.0), 12, 0.5)],
        (_beta_const(0.6), (1.0, 3.0))),
    "m3_top_gap": (
        [("src_A", TrueTopGap(1.6, 0.5), (1.0, 4.0), 200, 0.0),
         ("src_B", TrueTopGap(1.6, 0.5), (2.0, 5.0), 200, 0.0),
         ("src_C", TrueTopGap(1.6, 0.5), (0.5, 3.0), 200, 0.0)],
        (TrueTopGap(1.6, 0.5), (0.5, 5.0))),
    "m2_shifted_target": (
        [("src_A", _beta_const(0.7), (1.0, 4.0), 200, 0.0),
         ("src_B", _beta_const(0.6), (1.0, 4.0), 200, 0.0),
         ("src_C", _beta_const(0.8), (1.0, 4.0), 200, 0.0)],
        (_beta_const(0.7), (1.0, 4.0))),
}


def generate_scenario(name: str, seed: int = 0, K: int = 4, n_target_cases: int = 300,
                      n_select_cases: int = 60, n_audit_cases: int = 60,
                      n_target_tasks: int = 2) -> tuple[dict[str, ProbSet], ProbSet]:
    sources, (t_fn, t_scale) = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    splits = {"source_train": [], "source_select": [], "source_audit": []}
    sizes = {"source_select": n_select_cases, "source_audit": n_audit_cases}
    for sid, fn, scale, n_train, noise in sources:
        for split in splits:
            n = n_train if split == "source_train" else max(4, min(sizes[split], n_train))
            splits[split].append(make_rows(rng, n_cases=n, K=K, source_id=sid, task_id=f"task_{sid}",
                                           beta_star=fn, scale=scale, prefix=f"{split}_{sid}",
                                           label_noise=noise))
    targets = [make_rows(rng, n_cases=n_target_cases // n_target_tasks, K=K, source_id="",
                         task_id=f"target_task_{t}", beta_star=t_fn, scale=t_scale,
                         prefix=f"target_t{t}") for t in range(n_target_tasks)]
    return {k: concat(v) for k, v in splits.items()}, concat(targets)


def corrupt_states(ps: ProbSet, frac: float, seed: int, by: str = "state") -> tuple[ProbSet, dict]:
    """Controlled label noise (spec 6.6). by='state' flips every row of chosen cases; by='row' flips rows."""
    rng = np.random.default_rng(seed)
    y = ps.y.copy()
    if by == "state":
        cases = np.array(sorted({str(c) for c in ps.case_id}))
        chosen = set(rng.choice(cases, size=int(round(frac * len(cases))), replace=False))
        rows = [i for i in range(ps.n) if str(ps.case_id[i]) in chosen]
    else:
        rows = list(rng.choice(ps.n, size=int(round(frac * ps.n)), replace=False))
    for i in rows:
        K = ps.log_p[i].size
        y[i] = (y[i] + int(rng.integers(1, K))) % K
    changed_states = len({str(ps.case_id[i]) for i in rows})
    n_states = len({str(c) for c in ps.case_id})
    return ps.with_labels(y), {"by": by, "frac_rows": len(rows) / ps.n,
                               "frac_states": changed_states / n_states, "rows": len(rows)}
