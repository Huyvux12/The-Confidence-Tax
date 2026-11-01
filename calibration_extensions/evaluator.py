"""Scoring (spec 8.4, 9.3, 9.4). The only module allowed to open target_labels_private."""
from __future__ import annotations

import numpy as np

from .numerics import ProbSet, gather, surprisal

UNIT_FIELDS = ("model_id", "prompt_id", "task_id")
METRIC_COLUMNS = ["method_id", "protocol", "model_id", "prompt_id", "task_id", "split_id", "seed",
                  "n_rows", "n_cases", "nll", "brier", "ece_15", "accuracy", "delta_nll", "gain_nll",
                  "harm_001", "harm_any", "fallback", "solver_valid", "target_labels_used",
                  "label_query_cases", "fit_seconds", "predict_seconds"]


def row_metrics(ps: ProbSet, log_q: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Per-row NLL (nats), Brier (sum over K), confidence and argmax correctness."""
    mats = gather(ps, log_q)
    out = {k: np.empty(ps.n) for k in ("nll", "brier", "conf", "correct")}
    for g, M in zip(ps.groups, mats):
        rows = np.arange(len(g.idx))
        q = np.exp(M)
        onehot = np.zeros_like(q)
        onehot[rows, g.y] = 1.0
        out["nll"][g.idx] = -M[rows, g.y]
        out["brier"][g.idx] = ((q - onehot) ** 2).sum(axis=1)
        out["conf"][g.idx] = q.max(axis=1)
        out["correct"][g.idx] = (np.argmax(M, axis=1) == g.y).astype(float)
    out["delta"] = out["nll"] - surprisal(ps)
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width bins [k/15, (k+1)/15), last bin closed at 1; empty bins contribute 0."""
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / max(len(conf), 1))


def unit_keys(ps: ProbSet) -> np.ndarray:
    return np.array(["|".join(str(getattr(ps, f)[i]) for f in UNIT_FIELDS) for i in range(ps.n)])


def summarise(m: dict[str, np.ndarray], mask: np.ndarray, ps: ProbSet) -> dict:
    d = float(m["delta"][mask].mean())
    return {"n_rows": int(mask.sum()), "n_cases": len({str(c) for c in ps.case_id[mask]}),
            "nll": float(m["nll"][mask].mean()), "brier": float(m["brier"][mask].mean()),
            "ece_15": ece(m["conf"][mask], m["correct"][mask]),
            "accuracy": float(m["correct"][mask].mean()), "delta_nll": d, "gain_nll": -d,
            "harm_001": int(d > 0.01), "harm_any": int(d > 1e-8)}


def score(ps: ProbSet, log_q: list[np.ndarray], meta: dict) -> list[dict]:
    """One row per model x prompt x task, plus macro and micro aggregates under their own task_id."""
    if not ps.labelled:
        raise ValueError("score needs labels")
    m = row_metrics(ps, log_q)
    keys = unit_keys(ps)
    rows = []
    for k in sorted(set(keys.tolist())):
        mod, pr, task = k.split("|")
        rows.append({**meta, "model_id": mod, "prompt_id": pr, "task_id": task, **summarise(m, keys == k, ps)})
    macro = {c: float(np.mean([r[c] for r in rows])) for c in ("nll", "brier", "ece_15", "accuracy", "delta_nll", "gain_nll", "harm_001", "harm_any")}
    rows.append({**meta, "model_id": "*", "prompt_id": "*", "task_id": "__macro__",
                 "n_rows": ps.n, "n_cases": len({str(c) for c in ps.case_id}), **macro})
    rows.append({**meta, "model_id": "*", "prompt_id": "*", "task_id": "__micro__", **summarise(m, np.ones(ps.n, bool), ps)})
    return rows


def case_means(ps: ProbSet, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cases = np.array([str(c) for c in ps.case_id])[mask]
    v = values[mask]
    uniq, inv = np.unique(cases, return_inverse=True)
    return np.bincount(inv, weights=v), np.bincount(inv).astype(float)


def paired_bootstrap(ps: ProbSet, loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> list[dict]:
    """Mean NLL difference a - b, resampling cases within each unit (same cases for both methods).
    The macro row resamples every unit simultaneously and averages per-unit means."""
    rng = np.random.default_rng(seed)
    keys = unit_keys(ps)
    diff = loss_a - loss_b
    units = sorted(set(keys.tolist()))
    out, boot_units = [], []
    for k in units:
        s, c = case_means(ps, diff, keys == k)
        point = s.sum() / c.sum()
        draws = rng.integers(0, len(s), size=(n_boot, len(s)))
        boots = s[draws].sum(axis=1) / c[draws].sum(axis=1)
        boot_units.append(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"unit": k, "n_cases": len(s), "mean_diff": float(point), "ci_low": float(lo), "ci_high": float(hi)})
    macro = np.mean(boot_units, axis=0)
    lo, hi = np.percentile(macro, [2.5, 97.5])
    out.append({"unit": "__macro__", "n_cases": len({str(c) for c in ps.case_id}),
                "mean_diff": float(np.mean([r["mean_diff"] for r in out])), "ci_low": float(lo), "ci_high": float(hi)})
    return out


def target_benefit_rule(paired_vs_mrts: dict, harm_method: float, harm_mrts: float) -> tuple[str, str]:
    """Default endpoint (spec 9.3): macro delta NLL lower than MRTS with a paired 95% CI
    excluding 0, and harm_001 not higher on the fixed roster."""
    ok = paired_vs_mrts["ci_high"] < 0 and harm_method <= harm_mrts + 1e-12
    why = (f"macro mean diff vs MRTS {paired_vs_mrts['mean_diff']:.4g} "
           f"[{paired_vs_mrts['ci_low']:.4g}, {paired_vs_mrts['ci_high']:.4g}]; "
           f"harm_001 {harm_method:.3g} vs {harm_mrts:.3g}")
    return ("PASS" if ok else "FAIL"), why
