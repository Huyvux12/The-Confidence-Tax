"""M2 - forecast the NLL change of a shared beta on an unlabelled target batch (spec section 4).

Delta_T(beta) = (beta - 1) M_T + mean_i log sum_k p_ik^beta; only M_T (the mean target surprisal)
is unknown. The interval for M_T below is an empirical source-LOSO band, not a target guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import logsumexp

from .calibrators import BatchBeta, FitResult
from .features import prob_features
from .numerics import ProbSet, surprisal
from .source_objectives import argmin_near_one, beta_grid
from .source_splits import loso_folds

INTERVAL_TYPES = ("bootstrap_envelope", "envelope_no_bootstrap", "point", "raw")


@dataclass
class M2Config:
    bin_options: tuple = (2, 4)
    n_boot: int = 500
    min_cases: int = 5
    min_valid_frac: float = 0.8
    grid_lo: float = 0.25
    grid_hi: float = 4.0
    grid_n: int = 241
    interval_type: str = "bootstrap_envelope"
    batch_fields: tuple = ("model_id", "prompt_id", "task_id")
    seed: int = 0


@dataclass
class Envelope:
    K: int
    edges: np.ndarray
    supported: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    per_source: dict = field(default_factory=dict)
    inflation: float = 0.0

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1


def assign_bins(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], h, side="right")


def raw_bounds(ps: ProbSet) -> tuple[np.ndarray, np.ndarray]:
    """-log max p <= s_i <= -log min p, no invariance assumption."""
    lo, hi = np.empty(ps.n), np.empty(ps.n)
    for g in ps.groups:
        lo[g.idx] = -g.logP.max(axis=1)
        hi[g.idx] = -g.logP.min(axis=1)
    return lo, hi


def fit_envelope(ps: ProbSet, n_bins: int, cfg: M2Config, rng) -> Envelope:
    K = int(ps.K[0])
    if np.any(ps.K != K):
        raise ValueError("M2 envelope is fitted on a single K")
    h = prob_features(ps)[:, 0]
    edges = np.quantile(h, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    bins = assign_bins(h, edges)
    s = surprisal(ps)
    keys = np.array([str(x) for x in ps.source_id])
    lower = np.full(n_bins, np.inf)
    upper = np.full(n_bins, -np.inf)
    per_source = {}
    for src in ps.sources():
        rows = np.flatnonzero(keys == src)
        cases = sorted({str(ps.case_id[i]) for i in rows})
        cpos = {c: k for k, c in enumerate(cases)}
        cidx = np.array([cpos[str(ps.case_id[i])] for i in rows])
        S = np.zeros((len(cases), n_bins))
        C = np.zeros((len(cases), n_bins))
        np.add.at(S, (cidx, bins[rows]), s[rows])
        np.add.at(C, (cidx, bins[rows]), 1.0)
        n_cases_bin = (C > 0).sum(axis=0)
        draws = rng.integers(0, len(cases), size=(cfg.n_boot, len(cases)))
        bs, bc = S[draws].sum(axis=1), C[draws].sum(axis=1)
        info = []
        for b in range(n_bins):
            point = S[:, b].sum() / C[:, b].sum() if C[:, b].sum() else None
            valid = bc[:, b] > 0
            ok = n_cases_bin[b] >= cfg.min_cases and valid.mean() >= cfg.min_valid_frac
            rec = {"bin": b, "n_cases": int(n_cases_bin[b]), "valid_draws": int(valid.sum()),
                   "supported": bool(ok), "mean": point, "seed": cfg.seed}
            if ok:
                if cfg.interval_type == "bootstrap_envelope":
                    lo, hi = np.percentile(bs[valid, b] / bc[valid, b], [2.5, 97.5])
                else:
                    lo = hi = point
                rec |= {"lower": float(lo), "upper": float(hi)}
                lower[b], upper[b] = min(lower[b], lo), max(upper[b], hi)
            info.append(rec)
        per_source[src] = info
    supported = np.isfinite(lower)
    return Envelope(K, edges, supported, lower, upper, per_source)


def predict_interval(env: Envelope | None, ps: ProbSet, interval_type: str) -> dict:
    """Interval for the mean surprisal of one unlabelled batch (probabilities only)."""
    rlo, rhi = raw_bounds(ps)
    raw_lo, raw_hi = float(rlo.mean()), float(rhi.mean())
    base = {"raw_lower": raw_lo, "raw_upper": raw_hi}
    if interval_type == "raw" or env is None or np.any(ps.K != env.K):
        status = "RAW_ONLY" if interval_type == "raw" else "UNSUPPORTED_K"
        return base | {"m_lower": raw_lo, "m_upper": raw_hi, "unsupported_mass": 1.0, "status": status}
    bins = assign_bins(prob_features(ps)[:, 0], env.edges)
    sup = env.supported[bins]
    lo = (np.where(sup, env.lower[bins], rlo)).mean() - env.inflation
    hi = (np.where(sup, env.upper[bins], rhi)).mean() + env.inflation
    out = base | {"unsupported_mass": float(1 - sup.mean())}
    lo_c, hi_c = max(lo, raw_lo), min(hi, raw_hi)
    if lo_c > hi_c:
        return out | {"m_lower": lo, "m_upper": hi, "status": "EMPTY_INTERSECTION"}
    return out | {"m_lower": float(lo_c), "m_upper": float(hi_c), "status": "OK"}


def mean_lse(ps: ProbSet, beta: float) -> float:
    return float(np.concatenate([logsumexp(beta * g.logP, axis=1) for g in ps.groups]).mean())


def upper_delta(a_bar: float, beta: float, m_lower: float, m_upper: float) -> float:
    """B_T(beta): the worst case of (beta-1) M over M in [L, U] picks U above 1 and L below 1."""
    return a_bar + (beta - 1.0) * (m_upper if beta >= 1.0 else m_lower)


def choose_beta(ps: ProbSet, interval: dict, grid: np.ndarray, interval_type: str) -> tuple[float, list[dict]]:
    rows = []
    for b in grid:
        a = mean_lse(ps, b)
        if interval_type == "point":
            mid = 0.5 * (interval["m_lower"] + interval["m_upper"])
            ub = a + (b - 1.0) * mid
        else:
            ub = upper_delta(a, b, interval["m_lower"], interval["m_upper"])
        rows.append({"beta": float(b), "A_bar": a, "upper_delta": ub})
    ubs = np.array([r["upper_delta"] for r in rows])
    j = argmin_near_one(ubs, grid)
    beta = float(grid[j]) if ubs[j] < 0 else 1.0
    for r in rows:
        r["selected"] = r["beta"] == beta
    return beta, rows


@dataclass
class M2Model:
    envelopes: dict
    cfg: M2Config
    loso: dict
    source_interval_gate: bool
    select_checks: list


def loso_errors(ps: ProbSet, n_bins: int, cfg: M2Config) -> dict:
    out = []
    for k, (held, train_fold, val) in enumerate(loso_folds(ps)):
        env = fit_envelope(train_fold, n_bins, cfg, np.random.default_rng(cfg.seed + k))
        itv = predict_interval(env, val, cfg.interval_type)
        true_m = float(surprisal(val).mean())
        e = max(0.0, itv["m_lower"] - true_m, true_m - itv["m_upper"])
        out.append({"heldout": held, "true_m": true_m, "m_lower": itv["m_lower"], "m_upper": itv["m_upper"],
                    "midpoint_abs_error": abs(0.5 * (itv["m_lower"] + itv["m_upper"]) - true_m), "excess": e})
    return {"n_bins": n_bins, "folds": out, "mae": float(np.mean([f["midpoint_abs_error"] for f in out])),
            "max_error": float(np.max([f["midpoint_abs_error"] for f in out])),
            "inflation": float(np.max([f["excess"] for f in out]))}


def fit_m2(source_train: ProbSet, source_select: ProbSet, cfg: M2Config = M2Config()) -> M2Model:
    envelopes, loso = {}, {}
    for K in sorted(set(source_train.K.tolist())):
        ps = source_train.subset(source_train.K == K)
        if cfg.interval_type == "raw" or len(ps.sources()) < 2:
            envelopes[K] = None
            continue
        scores = [loso_errors(ps, B, cfg) for B in cfg.bin_options]
        best = min(scores, key=lambda s: (s["mae"], s["max_error"], s["n_bins"]))
        env = fit_envelope(ps, best["n_bins"], cfg, np.random.default_rng(cfg.seed))
        env.inflation = best["inflation"]
        envelopes[K], loso[K] = env, {"candidates": scores, "chosen_bins": best["n_bins"]}
    checks, gate = [], True
    for src in source_select.sources():
        for K in sorted(set(source_select.K.tolist())):
            ps = source_select.subset(np.array([str(s) == src for s in source_select.source_id]) & (source_select.K == K))
            if not ps.n:
                continue
            itv = predict_interval(envelopes.get(K), ps, cfg.interval_type)
            true_m = float(surprisal(ps).mean())
            covered = itv["status"] != "EMPTY_INTERSECTION" and itv["m_lower"] - 1e-12 <= true_m <= itv["m_upper"] + 1e-12
            gate &= covered
            checks.append({"source_id": src, "K": K, "true_m": true_m, "covered": bool(covered), **itv})
    return M2Model(envelopes, cfg, loso, bool(gate), checks)


def batches(ps: ProbSet, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    keys = np.array(["|".join(str(getattr(ps, f)[i]) for f in fields) for i in range(ps.n)])
    return {k: np.flatnonzero(keys == k) for k in sorted(set(keys.tolist()))}


def adapt_unlabeled(model: M2Model, target_inputs: ProbSet) -> FitResult:
    if target_inputs.labelled:
        raise ValueError("adapt_unlabeled must not receive target labels")
    cfg = model.cfg
    grid = beta_grid(cfg.grid_lo, cfg.grid_hi, cfg.grid_n)
    betas, rows, intervals = {}, [], {}
    for key, idx in batches(target_inputs, cfg.batch_fields).items():
        ps = target_inputs.subset(idx)
        K = int(ps.K[0]) if np.all(ps.K == ps.K[0]) else None
        itv = predict_interval(model.envelopes.get(K), ps, cfg.interval_type) if K else \
            {"status": "MIXED_K", "m_lower": None, "m_upper": None}
        if itv["status"] in ("EMPTY_INTERSECTION", "MIXED_K", "UNSUPPORTED_K") and cfg.interval_type != "raw":
            betas[key] = 1.0
            intervals[key] = itv | {"selected_beta": 1.0}
            continue
        beta, table = choose_beta(ps, itv, grid, cfg.interval_type)
        betas[key] = beta
        mid = 0.5 * (itv["m_lower"] + itv["m_upper"])
        intervals[key] = itv | {"m_hat": mid, "selected_beta": beta, "interval_type": cfg.interval_type,
                                "guarantee_type": "conditional_on_interval_coverage",
                                "target_label_access": False}
        rows += [{"batch": key, **r, "m_lower": itv["m_lower"], "m_upper": itv["m_upper"]} for r in table]
    fitted = BatchBeta(cfg.batch_fields, betas)
    deployed = fitted if model.source_interval_gate else BatchBeta(cfg.batch_fields, {})
    method_id = "M2" if cfg.interval_type == "bootstrap_envelope" else f"M2_{cfg.interval_type.upper()}"
    return FitResult(method_id, "target_unlabeled", deployed, fitted, {"betas": betas},
                     {"interval_type": cfg.interval_type,
                      "n_bins": {int(k): v["chosen_bins"] for k, v in model.loso.items()},
                      "inflation": {int(k): (e.inflation if e else None) for k, e in model.envelopes.items()}},
                     "closed_form", True, {"accept": model.source_interval_gate, "checks": model.select_checks},
                     None if model.source_interval_gate else "SOURCE_INTERVAL_GATE_FALSE",
                     "conditional_on_interval_coverage",
                     diagnostics={"intervals": intervals, "bounds_table": rows, "loso": model.loso,
                                  "assumption": "conditional mean surprisal of target lies in the source envelope per entropy bin"})


def config_dict(cfg: M2Config) -> dict:
    return asdict(cfg)
