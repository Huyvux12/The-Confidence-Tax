"""M5 - query a few target labels to certify an NLL gain on the current batch (spec section 7).

Bounds are deterministic for this finite batch: unknown rows contribute min_k / max_k of
delta_ia(k) = -log q_ia,k + log p_ik. They say nothing about future batches.
"""
from __future__ import annotations

import numpy as np

from .calibrators import Calibrator
from .numerics import ProbSet
from .source_splits import cases_of

STOP_REASONS = ("CERTIFIED", "BUDGET_EXHAUSTED", "NO_POTENTIAL", "ALL_LABELS_QUERIED", "ORACLE_FAILURE")


class OracleFailure(RuntimeError):
    pass


class LabelOracle:
    """Holds private labels; the method only sees what `query` returns. Cost counted per case."""

    def __init__(self, ps: ProbSet, private: dict[str, dict], cost: dict[str, float] | None = None):
        self._ps = ps
        self._private = private
        self._cases = cases_of(ps)
        self._cost = cost or {}
        self.queried: list[str] = []
        self.total_cost = 0.0

    def cost(self, case_id: str) -> float:
        return float(self._cost.get(case_id, 1.0))

    def query(self, case_id: str) -> dict[int, int]:
        """Return {row position: label position in that row's own answer order}."""
        if case_id not in self._cases:
            raise OracleFailure(f"unknown case {case_id}")
        out = {}
        for i in self._cases[case_id]:
            rec = self._private.get(str(self._ps.sample_id[i]))
            if rec is None:
                raise OracleFailure(f"no label for {self._ps.sample_id[i]}")
            lid = rec.get("label_id")
            out[int(i)] = self._ps.label_ids[i].index(lid) if lid is not None else int(rec["y"])
        if case_id not in self.queried:
            self.queried.append(case_id)
            self.total_cost += self.cost(case_id)
        return out


def candidate_deltas(ps: ProbSet, cal: Calibrator) -> list[np.ndarray]:
    log_q = cal.predict_log_proba(ps)
    return [-lq + lp for lq, lp in zip(log_q, ps.log_p)]


class BatchBounds:
    def __init__(self, ps: ProbSet, candidates: dict[str, Calibrator]):
        self.names = sorted(candidates)
        self.N = ps.n
        self.delta = {a: candidate_deltas(ps, candidates[a]) for a in self.names}
        self.l = {a: np.array([d.min() for d in self.delta[a]]) for a in self.names}
        self.u = {a: np.array([d.max() for d in self.delta[a]]) for a in self.names}
        self.known = {a: 0.0 for a in self.names}
        self.sum_l = {a: float(self.l[a].sum()) for a in self.names}
        self.sum_u = {a: float(self.u[a].sum()) for a in self.names}
        self.is_known = np.zeros(ps.n, dtype=bool)

    def reveal(self, labels: dict[int, int]) -> None:
        for i, y in labels.items():
            if self.is_known[i]:
                continue
            self.is_known[i] = True
            for a in self.names:
                self.known[a] += float(self.delta[a][i][y])
                self.sum_l[a] -= self.l[a][i]
                self.sum_u[a] -= self.u[a][i]

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {a: ((self.known[a] + self.sum_l[a]) / self.N, (self.known[a] + self.sum_u[a]) / self.N)
                for a in self.names}


def select_with_oracle(candidates: dict[str, Calibrator], target_inputs: ProbSet, oracle: LabelOracle,
                       budget: float, eta: float = 1e-6, rule: str = "active", seed: int = 0,
                       identity_name: str = "identity") -> dict:
    """rule='active': widest-bound case for the most promising candidate (spec 7.4); 'random' baseline."""
    if target_inputs.labelled:
        raise ValueError("select_with_oracle must not receive target labels")
    if identity_name not in candidates:
        raise ValueError("candidate set must include identity")
    rng = np.random.default_rng(seed)
    bb = BatchBounds(target_inputs, candidates)
    cases = cases_of(target_inputs)
    unqueried = sorted(cases)
    trace, stop, spent = [], None, 0.0
    others = [a for a in bb.names if a != identity_name]
    while True:
        bounds = bb.bounds()
        trace.append({"step": len(trace), "cost": spent, "n_queried": len(oracle.queried),
                      "bounds": {a: {"L": l, "U": u} for a, (l, u) in bounds.items()}})
        certified = [a for a in others if bounds[a][1] <= -eta]
        if certified:
            stop = "CERTIFIED"
            break
        if not unqueried:
            stop = "ALL_LABELS_QUERIED"
            break
        potential = [a for a in others if bounds[a][0] < -eta]
        if not potential:
            stop = "NO_POTENTIAL"
            break
        affordable = [c for c in unqueried if spent + oracle.cost(c) <= budget + 1e-12]
        if not affordable:
            stop = "BUDGET_EXHAUSTED"
            break
        if rule == "active":
            a_star = min(potential, key=lambda a: (bounds[a][1], a))
            width = bb.u[a_star] - bb.l[a_star]
            score = [(width[cases[c]].sum() / oracle.cost(c), c) for c in affordable]
            best = max(s for s, _ in score)
            pick = min(c for s, c in score if s == best)
        else:
            pick = affordable[int(rng.integers(len(affordable)))]
        try:
            labels = oracle.query(pick)
        except OracleFailure as e:
            stop = "ORACLE_FAILURE"
            trace[-1]["error"] = str(e)
            break
        spent += oracle.cost(pick)
        bb.reveal(labels)
        unqueried.remove(pick)
        trace[-1]["queried_case"] = pick
    bounds = bb.bounds()
    best_by_bound = min(others, key=lambda a: (bounds[a][1], a)) if others else identity_name
    certified = stop == "CERTIFIED" or (stop == "ALL_LABELS_QUERIED" and bounds[best_by_bound][1] <= -eta)
    selected = best_by_bound if certified else identity_name
    if certified and stop == "ALL_LABELS_QUERIED":
        stop = "CERTIFIED"
    return {"selected": selected, "certified": bool(certified), "certificate_scope": "current_finite_batch",
            "stop_reason": stop, "queried_cases": list(oracle.queried), "total_cost": spent,
            "best_candidate_by_bound": best_by_bound, "guaranteed_gain": -bounds[best_by_bound][1],
            "final_bounds": {a: {"L": l, "U": u} for a, (l, u) in bounds.items()},
            "queried_rows": np.flatnonzero(bb.is_known).tolist(), "trace": trace, "rule": rule, "eta": eta}
