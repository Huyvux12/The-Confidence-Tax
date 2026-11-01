"""Frozen calibrators: predict, serialise and reload without refitting (spec 8.2)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatureMap
from .numerics import ProbSet, normalise_rows, scatter, temper


class Calibrator:
    kind = "base"

    def log_q_groups(self, ps: ProbSet) -> list[np.ndarray]:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def arrays(self) -> dict:
        return {}

    def predict_log_proba(self, ps: ProbSet) -> list[np.ndarray]:
        return scatter(ps, self.log_q_groups(ps))

    def row_params(self, ps: ProbSet) -> list[dict]:
        return [{} for _ in range(ps.n)]


class IdentityCalibrator(Calibrator):
    kind = "identity"

    def log_q_groups(self, ps):
        return [g.logP.copy() for g in ps.groups]

    def row_params(self, ps):
        return [{"beta": 1.0} for _ in range(ps.n)]


class ScalarTS(Calibrator):
    kind = "scalar_ts"

    def __init__(self, beta: float):
        self.beta = float(beta)

    def log_q_groups(self, ps):
        return [temper(g.logP, self.beta) for g in ps.groups]

    def params(self):
        return {"beta": self.beta}

    def row_params(self, ps):
        return [{"beta": self.beta} for _ in range(ps.n)]


class FieldBeta(Calibrator):
    """beta(p) = phi(z(p)) @ theta (M1, M4-FIELD)."""
    kind = "field_beta"

    def __init__(self, fmap: FeatureMap, theta: np.ndarray):
        self.fmap, self.theta = fmap, np.asarray(theta, dtype=np.float64)

    def log_q_groups(self, ps):
        beta = self.fmap.beta(ps, self.theta)
        return [temper(g.logP, beta[g.idx]) for g in ps.groups]

    def params(self):
        return {"feature_map": self.fmap.to_dict(), "b0": float(self.theta[0]),
                "field_norm": float(np.linalg.norm(self.theta[1:]))}

    def arrays(self):
        out = {"theta": self.theta}
        if self.fmap.centers is not None:
            out["centers"] = self.fmap.centers
        return out

    def row_params(self, ps):
        return [{"beta": float(b)} for b in self.fmap.beta(ps, self.theta)]


def top_gap_parts(logP: np.ndarray):
    """Z' = z - d*1[k*], D = d*1[k*] so that u = a Z' + b D (spec 5.2)."""
    n = logP.shape[0]
    top = np.argmax(logP, axis=1)
    part = np.sort(logP, axis=1)
    d = part[:, -1] - part[:, -2]
    D = np.zeros_like(logP)
    D[np.arange(n), top] = d
    return logP - D, D, top, d


class TopGap(Calibrator):
    kind = "top_gap"

    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    def log_q_groups(self, ps):
        out = []
        for g in ps.groups:
            Zp, D, _, _ = top_gap_parts(g.logP)
            out.append(normalise_rows(self.a * Zp + self.b * D))
        return out

    def params(self):
        return {"a": self.a, "b": self.b}

    def row_params(self, ps):
        return [{"a": self.a, "b": self.b} for _ in range(ps.n)]


class BatchBeta(Calibrator):
    """One beta per target batch (M2); batches never seen keep beta = 1."""
    kind = "batch_beta"

    def __init__(self, batch_fields: tuple[str, ...], betas: dict[str, float]):
        self.batch_fields, self.betas = tuple(batch_fields), dict(betas)

    def batch_keys(self, ps):
        return np.array(["|".join(str(getattr(ps, f)[i]) for f in self.batch_fields) for i in range(ps.n)])

    def _beta(self, ps):
        return np.array([self.betas.get(k, 1.0) for k in self.batch_keys(ps)])

    def log_q_groups(self, ps):
        beta = self._beta(ps)
        return [temper(g.logP, beta[g.idx]) for g in ps.groups]

    def params(self):
        return {"batch_fields": list(self.batch_fields), "betas": self.betas}

    def row_params(self, ps):
        return [{"beta": float(b)} for b in self._beta(ps)]


@dataclass
class FitResult:
    """Spec 8.2. `deployed` is what predicts; `fitted` is the pre-gate solution.
    A fallback to identity is a valid method state; SOLVER_FAILURE is an implementation error."""
    method_id: str
    protocol: str
    deployed: Calibrator
    fitted: Calibrator | None
    parameters: dict
    selected_hyperparameters: dict
    solver_status: str
    solver_valid: bool
    source_gate: dict | None
    fallback_reason: str | None
    diagnostic_scope: str
    feature_map: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    input_fingerprints: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in (
            "method_id", "protocol", "parameters", "selected_hyperparameters", "solver_status",
            "solver_valid", "source_gate", "fallback_reason", "diagnostic_scope", "feature_map",
            "input_fingerprints")} | {"deployed_kind": self.deployed.kind}


def save_fitted(cal: Calibrator, destination, extra: dict | None = None) -> None:
    d = Path(destination)
    d.mkdir(parents=True, exist_ok=True)
    art = {"kind": cal.kind, "params": cal.params(), **(extra or {})}
    (d / "fit_artifact.json").write_text(json.dumps(art, indent=2, default=float), encoding="utf-8")
    np.savez(d / "fit_arrays.npz", **cal.arrays())


def load_fitted(destination) -> Calibrator:
    d = Path(destination)
    art = json.loads((d / "fit_artifact.json").read_text(encoding="utf-8"))
    arr = np.load(d / "fit_arrays.npz")
    p = art["params"]
    if art["kind"] == "identity":
        return IdentityCalibrator()
    if art["kind"] == "scalar_ts":
        return ScalarTS(p["beta"])
    if art["kind"] == "top_gap":
        return TopGap(p["a"], p["b"])
    if art["kind"] == "batch_beta":
        return BatchBeta(tuple(p["batch_fields"]), p["betas"])
    if art["kind"] == "field_beta":
        fm = p["feature_map"]
        fmap = FeatureMap(fm["kind"], arr["centers"] if "centers" in arr else None, fm["sigma"])
        return FieldBeta(fmap, arr["theta"])
    raise ValueError(art["kind"])
