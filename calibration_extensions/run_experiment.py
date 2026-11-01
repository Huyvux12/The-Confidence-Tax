"""Run one method branch end to end and write the artifacts of spec section 8.3.

    python -m calibration_extensions.run_experiment --config configs/toy_m1.json

Target labels are opened only after every source-only / target-unlabelled prediction
is frozen (a hash of those predictions is recorded first). M5 sees labels only via its oracle.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import data_adapter as da
from . import m1_functional as m1
from . import m2_target_gain as m2
from . import m3_top_gap as m3
from . import m4_label_budget as m4
from . import m5_active_certificate as m5
from .calibrators import FieldBeta, FitResult, IdentityCalibrator, TopGap, load_fitted, save_fitted
from .evaluator import METRIC_COLUMNS, paired_bootstrap, row_metrics, score, target_benefit_rule
from .numerics import ProbSet, check_log_q, concat, gather, losses, source_deltas, surprisal
from .source_objectives import beta_grid, calibrator_deltas, fit_scalar, fixed_gate
from .source_splits import assert_case_disjoint, calibrator_units, unit_subset
from .toy_data import corrupt_states, generate_scenario

SPLITS = ("source_train", "source_select", "source_audit")


# ---------------------------------------------------------------- small utilities

def make_cfg(cls, overrides: dict | None):
    names = {f.name for f in fields(cls)}
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in (overrides or {}).items() if k in names}
    return cls(**kw)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsonable(r.get(k)) for k in cols})


def peak_memory_mb() -> float | None:
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2 ** 20
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def unit_name(u) -> str:
    return f"{u[0]}__{u[1]}"


def unit_mask(ps: ProbSet, u) -> np.ndarray:
    return np.array([str(a) == u[0] and str(b) == u[1] for a, b in zip(ps.model_id, ps.prompt_id)])


# ---------------------------------------------------------------- the run

class Experiment:
    def __init__(self, config: dict):
        self.config = config
        self.method = config["method"].upper()
        self.out = Path(config["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(config["data_dir"])
        self.seed = int(config.get("seed", 0))
        g = config.get("grid", {})
        self.grid = beta_grid(g.get("lo", 0.25), g.get("hi", 4.0), g.get("n", 241))
        self.fits: dict[str, dict] = {}          # method_id -> {unit: FitResult}
        self.family: dict[str, str] = {}         # method_id -> role
        self.controls: dict[str, list[str]] = {}
        self.timing: dict[str, dict] = {}
        self.checks: list[dict] = []
        self.mechanism: list[dict] = []
        self.extra: dict[str, list] = {k: [] for k in ("target_gain_bounds", "adversary_budget",
                                                        "query_trace", "batch_certificates")}
        self.target_preds: dict[str, list] = {}
        self.target_meta: dict[str, dict] = {}

    # ---- data
    def load(self):
        gen = self.config.get("generate_toy")
        if gen:
            splits, target = generate_scenario(gen["scenario"], seed=gen.get("seed", self.seed),
                                               K=gen.get("K", 4))
            da.write_dataset(self.data_dir, splits, target,
                             {"generator": "toy_data", **gen, "label_provenance": "synthetic_known_truth"})
        self.src = da.load_sources(self.data_dir)
        self.target = da.load_split(self.data_dir, "target_inputs")
        assert_case_disjoint(*self.src.values(), self.target)
        self.corruption = None
        corr = self.config.get("m4", {}).get("corruption") if self.method == "M4" else None
        if corr:
            for k, split in enumerate(("source_train", "source_select")):
                self.src[split], info = corrupt_states(self.src[split], corr["frac"], corr.get("seed", 0) + k,
                                                       corr.get("by", "state"))
                self.corruption = (self.corruption or {}) | {split: info}
        self.units = calibrator_units(self.src["source_train"])

    def add(self, method_id: str, unit, fr: FitResult, role: str, controls=()):
        self.fits.setdefault(method_id, {})[unit] = fr
        self.family[method_id] = role
        self.controls[method_id] = list(controls)

    def timed(self, method_id: str, fn, *args, **kw):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        self.timing.setdefault(method_id, {"fit_seconds": 0.0, "n_solves": 0})
        self.timing[method_id]["fit_seconds"] += time.perf_counter() - t0
        self.timing[method_id]["n_solves"] += 1
        return out

    # ---- shared baselines (identical sources for every control)
    def fit_baselines(self, u, tr, sel):
        self.add("IDENTITY", u, FitResult("IDENTITY", "source_only", IdentityCalibrator(), IdentityCalibrator(),
                                          {}, {}, "none", True, None, None, "none"), "reference")
        for mid, rule in (("POOLED_TS", "pooled"), ("MRTS", "minimax")):
            cal, info = self.timed(mid, fit_scalar, tr, self.grid, rule)
            self.add(mid, u, FitResult(mid, "source_only", cal, cal, cal.params(), {"grid": len(self.grid)},
                                       "grid", True, None, None, "observed_source_losses", diagnostics=info),
                     "reference")
            gate = fixed_gate(sel, cal)
            self.add(f"{mid}_GATED", u, FitResult(f"{mid}_GATED", "source_only",
                                                  cal if gate["accept"] else IdentityCalibrator(), cal,
                                                  cal.params(), {}, "grid", True, gate,
                                                  None if gate["accept"] else "SOURCE_GATE_REJECTED",
                                                  "observed_source_losses"), "reference_same_gate")

    # ---- branches
    def run_m1(self, u, tr, sel):
        base = self.config.get("m1", {})
        for obj in ("MINIMAX", "MEAN_SAFE", "POOLED"):
            cfg = make_cfg(m1.M1Config, base | {"objective": obj, "seed": self.seed})
            fr = self.timed(f"M1_{obj}", m1.fit_m1, tr, sel, cfg, diagnose_classes=obj == "MINIMAX")
            self.add(fr.method_id, u, fr, "method" if obj != "POOLED" else "same_capacity_control",
                     ["MRTS", "MRTS_GATED", "POOLED_TS", "M1_POOLED"])
        cfg = make_cfg(m1.M1Config, base | {"objective": "MINIMAX", "function_class": "entropy_affine"})
        fr = self.timed("M1_MINIMAX_ENTROPY_AFFINE", m1.fit_m1, tr, sel, cfg, diagnose_classes=False)
        self.add(fr.method_id, u, fr, "ablation", ["MRTS", "M1_MINIMAX"])

    def run_m3(self, u, tr, sel):
        base = self.config.get("m3", {})
        for obj in ("MINIMAX", "MEAN_SAFE", "POOLED"):
            fr = self.timed(f"M3_{obj}", m3.fit_m3, tr, sel, make_cfg(m3.M3Config, base | {"objective": obj}))
            self.add(fr.method_id, u, fr, "method" if obj != "POOLED" else "same_capacity_control",
                     ["MRTS", "MRTS_GATED", "POOLED_TS", "M3_POOLED", "ENTROPY_AFFINE_POOLED"])
        cfg = m1.M1Config(objective="POOLED", function_class="entropy_affine", field_l2=(0.0,))
        fr = self.timed("ENTROPY_AFFINE_POOLED", m1.fit_m1, tr, sel, cfg, diagnose_classes=False)
        fr.method_id = "ENTROPY_AFFINE_POOLED"
        self.add("ENTROPY_AFFINE_POOLED", u, fr, "same_parameter_count_control")

    def run_m4(self, u, tr, sel):
        base = self.config.get("m4", {})
        for fc in base.get("function_classes", ["scalar", "field"]):
            for rho in base.get("rhos", [0.0, 0.05, 0.10, 0.20]):
                cfg = make_cfg(m4.M4Config, base | {"rho": rho, "function_class": fc})
                fr = self.timed(f"M4_{fc}_{rho}", m4.fit_m4, tr, sel, cfg)
                ctrl = f"M4_{fc.upper()}_rho0.00"
                self.add(fr.method_id, u, fr, "method" if rho > 0 else "rho0_control", ["MRTS", ctrl])
                if fr.fitted is not None:
                    for split in SPLITS:
                        risk = m4.robust_risk(self.src_u[split], fr.fitted, rho)
                        self.extra["adversary_budget"] += [r | {"unit": unit_name(u)} for r in
                                                           m4.adversary_budget_rows(fr.method_id, risk, split)]

    def run_m2(self, u, tr, sel, tgt):
        base = self.config.get("m2", {})
        for itype in m2.INTERVAL_TYPES:
            cfg = make_cfg(m2.M2Config, base | {"interval_type": itype, "seed": self.seed})
            model = self.timed(f"M2_{itype}", m2.fit_m2, tr, sel, cfg)
            fr = self.timed(f"M2_{itype}", m2.adapt_unlabeled, model, tgt)
            self.add(fr.method_id, u, fr, "method" if itype == "bootstrap_envelope" else "same_information_ablation",
                     ["MRTS", "MRTS_GATED", "M2_POINT", "M2_RAW", "M2_ENVELOPE_NO_BOOTSTRAP"])
            self.extra["target_gain_bounds"] += [r | {"method_id": fr.method_id, "unit": unit_name(u)}
                                                 for r in fr.diagnostics["bounds_table"]]

    # ---- predictions
    def predict_all(self):
        for mid, per_unit in self.fits.items():
            preds, pre = [None] * self.target.n, [None] * self.target.n
            params, fallback = [None] * self.target.n, [False] * self.target.n
            t0 = time.perf_counter()
            for u, fr in per_unit.items():
                idx = np.flatnonzero(unit_mask(self.target, u))
                if not idx.size:
                    continue
                tu = self.target.subset(idx)
                for j, (lq, rp) in enumerate(zip(fr.deployed.predict_log_proba(tu), fr.deployed.row_params(tu))):
                    preds[idx[j]], params[idx[j]], fallback[idx[j]] = lq, rp, fr.fallback_reason is not None
                if fr.fitted is not None:
                    for j, lq in enumerate(fr.fitted.predict_log_proba(tu)):
                        pre[idx[j]] = lq
            self.timing.setdefault(mid, {"fit_seconds": 0.0, "n_solves": 0})["predict_seconds"] = time.perf_counter() - t0
            self.target_preds[mid] = preds
            self.target_meta[mid] = {"params": params, "fallback": fallback}
            if all(p is not None for p in pre) and any(f for f in fallback):
                self.target_preds[f"{mid}__PREGATE"] = pre
                self.family[f"{mid}__PREGATE"] = "pregate_view"

    def prediction_hash(self) -> str:
        h = hashlib.sha256()
        for mid in sorted(self.target_preds):
            for lq in self.target_preds[mid]:
                h.update(np.asarray(lq, dtype=np.float64).tobytes())
        return h.hexdigest()

    # ---- M5 (needs the oracle, so runs after the frozen predictions)
    def run_m5(self, private):
        base = self.config.get("m5", {})
        budgets = base.get("budgets", [0, 8, 16, 32, 64])
        n_random = int(base.get("random_seeds", 20))
        eta = float(base.get("eta", 1e-6))
        cand_ids = base.get("candidates", ["IDENTITY", "POOLED_TS", "MRTS", "M1_MINIMAX", "M3_MINIMAX"])
        for u in self.units:
            idx_u = np.flatnonzero(unit_mask(self.target, u))
            tu = self.target.subset(idx_u)
            cands = {c.lower(): self.fits[c][u].fitted or IdentityCalibrator() for c in cand_ids if c in self.fits}
            cands["identity"] = IdentityCalibrator()
            for task in sorted({str(t) for t in tu.task_id}):
                rows = np.flatnonzero(np.array([str(t) == task for t in tu.task_id]))
                batch = tu.subset(rows)
                n_cases = len({str(c) for c in batch.case_id})
                for b in sorted({min(b, n_cases) for b in budgets}):
                    for rule, seeds in (("active", [0]), ("random", list(range(n_random)))):
                        for s in seeds:
                            oracle = m5.LabelOracle(batch, private)
                            t0 = time.perf_counter()
                            res = m5.select_with_oracle(cands, batch, oracle, b, eta, rule, self.seed + s)
                            dt = time.perf_counter() - t0
                            mid = f"M5_{rule.upper()}_b{b}"
                            self.timing.setdefault(mid, {"fit_seconds": 0.0, "n_solves": 0})
                            self.timing[mid]["fit_seconds"] += dt
                            self.timing[mid]["n_solves"] += 1
                            sel = cands[res["selected"]]
                            lq = sel.predict_log_proba(batch)
                            labelled = da.attach_labels(batch, private)
                            loss = losses(labelled, gather(labelled, lq))
                            true_delta = float((loss - surprisal(labelled)).mean())
                            unq = np.setdiff1d(np.arange(batch.n), res["queried_rows"])
                            unq_delta = float((loss - surprisal(labelled))[unq].mean()) if unq.size else None
                            all_true = {a: float((losses(labelled, cands[a].log_q_groups(labelled)) -
                                                  surprisal(labelled)).mean()) for a in cands}
                            false_cert = res["certified"] and all_true[res["selected"]] > -eta + 1e-9
                            self.extra["batch_certificates"].append({
                                "method_id": mid, "unit": unit_name(u), "task_id": task, "rule": rule, "seed": s,
                                "budget_cases": b, "n_cases": n_cases, "selected": res["selected"],
                                "certified": res["certified"], "stop_reason": res["stop_reason"],
                                "labels_used": len(res["queried_cases"]), "cost": res["total_cost"],
                                "guaranteed_gain": res["guaranteed_gain"], "true_delta_batch": true_delta,
                                "true_delta_unqueried": unq_delta, "false_certificate": false_cert,
                                "best_candidate_by_bound": res["best_candidate_by_bound"], "seconds": dt})
                            if rule == "active":
                                self.extra["query_trace"] += [{"method_id": mid, "unit": unit_name(u), "task_id": task,
                                                               "budget_cases": b, **t} for t in res["trace"]]
                            key = f"{mid}__seed{s}"
                            store = self.target_preds.setdefault(key, [None] * self.target.n)
                            for j, i in enumerate(idx_u[rows]):
                                store[i] = lq[j]
                            self.family[key] = "method" if rule == "active" else "random_query_baseline"
                            self.target_meta.setdefault(key, {"seed": s, "queries": 0, "protocol": "target_label_budget"})
                            self.target_meta[key]["queries"] += len(res["queried_cases"])

    # ---- outputs
    def source_metrics(self) -> list[dict]:
        rows = []
        for mid, per_unit in self.fits.items():
            for u, fr in per_unit.items():
                for split in SPLITS:
                    ps = unit_subset(self.src[split], u)
                    loss = losses(ps, fr.deployed.log_q_groups(ps)) if not isinstance(fr.deployed, m2.BatchBeta) \
                        else losses(ps, IdentityCalibrator().log_q_groups(ps))
                    base = surprisal(ps)
                    keys = np.array([str(s) for s in ps.source_id])
                    for s in sorted(set(keys.tolist())):
                        m = keys == s
                        rows.append({"method_id": mid, "model_id": u[0], "prompt_id": u[1], "source_id": s,
                                     "split": split, "n_rows": int(m.sum()),
                                     "n_cases": len({str(c) for c in ps.case_id[m]}),
                                     "nll": float(loss[m].mean()), "delta_nll": float((loss - base)[m].mean()),
                                     "labels_corrupted": bool(self.corruption and split in self.corruption)})
        return rows

    def run_checks(self, labelled_target: ProbSet):
        tid = self.target_preds["IDENTITY"]
        err = max(float(np.max(np.abs(a - b))) for a, b in zip(tid, self.target.log_p))
        self.checks.append({"check": "identity_log_q_equals_log_p", "method_id": "IDENTITY", "passed": err <= 1e-12,
                            "detail": err})
        for mid, preds in self.target_preds.items():
            if any(p is None for p in preds):
                self.checks.append({"check": "prediction_complete", "method_id": mid, "passed": False})
                continue
            c = check_log_q(gather(self.target, preds))
            self.checks.append({"check": "q_sums_to_one_and_finite", "method_id": mid, "passed": c["ok"], "detail": c})
        for mid, per_unit in self.fits.items():
            for u, fr in per_unit.items():
                self.checks.append({"check": "solver_valid", "method_id": mid, "unit": unit_name(u),
                                    "passed": fr.solver_valid, "detail": fr.solver_status})
                d = self.out / "fits" / mid / unit_name(u)
                save_fitted(fr.deployed, d / "deployed", {"method_id": mid, "summary": jsonable(fr.summary())})
                if fr.fitted is not None:
                    save_fitted(fr.fitted, d / "fitted", {"method_id": mid})
                tu = self.target.subset(unit_mask(self.target, u))
                a = fr.deployed.predict_log_proba(tu)
                b = load_fitted(d / "deployed").predict_log_proba(tu)
                err = max((float(np.max(np.abs(x - y))) for x, y in zip(a, b)), default=0.0)
                self.checks.append({"check": "save_load_roundtrip", "method_id": mid, "unit": unit_name(u),
                                    "passed": err <= 1e-12, "detail": err})
                if mid.startswith("M1") and "selected" in fr.diagnostics:
                    dg = fr.diagnostics["selected"]
                    self.checks.append({"check": "m1_gradient_finite_difference", "method_id": mid, "unit": unit_name(u),
                                        "passed": dg["gradient_check_max_abs_error"] <= 1e-6,
                                        "detail": dg["gradient_check_max_abs_error"]})
                    self.checks.append({"check": "m1_simplex_qp_kkt", "method_id": mid, "unit": unit_name(u),
                                        "passed": dg["qp"]["sum_alpha_error"] <= 1e-9 and dg["qp"]["min_alpha"] >= 0,
                                        "detail": dg["qp"]})
                if isinstance(fr.fitted, TopGap):
                    mats = fr.fitted.log_q_groups(tu)
                    rc = m3.rank_change_rate(tu, mats)
                    self.checks.append({"check": "m3_rank_preserved", "method_id": mid, "unit": unit_name(u),
                                        "passed": rc == 0.0, "detail": rc})
                if mid.startswith("M4") and fr.fitted is not None:
                    r = m4.robust_risk(unit_subset(self.src["source_train"], u), IdentityCalibrator(),
                                       fr.selected_hyperparameters["rho_assumed"])
                    worst = max(abs(v["R_rho"]) for v in r.values())
                    self.checks.append({"check": "m4_identity_robust_risk_zero", "method_id": mid, "unit": unit_name(u),
                                        "passed": worst <= 1e-12, "detail": worst})
        self.checks.append({"check": "target_inputs_have_no_y", "passed": not self.target.labelled})
        self.checks.append({"check": "labels_opened_after_freeze", "passed": True,
                            "detail": {"prediction_sha256_before_labels": self.pred_hash}})

    def mechanism_metrics(self, labelled_target: ProbSet):
        for mid, per_unit in self.fits.items():
            for u, fr in per_unit.items():
                base = {"method_id": mid, "unit": unit_name(u)}
                if mid.startswith("M1") and "selected" in fr.diagnostics:
                    for k, d in fr.diagnostics.items():
                        if k.startswith("class_") or k == "selected":
                            self.mechanism.append(base | {"scope": k, "metric": "common_descent_status", "value": d["status"]})
                            self.mechanism.append(base | {"scope": k, "metric": "rho", "value": d["rho"]})
                    if isinstance(fr.fitted, FieldBeta):
                        for split in SPLITS + ("target",):
                            ps = unit_subset(self.src[split], u) if split != "target" else \
                                self.target.subset(unit_mask(self.target, u))
                            beta = fr.fitted.fmap.beta(ps, fr.fitted.theta)
                            self.mechanism.append(base | {"scope": split, "metric": "beta_spread", "value": float(beta.max() - beta.min())})
                if isinstance(fr.fitted, TopGap):
                    lt = labelled_target.subset(unit_mask(labelled_target, u))
                    for which, cal in (("fitted", fr.fitted), ("identity", IdentityCalibrator())):
                        dec = m3.nll_decomposition(lt, cal.log_q_groups(lt))
                        for k, v in dec.items():
                            self.mechanism.append(base | {"scope": f"target_{which}", "metric": k, "value": v})
                    self.mechanism.append(base | {"scope": "fit", "metric": "log_a_over_b",
                                                  "value": float(np.log(fr.fitted.a / fr.fitted.b))})
                    self.mechanism.append(base | {"scope": "fit", "metric": "effective_dof", "value": fr.parameters.get("effective_dof")})
                if mid.startswith("M2"):
                    lt = labelled_target.subset(unit_mask(labelled_target, u))
                    for key, idx in m2.batches(lt, fr.fitted.batch_fields).items():
                        b = lt.subset(idx)
                        itv = fr.diagnostics["intervals"][key]
                        true_m = float(surprisal(b).mean())
                        beta = fr.fitted.betas.get(key, 1.0)
                        true_delta = float(m2.mean_lse(b, beta) + (beta - 1) * true_m)
                        pred_ub = None
                        if itv.get("m_lower") is not None:
                            pred_ub = m2.upper_delta(m2.mean_lse(b, beta), beta, itv["m_lower"], itv["m_upper"])
                        covered = itv.get("m_lower") is not None and itv["m_lower"] <= true_m <= itv["m_upper"]
                        self.mechanism.append(base | {"scope": key, "metric": "interval", "value": json.dumps(jsonable({
                            "true_m": true_m, "m_lower": itv.get("m_lower"), "m_upper": itv.get("m_upper"),
                            "interval_covered": covered, "selected_beta": beta, "true_delta": true_delta,
                            "predicted_upper_delta": pred_ub, "unsupported_mass": itv.get("unsupported_mass"),
                            "false_acceptance": bool(beta != 1.0 and true_delta > 0),
                            "abs_error_m": abs(0.5 * (itv["m_lower"] + itv["m_upper"]) - true_m) if itv.get("m_lower") is not None else None}))})
                        for r in self.extra["target_gain_bounds"]:
                            if r["method_id"] == mid and r["batch"] == key and r["unit"] == unit_name(u):
                                r["true_m"], r["interval_covered"] = true_m, covered
                                r["true_delta"] = float(r["A_bar"] + (r["beta"] - 1) * true_m)

    def mechanism_value(self, mid, metric, scope=None):
        return [r for r in self.mechanism if r["method_id"] == mid and r["metric"] == metric
                and (scope is None or r["scope"] == scope)]

    def audit_worst(self, mid) -> float:
        out = []
        for u, fr in self.fits[mid].items():
            ps = unit_subset(self.src["source_audit"], u)
            out.append(max(calibrator_deltas(ps, fr.deployed).values()))
        return max(out)

    def decide(self, metrics: dict, paired: dict) -> list[dict]:
        decisions = []
        failed = {c.get("method_id") for c in self.checks if not c["passed"]}
        exploratory = bool(self.config.get("generate_toy")) or self.config.get("exploratory", True)
        for mid, role in self.family.items():
            if role not in ("method",) or mid.endswith("__PREGATE"):
                continue
            base_mid = mid.split("__seed")[0]
            per_unit = self.fits.get(mid, {})
            invalid = any(not fr.solver_valid for fr in per_unit.values())
            impl = "INVALID" if invalid else ("FAIL" if mid in failed else "PASS")
            mech, mech_why = self.mechanism_decision(base_mid)
            tb, tb_why = "NOT_RUN", "MRTS comparison missing"
            if mid in paired and "MRTS" in metrics:
                tb, tb_why = target_benefit_rule(paired[mid], metrics[mid]["harm_001"], metrics["MRTS"]["harm_001"])
            decisions.append({"method_id": base_mid, "implementation": impl, "mechanism": mech, "target_benefit": tb,
                              "reason": f"mechanism: {mech_why} | target: {tb_why}",
                              "claim_scope": "exploratory_toy_data" if exploratory else "confirmatory_preregistered",
                              "evidence_files": ["checks.json", "mechanism_metrics.csv", "paired_comparisons.csv",
                                                 "target_metrics.csv"]})
        return decisions

    def mechanism_decision(self, mid) -> tuple[str, str]:
        """Pre-declared, method-specific rules on held-out (audit or frozen-target) evidence."""
        try:
            if mid.startswith("M1"):
                w, r = self.audit_worst(mid), self.audit_worst("MRTS_GATED")
                ok = w <= 1e-8 and w < r - 1e-8
                return ("PASS" if ok else "FAIL"), f"rule: audit worst-source delta {w:.4g} <= 0 and < MRTS_GATED {r:.4g}"
            if mid.startswith("M3"):
                lab = [abs(x["value"]) for x in self.mechanism_value(mid, "log_a_over_b")]
                w, r = self.audit_worst(mid), self.audit_worst("MRTS_GATED")
                ok = min(lab) > 0.05 and w < r - 1e-8
                return ("PASS" if ok else "FAIL"), f"rule: |log a/b| {min(lab):.3g} > 0.05 and audit worst {w:.4g} < MRTS_GATED {r:.4g}"
            if mid.startswith("M4"):
                ctrl = self.controls[mid][1]
                w, r = self.audit_worst(mid), self.audit_worst(ctrl)
                ok = w < r - 1e-8
                return ("PASS" if ok else "FAIL"), f"rule: clean-audit worst delta {w:.4g} < {ctrl} {r:.4g} (train/select corrupted: {bool(self.corruption)})"
            if mid.startswith("M2"):
                recs = [json.loads(x["value"]) for x in self.mechanism_value(mid, "interval")]
                cov = np.mean([r["interval_covered"] for r in recs])
                fa = np.mean([r["false_acceptance"] for r in recs])
                moved = np.mean([r["selected_beta"] != 1.0 for r in recs])
                ok = cov >= 0.9 and fa == 0 and moved > 0
                return ("PASS" if ok else "FAIL"), f"rule: coverage {cov:.2f} >= 0.9, false acceptance {fa:.2f} == 0, non-identity rate {moved:.2f} > 0"
            if mid.startswith("M5"):
                b = mid.split("_b")[-1].split("__")[0]
                rows = [r for r in self.extra["batch_certificates"] if r["budget_cases"] == int(b)]
                act = [r for r in rows if r["rule"] == "active"]
                rnd = [r for r in rows if r["rule"] == "random"]
                fc = any(r["false_certificate"] for r in rows)
                a, rr = np.mean([r["certified"] for r in act]), np.mean([r["certified"] for r in rnd])
                ok = not fc and a > rr
                return ("PASS" if ok else "FAIL"), f"rule: no false certificate ({not fc}) and active certified {a:.2f} > random {rr:.2f}"
        except (KeyError, ValueError) as e:
            return "NOT_RUN", f"missing evidence: {e}"
        return "NOT_RUN", "no mechanism rule"

    def run(self):
        t_start = time.perf_counter()
        self.load()
        for u in self.units:
            self.src_u = {k: unit_subset(v, u) for k, v in self.src.items()}
            tr, sel = self.src_u["source_train"], self.src_u["source_select"]
            self.fit_baselines(u, tr, sel)
            if self.method in ("M1", "M5"):
                self.run_m1(u, tr, sel)
            if self.method in ("M3", "M5"):
                self.run_m3(u, tr, sel)
            if self.method == "M4":
                self.run_m4(u, tr, sel)
            if self.method == "M2":
                tgt = self.target.subset(unit_mask(self.target, u))
                self.run_m2(u, tr, sel, tgt)
        self.predict_all()
        self.pred_hash = self.prediction_hash()
        # ---- labels are opened only from here on
        private = da.load_private_labels(self.data_dir)
        if self.method == "M5":
            for mid in list(self.family):
                if self.family[mid] == "method" and not mid.startswith("M5"):
                    self.family[mid] = "m5_candidate"
            self.run_m5(private)
        labelled = da.attach_labels(self.target, private)
        self.run_checks(labelled)
        self.mechanism_metrics(labelled)

        metric_rows, macro, loss_by = [], {}, {}
        for mid, preds in self.target_preds.items():
            if any(p is None for p in preds):
                continue
            meta = self.target_meta.get(mid, {})
            per_unit = self.fits.get(mid.split("__")[0], {})
            frs = list(per_unit.values())
            proto = meta.get("protocol") or (frs[0].protocol if frs else "source_only")
            row_meta = {"method_id": mid.split("__seed")[0], "protocol": proto, "split_id": "target",
                        "seed": meta.get("seed", self.seed), "fallback": any(fr.fallback_reason for fr in frs),
                        "solver_valid": all(fr.solver_valid for fr in frs) if frs else True,
                        "target_labels_used": proto == "target_label_budget",
                        "label_query_cases": meta.get("queries", 0),
                        "fit_seconds": self.timing.get(mid.split("__")[0], {}).get("fit_seconds"),
                        "predict_seconds": self.timing.get(mid.split("__")[0], {}).get("predict_seconds")}
            rows = score(labelled, preds, row_meta)
            metric_rows += rows
            macro[mid] = next(r for r in rows if r["task_id"] == "__macro__")
            loss_by[mid] = row_metrics(labelled, preds)["nll"]

        paired_rows, paired_macro = [], {}
        refs = ["MRTS", "MRTS_GATED", "POOLED_TS", "IDENTITY"]
        for mid in loss_by:
            if mid in refs or (self.family.get(mid) == "random_query_baseline" and not mid.endswith("__seed0")):
                continue
            base_mid = mid.split("__")[0]
            for ref in refs + [c for c in self.controls.get(base_mid, []) if c not in refs]:
                if ref not in loss_by or ref == base_mid:
                    continue
                res = paired_bootstrap(labelled, loss_by[mid], loss_by[ref], seed=self.seed)
                paired_rows += [{"method_id": mid, "control_id": ref, **r} for r in res]
                if ref == "MRTS":
                    paired_macro[mid] = res[-1]

        decisions = self.decide(macro, paired_macro)
        self.write_outputs(metric_rows, paired_rows, decisions, time.perf_counter() - t_start)
        return decisions

    def write_outputs(self, metric_rows, paired_rows, decisions, total_seconds):
        o = self.out
        write_json(o / "resolved_config.json", {
            "config": self.config, "method": self.method, "seed": self.seed, "beta_grid": [float(self.grid[0]), float(self.grid[-1]), len(self.grid)],
            "units": [list(u) for u in self.units], "corruption": self.corruption,
            "input_fingerprints": da.input_fingerprints(self.data_dir),
            "protocols": {mid: next(iter(p.values())).protocol for mid, p in self.fits.items()},
            "gate_rule": "max_w Delta_w(select) <= epsilon_select + 1e-8, fixed before select is read",
            "endpoint": "macro delta NLL vs MRTS, paired case bootstrap 95% CI excludes 0, harm_001 not higher"})
        write_json(o / "fit_artifact.json", {mid: {unit_name(u): fr.summary() for u, fr in p.items()}
                                             for mid, p in self.fits.items()})
        arrays = {}
        for mid, p in self.fits.items():
            for u, fr in p.items():
                for k, v in (fr.fitted.arrays() if fr.fitted is not None else {}).items():
                    arrays[f"{mid}/{unit_name(u)}/{k}"] = v
        np.savez(o / "fit_arrays.npz", **arrays)
        write_json(o / "diagnostics.json", {mid: {unit_name(u): {"diagnostic_scope": fr.diagnostic_scope,
                                                                  "fallback_reason": fr.fallback_reason,
                                                                  "solver_status": fr.solver_status,
                                                                  "source_gate": fr.source_gate,
                                                                  **fr.diagnostics} for u, fr in p.items()}
                                            for mid, p in self.fits.items()})
        write_csv(o / "source_metrics.csv", self.source_metrics())
        with open(o / "predictions.jsonl", "w", encoding="utf-8") as f:
            for mid, preds in self.target_preds.items():
                meta = self.target_meta.get(mid, {})
                for i, lq in enumerate(preds):
                    if lq is None:
                        continue
                    f.write(json.dumps(jsonable({
                        "sample_id": self.target.sample_id[i], "method_id": mid, "log_q": lq,
                        "argmax": int(np.argmax(lq)), "params": (meta.get("params") or [None] * self.target.n)[i],
                        "fallback": (meta.get("fallback") or [False] * self.target.n)[i]})) + "\n")
        write_csv(o / "target_metrics.csv", metric_rows, METRIC_COLUMNS)
        write_csv(o / "paired_comparisons.csv", paired_rows)
        write_csv(o / "mechanism_metrics.csv", self.mechanism)
        write_json(o / "runtime.json", {"total_seconds": total_seconds, "methods": self.timing,
                                        "peak_memory_mb": peak_memory_mb(), "cpu": platform.processor(),
                                        "python": sys.version, "n_target_rows": self.target.n,
                                        "n_target_tasks": len({str(t) for t in self.target.task_id}),
                                        "gpu": None, "llm_inference": None})
        write_json(o / "checks.json", {"all_passed": all(c["passed"] for c in self.checks), "checks": self.checks})
        write_json(o / "decision.json", decisions)
        if self.method == "M2":
            write_csv(o / "target_gain_bounds.csv", self.extra["target_gain_bounds"])
        if self.method == "M4":
            write_csv(o / "adversary_budget.csv", self.extra["adversary_budget"])
        if self.method == "M5":
            with open(o / "query_trace.jsonl", "w", encoding="utf-8") as f:
                for r in self.extra["query_trace"]:
                    f.write(json.dumps(jsonable(r)) + "\n")
            write_csv(o / "batch_certificates.csv", self.extra["batch_certificates"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    decisions = Experiment(config).run()
    for d in decisions:
        print(f"{d['method_id']:<34} impl={d['implementation']:<7} mech={d['mechanism']:<7} target={d['target_benefit']}")


if __name__ == "__main__":
    main()
