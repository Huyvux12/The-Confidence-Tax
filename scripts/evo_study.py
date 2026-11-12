"""EvoLearn extension experiments (jobs run on Modal via scripts/modal_v4.py, job["module"] = "evo_study").

front   per unit: for an eps grid, solve the v4b problem on source_train with SLSQP and with CMA-ES (seed 1);
        log every CMA-ES evaluation as (pooled regret, worst-source bound) -> empirical Pareto front.
centers per unit (and per LOTO job): "Evo-RB" = CMA-ES over the RBF centres and log-bandwidth of the v4b field
        (outer, non-convex); inner SLSQP solves theta under the budget on source_train; fitness = pooled regret on
        source_select plus a penalty when some select source violates the G-sig rule; the deployment gate then uses
        source_audit (not used for fitting or fitness).

    python scripts/evo_study.py summarize-front
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v4_study  # noqa: E402

EPS_GRID = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
Z = 1.645
OUT = ROOT / "runs/evo_study"


def front_job(unit, spec_tuple):
    from calibration_extensions.evo_calibration import solve_cma
    from calibration_extensions.rb_calibration import Spec, solve
    spec = Spec(*spec_tuple)
    tr = v4_study.load_unit("dev", unit, ["source_train"])["source_train"]
    rows, front = [], []
    for eps in EPS_GRID:
        cal, info = solve(tr, spec, eps, Z)
        cal2, info2 = solve_cma(tr, spec, eps, Z, seed=1, log_front=True)
        front += info2.pop("front")
        rows.append({"eps": eps, "slsqp_valid": info["valid"], "slsqp_mean": info["train_mean"],
                     "slsqp_worst": info["train_worst"], "cma_valid": info2["valid"], "cma_mean": info2["train_mean"],
                     "cma_worst": info2["train_worst"], "cma_bound": info2["worst_bound"]})
    # non-dominated subset of all evaluated points (minimise both)
    pts = sorted(set((round(a, 6), round(b, 6)) for a, b in front))
    nd, best_b = [], float("inf")
    for a, b in pts:
        if b < best_b:
            nd.append((a, b))
            best_b = b
    return {"stage": "front", "unit": list(unit), "spec": list(spec.__dict__.values()), "rows": rows,
            "front": nd, "n_evals": len(front)}


# ---------------------------------------------------------------- Evo-RB: evolved centres and bandwidth

def evo_rb_fit(tr, sel, spec_tuple, eps=0.01, seed=1, maxfevals=300):
    """Outer CMA-ES over (centres, log sigma) of an rbf_oct field with J centres; returns the fitted calibrator."""
    import cma
    import numpy as np
    import gate_study
    from calibration_extensions.calibrators import FieldBeta
    from calibration_extensions.features import FeatureMap, features_for, fit_feature_map
    from calibration_extensions.rb_calibration import Spec, solve
    kind, J, sigma, lam = spec_tuple
    if kind == "constant" or J == 0:
        cal, info = solve(tr, Spec(*spec_tuple), eps, Z)
        return cal, {"status": "constant spec, no structure to evolve", "evals": 0}
    base = fit_feature_map(tr, kind, J, sigma)                  # farthest-point start
    d = base.centers.shape[1]
    x0 = np.r_[base.centers.ravel(), np.log(sigma)]

    def make_spec_cal(x):
        C = np.clip(x[:-1].reshape(J, d), -0.5, 1.5)
        s = float(np.exp(np.clip(x[-1], np.log(0.05), np.log(3.0))))
        return FeatureMap(kind, C, s)

    cache = {}

    def inner(x):
        key = np.round(x, 8).tobytes()
        if key not in cache:
            fmap = make_spec_cal(x)
            cal, info = solve_with_map(tr, fmap, lam, eps)
            cache[key] = cal
        return cache[key]

    def fitness(x):
        cal = inner(x)
        if cal is None:
            return 1e3
        st = gate_study.select_stats(sel, cal)
        pi = {w: v for w, v in st.items()}
        mean = float(np.mean([v["delta"] for v in st.values()]))
        viol = sum(max(v["delta"] - 1.645 * v["se"] - eps, 0.0) for v in st.values())
        return mean + 100.0 * viol

    best_x, best_f = x0.copy(), fitness(x0)
    es = cma.CMAEvolutionStrategy(x0, 0.05, {"seed": seed, "maxfevals": maxfevals, "verbose": -9})
    n = 0
    while not es.stop():
        X = es.ask()
        F = [fitness(x) for x in X]
        n += len(X)
        for x, f in zip(X, F):
            if f < best_f:
                best_x, best_f = np.array(x), f
        es.tell(X, F)
    cal = inner(best_x)
    return cal, {"status": f"CMA-ES outer {n} evals", "evals": n, "fitness_start": fitness(x0), "fitness_best": best_f,
                 "sigma": float(make_spec_cal(best_x).sigma)}


def solve_with_map(tr, fmap, lam, eps):
    """rb_calibration.solve for a given (evolved) feature map: same objective, constraints and solver."""
    import numpy as np
    from scipy.optimize import minimize
    from calibration_extensions.calibrators import FieldBeta
    from calibration_extensions.features import project_to_bounds
    from calibration_extensions.rb_calibration import BOUNDS, LinearLogitOracle
    lo, hi = BOUNDS
    Phi = fmap.phi(tr)
    oracle = LinearLogitOracle(tr, lambda g: Phi[g.idx][:, :, None] * g.logP[:, None, :])
    x0 = fmap.identity_theta()
    D = len(x0)
    fun = lambda x: float(oracle.pi @ oracle(x)[0]) + lam * float(x[1:] @ x[1:])
    jac = lambda x: oracle.pi @ oracle(x)[1] + lam * np.r_[0.0, 2 * x[1:]]
    cons = [{"type": "ineq", "fun": lambda x: eps - oracle.ucb(x, Z)[0], "jac": lambda x: -oracle.ucb(x, Z)[1]},
            {"type": "ineq", "fun": lambda x: np.array([(x[0] - lo) ** 2 - x[1:] @ x[1:], (hi - x[0]) ** 2 - x[1:] @ x[1:]]),
             "jac": lambda x: np.vstack([np.r_[2 * (x[0] - lo), -2 * x[1:]], np.r_[-2 * (hi - x[0]), -2 * x[1:]]])}]
    bounds = [(lo, hi)] + [(None, None)] * (D - 1)
    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    th = project_to_bounds(fmap, res.x, lo, hi)
    ok = bool(np.all(np.isfinite(th))) and float(np.max(oracle.ucb(th, Z)[0])) <= eps + 1e-7 and fun(th) <= fun(x0) + 1e-9
    return (FieldBeta(fmap, th) if ok else None), {"status": f"SLSQP:{res.status}"}


def evo_unit_job(phase, unit, spec):
    from calibration_extensions import data_adapter as da
    from calibration_extensions.calibrators import IdentityCalibrator
    from calibration_extensions.evaluator import row_metrics, score
    import gate_study
    s = v4_study.load_unit(phase, unit, ["source_train", "source_select", "source_audit", "target_inputs"])
    tgt = da.attach_labels(s["target_inputs"], da.load_private_labels(str(ROOT / v4_study.DATA[phase])))
    cal, info = evo_rb_fit(s["source_train"], s["source_select"], spec)
    gate = None if cal is None else gate_study.select_stats(s["source_audit"], cal)
    target, row_nll = {}, {}
    sids = [str(x) for x in tgt.sample_id]
    for name, c in (("IDENTITY", IdentityCalibrator()), ("V", cal)):
        if c is None:
            continue
        lq = c.predict_log_proba(tgt)
        rows = [r for r in score(tgt, lq, {}) if not r["task_id"].startswith("__")]
        target[name] = {r["task_id"]: {k: r[k] for k in ("nll", "brier", "ece_15", "accuracy", "delta_nll")} for r in rows}
        row_nll[name] = dict(zip(sids, row_metrics(tgt, lq)["nll"].tolist()))
    return {"stage": "evo_unit", "phase": phase, "unit": list(unit), "variant": "evo_rb", "spec": spec,
            "select": {"V": gate}, "target": target, "row_nll": row_nll, "fit": info}


def evo_loto_job(unit, task, spec=None, nested=False):
    from calibration_extensions.calibrators import IdentityCalibrator
    from calibration_extensions.evaluator import ece, row_metrics
    from calibration_extensions.numerics import concat
    import gate_study
    s = v4_study.load_unit("loto", unit, ["source_train", "source_select", "source_audit"])
    keep = lambda ps, inside: ps.subset((ps.source_id == task) == inside)
    tr, sel, aud = keep(s["source_train"], False), keep(s["source_select"], False), keep(s["source_audit"], False)
    if nested:
        if spec is not None:
            raise ValueError("Nested Evo-RB LOTO must not use the global spec")
        spec = v4_study.nested_loto_spec(tr, "v4b", task)
    elif spec is None:
        raise ValueError("Historical Evo-RB LOTO requires its global spec")
    ev = concat([keep(s[k], True) for k in s])
    cal, info = evo_rb_fit(tr, sel, spec)
    gate = None if cal is None else gate_study.select_stats(aud, cal)
    target = {}
    for name, c in (("IDENTITY", IdentityCalibrator()), ("V", cal)):
        if c is None:
            continue
        rm = row_metrics(ev, c.predict_log_proba(ev))
        target[name] = {task: {"nll": float(rm["nll"].mean()), "brier": float(rm["brier"].mean()),
                               "ece_15": ece(rm["conf"], rm["correct"]), "accuracy": float(rm["correct"].mean()),
                               "delta_nll": float(rm["delta"].mean())}}
    result = {"stage": "evo_loto_nested" if nested else "evo_loto",
              "unit": list(unit), "task": task, "variant": "evo_rb", "spec": spec,
              "select": {"V": gate}, "target": target, "fit": info}
    if nested:
        result["selection_protocol"] = "nested_loso_on_24_sources"
    return result


def run_job(job):
    if job["stage"] == "front":
        return front_job(tuple(job["unit"]), job["spec"])
    if job["stage"] == "evo_unit":
        return evo_unit_job(job["phase"], tuple(job["unit"]), job["spec"])
    if job["stage"] == "evo_loto_nested":
        return evo_loto_job(tuple(job["unit"]), job["task"], nested=True)
    if job["stage"] == "evo_loto":
        return evo_loto_job(tuple(job["unit"]), job["task"], job["spec"])
    raise ValueError(f"Unknown Evo-RB stage: {job['stage']}")


def jobs_for(stage, phase="dev"):
    hy = json.loads((ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8"))
    units = [u.split("|") for u in sorted(hy["units"])]
    if stage == "evo_loto_nested":
        return [{"module": "evo_study", "stage": stage, "unit": u, "task": t, "variant": "evo_rb"}
                for u in units for t in hy["tasks"]]
    if stage not in ("front", "evo_unit", "evo_loto"):
        raise ValueError(f"Unknown Evo-RB stage: {stage}")
    chosen = json.loads((ROOT / "runs/v4_study/chosen.json").read_text(encoding="utf-8"))["v4b"]
    if stage == "front":
        return [{"module": "evo_study", "stage": "front", "unit": u, "variant": "front", "spec": chosen["|".join(u)]}
                for u in units]
    if stage == "evo_unit":
        return [{"module": "evo_study", "stage": stage, "phase": phase, "unit": u, "variant": "evo_rb",
                 "spec": chosen["|".join(u)]} for u in units]
    return [{"module": "evo_study", "stage": stage, "unit": u, "task": t, "variant": "evo_rb",
             "spec": chosen["|".join(u)]} for u in units for t in hy["tasks"]]


def summarize_front():
    recs = [json.loads(l) for l in (OUT / "front.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gaps = []
    for r in recs:
        for row in r["rows"]:
            if row["slsqp_valid"] and row["cma_valid"]:
                gaps.append(row["cma_mean"] - row["slsqp_mean"])
    import numpy as np
    g = np.array(gaps)
    print(f"{len(recs)} units, {len(g)} (unit, eps) pairs: CMA - SLSQP pooled train regret: mean {g.mean():+.2e}, "
          f"min {g.min():+.2e}, max {g.max():+.2e}; CMA better by >1e-4 in {(g < -1e-4).sum()}, worse in {(g > 1e-4).sum()}")


if __name__ == "__main__":
    if sys.argv[1:] == ["summarize-front"]:
        summarize_front()
