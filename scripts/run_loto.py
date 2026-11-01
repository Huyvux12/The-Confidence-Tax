"""Secondary analysis (todo.md item 3): leave-one-task-out over the 25 source tasks.

    python scripts/run_loto.py --out runs/loto --workers 5

For every unit (model x prompt) and source task t: fit on the other 24 tasks (source_train; source_select
for gates) and score every row of t (train + select + audit). RB-Cal-RBF keeps the spec chosen on all 25
sources in runs/rbcal_dev (t took part in that choice of four discrete hyperparameters; see todo.md).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_extensions import data_adapter as da  # noqa: E402
from calibration_extensions.calibrators import IdentityCalibrator  # noqa: E402
from calibration_extensions.evaluator import ece, row_metrics  # noqa: E402
from calibration_extensions.extra_baselines import fit_ats_p, fit_thermometer_p  # noqa: E402
from calibration_extensions.numerics import concat  # noqa: E402
from calibration_extensions.rb_calibration import CONSTANT_MENU, fit_rbcal, fit_shrinkage_loso  # noqa: E402
from calibration_extensions.source_objectives import beta_grid, fit_scalar  # noqa: E402

DATA = "data/real/dev"
FULL_RUN = Path("runs/rbcal_dev/units")
PRIMARY = "RBCAL_UCB_RBF_e0.01"
METHODS = ["IDENTITY", "POOLED_TS", "MRTS", "SHRINKAGE_LOSO", "RBCAL_UCB_CONST_e0.01", PRIMARY, "THERMO_P", "ATS_P"]
UCB_Z, EPS = 1.645, 0.01
_UNIT = {}


def unit_file(unit):
    return unit[0].replace("/", "__") + "__" + unit[1]


def load_unit(unit):
    if unit not in _UNIT:
        where = {"model_id": unit[0], "prompt_id": unit[1]}
        s = {k: da.load_split(DATA, k, where) for k in ("source_train", "source_select", "source_audit")}
        prev = pickle.loads((FULL_RUN / (unit_file(unit) + ".pkl")).read_bytes())
        _UNIT.clear()
        _UNIT[unit] = (s, prev["methods"][PRIMARY]["hyper"], prev["regime"])
    return _UNIT[unit]


def run_job(job):
    unit, task = job
    s, hyper, regime = load_unit(unit)
    keep = lambda ps, inside: ps.subset((ps.source_id == task) == inside)
    tr, sel = keep(s["source_train"], False), keep(s["source_select"], False)
    ev = concat([keep(s[k], True) for k in ("source_train", "source_select", "source_audit")])
    grid = beta_grid()
    cals = {"IDENTITY": IdentityCalibrator(),
            "POOLED_TS": fit_scalar(tr, grid, "pooled")[0],
            "MRTS": fit_scalar(tr, grid, "minimax")[0],
            "SHRINKAGE_LOSO": fit_shrinkage_loso(tr, sel).deployed,
            "RBCAL_UCB_CONST_e0.01": fit_rbcal(tr, sel, CONSTANT_MENU, EPS, "C", z=UCB_Z).deployed,
            PRIMARY: fit_rbcal(tr, sel, [(hyper["kind"], hyper["J"], hyper["sigma"], hyper["lam"])], EPS, "R",
                               z=UCB_Z).deployed,
            "THERMO_P": fit_thermometer_p(tr)[0],
            "ATS_P": fit_ats_p(tr)[0]}
    out = {"unit": list(unit), "task": task, "regime": regime, "n_rows": ev.n,
           "n_cases": len({str(c) for c in ev.case_id}), "methods": {}}
    for m, cal in cals.items():
        rm = row_metrics(ev, cal.predict_log_proba(ev))
        d = float(rm["delta"].mean())
        out["methods"][m] = {"delta_nll": d, "harm_001": int(d > 0.01), "ece_15": ece(rm["conf"], rm["correct"]),
                             "deployed": cal.kind}
    return out


def task_bootstrap(res, a, b, n_boot, seed=0):
    """Mean over (unit, task) of delta_a - delta_b; CI resamples tasks (all units of a task move together)."""
    tasks = sorted({r["task"] for r in res})
    per_task = np.array([np.mean([r["methods"][a]["delta_nll"] - r["methods"][b]["delta_nll"]
                                  for r in res if r["task"] == t]) for t in tasks])
    rng = np.random.default_rng(seed)
    boots = per_task[rng.integers(0, len(tasks), size=(n_boot, len(tasks)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    diffs = np.array([r["methods"][a]["delta_nll"] - r["methods"][b]["delta_nll"] for r in res])
    return {"mean_diff": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_tasks": len(tasks),
            "win": int(np.sum(diffs < -1e-4)), "tie": int(np.sum(np.abs(diffs) <= 1e-4)), "loss": int(np.sum(diffs > 1e-4))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/loto")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    rows = da.read_jsonl(Path(DATA) / "source_select.jsonl")
    units = sorted({(r["model_id"], r["prompt_id"]) for r in rows})
    tasks = sorted({r["source_id"] for r in rows})
    jfile = lambda j: out / "jobs" / f"{unit_file(j[0])}__{j[1]}.json"
    jobs = [(u, t) for u in units for t in tasks]          # grouped by unit so each worker reloads rarely
    todo = [j for j in jobs if not jfile(j).exists()]
    print(f"{len(units)} units x {len(tasks)} tasks = {len(jobs)} jobs, {len(jobs) - len(todo)} cached", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            jfile((tuple(r["unit"]), r["task"])).write_text(json.dumps(r, default=float))
            if k % 10 == 0 or k == len(futs):
                print(f"{k}/{len(futs)} jobs done", flush=True)
    res = [json.loads(jfile(j).read_text()) for j in jobs]

    report = {"n_jobs": len(res), "tasks": tasks, "summaries": {}, "primary_vs": {}}
    for reg in (None, "conflict", "consensus"):
        rs = [r for r in res if reg is None or r["regime"] == reg]
        key = reg or "all"
        report["summaries"][key] = {m: {"macro_delta": float(np.mean([r["methods"][m]["delta_nll"] for r in rs])),
                                        "worst_delta": float(np.max([r["methods"][m]["delta_nll"] for r in rs])),
                                        "harm_001": int(sum(r["methods"][m]["harm_001"] for r in rs)),
                                        "ece_15": float(np.mean([r["methods"][m]["ece_15"] for r in rs])),
                                        "n": len(rs)} for m in METHODS}
        report["primary_vs"][key] = {m: task_bootstrap(rs, PRIMARY, m, args.n_boot) for m in METHODS if m != PRIMARY}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key in ("all", "conflict", "consensus"):
        print(f"\n== {key} (n={report['summaries'][key][PRIMARY]['n']})")
        for m in METHODS:
            s = report["summaries"][key][m]
            print(f"  {m:<24}{s['macro_delta']:+.4f} worst={s['worst_delta']:+.4f} harm={s['harm_001']:>3} ece={s['ece_15']:.4f}")
        for m, v in report["primary_vs"][key].items():
            print(f"  primary - {m:<22}{v['mean_diff']:+.4f} [{v['ci_low']:+.4f},{v['ci_high']:+.4f}] "
                  f"W/T/L {v['win']}/{v['tie']}/{v['loss']}")


if __name__ == "__main__":
    main()
