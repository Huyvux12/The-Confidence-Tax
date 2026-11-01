"""RB-Cal experiment on real data (PREREGISTRATION.md).

    python scripts/run_rbcal.py --data data/real/dev --out runs/rbcal_dev --workers 6

Fitting reads only source splits. Target predictions for every method are written and hashed
before target labels are opened by the evaluator.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_extensions import data_adapter as da  # noqa: E402
from calibration_extensions.calibrators import FitResult, IdentityCalibrator  # noqa: E402
from calibration_extensions.evaluator import paired_bootstrap, row_metrics, score  # noqa: E402
from calibration_extensions.features import fit_feature_map  # noqa: E402
from calibration_extensions.m1_functional import M1Config, diagnose_at_identity  # noqa: E402
from calibration_extensions.numerics import gather  # noqa: E402
from calibration_extensions.rb_calibration import (CONSTANT_MENU, RBF_MENU, TOPGAP_MENU, fit_rbcal,  # noqa: E402
                                                   fit_shrinkage_loso)
from calibration_extensions.source_objectives import beta_grid, calibrator_deltas, fit_scalar, fixed_gate  # noqa: E402
from calibration_extensions.source_splits import calibrator_units, unit_subset  # noqa: E402

EPS_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
UCB_EPS = (0.005, 0.01, 0.02)
UCB_Z = 1.645
PRIMARY = "RBCAL_UCB_RBF_e0.01"          # revised after dev, see PROTOCOL_LOG
PRIMARY_CONST = "RBCAL_UCB_CONST_e0.01"


def scalar_fit(mid, cal, info):
    return FitResult(mid, "source_only", cal, cal, cal.params(), {}, "grid", True, None, None,
                     "observed_source_losses", diagnostics=info)


def run_unit(args):
    data, unit, eps_grid, chosen = args
    t0 = time.time()

    def rbcal(menu, eps, mid, z=0.0):
        # `chosen` (method -> hyperparameters of an earlier run on the same source data) skips the
        # LOSO menu search and refits only the selected spec; selection depends on sources alone.
        if chosen is not None and len(menu) > 1:
            h = chosen[mid]
            menu = [(h["kind"], h["J"], h["sigma"], h["lam"])]
        return fit_rbcal(tr, sel, menu, eps, mid, z=z)

    where = {"model_id": unit[0], "prompt_id": unit[1]}
    src = {s: da.load_split(data, s, where) for s in ("source_train", "source_select", "source_audit")}
    tgt = da.load_split(data, "target_inputs", where)
    tr, sel = src["source_train"], src["source_select"]
    grid = beta_grid()
    fits = {"IDENTITY": scalar_fit("IDENTITY", IdentityCalibrator(), {})}
    for mid, rule in (("POOLED_TS", "pooled"), ("MRTS", "minimax")):
        cal, info = fit_scalar(tr, grid, rule)
        fits[mid] = scalar_fit(mid, cal, info)
        g = fixed_gate(sel, cal)
        fits[mid + "_GATED"] = FitResult(mid + "_GATED", "source_only", cal if g["accept"] else IdentityCalibrator(),
                                         cal, cal.params(), {}, "grid", True, g,
                                         None if g["accept"] else "SOURCE_GATE_REJECTED", "observed_source_losses")
    fits["SHRINKAGE_LOSO"] = fit_shrinkage_loso(tr, sel)
    for e in eps_grid:
        fits[f"RBCAL_CONST_e{e}"] = rbcal(CONSTANT_MENU, e, f"RBCAL_CONST_e{e}")
        fits[f"RBCAL_RBF_e{e}"] = rbcal(RBF_MENU, e, f"RBCAL_RBF_e{e}")
        fits[f"RBCAL_TOPGAP_e{e}"] = rbcal(TOPGAP_MENU, e, f"RBCAL_TOPGAP_e{e}")
    for e in UCB_EPS:
        for name, menu in (("CONST", CONSTANT_MENU), ("RBF", RBF_MENU), ("TOPGAP", TOPGAP_MENU)):
            mid = f"RBCAL_UCB_{name}_e{e}"
            fits[mid] = rbcal(menu, e, mid, z=UCB_Z)
    fits["POOLED_RBF"] = rbcal(RBF_MENU, None, "POOLED_RBF")
    fits["POOLED_TOPGAP"] = rbcal(TOPGAP_MENU, None, "POOLED_TOPGAP")
    fits["POOLED_AFFINE"] = fit_rbcal(tr, sel, [("entropy_affine", 0, None, 0.0)], None, "POOLED_AFFINE")
    diag = diagnose_at_identity(tr, fit_feature_map(tr, "constant"), M1Config())
    out = {"unit": unit, "regime": "consensus" if diag["common_descent"] else "conflict",
           "diag_status": diag["status"], "methods": {}, "seconds": None}
    for mid, fr in fits.items():
        rec = {"fallback": fr.fallback_reason, "solver_valid": fr.solver_valid,
               "hyper": fr.selected_hyperparameters, "params": fr.parameters,
               "gate": None if fr.source_gate is None else {k: fr.source_gate[k] for k in ("accept", "max_delta")},
               "audit_deltas": calibrator_deltas(src["source_audit"], fr.deployed),
               "train_deltas": calibrator_deltas(tr, fr.deployed),
               "target_log_q": fr.deployed.predict_log_proba(tgt),
               "target_log_q_pregate": (fr.fitted or fr.deployed).predict_log_proba(tgt)}
        out["methods"][mid] = rec
    out["target_sample_ids"] = [str(s) for s in tgt.sample_id]
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fit-only", action="store_true", help="cache per-unit fits and stop before labels")
    ap.add_argument("--reuse-selection", nargs=2, metavar=("DATA", "UNITS_DIR"),
                    help="take the menu choices from a finished run on DATA (source files must be identical)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if json.loads((Path(args.data) / "manifest.json").read_text()).get("sealed"):
        print("NOTE: running on the sealed confirmation split", flush=True)

    units = sorted({(r["model_id"], r["prompt_id"]) for r in da.read_jsonl(Path(args.data) / "source_select.jsonl")})
    cache = out / "units"
    cache.mkdir(exist_ok=True)
    cfile = lambda u: cache / (u[0].replace("/", "__") + "__" + u[1] + ".pkl")
    todo = [u for u in units if not cfile(u).exists()]
    print(f"{len(units)} units, {len(units) - len(todo)} cached", flush=True)
    chosen = {u: None for u in units}
    if args.reuse_selection:
        rdata, rdir = args.reuse_selection
        srcs = [da.SPLIT_FILES[s] for s in ("source_train", "source_select", "source_audit")]
        a, b = da.input_fingerprints(args.data), da.input_fingerprints(rdata)
        if any(a[f] != b[f] for f in srcs):
            raise SystemExit("source files differ from the reused run; selection cannot be reused")
        for u in units:
            prev = pickle.loads((Path(rdir) / cfile(u).name).read_bytes())
            chosen[u] = {m: rec["hyper"] for m, rec in prev["methods"].items()}
        print(f"reusing menu selection from {rdir} (sources identical)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(run_unit, (args.data, u, EPS_GRID, chosen[u])) for u in todo]
        for fut in as_completed(futures):       # cache each unit as soon as it finishes
            r = fut.result()
            print(f"done {r['unit']} regime={r['regime']} in {r['seconds'] / 60:.1f} min", flush=True)
            cfile(r["unit"]).write_bytes(pickle.dumps(r))
    if args.fit_only:
        print("FIT DONE", flush=True)
        return
    results = [pickle.loads(cfile(u).read_bytes()) for u in units]
    (out / "unit_results.pkl").write_bytes(pickle.dumps(results))

    # ---- assemble frozen target predictions, hash them, then open labels
    tgt = da.load_split(args.data, "target_inputs")
    pos = {str(s): i for i, s in enumerate(tgt.sample_id)}
    methods = sorted(results[0]["methods"])
    preds = {}
    for view in ("", "__PREGATE"):
        for m in methods:
            arr = [None] * tgt.n
            for r in results:
                key = "target_log_q" + ("_pregate" if view else "")
                for sid, lq in zip(r["target_sample_ids"], r["methods"][m][key]):
                    arr[pos[sid]] = lq
            preds[m + view] = arr
    h = hashlib.sha256(b"".join(np.asarray(lq).tobytes() for m in sorted(preds) for lq in preds[m])).hexdigest()
    (out / "prediction_hash.txt").write_text(h)
    labelled = da.attach_labels(tgt, da.load_private_labels(args.data))

    regime = {"|".join(r["unit"]): r["regime"] for r in results}
    metric_rows, loss = [], {}
    for m, p in preds.items():
        rows = score(labelled, p, {"method_id": m})
        for row in rows:
            row["regime"] = regime.get(f"{row['model_id']}|{row['prompt_id']}", "all")
        metric_rows += rows
        loss[m] = row_metrics(labelled, p)["nll"]
    import csv
    cols = list(dict.fromkeys(k for r in metric_rows for k in r))
    with open(out / "target_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(metric_rows)

    def unit_rows(m, reg=None):
        return [r for r in metric_rows if r["method_id"] == m and not r["task_id"].startswith("__")
                and (reg is None or r["regime"] == reg)]

    def summary(m, reg=None):
        rs = unit_rows(m, reg)
        d = np.array([r["delta_nll"] for r in rs])
        return {"n_units": len(rs), "macro_delta": float(d.mean()), "worst_delta": float(d.max()),
                "harm_001": int(sum(r["harm_001"] for r in rs)), "ece_15": float(np.mean([r["ece_15"] for r in rs]))}

    def paired(a, b, reg=None):
        mask = np.ones(labelled.n, bool) if reg is None else np.array(
            [regime[f"{labelled.model_id[i]}|{labelled.prompt_id[i]}"] == reg for i in range(labelled.n)])
        sub = labelled.subset(mask)
        return paired_bootstrap(sub, loss[a][mask], loss[b][mask], n_boot=args.n_boot)[-1]

    report = {"prediction_sha256": h, "data": args.data, "regimes": regime, "summaries": {}, "endpoints": {}}
    for reg in (None, "conflict", "consensus"):
        key = reg or "all"
        report["summaries"][key] = {m: summary(m, reg) for m in preds if unit_rows(m, reg)}
        if not unit_rows(PRIMARY, reg):
            continue
        p, mr, pt, rc = (report["summaries"][key][x] for x in (PRIMARY, "MRTS", "POOLED_TS", PRIMARY_CONST))
        e1 = paired(PRIMARY, "MRTS", reg)
        e3 = paired(PRIMARY, PRIMARY_CONST, reg)
        report["endpoints"][key] = {
            "E1_vs_MRTS": {"paired": e1, "harm": [p["harm_001"], mr["harm_001"]],
                           "pass": bool(e1["ci_high"] < 0 and p["harm_001"] <= mr["harm_001"])},
            "E2_vs_POOLED_TS": {"harm": [p["harm_001"], pt["harm_001"]], "worst": [p["worst_delta"], pt["worst_delta"]],
                                "gain_retained": (p["macro_delta"] / pt["macro_delta"]) if pt["macro_delta"] < 0 else None,
                                "pass": bool(p["harm_001"] <= pt["harm_001"] and p["worst_delta"] < pt["worst_delta"])},
            "E3_vs_RBCAL_CONST": {"paired": e3, "harm": [p["harm_001"], rc["harm_001"]],
                                  "pass": bool(e3["ci_high"] < 0 and p["harm_001"] <= rc["harm_001"])}}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps(report["endpoints"], indent=1, default=float))
    show = ["IDENTITY", "MRTS", "MRTS_GATED", "POOLED_TS", "SHRINKAGE_LOSO", "POOLED_AFFINE", "POOLED_RBF",
            "POOLED_TOPGAP"] + [f"RBCAL_{c}_e{e}" for c in ("CONST", "RBF", "TOPGAP") for e in EPS_GRID] +         [f"RBCAL_UCB_{c}_e{e}" for c in ("CONST", "RBF", "TOPGAP") for e in UCB_EPS]
    print(f"\n{'method':<24}{'macro dNLL':>11}{'worst':>9}{'harm':>6}{'ECE':>8}")
    for m in show:
        s = report["summaries"]["all"].get(m)
        if s:
            print(f"{m:<24}{s['macro_delta']:>+11.4f}{s['worst_delta']:>+9.4f}{s['harm_001']:>6}{s['ece_15']:>8.4f}")


if __name__ == "__main__":
    main()
