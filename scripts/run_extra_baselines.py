"""Secondary analysis (todo.md item 1): THERMO_P and ATS_P against RB-Cal on the dev and confirm targets.

    python scripts/run_extra_baselines.py --out runs/extra_baselines

Fits read source_train only (byte-identical in dev and confirm_sealed). RB-Cal and the other methods
are taken from the frozen runs runs/rbcal_dev and runs/rbcal_confirm; nothing there is refitted.
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
from calibration_extensions.evaluator import paired_bootstrap, row_metrics, score  # noqa: E402
from calibration_extensions.extra_baselines import fit_ats_p, fit_thermometer_p  # noqa: E402

SPLITS = {"dev": ("data/real/dev", "runs/rbcal_dev"), "confirm": ("data/real/confirm_sealed", "runs/rbcal_confirm")}
PRIMARY = "RBCAL_UCB_RBF_e0.01"
COMPARE = ["IDENTITY", "MRTS", "POOLED_TS", "SHRINKAGE_LOSO", "RBCAL_UCB_CONST_e0.01", PRIMARY]
NEW = ["THERMO_P", "ATS_P"]


def fit_unit(unit):
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    tr = da.load_split(SPLITS["dev"][0], "source_train", where)
    out = {"unit": unit, "fits": {}, "pred": {}}
    for mid, fit in (("THERMO_P", fit_thermometer_p), ("ATS_P", fit_ats_p)):
        cal, info = fit(tr)
        out["fits"][mid] = {"params": cal.params(), "info": info}
        for split, (data, _) in SPLITS.items():
            tgt = da.load_split(data, "target_inputs", where)
            out["pred"][(split, mid)] = dict(zip([str(s) for s in tgt.sample_id], cal.predict_log_proba(tgt)))
    return out


def sign_counts(rows, a, b):
    d = {}
    for r in rows:
        d.setdefault((r["model_id"], r["prompt_id"], r["task_id"]), {})[r["method_id"]] = r["delta_nll"]
    w = sum(v[a] < v[b] - 1e-4 for v in d.values())
    l = sum(v[a] > v[b] + 1e-4 for v in d.values())
    return {"win": int(w), "tie": int(len(d) - w - l), "loss": int(l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/extra_baselines")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units = sorted({(r["model_id"], r["prompt_id"]) for r in da.read_jsonl(Path(SPLITS["dev"][0]) / "source_select.jsonl")})
    fits = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fit_unit, u) for u in units]):
            r = fut.result()
            fits[r["unit"]] = r
            print("fitted", r["unit"], {m: f["info"]["status"] for m, f in r["fits"].items()}, flush=True)
    (out / "fits.json").write_text(json.dumps({"|".join(u): f["fits"] for u, f in fits.items()}, indent=1, default=float))

    report = {}
    for split, (data, run) in SPLITS.items():
        tgt = da.load_split(data, "target_inputs")
        labelled = da.attach_labels(tgt, da.load_private_labels(data))
        sids = [str(s) for s in tgt.sample_id]
        unit_res = pickle.loads((Path(run) / "unit_results.pkl").read_bytes())
        regime = {"|".join(r["unit"]): r["regime"] for r in unit_res}
        frozen = {}
        for r in unit_res:
            for m in COMPARE:
                frozen.setdefault(m, {}).update(zip(r["target_sample_ids"], r["methods"][m]["target_log_q"]))
        preds = {m: [frozen[m][s] for s in sids] for m in COMPARE}
        for m in NEW:
            merged = {}
            for f in fits.values():
                merged.update(f["pred"][(split, m)])
            preds[m] = [merged[s] for s in sids]
        rows, loss = [], {}
        for m, p in preds.items():
            for row in score(labelled, p, {"method_id": m}):
                if not row["task_id"].startswith("__"):
                    row["regime"] = regime[f"{row['model_id']}|{row['prompt_id']}"]
                    rows.append(row)
            loss[m] = row_metrics(labelled, p)["nll"]
        rep = {}
        for reg in (None, "conflict", "consensus"):
            rs = [r for r in rows if reg is None or r["regime"] == reg]
            mask = np.array([reg is None or regime[f"{labelled.model_id[i]}|{labelled.prompt_id[i]}"] == reg
                             for i in range(labelled.n)])
            sub = labelled.subset(mask)
            summ = {}
            for m in preds:
                d = np.array([r["delta_nll"] for r in rs if r["method_id"] == m])
                summ[m] = {"macro_delta": float(d.mean()), "worst_delta": float(d.max()),
                           "harm_001": int(sum(r["harm_001"] for r in rs if r["method_id"] == m)),
                           "ece_15": float(np.mean([r["ece_15"] for r in rs if r["method_id"] == m])), "n": len(d)}
            vs = {}
            for m in NEW + ["POOLED_TS"]:
                pb = paired_bootstrap(sub, loss[PRIMARY][mask], loss[m][mask], n_boot=args.n_boot)[-1]
                vs[m] = {"paired_primary_minus_m": pb, "signs": sign_counts(rs, PRIMARY, m)}
            rep[reg or "all"] = {"summary": summ, "primary_vs": vs}
        report[split] = rep
        with open(out / f"metrics_{split}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=0, default=float)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for split in SPLITS:
        print(f"\n== {split}")
        for reg in ("all", "conflict", "consensus"):
            s = report[split][reg]["summary"]
            print(f"  [{reg}] " + "  ".join(f"{m}:{s[m]['macro_delta']:+.4f}/h{s[m]['harm_001']}" for m in COMPARE + NEW))
            for m, v in report[split][reg]["primary_vs"].items():
                p = v["paired_primary_minus_m"]
                print(f"     primary - {m:<10} {p['mean_diff']:+.4f} [{p['ci_low']:+.4f},{p['ci_high']:+.4f}] {v['signs']}")


if __name__ == "__main__":
    main()
