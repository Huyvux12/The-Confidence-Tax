"""Pre-registered endpoints Q1-Q3 on the confirm3 roster (PROTOCOL_LOG 2026-11-06).

Needs runs/rbcal_confirm3/{units,extra} (all methods + THERMO_P/ATS_P), runs/gate_study/confirm3.jsonl (v3 gate
statistics) and runs/v4_study/final_confirm3.jsonl (v4b fits with row-level NLL).

    python scripts/score_confirm3.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibration_extensions import data_adapter as da  # noqa: E402
from calibration_extensions.evaluator import row_metrics, score  # noqa: E402
import gate_study  # noqa: E402

PHASE = "confirm3"
DATA = ROOT / "data/real/confirm3_sealed"
RUN = ROOT / "runs/rbcal_confirm3"
V1, V3, V4 = "RBCAL_UCB_RBF_e0.01", "V3_UCB_RBF_GSIG", "V4B_ORDER_CONSISTENCY"
SHOW = ["IDENTITY", "POOLED_TS", "MRTS", "SHRINKAGE_LOSO", "THERMO_P", "ATS_P", "RBCAL_UCB_CONST_e0.01", V1, V3, V4]
HARM, N_BOOT, SEED = 0.01, 5000, 20261107


def main():
    tgt = da.load_split(str(DATA), "target_inputs")
    lab = da.attach_labels(tgt, da.load_private_labels(str(DATA)))
    sids = [str(s) for s in tgt.sample_id]
    pos = {s: i for i, s in enumerate(sids)}
    units = [pickle.loads(Path(p).read_bytes()) for p in sorted(glob.glob(str(RUN / "units/*.pkl")))]
    extra = [pickle.loads(Path(p).read_bytes()) for p in sorted(glob.glob(str(RUN / "extra/*.pkl")))]
    gates = {"|".join(r["unit"]): r for r in gate_study.load_records(PHASE)}
    v4 = [json.loads(l) for l in (ROOT / "runs/v4_study/final_confirm3.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip() and json.loads(l)["variant"] == "v4b"]
    assert len(units) == len(extra) == len(gates) == len(v4) == 10

    preds = {m: [None] * tgt.n for m in SHOW if m != V4}
    accept = {"v3": {}, "v4b": {}}
    for u in units:
        key = "|".join(u["unit"])
        acc = gate_study.accept(gates[key]["select"]["C1"], "G-sig")
        accept["v3"][key] = acc
        for m in preds:
            if m in ("THERMO_P", "ATS_P"):
                continue
            src, field = ((V1, "target_log_q_pregate") if acc else ("IDENTITY", "target_log_q")) if m == V3 else (m, "target_log_q")
            for sid, lq in zip(u["target_sample_ids"], u["methods"][src][field]):
                preds[m][pos[sid]] = lq
    for e in extra:
        for m in ("THERMO_P", "ATS_P"):
            for sid, lq in e[m]["pred"].items():
                preds[m][pos[sid]] = lq
    assert all(x is not None for p in preds.values() for x in p)

    rows, loss = {}, {}
    for m, p in preds.items():
        rows[m] = [r for r in score(lab, p, {"method_id": m}) if not r["task_id"].startswith("__")]
        loss[m] = row_metrics(lab, p)["nll"]
    # v4b: per-task metrics and row NLL from its own fits (candidate if the G-sig gate accepts, else identity)
    rows[V4], loss[V4] = [], np.full(tgt.n, np.nan)
    for r in v4:
        key = "|".join(r["unit"])
        acc = gate_study.accept(r["select"]["V"], "G-sig")
        accept["v4b"][key] = acc
        src = "V" if acc else "IDENTITY"
        for t, m in r["target"][src].items():
            d = m["delta_nll"]
            rows[V4].append({"method_id": V4, "model_id": r["unit"][0], "prompt_id": r["unit"][1], "task_id": t, **m,
                             "harm_001": int(d > HARM), "harm_any": int(d > 1e-8)})
        for sid, v in r["row_nll"][src].items():
            loss[V4][pos[sid]] = v
    assert not np.isnan(loss[V4]).any()
    ident = {(r["model_id"], r["prompt_id"], r["task_id"]): r for r in rows["IDENTITY"]}

    summ = {}
    for m in SHOW:
        rs = rows[m]
        d = np.array([r["delta_nll"] for r in rs])
        ece_down = sum(r["ece_15"] < ident[(r["model_id"], r["prompt_id"], r["task_id"])]["ece_15"] for r in rs)
        summ[m] = {"n_units": len(rs), "macro_delta": float(d.mean()), "worst": float(d.max()),
                   "harm_001": int((d > HARM).sum()), "harm_any": int((d > 1e-8).sum()), "ece_decreases": int(ece_down),
                   **{k: float(np.mean([r[k] for r in rs])) for k in ("nll", "brier", "ece_15", "accuracy")}}
    pooled = summ["POOLED_TS"]["macro_delta"]
    for m in summ:
        summ[m]["gain_retained_vs_pooled"] = summ[m]["macro_delta"] / pooled if pooled < 0 else None

    unit = np.array([f"{lab.model_id[i]}|{lab.prompt_id[i]}" for i in range(lab.n)])
    task = np.array([str(t) for t in lab.task_id])
    qid = np.array([str(c) for c in lab.case_id])

    def sync_ci(a, b):
        """Questions resampled per task, shared by all model x prompt units."""
        diff = loss[a] - loss[b]
        qs = {t: sorted(set(qid[task == t].tolist())) for t in sorted(set(task.tolist()))}
        cells = {}
        for k in sorted(set(zip(unit.tolist(), task.tolist()))):
            m = (unit == k[0]) & (task == k[1])
            idx = {x: j for j, x in enumerate(qs[k[1]])}
            S, N = np.zeros(len(qs[k[1]])), np.zeros(len(qs[k[1]]))
            np.add.at(S, [idx[x] for x in qid[m]], diff[m])
            np.add.at(N, [idx[x] for x in qid[m]], 1)
            cells[k] = (S, N)
        point = float(np.mean([S.sum() / N.sum() for S, N in cells.values()]))
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            draw = {t: rng.integers(0, len(v), len(v)) for t, v in qs.items()}
            boots[b] = np.mean([S[draw[k[1]]].sum() / max(N[draw[k[1]]].sum(), 1) for k, (S, N) in cells.items()])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean_diff": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": N_BOOT, "seed": SEED}

    q1 = sync_ci(V4, "MRTS")
    s4, sm, sp, s3 = summ[V4], summ["MRTS"], summ["POOLED_TS"], summ[V3]
    endpoints = {
        "Q1_v4b_vs_MRTS": {"paired": q1, "harm": [s4["harm_001"], sm["harm_001"]],
                           "pass": bool(q1["ci_high"] < 0 and s4["harm_001"] <= sm["harm_001"])},
        "Q2_v4b_vs_POOLED_TS": {"harm": [s4["harm_001"], sp["harm_001"]], "worst": [s4["worst"], sp["worst"]],
                                "pass": bool(s4["harm_001"] <= sp["harm_001"] and s4["worst"] < sp["worst"])},
        "Q3_v4b_vs_v3": {"macro": [s4["macro_delta"], s3["macro_delta"]], "harm": [s4["harm_001"], s3["harm_001"]],
                         "paired_sync": sync_ci(V4, V3),
                         "pass": bool(s4["macro_delta"] <= s3["macro_delta"] + 1e-12 and s4["harm_001"] <= s3["harm_001"])}}
    report = {"data": str(DATA.relative_to(ROOT)), "gate_accept": accept, "summary": summ, "endpoints": endpoints,
              "units": rows}
    (RUN / "confirm3_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"{'method':24} {'delta':>8} {'worst':>8} {'h':>3} {'h_any':>5} {'ECE<':>5} {'NLL':>6} {'Brier':>6} {'ECE':>6}")
    for m in SHOW:
        s = summ[m]
        print(f"{m:24} {s['macro_delta']:+.4f} {s['worst']:+.4f} {s['harm_001']:3d} {s['harm_any']:5d} "
              f"{s['ece_decreases']:5d} {s['nll']:.3f} {s['brier']:.3f} {s['ece_15']:.3f}")
    print("gate accepted:", {k: sum(v.values()) for k, v in accept.items()})
    print(json.dumps(endpoints, indent=1, default=float))


if __name__ == "__main__":
    main()
