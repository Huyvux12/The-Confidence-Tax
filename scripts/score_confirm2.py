"""Pre-registered endpoints P1-P3 on the confirm2 roster (PROTOCOL_LOG 2026-11-06).

Needs runs/rbcal_confirm2/units (all methods, fitted on Modal), runs/rbcal_confirm2/extra (THERMO_P, ATS_P)
and runs/gate_study/confirm2.jsonl (select statistics of the C1 candidate for the G-sig gate).

    python scripts/score_confirm2.py
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

DATA = ROOT / "data/real/confirm2_sealed"
RUN = ROOT / "runs/rbcal_confirm2"
V1, V3 = "RBCAL_UCB_RBF_e0.01", "V3_UCB_RBF_GSIG"
SHOW = ["IDENTITY", "POOLED_TS", "MRTS", "SHRINKAGE_LOSO", "THERMO_P", "ATS_P", "RBCAL_UCB_CONST_e0.01", V1, V3,
        V1 + "__PREGATE"]
HARM, N_BOOT, SEED = 0.01, 5000, 20261106


def main():
    tgt = da.load_split(str(DATA), "target_inputs")
    lab = da.attach_labels(tgt, da.load_private_labels(str(DATA)))
    pos = {str(s): i for i, s in enumerate(tgt.sample_id)}
    units = [pickle.loads(Path(p).read_bytes()) for p in sorted(glob.glob(str(RUN / "units/*.pkl")))]
    extra = [pickle.loads(Path(p).read_bytes()) for p in sorted(glob.glob(str(RUN / "extra/*.pkl")))]
    gates = {"|".join(r["unit"]): r for r in gate_study.load_records("confirm2")}
    assert len(units) == len(extra) == len(gates) == 10

    preds = {m: [None] * tgt.n for m in SHOW}
    v3_accept = {}
    for u in units:
        key = "|".join(u["unit"])
        acc = gate_study.accept(gates[key]["select"]["C1"], "G-sig")
        v3_accept[key] = acc
        for m in SHOW:
            if m in ("THERMO_P", "ATS_P"):
                continue
            if m == V3:
                src, field = (V1, "target_log_q_pregate") if acc else ("IDENTITY", "target_log_q")
            elif m.endswith("__PREGATE"):
                src, field = m[:-9], "target_log_q_pregate"
            else:
                src, field = m, "target_log_q"
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
    summ = {}
    for m, rs in rows.items():
        d = np.array([r["delta_nll"] for r in rs])
        summ[m] = {"n_units": len(rs), "macro_delta": float(d.mean()), "worst": float(d.max()),
                   "harm_001": int((d > HARM).sum()), "harm_any": int((d > 1e-8).sum()),
                   **{k: float(np.mean([r[k] for r in rs])) for k in ("nll", "brier", "ece_15", "accuracy")}}
    pooled = summ["POOLED_TS"]["macro_delta"]
    for m in summ:
        summ[m]["gain_retained_vs_pooled"] = summ[m]["macro_delta"] / pooled if pooled < 0 else None

    # synchronized case bootstrap: questions resampled per task, shared by all model x prompt units
    def sync_ci(a, b):
        diff = loss[a] - loss[b]
        unit = np.array([f"{lab.model_id[i]}|{lab.prompt_id[i]}" for i in range(lab.n)])
        task = np.array([str(t) for t in lab.task_id])
        q = np.array([str(c).split("::", 1)[1] if "::" in str(c) else str(c) for c in lab.case_id])
        cells, qs = {}, {}
        for t in sorted(set(task.tolist())):
            qs[t] = sorted(set(q[task == t].tolist()))
        for k in sorted(set(zip(unit.tolist(), task.tolist()))):
            m = (unit == k[0]) & (task == k[1])
            idx = {x: j for j, x in enumerate(qs[k[1]])}
            S, N = np.zeros(len(qs[k[1]])), np.zeros(len(qs[k[1]]))
            np.add.at(S, [idx[x] for x in q[m]], diff[m])
            np.add.at(N, [idx[x] for x in q[m]], 1)
            cells[k] = (S, N)
        point = float(np.mean([S.sum() / N.sum() for S, N in cells.values()]))
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            draw = {t: rng.integers(0, len(v), len(v)) for t, v in qs.items()}
            boots[b] = np.mean([S[draw[k[1]]].sum() / max(N[draw[k[1]]].sum(), 1) for k, (S, N) in cells.items()])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean_diff": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": N_BOOT, "seed": SEED}

    p1 = sync_ci(V3, "MRTS")
    s3, sm, sp, s1 = summ[V3], summ["MRTS"], summ["POOLED_TS"], summ[V1]
    endpoints = {
        "P1_v3_vs_MRTS": {"paired": p1, "harm": [s3["harm_001"], sm["harm_001"]],
                          "pass": bool(p1["ci_high"] < 0 and s3["harm_001"] <= sm["harm_001"])},
        "P2_v3_vs_POOLED_TS": {"harm": [s3["harm_001"], sp["harm_001"]], "worst": [s3["worst"], sp["worst"]],
                               "pass": bool(s3["harm_001"] <= sp["harm_001"] and s3["worst"] < sp["worst"])},
        "P3_v3_vs_v1": {"macro": [s3["macro_delta"], s1["macro_delta"]], "harm": [s3["harm_001"], s1["harm_001"]],
                        "paired_sync": sync_ci(V3, V1),
                        "pass": bool(s3["macro_delta"] <= s1["macro_delta"] + 1e-12 and s3["harm_001"] <= s1["harm_001"])}}
    per_task = {m: {t: float(np.mean([r["delta_nll"] for r in rs if r["task_id"] == t]))
                    for t in sorted({r["task_id"] for r in rs})} for m, rs in rows.items()}
    report = {"data": str(DATA.relative_to(ROOT)), "v3_gate_accept": v3_accept, "summary": summ,
              "endpoints": endpoints, "per_task_mean_delta": per_task,
              "units": {m: rs for m, rs in rows.items()}}
    (RUN / "confirm2_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"{'method':28} {'delta':>8} {'worst':>8} {'h':>3} {'h_any':>5} {'NLL':>6} {'Brier':>6} {'ECE':>6} {'acc':>6}")
    for m in SHOW:
        s = summ[m]
        print(f"{m:28} {s['macro_delta']:+.4f} {s['worst']:+.4f} {s['harm_001']:3d} {s['harm_any']:5d} "
              f"{s['nll']:.3f} {s['brier']:.3f} {s['ece_15']:.3f} {s['accuracy']:.3f}")
    print("v3 gate accepted units:", sum(v3_accept.values()), "/ 10")
    print(json.dumps(endpoints, indent=1, default=float))


if __name__ == "__main__":
    main()
