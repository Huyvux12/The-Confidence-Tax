"""Pre-registered endpoints S1-S3 for RB-Switch on the confirm4 roster (PROTOCOL_LOG 2026-11-06).

RB-Switch: per target task batch (model x prompt x task) compute the frozen label-free signal from the unlabelled
target inputs (scripts/step0_diagnostics.task_signals) and deploy pooled TS if it reaches the frozen threshold,
else v4b. Needs runs/rbcal_confirm4/{units,extra}, runs/v4_study/final_confirm4.jsonl (v4b fits with row NLL).

    python scripts/score_confirm4.py
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
import step0_diagnostics as s0  # noqa: E402

DATA = ROOT / "data/real/confirm4_sealed"
RUN = ROOT / "runs/rbcal_confirm4"
FROZEN = json.loads((ROOT / "runs/step0/switch_frozen.json").read_text(encoding="utf-8"))
HARM, N_BOOT, SEED = 0.01, 5000, 20261108
BASE = ["IDENTITY", "POOLED_TS", "MRTS", "THERMO_P", "ATS_P", "RBCAL_UCB_RBF_e0.01"]


def signal(s, kind):
    if kind == "combined":
        return (s["template_tv"] + s["order_tv"] + (1.0 - s["peer_prob_on_top"])) / 3.0
    return (s["template_tv"] + s["order_tv"]) / 2.0


def main():
    tgt = da.load_split(str(DATA), "target_inputs")
    lab = da.attach_labels(tgt, da.load_private_labels(str(DATA)))
    sids = [str(s) for s in tgt.sample_id]
    pos = {s: i for i, s in enumerate(sids)}
    unit_key = [(str(lab.model_id[i]), str(lab.prompt_id[i]), str(lab.task_id[i])) for i in range(lab.n)]

    preds = {m: [None] * tgt.n for m in BASE if m not in ("THERMO_P", "ATS_P")}
    for p in sorted(glob.glob(str(RUN / "units/*.pkl"))):
        u = pickle.loads(Path(p).read_bytes())
        for m in preds:
            for sid, lq in zip(u["target_sample_ids"], u["methods"][m]["target_log_q"]):
                preds[m][pos[sid]] = lq
    for m in ("THERMO_P", "ATS_P"):
        preds[m] = [None] * tgt.n
    for p in sorted(glob.glob(str(RUN / "extra/*.pkl"))):
        e = pickle.loads(Path(p).read_bytes())
        for m in ("THERMO_P", "ATS_P"):
            for sid, lq in e[m]["pred"].items():
                preds[m][pos[sid]] = lq
    assert all(x is not None for v in preds.values() for x in v)

    per_unit, loss = {}, {}
    for m, p in preds.items():
        per_unit[m] = {(r["model_id"], r["prompt_id"], r["task_id"]): r
                       for r in score(lab, p, {"method_id": m}) if not r["task_id"].startswith("__")}
        loss[m] = row_metrics(lab, p)["nll"]
    # v4b from its own fits (candidate if the G-sig gate accepts, else identity)
    v4 = [json.loads(l) for l in (ROOT / "runs/v4_study/final_confirm4.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip() and json.loads(l)["variant"] == "v4b"]
    assert len(v4) == 10
    per_unit["V4B"], loss["V4B"] = {}, np.full(lab.n, np.nan)
    for r in v4:
        src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
        for t, m in r["target"][src].items():
            per_unit["V4B"][(r["unit"][0], r["unit"][1], t)] = {**m, "harm_001": int(m["delta_nll"] > HARM)}
        for sid, v in r["row_nll"][src].items():
            loss["V4B"][pos[sid]] = v
    assert not np.isnan(loss["V4B"]).any()

    # label-free task signals on the target inputs, then the frozen switches
    sig = s0.task_signals(DATA, ("target_inputs",))
    switches = {"RB_SWITCH": ("primary", "POOLED_TS", "combined"),
                "RB_SWITCH_NOPEER": ("no_peer", "POOLED_TS", "order_template"),
                "RB_SWITCH_THERMO": ("thermo", "THERMO_P", "combined")}
    choice = {}
    for name, (fz, strong, kind) in switches.items():
        tau = FROZEN[fz]["tau"]
        choice[name] = {k: (strong if signal(sig[k], kind) >= tau else "V4B") for k in per_unit["IDENTITY"]}
        per_unit[name] = {k: per_unit[c][k] for k, c in choice[name].items()}
        loss[name] = np.array([loss[choice[name][unit_key[i]]][i] for i in range(lab.n)])

    ident = per_unit["IDENTITY"]
    summ = {}
    for m, rows in per_unit.items():
        ks = sorted(rows)
        d = np.array([rows[k]["delta_nll"] for k in ks])
        e = np.array([rows[k]["ece_15"] for k in ks])
        e0 = np.array([ident[k]["ece_15"] for k in ks])
        summ[m] = {"n_units": len(ks), "macro_delta": float(d.mean()), "worst": float(d.max()),
                   "harm_001": int((d > HARM).sum()), "nll": float(np.mean([rows[k]["nll"] for k in ks])),
                   "brier": float(np.mean([rows[k]["brier"] for k in ks])), "ece_15": float(e.mean()),
                   "ece_decreases": int((e < e0).sum()), "ece_slope": float(np.polyfit(e0, e, 1)[0])}
    for name in switches:
        summ[name]["share_strong"] = float(np.mean([c != "V4B" for c in choice[name].values()]))

    task = np.array([k[2] for k in unit_key])
    unit = np.array([f"{k[0]}|{k[1]}" for k in unit_key])
    qid = np.array([str(c) for c in lab.case_id])

    def sync_ci(a, b):
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
        return {"mean_diff": point, "ci_low": float(lo), "ci_high": float(hi)}

    s, sp, st, sa = summ["RB_SWITCH"], summ["POOLED_TS"], summ["THERMO_P"], summ["ATS_P"]
    endpoints = {
        "S1_harm": {"switch": s["harm_001"], "pooled": sp["harm_001"], "thermo": st["harm_001"], "ats": sa["harm_001"],
                    "pass": bool(s["harm_001"] <= sp["harm_001"] // 3 and s["harm_001"] <= st["harm_001"]
                                 and s["harm_001"] <= sa["harm_001"])},
        "S2_gain": {"switch": s["macro_delta"], "pooled": sp["macro_delta"],
                    "pass": bool(s["macro_delta"] <= sp["macro_delta"] + 0.005)},
        "S3_ece": {"switch": s["ece_15"], "pooled": sp["ece_15"], "pass": bool(s["ece_15"] <= sp["ece_15"] + 0.005)},
        "secondary_sync_ci": {"switch_minus_pooled": sync_ci("RB_SWITCH", "POOLED_TS"),
                              "switch_minus_v4b": sync_ci("RB_SWITCH", "V4B"),
                              "switch_minus_thermo": sync_ci("RB_SWITCH", "THERMO_P")}}
    units_out = {m: [{"model_id": k[0], "prompt_id": k[1], "task_id": k[2], **{x: r[x] for x in
                      ("nll", "brier", "ece_15", "delta_nll")}} for k, r in sorted(rows.items())]
                 for m, rows in per_unit.items()}
    report = {"data": str(DATA.relative_to(ROOT)), "frozen": FROZEN, "summary": summ, "endpoints": endpoints,
              "choices": {n: {"|".join(k): c for k, c in ch.items()} for n, ch in choice.items()}, "units": units_out}
    (RUN / "confirm4_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"{'method':22} {'delta':>8} {'worst':>8} {'h':>3} {'NLL':>6} {'Brier':>6} {'ECE':>6} {'ECE<':>5} {'slope':>6}")
    for m in ["IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS", "RBCAL_UCB_RBF_e0.01", "V4B",
              "RB_SWITCH", "RB_SWITCH_NOPEER", "RB_SWITCH_THERMO"]:
        x = summ[m]
        print(f"{m:22} {x['macro_delta']:+.4f} {x['worst']:+.4f} {x['harm_001']:3d} {x['nll']:.3f} {x['brier']:.3f} "
              f"{x['ece_15']:.3f} {x['ece_decreases']:3d}/{x['n_units']} {x['ece_slope']:.2f}"
              + (f"  strong {x['share_strong']:.0%}" if "share_strong" in x else ""))
    print(json.dumps(endpoints, indent=1, default=float))


if __name__ == "__main__":
    main()
