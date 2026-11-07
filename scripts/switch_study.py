"""Feasibility of a task-level switch (design data only: LOTO, 250 units).

Policy: on a target task use a strong unconstrained calibrator S (pooled TS or Thermometer-P) when a label-free
task signal says the task is NOT of the "already calibrated" kind, otherwise the safe calibrator v4b. The
threshold for each held-out task is chosen on the other 24 tasks only (nested leave-one-task-out): the largest
gain subject to at most `max_harm` harmed units (Delta > 0.01) among those 24 tasks' 240 units.

    python scripts/switch_study.py      # -> runs/step0/switch_study.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_study  # noqa: E402

HARM = 0.01


def load():
    d = {}
    for p in glob.glob(str(ROOT / "runs/loto/jobs/*.json")):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        d[(j["unit"][0], j["unit"][1], j["task"])] = {m: (v["delta_nll"], v["ece_15"]) for m, v in j["methods"].items()}
    for l in (ROOT / "runs/v4_study/loto.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["variant"] == "v4b":
            src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
            m = r["target"][src][r["task"]]
            d[(r["unit"][0], r["unit"][1], r["task"])]["V4B"] = (m["delta_nll"], m["ece_15"])
    sig = {tuple(k.split("|")): v for k, v in
           json.loads((ROOT / "runs/step0/task_signals.json").read_text(encoding="utf-8")).items()}
    keys = sorted(k for k in d if "V4B" in d[k])
    return d, sig, keys


def score_of(sig, k, name):
    s = sig[k]
    if name == "template_tv":
        return s["template_tv"]
    if name == "order_tv":
        return s["order_tv"]
    if name == "peer_disagree":
        return 1.0 - s["peer_prob_on_top"]
    # rank-free combination: mean of the three, each already in [0, 1]
    return (s["template_tv"] + s["order_tv"] + (1.0 - s["peer_prob_on_top"])) / 3.0


def evaluate(d, sig, keys, strong, signal, max_harm):
    tasks = sorted({k[2] for k in keys})
    x = np.array([score_of(sig, k, signal) for k in keys])
    dS = np.array([d[k][strong][0] for k in keys])
    dV = np.array([d[k]["V4B"][0] for k in keys])
    eS = np.array([d[k][strong][1] for k in keys])
    eV = np.array([d[k]["V4B"][1] for k in keys])
    e0 = np.array([d[k]["IDENTITY"][1] for k in keys])
    task = np.array([k[2] for k in keys])
    use_strong = np.zeros(len(keys), bool)
    taus = {}
    for t in tasks:
        tr = task != t
        best = (np.inf, np.inf)          # (macro delta, tau) on the 24 training tasks
        for tau in np.r_[np.unique(x[tr]), np.inf]:
            pick = x[tr] >= tau
            dd = np.where(pick, dS[tr], dV[tr])
            if (dd > HARM).sum() <= max_harm and dd.mean() < best[0]:
                best = (dd.mean(), tau)
        taus[t] = float(best[1])
        use_strong[task == t] = x[task == t] >= best[1]
    dd = np.where(use_strong, dS, dV)
    ee = np.where(use_strong, eS, eV)
    slope = float(np.polyfit(e0, ee, 1)[0])
    return {"strong": strong, "signal": signal, "max_harm_train": max_harm, "macro_delta": float(dd.mean()),
            "harm": int((dd > HARM).sum()), "worst": float(dd.max()), "ece": float(ee.mean()),
            "ece_decreases": int((ee < e0).sum()), "ece_slope": slope, "share_strong": float(use_strong.mean()),
            "thresholds": taus}


def reference(d, keys, m):
    dd = np.array([d[k][m][0] for k in keys])
    ee = np.array([d[k][m][1] for k in keys])
    e0 = np.array([d[k]["IDENTITY"][1] for k in keys])
    return {"macro_delta": float(dd.mean()), "harm": int((dd > HARM).sum()), "worst": float(dd.max()),
            "ece": float(ee.mean()), "ece_decreases": int((ee < e0).sum()), "ece_slope": float(np.polyfit(e0, ee, 1)[0])}


def main():
    d, sig, keys = load()
    out = {"reference": {m: reference(d, keys, m) for m in ("POOLED_TS", "THERMO_P", "ATS_P", "V4B")}, "switch": []}
    for strong in ("POOLED_TS", "THERMO_P"):
        for signal in ("template_tv", "order_tv", "peer_disagree", "combined"):
            for mh in (0, 2):
                out["switch"].append(evaluate(d, sig, keys, strong, signal, mh))
    (ROOT / "runs/step0/switch_study.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda n, r: print(f"{n:42s} delta {r['macro_delta']:+.4f} harm {r['harm']:3d} worst {r['worst']:+.4f} "
                           f"ECE {r['ece']:.3f} ECE< {r['ece_decreases']:3d}/250 slope {r['ece_slope']:.2f}"
                           + (f" strong {r['share_strong']:.0%}" if "share_strong" in r else ""))
    for m, r in out["reference"].items():
        f(m, r)
    for r in out["switch"]:
        f(f"switch {r['strong']}|{r['signal']}|h<={r['max_harm_train']}", r)


if __name__ == "__main__":
    main()
