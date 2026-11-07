"""Phase 10.0: go/no-go diagnostics on LOTO (design data only; no confirm roster is read).

(a) oracle task-level selector: per held-out (unit, task) pick the best calibrator among a set, with and without
    a no-harm restriction, and compare with v4b;
(b) AUROC of label-free task-level signals for two labels: "pooled TS harms" (D_pooled > 0.01) and
    "pooled TS beats v4b by > 0.01".

    python scripts/step0_diagnostics.py      # -> runs/step0/diagnostics.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_study  # noqa: E402

OUT = ROOT / "runs/step0"
HARM = 0.01


def loto_deltas():
    d = {}
    for p in glob.glob(str(ROOT / "runs/loto/jobs/*.json")):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        d[(j["unit"][0], j["unit"][1], j["task"])] = {m: v["delta_nll"] for m, v in j["methods"].items()}
    for l in (ROOT / "runs/v4_study/loto.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["variant"] != "v4b":
            continue
        src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
        d[(r["unit"][0], r["unit"][1], r["task"])]["V4B"] = r["target"][src][r["task"]]["delta_nll"]
    return d


def semantic(probs, label_ids):
    out = np.zeros(len(probs))
    for pos, lab in enumerate(label_ids):
        out[int(lab)] = probs[pos]
    return out


def task_signals(data_dir=None, files=("source_train", "source_select", "source_audit")):
    """Label-free statistics per (model, prompt, task) over all rows of that task in the given files
    (default: the 25 source tasks of data/real/dev; targets: files=("target_inputs",)). Labels are never read."""
    data_dir = Path(data_dir) if data_dir else ROOT / "data/real/dev"
    rows = {}
    for s in files:
        for line in open(data_dir / f"{s}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            o = r["sample_id"].split("::")[-2]
            rows[(r["model_id"], r["prompt_id"], r["case_id"], o)] = (r["task_id"], np.asarray(r["probs"]),
                                                                       semantic(r["probs"], r["label_ids"]))
    models = sorted({k[0] for k in rows})
    acc = defaultdict(lambda: defaultdict(list))
    for (m, p, c, o), (task, probs, sem) in rows.items():
        key = (m, p, task)
        K = len(probs)
        a = acc[key]
        a["entropy"].append(float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / np.log(K)))
        a["maxprob"].append(float(probs.max()))
        other_o = "o1" if o == "o0" else "o0"
        if (m, p, c, other_o) in rows:
            a["order_tv"].append(0.5 * float(np.abs(sem - rows[(m, p, c, other_o)][2]).sum()))
        other_p = "paren" if p == "letter_colon" else "letter_colon"
        if (m, other_p, c, o) in rows:
            a["template_tv"].append(0.5 * float(np.abs(sem - rows[(m, other_p, c, o)][2]).sum()))
        top = int(np.argmax(sem))
        peers = [rows[(q, p, c, o)][2] for q in models if q != m and (q, p, c, o) in rows]
        if peers:
            a["peer_prob_on_top"].append(float(np.mean([q[top] for q in peers])))
            a["peer_argmax_agree"].append(float(np.mean([int(np.argmax(q)) == top for q in peers])))
        a.setdefault("_pos", []).append(probs) if K == 4 else None
    sig = {}
    for key, a in acc.items():
        s = {k: float(np.mean(v)) for k, v in a.items() if not k.startswith("_")}
        if a.get("_pos"):
            prior = np.mean(a["_pos"], axis=0)
            s["position_prior_entropy"] = float(-(prior * np.log(prior)).sum() / np.log(4))
        sig[key] = s
    return sig


def auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = loto_deltas()
    keys = sorted(k for k in d if "V4B" in d[k])
    D = {m: np.array([d[k][m] for k in keys]) for m in ("IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS", "V4B")}
    rep = {"n_units": len(keys), "macro": {m: float(v.mean()) for m, v in D.items()},
           "harms": {m: int((v > HARM).sum()) for m, v in D.items()}}
    cands = ["IDENTITY", "V4B", "POOLED_TS", "THERMO_P", "ATS_P"]
    M = np.stack([D[m] for m in cands], axis=1)
    rep["oracle_any"] = float(M.min(axis=1).mean())
    safe = np.where(M <= HARM, M, np.inf)
    rep["oracle_safe"] = float(safe.min(axis=1).mean())
    pv = np.minimum(D["V4B"], np.where(D["POOLED_TS"] <= HARM, D["POOLED_TS"], np.inf))
    rep["oracle_v4b_or_pooled_safe"] = float(pv.mean())
    rep["oracle_gain_over_v4b"] = float(D["V4B"].mean() - rep["oracle_v4b_or_pooled_safe"])
    # where does pooled's extra gain come from?
    better = D["POOLED_TS"] < D["V4B"]
    harmed_tasks = {k[2] for k, h in zip(keys, D["POOLED_TS"] > HARM) if h}
    in_h = np.array([k[2] in harmed_tasks for k in keys])
    gap = D["V4B"] - D["POOLED_TS"]
    rep["gap_decomposition"] = {"total_gap_mean": float(gap.mean()),
                                "share_from_units_where_pooled_harms": float(gap[D["POOLED_TS"] > HARM].sum() / gap.sum()),
                                "share_from_tasks_where_pooled_harms_some_unit": float(gap[in_h].sum() / gap.sum()),
                                "units_pooled_better": int(better.sum())}
    sig = task_signals()
    names = sorted({n for s in sig.values() for n in s})
    labels = {"pooled_harms": D["POOLED_TS"] > HARM, "pooled_beats_v4b_by_0.01": gap > 0.01}
    rep["auroc"] = {}
    for lab, y in labels.items():
        rep["auroc"][lab] = {"n_pos": int(y.sum())}
        for n in names:
            x = np.array([sig[k].get(n, np.nan) for k in keys])
            ok = ~np.isnan(x)
            a = auroc(x[ok], y[ok])
            rep["auroc"][lab][n] = max(a, 1 - a) if not np.isnan(a) else a      # direction-free
    rep["go"] = bool(rep["oracle_gain_over_v4b"] >= 0.01 and
                     max(v for lab in rep["auroc"].values() for k, v in lab.items() if k != "n_pos") >= 0.75)
    (OUT / "diagnostics.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
