"""Phase 6 (todo.md): deployment gates for RB-Cal-RBF (eps = 0.01).

The RBF candidates are refitted with the spec already chosen on all sources in runs/rbcal_dev (no menu
search), so fits depend on sources only. Each job returns per-source select regrets with case-clustered
SEs plus target metrics of every candidate and fallback; gate variants are applied afterwards:

    candidates  C0: z = 0            C1: z = 1.645 (current primary)
    gates       G-mean: max_w D_w^select <= eps                                   (current)
                G-sig:  max_w (D_w^select - 1.645 SE_w^select) <= eps             (reject only significant violations)
    fallback    identity | MRTS

    python scripts/gate_study.py hypers                       # runs/gate_study/hypers.json from runs/rbcal_dev/units
    python scripts/gate_study.py local --phase dev            # or run on Modal: scripts/modal_gate_study.py
    python scripts/gate_study.py summarize --phase dev
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPS, Z_FIT, Z_GATE, HARM = 0.01, 1.645, 1.645, 0.01
CANDS = {"C0": ("RBCAL_RBF_e0.01", 0.0), "C1": ("RBCAL_UCB_RBF_e0.01", Z_FIT)}
OUT = ROOT / "runs/gate_study"
DATA = {"dev": "data/real/dev", "confirm": "data/real/confirm_sealed", "confirm2": "data/real/confirm2_sealed", "confirm3": "data/real/confirm3_sealed", "confirm4": "data/real/confirm4_sealed",
        "loto": "data/real/dev"}


def unit_file(unit):
    return unit[0].replace("/", "__") + "__" + unit[1]


# ---------------------------------------------------------------- fitting (needs numpy; runs remotely)

def select_stats(sel, cal):
    """Per source: mean regret on select and its case-clustered SE (Eq. 6 without smoothing)."""
    import numpy as np
    from calibration_extensions.numerics import losses, surprisal
    d = losses(sel, cal.log_q_groups(sel)) - surprisal(sel)
    src = np.array([str(s) for s in sel.source_id])
    cases = np.array([str(c) for c in sel.case_id])
    out = {}
    for w in sorted(set(src.tolist())):
        m = src == w
        _, inv = np.unique(cases[m], return_inverse=True)
        s = np.bincount(inv, weights=d[m])
        cnt = np.bincount(inv).astype(float)
        n, C = m.sum(), len(s)
        mean = s.sum() / n
        e = s - cnt * mean
        se = float(np.sqrt(C / max(C - 1, 1) / n ** 2 * (e @ e)))
        out[w] = {"delta": float(mean), "se": se}
    return out


def fit_candidates(tr, sel, hypers):
    from calibration_extensions.rb_calibration import fit_rbcal
    from calibration_extensions.source_objectives import beta_grid, fit_scalar
    from calibration_extensions.calibrators import IdentityCalibrator
    cals, stats = {"IDENTITY": IdentityCalibrator(), "MRTS": fit_scalar(tr, beta_grid(), "minimax")[0]}, {}
    for c, (mid, z) in CANDS.items():
        h = hypers[c]
        fr = fit_rbcal(tr, sel, [(h["kind"], h["J"], h["sigma"], h["lam"])], EPS, mid, z=z)
        cals[c] = fr.fitted                                    # None on solver failure -> always rejected
        stats[c] = None if fr.fitted is None else select_stats(sel, fr.fitted)
    return cals, stats


def unit_job(phase, unit, hypers):
    from calibration_extensions import data_adapter as da
    from calibration_extensions.evaluator import score
    data = str(ROOT / DATA[phase])
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    tr, sel = (da.load_split(data, s, where) for s in ("source_train", "source_select"))
    tgt = da.attach_labels(da.load_split(data, "target_inputs", where), da.load_private_labels(data))
    cals, stats = fit_candidates(tr, sel, hypers)
    target = {}
    for name, cal in cals.items():
        if cal is None:
            continue
        rows = [r for r in score(tgt, cal.predict_log_proba(tgt), {}) if not r["task_id"].startswith("__")]
        target[name] = {r["task_id"]: {k: r[k] for k in ("nll", "brier", "ece_15", "accuracy", "delta_nll")}
                        for r in rows}
    return {"phase": phase, "unit": list(unit), "select": stats, "target": target}


def loto_job(unit, task, hypers):
    from calibration_extensions import data_adapter as da
    from calibration_extensions.evaluator import ece, row_metrics
    from calibration_extensions.numerics import concat
    data = str(ROOT / DATA["loto"])
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    s = {k: da.load_split(data, k, where) for k in ("source_train", "source_select", "source_audit")}
    keep = lambda ps, inside: ps.subset((ps.source_id == task) == inside)
    tr, sel = keep(s["source_train"], False), keep(s["source_select"], False)
    ev = concat([keep(s[k], True) for k in s])
    cals, stats = fit_candidates(tr, sel, hypers)
    target = {}
    for name, cal in cals.items():
        if cal is None:
            continue
        rm = row_metrics(ev, cal.predict_log_proba(ev))
        target[name] = {task: {"nll": float(rm["nll"].mean()), "brier": float(rm["brier"].mean()),
                               "ece_15": ece(rm["conf"], rm["correct"]), "accuracy": float(rm["correct"].mean()),
                               "delta_nll": float(rm["delta"].mean())}}
    return {"phase": "loto", "unit": list(unit), "task": task, "select": stats, "target": target}


def run_job(job: dict) -> dict:
    if job["phase"] == "loto":
        return loto_job(tuple(job["unit"]), job["task"], job["hypers"])
    return unit_job(job["phase"], tuple(job["unit"]), job["hypers"])


def jobs_for(phase: str, hy: dict) -> list[dict]:
    units = [tuple(u.split("|")) for u in sorted(hy["units"])]
    if phase == "loto":
        return [{"phase": phase, "unit": list(u), "task": t, "hypers": hy["units"]["|".join(u)]}
                for u in units for t in hy["tasks"]]
    return [{"phase": phase, "unit": list(u), "hypers": hy["units"]["|".join(u)]} for u in units]


# ---------------------------------------------------------------- gate variants (pure post-processing)

def accept(stats, gate, eps=EPS):
    if stats is None:
        return False
    if gate == "G-mean":
        return max(v["delta"] for v in stats.values()) <= eps + 1e-8
    return max(v["delta"] - Z_GATE * v["se"] for v in stats.values()) <= eps + 1e-8


VARIANTS = [(c, g, f) for c in ("C0", "C1") for g in ("G-mean", "G-sig") for f in ("IDENTITY", "MRTS")]


def variant_rows(records, variant):
    c, g, f = variant
    rows = []
    for r in records:
        ok = accept(r["select"].get(c), g)
        src = r["target"][c] if ok else r["target"][f]
        rows += [{"unit": tuple(r["unit"]), "task": t, "accepted": ok, **m} for t, m in src.items()]
    return rows


def summarize_rows(rows):
    import numpy as np
    d = np.array([x["delta_nll"] for x in rows])
    return {"n": len(rows), "macro_delta": float(d.mean()), "worst": float(d.max()), "harm_001": int((d > HARM).sum()),
            "harm_any": int((d > 1e-8).sum()), "nll": float(np.mean([x["nll"] for x in rows])),
            "brier": float(np.mean([x["brier"] for x in rows])), "ece_15": float(np.mean([x["ece_15"] for x in rows])),
            "accept_rate": float(np.mean([x["accepted"] for x in rows]))}


def choose(table):
    """Phase-6 rule: zero dev harms, then largest gain; ties -> identity fallback, G-mean, C1."""
    pref = lambda v: (v[2] != "IDENTITY", v[1] != "G-mean", v[0] != "C1")
    ok = [v for v in VARIANTS if table[v]["harm_001"] == 0]
    best = min(table[v]["macro_delta"] for v in ok)
    return min([v for v in ok if table[v]["macro_delta"] <= best + 1e-8], key=pref)


def load_records(phase):
    return [json.loads(l) for l in (OUT / f"{phase}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hypers", "local", "summarize"])
    ap.add_argument("--phase", choices=["dev", "confirm", "confirm2", "confirm3", "confirm4", "loto"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "hypers":
        hy = {"units": {}, "tasks": None}
        for p in sorted(glob.glob(str(ROOT / "runs/rbcal_dev/units/*.pkl"))):
            u = pickle.loads(Path(p).read_bytes())
            hy["units"]["|".join(u["unit"])] = {c: u["methods"][mid]["hyper"] for c, (mid, _) in CANDS.items()}
            hy["tasks"] = sorted(u["methods"]["IDENTITY"]["train_deltas"])
        (OUT / "hypers.json").write_text(json.dumps(hy, indent=1), encoding="utf-8")
        print(len(hy["units"]), "units,", len(hy["tasks"]), "tasks")
        return
    if args.cmd == "local":
        hy = json.loads((OUT / "hypers.json").read_text(encoding="utf-8"))
        with open(OUT / f"{args.phase}.jsonl", "w", encoding="utf-8") as f:
            for j in jobs_for(args.phase, hy):
                f.write(json.dumps(run_job(j)) + "\n")
        return
    recs = load_records(args.phase)
    table = {v: summarize_rows(variant_rows(recs, v)) for v in VARIANTS}
    print(f"{args.phase}: {len(recs)} jobs")
    print(f"{'candidate':4} {'gate':7} {'fallback':9} {'delta':>8} {'worst':>8} {'h':>3} {'h_any':>5} "
          f"{'NLL':>6} {'Brier':>6} {'ECE':>6} {'accept':>6}")
    for v, s in table.items():
        print(f"{v[0]:9} {v[1]:7} {v[2]:9} {s['macro_delta']:+.4f} {s['worst']:+.4f} {s['harm_001']:3d} "
              f"{s['harm_any']:5d} {s['nll']:.3f} {s['brier']:.3f} {s['ece_15']:.3f} {s['accept_rate']:.2f}")
    (OUT / f"summary_{args.phase}.json").write_text(
        json.dumps({"|".join(v): s for v, s in table.items()}, indent=1), encoding="utf-8")
    if args.phase == "dev":
        print("rule choice on dev:", choose(table))


if __name__ == "__main__":
    main()
