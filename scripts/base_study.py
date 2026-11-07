"""Phase 10: unconstrained baselines on a given data base (fair comparison with RB-Cal variants on the same base).

Methods: IDENTITY, POOLED_TS, MRTS, THERMO_P, ATS_P and, with peer=True, THERMO_P_PEER / ATS_P_PEER (their
RBF features extended with order consistency + peer consensus, kind rbf_ocp). Jobs run on Modal through
scripts/modal_v4.py (job["module"] = "base_study").

    python scripts/base_study.py summarize --phase dev|loto      # compares bases and RB-Cal variants
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v4_study  # noqa: E402

OUT = ROOT / "runs/v4_study"
HARM = 0.01


def fit_all(tr, peer):
    from calibration_extensions.calibrators import IdentityCalibrator
    from calibration_extensions.extra_baselines import fit_ats_p, fit_thermometer_p
    from calibration_extensions.source_objectives import beta_grid, fit_scalar
    cals = {"IDENTITY": IdentityCalibrator(), "POOLED_TS": fit_scalar(tr, beta_grid(), "pooled")[0],
            "MRTS": fit_scalar(tr, beta_grid(), "minimax")[0], "THERMO_P": fit_thermometer_p(tr)[0],
            "ATS_P": fit_ats_p(tr)[0]}
    if peer:
        cals["THERMO_P_PEER"] = fit_thermometer_p(tr, "rbf_ocp")[0]
        cals["ATS_P_PEER"] = fit_ats_p(tr, "rbf_ocp")[0]
    return cals


def unit_job(phase, base, unit, peer):
    from calibration_extensions import data_adapter as da
    from calibration_extensions.evaluator import row_metrics, score
    s = v4_study.load_unit(phase, unit, ["source_train", "target_inputs"], base)
    tgt = da.attach_labels(s["target_inputs"], da.load_private_labels(str(ROOT / (v4_study.DATA[phase] + base))))
    target, row_nll = {}, {}
    sids = [str(x) for x in tgt.sample_id]
    for name, cal in fit_all(s["source_train"], peer).items():
        lq = cal.predict_log_proba(tgt)
        rows = [r for r in score(tgt, lq, {}) if not r["task_id"].startswith("__")]
        target[name] = {r["task_id"]: {k: r[k] for k in ("nll", "brier", "ece_15", "accuracy", "delta_nll")} for r in rows}
        row_nll[name] = dict(zip(sids, row_metrics(tgt, lq)["nll"].tolist()))
    return {"stage": "base_unit", "phase": phase, "base": base, "unit": list(unit), "target": target,
            "row_nll": row_nll, "identity_raw_nll": v4_study.raw_identity_nll(phase, unit)}


def loto_job(base, unit, task, peer):
    from calibration_extensions.evaluator import ece, row_metrics
    from calibration_extensions.numerics import concat
    s = v4_study.load_unit("loto", unit, ["source_train", "source_select", "source_audit"], base)
    keep = lambda ps, inside: ps.subset((ps.source_id == task) == inside)
    tr = keep(s["source_train"], False)
    ev = concat([keep(s[k], True) for k in s])
    target = {}
    for name, cal in fit_all(tr, peer).items():
        rm = row_metrics(ev, cal.predict_log_proba(ev))
        target[name] = {task: {"nll": float(rm["nll"].mean()), "brier": float(rm["brier"].mean()),
                               "ece_15": ece(rm["conf"], rm["correct"]), "accuracy": float(rm["correct"].mean()),
                               "delta_nll": float(rm["delta"].mean())}}
    return {"stage": "base_loto", "base": base, "unit": list(unit), "task": task, "target": target,
            "identity_raw_nll": v4_study.raw_identity_nll("loto", unit, task)}


def run_job(job):
    if job["stage"] == "base_unit":
        return unit_job(job["phase"], job["base"], tuple(job["unit"]), job["peer"])
    return loto_job(job["base"], tuple(job["unit"]), job["task"], job["peer"])


def jobs_for(stage, phase, base, peer):
    hy = json.loads((ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8"))
    units = [u.split("|") for u in sorted(hy["units"])]
    if stage == "base_unit":
        return [{"module": "base_study", "stage": stage, "phase": phase, "base": base, "unit": u, "peer": peer,
                 "variant": "base" + base} for u in units]
    return [{"module": "base_study", "stage": stage, "base": base, "unit": u, "task": t, "peer": peer,
             "variant": "base" + base} for u in units for t in hy["tasks"]]


# ---------------------------------------------------------------- summary

def rows_of(records, method, gate_variant=None):
    """Per (unit, task): metrics, delta vs same-base identity, delta vs the original single-order identity."""
    import gate_study
    out = []
    for r in records:
        if gate_variant is not None:
            eps = v4_study.VARIANTS[gate_variant][1]
            src = "V" if gate_study.accept(r["select"]["V"], "G-sig", eps) else "IDENTITY"
        else:
            src = method
        for t, m in r["target"][src].items():
            raw = (r.get("identity_raw_nll") or {}).get(t)
            out.append({"unit": tuple(r["unit"]), "task": t, **m,
                        "delta_raw": None if raw is None else m["nll"] - raw})
    return out


def describe(name, rows):
    import numpy as np
    d = np.array([x["delta_nll"] for x in rows])
    line = (f"{name:26s} n={len(rows):3d} delta {d.mean():+.4f} harm {int((d > HARM).sum()):3d} worst {d.max():+.4f} "
            f"NLL {np.mean([x['nll'] for x in rows]):.3f} ECE {np.mean([x['ece_15'] for x in rows]):.3f}")
    if rows and rows[0]["delta_raw"] is not None:
        dr = np.array([x["delta_raw"] for x in rows])
        line += f" | vs raw identity: delta {dr.mean():+.4f} harm {int((dr > HARM).sum()):3d}"
    print(line)
    return {"n": len(rows), "macro_delta": float(d.mean()), "harm": int((d > HARM).sum()), "worst": float(d.max())}


def summarize(phase, base):
    name = f"base_{phase}{base}"
    recs = v4_study.load(name)
    print(f"== {phase} on base '{base}' ({len(recs)} jobs)")
    for m in recs[0]["target"]:
        describe(m, rows_of(recs, m))
    for v, (_, _, _, b) in v4_study.VARIANTS.items():
        f = OUT / (("loto" if phase == "loto" else f"final_{phase}") + ".jsonl")
        if b == base and f.exists():
            rs = [r for r in v4_study.load(f.stem) if r["variant"] == v]
            if rs:
                describe(v, rows_of(rs, None, gate_variant=v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize"])
    ap.add_argument("--phase", default="dev")
    ap.add_argument("--base", default="_avg")
    a = ap.parse_args()
    summarize(a.phase, a.base)


if __name__ == "__main__":
    main()
