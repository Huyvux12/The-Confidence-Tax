"""Phase 8 (todo.md): RB-Cal with answer-order consistency features (v4a: rbf_oc, v4b: rbf_oct).

Same candidate as v3 (eps = 0.01, z = 1.645, LOSO menu selection on source_train, G-sig gate, identity
fallback); only the feature map of the field changes. Stages (jobs run on Modal via scripts/modal_v4.py):

    loso    score one menu spec of one variant on one unit (25 LOSO folds of dev source_train)
    choose  pick each unit's spec with the select_by_loso rule            (local: python scripts/v4_study.py choose)
    final   refit the chosen spec on source_train, gate stats on source_select, target metrics (phase dev|confirm|confirm2)
    loto    refit on 24 source tasks, evaluate the held-out task (250 jobs)
    summarize --phase dev|loto|...                                         (local)
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EPS, Z_FIT = 0.01, 1.645
# variant -> (menu, budget eps, fitting margin z, data base); v5 = relaxed operating point (phase 9);
# phase 10: base "_avg" = probabilities averaged over the two answer orders, "_aug" = original probabilities, both
# with precomputed order_tv_raw / peer_top / peer_top1 fields (data_gen/make_aug.py)
VARIANTS = {"v4a": ("RBF_OC_MENU", 0.01, 1.645, ""), "v4b": ("RBF_OCT_MENU", 0.01, 1.645, ""),
            "v5a": ("RBF_OCT_MENU", 0.01, 0.0, ""), "v5b": ("RBF_OCT_MENU", 0.02, 0.0, ""),
            "v4b_avg": ("RBF_OCT_MENU", 0.01, 1.645, "_avg"),
            "v6_aug": ("RBF_OCP_MENU", 0.01, 1.645, "_aug"), "v6_avg": ("RBF_OCP_MENU", 0.01, 1.645, "_avg"),
            "v6p1_aug": ("RBF_OCP1_MENU", 0.01, 1.645, "_aug"), "v6p1_avg": ("RBF_OCP1_MENU", 0.01, 1.645, "_avg")}
OUT = ROOT / "runs/v4_study"
DATA = {"dev": "data/real/dev", "confirm": "data/real/confirm_sealed", "confirm2": "data/real/confirm2_sealed", "confirm3": "data/real/confirm3_sealed", "confirm4": "data/real/confirm4_sealed",
        "loto": "data/real/dev"}


def menu(variant):
    from calibration_extensions import rb_calibration as rb
    return getattr(rb, VARIANTS[variant][0])


def load_unit(phase, unit, splits, base=""):
    from calibration_extensions import data_adapter as da
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    return {s: da.load_split(str(ROOT / (DATA[phase] + base)), s, where) for s in splits}


def raw_identity_nll(phase, unit, task=None):
    """Per-task NLL of the original single-order model (reference for harm when the base is averaged)."""
    import numpy as np
    from calibration_extensions import data_adapter as da
    from calibration_extensions.numerics import concat, surprisal
    if task is None:
        t = load_unit(phase, unit, ["target_inputs"])["target_inputs"]
        t = da.attach_labels(t, da.load_private_labels(str(ROOT / DATA[phase])))
    else:
        s = load_unit(phase, unit, ["source_train", "source_select", "source_audit"])
        t = concat([ps.subset(ps.source_id == task) for ps in s.values()])
    nll = surprisal(t)
    keys = np.array([str(x) for x in t.task_id])
    return {k: float(nll[keys == k].mean()) for k in sorted(set(keys.tolist()))}


# ---------------------------------------------------------------- remote stages

def loso_job(unit, variant, k):
    import numpy as np
    from calibration_extensions.numerics import TOL
    from calibration_extensions.rb_calibration import Spec, solve
    from calibration_extensions.source_objectives import calibrator_deltas
    from calibration_extensions.source_splits import loso_folds
    _, eps, z, base = VARIANTS[variant]
    tr = load_unit("dev", unit, ["source_train"], base)["source_train"]
    spec = Spec(*menu(variant)[k])
    d = []
    for held, trf, val in loso_folds(tr):
        cal, _ = solve(trf, spec, eps, z)
        d.append(np.inf if cal is None else calibrator_deltas(val, cal)[held])
    d = np.array(d)
    return {"stage": "loso", "unit": list(unit), "variant": variant, "k": k, "spec": list(menu(variant)[k]),
            "violations": int(np.sum(d > EPS + TOL)), "mean_heldout": float(np.mean(d)), "max_heldout": float(np.max(d))}


def fit_candidate(tr, sel, spec, variant):
    import gate_study
    from calibration_extensions.calibrators import IdentityCalibrator
    from calibration_extensions.rb_calibration import fit_rbcal
    _, eps, z, _base = VARIANTS[variant]
    fr = fit_rbcal(tr, sel, [tuple(spec)], eps, "V4", z=z)
    cals = {"IDENTITY": IdentityCalibrator(), "V": fr.fitted}
    return cals, (None if fr.fitted is None else gate_study.select_stats(sel, fr.fitted))


def final_job(phase, unit, variant, spec):
    from calibration_extensions import data_adapter as da
    from calibration_extensions.evaluator import score
    base = VARIANTS[variant][3]
    s = load_unit(phase, unit, ["source_train", "source_select", "target_inputs"], base)
    tgt = da.attach_labels(s["target_inputs"], da.load_private_labels(str(ROOT / (DATA[phase] + base))))
    from calibration_extensions.evaluator import row_metrics
    cals, stats = fit_candidate(s["source_train"], s["source_select"], spec, variant)
    target, row_nll = {}, {}
    sids = [str(x) for x in tgt.sample_id]
    for name, cal in cals.items():
        if cal is None:
            continue
        lq = cal.predict_log_proba(tgt)
        rows = [r for r in score(tgt, lq, {}) if not r["task_id"].startswith("__")]
        target[name] = {r["task_id"]: {k: r[k] for k in ("nll", "brier", "ece_15", "accuracy", "delta_nll")} for r in rows}
        row_nll[name] = dict(zip(sids, row_metrics(tgt, lq)["nll"].tolist()))
    return {"stage": "final", "phase": phase, "unit": list(unit), "variant": variant, "spec": spec,
            "select": {"V": stats}, "target": target, "row_nll": row_nll,
            "identity_raw_nll": raw_identity_nll(phase, unit) if base else None}


def nested_loto_spec(train, variant, heldout_task):
    """Select the discrete spec using inner LOSO on 24 outer-training tasks only.

    This must NOT use the global runs/v4_study/chosen.json configuration.
    Only source_train rows of the 24 remaining tasks enter the selection.
    """
    from calibration_extensions.rb_calibration import Spec, select_by_loso

    if any(str(s) == heldout_task for s in train.source_id):
        raise ValueError("Held-out task leaked into nested LOTO selection")
    if len(train.sources()) != 24:
        raise ValueError("Nested LOTO requires exactly 24 remaining source tasks")
    _name, eps, z, _base = VARIANTS[variant]
    specs = [Spec(*entry) for entry in menu(variant)]
    selected, _scores = select_by_loso(train, specs, eps, EPS, z)
    return [selected.kind, selected.J, selected.sigma, selected.lam]


def loto_job(unit, task, variant, spec=None, nested=False):
    from calibration_extensions.evaluator import ece, row_metrics
    from calibration_extensions.numerics import concat
    base = VARIANTS[variant][3]
    s = load_unit("loto", unit, ["source_train", "source_select", "source_audit"], base)
    keep = lambda ps, inside: ps.subset((ps.source_id == task) == inside)
    tr, sel = keep(s["source_train"], False), keep(s["source_select"], False)
    if nested:
        if spec is not None:
            raise ValueError("Nested LOTO must not receive a globally selected spec")
        spec = nested_loto_spec(tr, variant, task)
    elif spec is None:
        raise ValueError("Legacy LOTO requires its historical global spec")
    ev = concat([keep(s[k], True) for k in s])
    cals, stats = fit_candidate(tr, sel, spec, variant)
    target = {}
    for name, cal in cals.items():
        if cal is None:
            continue
        rm = row_metrics(ev, cal.predict_log_proba(ev))
        target[name] = {task: {"nll": float(rm["nll"].mean()), "brier": float(rm["brier"].mean()),
                               "ece_15": ece(rm["conf"], rm["correct"]), "accuracy": float(rm["correct"].mean()),
                               "delta_nll": float(rm["delta"].mean())}}
    result = {"stage": "loto_nested" if nested else "loto", "unit": list(unit), "task": task,
              "variant": variant, "spec": spec, "select": {"V": stats}, "target": target,
              "identity_raw_nll": raw_identity_nll("loto", unit, task) if base else None}
    if nested:
        result["selection_protocol"] = "nested_loso_on_24_sources"
    return result


def run_job(job: dict) -> dict:
    st = job["stage"]
    if st == "loso":
        return loso_job(tuple(job["unit"]), job["variant"], job["k"])
    if st == "final":
        return final_job(job["phase"], tuple(job["unit"]), job["variant"], job["spec"])
    if st == "loto_nested":
        return loto_job(tuple(job["unit"]), job["task"], job["variant"], nested=True)
    if st == "loto":
        return loto_job(tuple(job["unit"]), job["task"], job["variant"], job["spec"])
    raise ValueError(f"Unknown v4 stage: {st}")


def jobs_for(stage, phase=None, variants=None):
    """Job dicts; needs only json (the Modal entrypoint has no numpy)."""
    hy = json.loads((ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8"))
    units = [u.split("|") for u in sorted(hy["units"])]
    variants = variants or list(VARIANTS)
    if stage == "loso":
        return [{"stage": "loso", "unit": u, "variant": v, "k": k} for u in units for v in variants for k in range(37)]
    if stage == "loto_nested":
        return [{"stage": stage, "unit": u, "task": t, "variant": v}
                for u in units for t in hy["tasks"] for v in variants]
    if stage not in ("final", "loto"):
        raise ValueError(f"Unknown v4 stage: {stage}")
    chosen = json.loads((OUT / "chosen.json").read_text(encoding="utf-8"))
    if stage == "final":
        return [{"stage": "final", "phase": phase, "unit": u, "variant": v, "spec": chosen[v]["|".join(u)]}
                for u in units for v in variants]
    return [{"stage": "loto", "unit": u, "task": t, "variant": v, "spec": chosen[v]["|".join(u)]}
            for u in units for t in hy["tasks"] for v in variants]


# ---------------------------------------------------------------- local stages

def load(name):
    return [json.loads(l) for l in (OUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def choose():
    """select_by_loso rule: fewest held-out violations, then mean held-out delta (1e-8 ties), then (J, -lam), order."""
    from calibration_extensions.numerics import TOL
    scores = load("loso")
    present = [v for v in VARIANTS if any(s["variant"] == v for s in scores)]
    out = {v: {} for v in present}
    for v in present:
        for u in sorted({"|".join(s["unit"]) for s in scores}):
            ss = sorted([s for s in scores if s["variant"] == v and "|".join(s["unit"]) == u], key=lambda s: s["k"])
            assert len(ss) == 37, (v, u, len(ss))
            pool = [s for s in ss if s["violations"] == min(x["violations"] for x in ss)]
            best = min(s["mean_heldout"] for s in pool)
            pool = [s for s in pool if s["mean_heldout"] <= best + TOL]
            pick = min(pool, key=lambda s: ((s["spec"][1], -s["spec"][3]), s["k"]))
            out[v][u] = pick["spec"]
    (OUT / "chosen.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for v in out:
        print(v, {u.split("/")[1][:14]: s for u, s in out[v].items()})


def deployed_rows(records, use_gate=True):
    import gate_study
    rows = []
    for r in records:
        eps = VARIANTS[r["variant"]][1]
        ok = gate_study.accept(r["select"]["V"], "G-sig", eps) if use_gate else r["select"]["V"] is not None
        src = r["target"]["V"] if ok else r["target"]["IDENTITY"]
        rows += [{"unit": tuple(r["unit"]), "task": t, "accepted": ok, **m} for t, m in src.items()]
    return rows


def summarize(phase):
    import gate_study
    name = "loto" if phase == "loto" else f"final_{phase}"
    recs = load(name)
    for v in VARIANTS:
        rs = [r for r in recs if r["variant"] == v]
        if rs:
            s = gate_study.summarize_rows(deployed_rows(rs))
            print(f"{v}: n={s['n']} delta {s['macro_delta']:+.4f} worst {s['worst']:+.4f} harm {s['harm_001']} "
                  f"h_any {s['harm_any']} NLL {s['nll']:.3f} Brier {s['brier']:.3f} ECE {s['ece_15']:.3f} "
                  f"accept {s['accept_rate']:.2f}")
    ref = phase
    try:
        v3 = gate_study.summarize_rows(gate_study.variant_rows(gate_study.load_records(ref), ("C1", "G-sig", "IDENTITY")))
        print(f"v3 : n={v3['n']} delta {v3['macro_delta']:+.4f} worst {v3['worst']:+.4f} harm {v3['harm_001']} "
              f"h_any {v3['harm_any']} NLL {v3['nll']:.3f} Brier {v3['brier']:.3f} ECE {v3['ece_15']:.3f}")
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["choose", "summarize"])
    ap.add_argument("--phase", default="dev")
    args = ap.parse_args()
    choose() if args.cmd == "choose" else summarize(args.phase)


if __name__ == "__main__":
    main()
