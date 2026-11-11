"""Fixed-structure Margin vs No-margin control over the 250 LOTO units.

This is an ablation of the fixed RBF coefficient optimizer (fit_rbcal), NOT
the outer CMA-ES Evo-RB structure-search method.

Each model/prompt reuses the frozen v4b structure selected on all 25 source
tasks, then refits coefficients with the held-out task excluded. Thus the
LOTO experiment is *not* nested hyperparameter selection.

Scientific behaviour is retained from the author's original script:
 - train-side UCB constraint uses Delta + z*SE <= eps for z=1.645 or 0;
 - the shared G-sig deployment gate uses Delta - 1.645*SE <= eps.
Do not silently change the gate sign, which would change reported numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_extensions import data_adapter as da
from calibration_extensions.rb_calibration import fit_rbcal
from calibration_extensions.calibrators import IdentityCalibrator
from calibration_extensions.numerics import concat
from calibration_extensions.evaluator import row_metrics, ece
import gate_study

EPS = 0.01
Z_GATE = 1.645


def holdout(ps, task, inside):
    return ps.subset((ps.source_id == task) == inside)


def one(job):
    unit, task, spec, data_dir = job
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    splits = {
        k: da.load_split(data_dir, k, where)
        for k in ("source_train", "source_select", "source_audit")
    }
    train = holdout(splits["source_train"], task, False)
    select = holdout(splits["source_select"], task, False)
    heldout = concat([
        holdout(splits[k], task, True)
        for k in ("source_train", "source_select", "source_audit")
    ])

    record = {"unit": list(unit), "task": task, "spec": spec}
    for name, z in (("margin", 1.645), ("no_margin", 0.0)):
        fit = fit_rbcal(
            train, select, [tuple(spec)], EPS, "FIXED_MARGIN_CONTROL", z=z
        )
        cal = fit.fitted
        stats = gate_study.select_stats(select, cal) if cal is not None else None
        accepted = gate_study.accept(stats, "G-sig", EPS) if stats is not None else False
        deployed = cal if accepted else IdentityCalibrator()

        metrics = row_metrics(heldout, deployed.predict_log_proba(heldout))
        delta = float(metrics["delta"].mean())
        record[name] = {
            "delta_nll": delta,
            "ece": ece(metrics["conf"], metrics["correct"]),
            "brier": float(metrics["brier"].mean()),
            "harm": int(delta > EPS),
            "accepted": accepted,
            # FitResult has solver_valid; there is no FitResult.valid.
            "fit_valid": fit.solver_valid,
            "gate_max": (
                None if stats is None
                else max(v["delta"] - Z_GATE * v["se"] for v in stats.values())
            ),
        }
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-structure Margin vs No-margin control for LOTO"
    )
    parser.add_argument(
        "--data", default=str(ROOT / "data/real/dev"),
        help="directory containing source_train/select/audit JSONL files",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out", default=str(ROOT / "runs/fixed_margin_control.jsonl")
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    chosen = json.loads(
        (ROOT / "runs/v4_study/chosen.json").read_text(encoding="utf-8")
    )["v4b"]
    hyper = json.loads(
        (ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8")
    )
    units = [tuple(key.split("|")) for key in sorted(hyper["units"])]
    tasks = sorted(hyper["tasks"])
    jobs = [
        (unit, task, chosen["|".join(unit)], args.data)
        for unit in units for task in tasks
    ]
    if len(jobs) != 250:
        raise ValueError(f"Expected 250 model–prompt–task LOTO jobs; got {len(jobs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            done[("|".join(record["unit"]), record["task"])] = record

    # Refuse to reuse a cached result under a different frozen field structure.
    for unit, task, spec, _ in jobs:
        key = ("|".join(unit), task)
        if key in done and done[key]["spec"] != spec:
            raise ValueError(f"Cached specification mismatch for {key}")
    todo = [
        job for job in jobs if ("|".join(job[0]), job[1]) not in done
    ]
    print(f"{len(jobs)} total, {len(done)} cached, {len(todo)} to run", flush=True)

    if todo:
        # Save completed tasks as they finish, allowing resumption after interruption.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(one, job) for job in todo]
            with out.open("a", encoding="utf-8") as checkpoint:
                for i, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    key = ("|".join(record["unit"]), record["task"])
                    done[key] = record
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    if i % 10 == 0 or i == len(futures):
                        print(f"{i}/{len(futures)} finished", flush=True)

    # Canonical deterministic order after all jobs finish.
    rows = [done[("|".join(unit), task)] for unit, task, _, _ in jobs]
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    for name in ("margin", "no_margin"):
        deltas = np.array([row[name]["delta_nll"] for row in rows])
        harmed = deltas > EPS
        pos_regret = float(deltas[harmed].mean()) if harmed.any() else None
        print(
            name,
            "n", len(deltas),
            "delta", float(deltas.mean()),
            "ece", float(np.mean([r[name]["ece"] for r in rows])),
            "brier", float(np.mean([r[name]["brier"] for r in rows])),
            "harm", int(harmed.sum()),
            "pos_regret", pos_regret,
            "accepted", float(np.mean([r[name]["accepted"] for r in rows])),
            flush=True,
        )


if __name__ == "__main__":
    main()
