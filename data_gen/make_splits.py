"""Turn raw per-model probability files into the canonical dataset of spec 2.3.

    python data_gen/make_splits.py --raw data_gen/raw --out data/real

Writes data/real/dev/ (sources + target_dev as target) and data/real/confirm_sealed/
(sources + target_confirm). The sealed directory must not be read until the method,
config and endpoints are frozen. Source tasks are split by case (60/20/20) into
train/select/audit; every row of a case (orders, prompts, models) lands on one side.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_extensions import data_adapter as da  # noqa: E402

FIELDS = ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id", "probs", "label_ids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data_gen/raw")
    ap.add_argument("--pattern", default="probs_*.jsonl")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--min-letter-mass", type=float, default=0.0,
                    help="rows below are kept; this only reports them (no outcome-based filtering)")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(str(Path(args.raw) / args.pattern))):
        rows += [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:  # raw sample ids are per model; make them unique across models
        if not r["sample_id"].startswith(r["model_id"] + "::"):
            r["sample_id"] = f"{r['model_id']}::{r['sample_id']}"
    if not rows:
        raise SystemExit("no raw rows found")
    # HellaSwag `ind` is not unique: two different items share an id. Drop every case whose
    # sample ids collide (decided from ids alone, before any outcome is looked at).
    seen, clash = {}, set()
    for r in rows:
        key = (r["model_id"], r["sample_id"])
        if key in seen:
            clash.add(r["case_id"])
        seen[key] = True
    rows = [r for r in rows if r["case_id"] not in clash]
    print("dropped colliding cases:", sorted(clash))
    cases = sorted({r["case_id"] for r in rows if r["role"] == "source"})
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    n = len(cases)
    split_of = {c: ("source_train" if k < 0.6 * n else "source_select" if k < 0.8 * n else "source_audit")
                for k, c in enumerate(cases)}

    def canon(r, with_y):
        out = {k: r[k] for k in FIELDS} | {"label_provenance": "dataset_gold", "split_id": r["role"]}
        if with_y:
            out["y"] = r["y"]
        return out

    src = {s: [canon(r, True) for r in rows if r["role"] == "source" and split_of[r["case_id"]] == s]
           for s in ("source_train", "source_select", "source_audit")}
    for phase, role in (("dev", "target_dev"), ("confirm_sealed", "target_confirm")):
        d = Path(args.out) / phase
        tgt = [r for r in rows if r["role"] == role]
        for name, rs in src.items():
            da.write_jsonl(d / da.SPLIT_FILES[name], rs)
        da.write_jsonl(d / da.SPLIT_FILES["target_inputs"], [canon(r, False) for r in tgt])
        da.write_jsonl(d / da.PRIVATE_FILE, [{"sample_id": r["sample_id"], "case_id": r["case_id"], "y": r["y"],
                                              "label_id": r["label_ids"][r["y"]]} for r in tgt])
        manifest = {"phase": phase, "seed": args.seed, "models": sorted({r["model_id"] for r in rows}),
                    "prompts": sorted({r["prompt_id"] for r in rows}),
                    "target_tasks": sorted({r["task_id"] for r in tgt}),
                    "source_tasks": sorted({r["task_id"] for r in rows if r["role"] == "source"}),
                    "n_rows": {k: len(v) for k, v in src.items()} | {"target": len(tgt)},
                    "low_letter_mass_rows": sum(r["letter_mass"] < 0.5 for r in rows),
                    "sealed": phase == "confirm_sealed"}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(phase, manifest["n_rows"], "targets:", manifest["target_tasks"])


if __name__ == "__main__":
    main()
