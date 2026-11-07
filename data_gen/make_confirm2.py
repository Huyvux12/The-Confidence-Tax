"""Build data/real/confirm2_sealed: source files copied byte-for-byte from data/real/dev (so the frozen
selection can be reused), targets from data_gen/raw_confirm2 (PROTOCOL_LOG 2026-11-06).

    python data_gen/make_confirm2.py --raw data_gen/raw_confirm2 --dev data/real/dev --out data/real/confirm2_sealed
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_extensions import data_adapter as da  # noqa: E402

FIELDS = ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id", "probs", "label_ids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data_gen/raw_confirm2")
    ap.add_argument("--dev", default="data/real/dev")
    ap.add_argument("--out", default="data/real/confirm2_sealed")
    ap.add_argument("--role", default="target_confirm2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("source_train", "source_select", "source_audit"):
        shutil.copyfile(Path(args.dev) / da.SPLIT_FILES[s], out / da.SPLIT_FILES[s])
    rows = []
    for path in sorted(glob.glob(str(Path(args.raw) / "probs_*.jsonl"))):
        rows += [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        if not r["sample_id"].startswith(r["model_id"] + "::"):
            r["sample_id"] = f"{r['model_id']}::{r['sample_id']}"
        assert r["role"] == args.role and r["source_id"] is None
    seen, clash = set(), set()                      # same id-collision rule as make_splits.py
    for r in rows:
        if r["sample_id"] in seen:
            clash.add(r["case_id"])
        seen.add(r["sample_id"])
    rows = [r for r in rows if r["case_id"] not in clash]
    print("dropped colliding cases:", sorted(clash))
    tgt = [{k: r[k] for k in FIELDS} | {"label_provenance": "dataset_gold", "split_id": r["role"]} for r in rows]
    da.write_jsonl(out / da.SPLIT_FILES["target_inputs"], tgt)
    da.write_jsonl(out / da.PRIVATE_FILE, [{"sample_id": r["sample_id"], "case_id": r["case_id"], "y": r["y"],
                                           "label_id": r["label_ids"][r["y"]]} for r in rows])
    dev_manifest = json.loads((Path(args.dev) / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"phase": out.name, "seed": dev_manifest["seed"], "models": sorted({r["model_id"] for r in rows}),
                "prompts": sorted({r["prompt_id"] for r in rows}), "target_tasks": sorted({r["task_id"] for r in rows}),
                "source_tasks": dev_manifest["source_tasks"],
                "n_rows": {k: v for k, v in dev_manifest["n_rows"].items() if k != "target"} | {"target": len(tgt)},
                "low_letter_mass_rows_target": sum(r["letter_mass"] < 0.5 for r in rows),
                "sources_copied_from": args.dev, "sealed": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(manifest["n_rows"], "targets:", manifest["target_tasks"])


if __name__ == "__main__":
    main()
