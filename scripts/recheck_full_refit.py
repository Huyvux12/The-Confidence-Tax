"""Recheck (todo.md phase 5C): refit one confirm unit from scratch, with the full leave-one-source-out
menu search (no --reuse-selection), and compare with the scored confirm run.

    python scripts/recheck_full_refit.py Qwen/Qwen3.5-9B paren
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rbcal import EPS_GRID, run_unit  # noqa: E402


def main():
    unit = (sys.argv[1], sys.argv[2])
    fresh = run_unit(("data/real/confirm_sealed", unit, EPS_GRID, None))
    name = unit[0].replace("/", "__") + "__" + unit[1] + ".pkl"
    scored = pickle.loads((Path("runs/rbcal_confirm/units") / name).read_bytes())
    assert fresh["target_sample_ids"] == scored["target_sample_ids"]
    rows = {}
    for m in scored["methods"]:
        a, b = fresh["methods"][m], scored["methods"][m]
        diff = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a["target_log_q"], b["target_log_q"]))
        rows[m] = {"max_abs_logq_diff": diff, "same_hyper": a["hyper"] == b["hyper"],
                   "same_gate": (a["gate"] or {}).get("accept") == (b["gate"] or {}).get("accept")}
    out = {"unit": list(unit), "minutes": fresh["seconds"] / 60, "regime_same": fresh["regime"] == scored["regime"],
           "methods": rows, "all_hyper_same": all(r["same_hyper"] for r in rows.values()),
           "all_gates_same": all(r["same_gate"] for r in rows.values()),
           "max_abs_logq_diff": max(r["max_abs_logq_diff"] for r in rows.values())}
    Path("runs/recheck").mkdir(parents=True, exist_ok=True)
    Path("runs/recheck/full_refit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=1))


if __name__ == "__main__":
    main()
