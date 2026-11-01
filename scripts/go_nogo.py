"""Go/no-go for M1 on real data, using source_train only (no target is read).

For each model x prompt: does a shared temperature have a common descent direction across
sources, do the entropy-affine / RBF classes have one, and where does MRTS land?

    python scripts/go_nogo.py --data data/real/dev --out runs/go_nogo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_extensions import data_adapter as da  # noqa: E402
from calibration_extensions.features import fit_feature_map  # noqa: E402
from calibration_extensions.m1_functional import M1Config, diagnose_at_identity  # noqa: E402
from calibration_extensions.source_objectives import beta_grid, fit_scalar, scalar_delta_table  # noqa: E402
from calibration_extensions.source_splits import calibrator_units, unit_subset  # noqa: E402

CLASSES = [("constant", 0, None), ("entropy_affine", 0, None), ("rbf", 4, 0.5), ("rbf", 8, 0.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/dev")
    ap.add_argument("--out", default="runs/go_nogo.json")
    args = ap.parse_args()
    tr_all = da.load_split(args.data, "source_train")
    cfg = M1Config()
    grid = beta_grid()
    report = []
    for u in calibrator_units(tr_all):
        tr = unit_subset(tr_all, u)
        table, pooled, srcs = scalar_delta_table(tr, grid)
        best_beta = grid[np.argmin(table, axis=0)]           # each source's own best shared beta
        mrts, info = fit_scalar(tr, grid, "minimax")
        pts, _ = fit_scalar(tr, grid, "pooled")
        rec = {"model": u[0], "prompt": u[1], "n_rows": tr.n, "n_sources": len(srcs),
               "mrts_beta": mrts.beta, "mrts_worst_delta": info["score"], "pooled_beta": pts.beta,
               "sources_want_colder": int(np.sum(best_beta < 1 - 1e-9)),
               "sources_want_hotter": int(np.sum(best_beta > 1 + 1e-9)),
               "per_source_best_beta": dict(zip(srcs, best_beta.round(3).tolist()))}
        for kind, J, s in CLASSES:
            d = diagnose_at_identity(tr, fit_feature_map(tr, kind, J, s), cfg)
            rec[f"{kind}{J or ''}"] = {"status": d["status"], "rho": d["rho"],
                                       "grad_check": d["gradient_check_max_abs_error"]}
        report.append(rec)
        print(f"{u[0]:<40} {u[1]:<13} MRTS beta={mrts.beta:.3f} worst={info['score']:+.4f} "
              f"colder/hotter={rec['sources_want_colder']}/{rec['sources_want_hotter']} | "
              + " ".join(f"{k}:{rec[k]['status'][:6]}" for k in ("constant", "entropy_affine", "rbf4", "rbf8")),
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    conflict = [r for r in report if r["constant"]["status"] != "COMMON_DESCENT_VERIFIED"]
    resolved = [r for r in conflict if any(r[k]["status"] == "COMMON_DESCENT_VERIFIED"
                                           for k in ("entropy_affine", "rbf4", "rbf8"))]
    print(f"\nunits={len(report)} constant-conflict={len(conflict)} resolved-by-richer-class={len(resolved)}")


if __name__ == "__main__":
    main()
