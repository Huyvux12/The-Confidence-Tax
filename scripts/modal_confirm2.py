"""Fit every method for the confirm2 roster on Modal CPU containers (PROTOCOL_LOG 2026-11-06).

Per unit: run_rbcal.run_unit with the selection frozen in runs/rbcal_dev (sources byte-identical), the
gate_study job (v3 gate statistics), and THERMO_P / ATS_P predictions. Labels are opened locally afterwards.

    MODAL_PROFILE=acc3 modal run scripts/modal_confirm2.py
    python scripts/run_rbcal.py --data data/real/confirm2_sealed --out runs/rbcal_confirm2 \
        --reuse-selection data/real/dev runs/rbcal_dev/units        # all units cached -> scoring only
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/proj"
DATA = {"confirm2": "data/real/confirm2_sealed", "confirm3": "data/real/confirm3_sealed", "confirm4": "data/real/confirm4_sealed"}
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy==2.5.4", "scipy==1.18.1", "cvxpy==1.9.3", "clarabel==0.11.1")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "calibration_extensions", f"{REMOTE}/calibration_extensions", ignore=IGNORE)
         .add_local_file(ROOT / "scripts/run_rbcal.py", f"{REMOTE}/scripts/run_rbcal.py")
         .add_local_file(ROOT / "scripts/gate_study.py", f"{REMOTE}/scripts/gate_study.py"))
for _d in DATA.values():
    if (ROOT / _d).exists():
        image = image.add_local_dir(ROOT / _d, f"{REMOTE}/{_d}")
app = modal.App("rbcal-confirm2-fit", image=image)


@app.function(cpu=1.0, memory=6144, timeout=4 * 3600, retries=1)
def fit_unit(unit: list, chosen: dict, hypers: dict, phase: str = "confirm2") -> dict:
    import os, pickle
    os.chdir(REMOTE)
    sys.path.insert(0, f"{REMOTE}/scripts")
    import gate_study
    import run_rbcal
    from calibration_extensions import data_adapter as da
    from calibration_extensions.extra_baselines import fit_ats_p, fit_thermometer_p
    data = f"{REMOTE}/{DATA[phase]}"
    res = run_rbcal.run_unit((data, tuple(unit), run_rbcal.EPS_GRID, chosen))
    gate = gate_study.unit_job(phase, tuple(unit), hypers)
    where = {"model_id": unit[0], "prompt_id": unit[1]}
    tr = da.load_split(data, "source_train", where)
    tgt = da.load_split(data, "target_inputs", where)
    extra = {}
    for mid, fit in (("THERMO_P", fit_thermometer_p), ("ATS_P", fit_ats_p)):
        cal, info = fit(tr)
        extra[mid] = {"status": info.get("status"), "pred": dict(zip([str(s) for s in tgt.sample_id],
                                                                     cal.predict_log_proba(tgt)))}
    return {"unit": pickle.dumps(res), "gate": json.dumps(gate), "extra": pickle.dumps(extra)}


@app.local_entrypoint()
def main(phase: str = "confirm2"):
    chosen = json.loads((ROOT / "runs/gate_study/chosen_dev.json").read_text(encoding="utf-8"))
    hypers = json.loads((ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8"))["units"]
    units = sorted(chosen)
    out = ROOT / f"runs/rbcal_{phase}"
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "extra").mkdir(exist_ok=True)
    gates = []
    args = [(u.split("|"), chosen[u], hypers[u], phase) for u in units]
    for u, res in zip(units, fit_unit.starmap(args)):
        stem = u.replace("/", "__").replace("|", "__")
        (out / "units" / f"{stem}.pkl").write_bytes(res["unit"])
        (out / "extra" / f"{stem}.pkl").write_bytes(res["extra"])
        gates.append(res["gate"])
        print("saved", u, flush=True)
    (ROOT / f"runs/gate_study/{phase}.jsonl").write_text("\n".join(gates) + "\n", encoding="utf-8")
    print("JOB DONE")
