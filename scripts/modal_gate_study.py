"""Run scripts/gate_study.py jobs on Modal CPU containers (one container per job).

    modal run -q scripts/modal_gate_study.py --phase dev          # -> runs/gate_study/dev.jsonl
    python scripts/gate_study.py summarize --phase dev
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/proj"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy==2.5.4", "scipy==1.18.1", "cvxpy==1.9.3", "clarabel==0.11.1")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "calibration_extensions", f"{REMOTE}/calibration_extensions", ignore=IGNORE)
         .add_local_file(ROOT / "scripts/gate_study.py", f"{REMOTE}/scripts/gate_study.py")
         .add_local_dir(ROOT / "data/real/dev", f"{REMOTE}/data/real/dev")
         .add_local_dir(ROOT / "data/real/confirm_sealed", f"{REMOTE}/data/real/confirm_sealed"))
app = modal.App("rbcal-gate-study", image=image)


@app.function(cpu=1.0, memory=4096, timeout=3600, retries=1)
def run(job: dict) -> str:
    sys.path.insert(0, f"{REMOTE}/scripts")
    import gate_study
    return json.dumps(gate_study.run_job(job))


@app.local_entrypoint()
def main(phase: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate_study                                   # top level needs no numpy
    hy = json.loads((ROOT / "runs/gate_study/hypers.json").read_text(encoding="utf-8"))
    jobs = gate_study.jobs_for(phase, hy)
    out = ROOT / f"runs/gate_study/{phase}.jsonl"
    print(f"{phase}: {len(jobs)} jobs")
    with open(out, "w", encoding="utf-8") as f:
        for k, res in enumerate(run.map(jobs), 1):
            f.write(res + "\n")
            if k % 10 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} done", flush=True)
    print("JOB DONE", out)
