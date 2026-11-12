"""Run scripts/v4_study.py jobs on Modal CPU containers; results are appended as they arrive (resumable).

    MODAL_PROFILE=acc3 modal run scripts/modal_v4.py --stage loso
    python scripts/v4_study.py choose
    MODAL_PROFILE=acc3 modal run scripts/modal_v4.py --stage final --phase dev
    MODAL_PROFILE=acc3 modal run scripts/modal_v4.py --stage loto --variants v4b
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/proj"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy==2.5.4", "scipy==1.18.1", "cvxpy==1.9.3", "clarabel==0.11.1", "cma==4.4.0")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_dir(ROOT / "calibration_extensions", f"{REMOTE}/calibration_extensions", ignore=IGNORE)
         .add_local_file(ROOT / "scripts/gate_study.py", f"{REMOTE}/scripts/gate_study.py")
         .add_local_file(ROOT / "scripts/v4_study.py", f"{REMOTE}/scripts/v4_study.py")
         .add_local_file(ROOT / "scripts/base_study.py", f"{REMOTE}/scripts/base_study.py")
         .add_local_file(ROOT / "scripts/evo_study.py", f"{REMOTE}/scripts/evo_study.py")
         .add_local_file(ROOT / "runs/gate_study/hypers.json", f"{REMOTE}/runs/gate_study/hypers.json"))
DATA_DIRS = ["dev", "dev_avg", "dev_aug", "confirm_sealed", "confirm2_sealed", "confirm3_sealed",
             "confirm4_sealed", "confirm4_sealed_avg", "confirm4_sealed_aug"]
for _d in DATA_DIRS:                                  # same list locally and in the container (mounted dirs)
    if (ROOT / "data/real" / _d).exists():
        image = image.add_local_dir(ROOT / "data/real" / _d, f"{REMOTE}/data/real/{_d}")
app = modal.App("rbcal-v4", image=image)


@app.function(cpu=1.0, memory=4096, timeout=3 * 3600, retries=1)
def run(job: dict) -> str:
    import importlib
    sys.path.insert(0, f"{REMOTE}/scripts")
    mod = importlib.import_module(job.get("module", "v4_study"))
    return json.dumps(mod.run_job(job))


def key(j):
    return json.dumps([j["stage"], j.get("phase"), j["unit"], j["variant"], j.get("k"), j.get("task"), j.get("base")])


@app.local_entrypoint()
def main(stage: str, phase: str = "dev", variants: str = "", base: str = "", peer: bool = False):
    sys.path.insert(0, str(ROOT / "scripts"))
    import v4_study
    outdir = "runs/v4_study"
    if stage in ("base_unit", "base_loto"):
        import base_study
        jobs = base_study.jobs_for(stage, phase, base, peer)
        name = f"base_{phase}{base}" if stage == "base_unit" else f"base_loto{base}"
    elif stage in ("front", "evo_unit", "evo_loto", "evo_loto_nested"):
        import evo_study
        jobs = evo_study.jobs_for(stage, phase)
        name = {"front": "front", "evo_unit": f"evo_{phase}",
                "evo_loto": "evo_loto", "evo_loto_nested": "evo_loto_nested"}[stage]
        outdir = "runs/evo_study"
    else:
        vs = [v for v in variants.split(",") if v] or None
        if stage == "final" and vs is None:
            vs = list(v4_study.VARIANTS)
        jobs = v4_study.jobs_for(stage, phase, vs)
        name = {"loso": "loso", "final": f"final_{phase}",
                "loto": "loto", "loto_nested": "loto_nested"}[stage]
    out = ROOT / outdir / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                r.setdefault("variant", "base" + r.get("base", ""))
                done.add(key({**r, "stage": stage, "phase": phase if stage in ("final", "base_unit", "evo_unit") else None}))
    todo = [j for j in jobs if key({**j, "phase": j.get("phase") if stage in ("final", "base_unit", "evo_unit") else None}) not in done]
    print(f"{stage}: {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    fails = 0
    with open(out, "a", encoding="utf-8") as f:
        for k, res in enumerate(run.map(todo, order_outputs=False, return_exceptions=True), 1):
            if isinstance(res, Exception):
                fails += 1
                print("FAILED:", repr(res)[:300], flush=True)
                continue
            f.write(res + "\n")
            f.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"{k}/{len(todo)}", flush=True)
    print(f"JOB DONE {stage} fails={fails}")
