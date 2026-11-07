"""Score the confirm4 items with the five models on Modal A100-40GB (one container per model).

    MODAL_PROFILE=acc3 modal run data_gen/modal_infer_confirm4.py --limit 2      # smoke test
    MODAL_PROFILE=acc3 modal run data_gen/modal_infer_confirm4.py                # full run -> data_gen/raw_confirm4/

Same infer_mcq.py, prompts, bf16 and pinned model commits as the 2026-10-31 run; transformers 5.21.0 without
flash-linear-attention / causal_conv1d (the original run used the reference kernels too).
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODELS = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "ibm-granite/granite-4.2-8b",
          "mistralai/Ministral-3-8B-Instruct-2512", "allenai/Olmo-3-7B-Instruct"]
REVISIONS = json.loads((HERE / "revisions.json").read_text(encoding="utf-8"))["models"]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch", "torchvision", "transformers==5.21.0", "accelerate", "sentencepiece",
                         "protobuf", "pillow")
         .add_local_file(HERE / "infer_mcq.py", "/root/infer_mcq.py")
         .add_local_file(HERE / "revisions.json", "/root/revisions.json")
         .add_local_file(HERE / "items_confirm4.jsonl", "/root/items.jsonl")
         .add_local_file(HERE / "items.jsonl", "/root/items_orig.jsonl"))
app = modal.App("rbcal-infer-confirm4", image=image)
vol = modal.Volume.from_name("rbcal-confirm2", create_if_missing=True)


def fname(model: str) -> str:
    return "probs_" + model.split("/")[1].replace(".", "_") + ".jsonl"


@app.function(gpu="A100-40GB", timeout=4 * 3600, cpu=4, memory=32768, retries=0, volumes={"/vol": vol})
def infer(model: str, limit: int = 0) -> str:
    """Score the confirm4 items, then 3 items per task of the original pool (numerics check vs data_gen/raw).
    Outputs go to the volume (resumable: infer_mcq.py skips rows already written)."""
    import os, subprocess, sys, torch
    tv = str(torch.__version__)
    print("torch", tv, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
    sub = "c4_smoke" if limit else "c4"
    for key, items, lim in ((sub, "/root/items.jsonl", limit), (sub + "_check", "/root/items_orig.jsonl", 3)):
        os.makedirs(f"/vol/{key}", exist_ok=True)
        out = f"/vol/{key}/{fname(model)}"
        cmd = [sys.executable, "/root/infer_mcq.py", "--items", items, "--model", model,
               "--out", out, "--revision", REVISIONS[model]] + (["--limit", str(lim)] if lim else [])
        rc = subprocess.call(cmd)
        vol.commit()
        if rc != 0:
            raise RuntimeError(f"{model} {key} failed rc={rc}")
    return f"{model} done, torch {tv}"


@app.local_entrypoint()
def main(limit: int = 0, only: str = ""):
    models = [m for m in MODELS if not only or only in m]
    for msg in infer.map(models, kwargs={"limit": limit}, return_exceptions=True):
        print(msg, flush=True)
    print("JOB DONE  ->  modal volume get rbcal-confirm2 c4 data_gen/raw_confirm4 (and main_check)")
