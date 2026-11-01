"""Score every model in sequence on the VM; resumable, frees the HF cache after each model."""
import json, os, shutil, subprocess, sys

MODELS = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "ibm-granite/granite-4.2-8b",
          "mistralai/Ministral-3-8B-Instruct-2512", "allenai/Olmo-3-7B-Instruct"]
# Commits used for the paper (recorded afterwards; see data_gen/revisions.json). Missing file = latest.
REV_FILE = "/content/revisions.json"
REVISIONS = json.load(open(REV_FILE))["models"] if os.path.exists(REV_FILE) else {}
for m in MODELS:
    out = "/content/probs_" + m.split("/")[1].replace(".", "_") + ".jsonl"
    print(f"=== START {m}", flush=True)
    pinned = REVISIONS.get(m)
    rc = subprocess.call([sys.executable, "/content/infer_mcq.py", "--items", "/content/items.jsonl",
                          "--model", m, "--out", out] + (["--revision", pinned] if pinned else []))
    print(f"=== END {m} rc={rc}", flush=True)
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub"), ignore_errors=True)
print("ALL DONE", flush=True)
