"""Record the Hugging Face commits of the models and datasets used for the paper.

    python scripts/record_revisions.py

Inference and item building ran on 2026-10-31 without pinned revisions. A repository whose last commit
is older than that run was used at its current commit, which is what this script records; it refuses to
record a repository that changed after the run.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

RUN_STARTED_UTC = "2026-10-31T07:30:00Z"   # item building and inference started after this time
MODELS = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "ibm-granite/granite-4.2-8b",
          "mistralai/Ministral-3-8B-Instruct-2512", "allenai/Olmo-3-7B-Instruct"]
DATASETS = ["cais/mmlu", "TIGER-Lab/MMLU-Pro", "allenai/ai2_arc", "allenai/openbookqa", "tau/commonsense_qa",
            "Rowan/hellaswag", "allenai/sciq", "openlifescienceai/medmcqa", "truthfulqa/truthful_qa",
            "google/boolq", "deepmind/aqua_rat", "allenai/winogrande", "baber/piqa", "lucasmccabe/logiqa"]


def info(kind, repo):
    raw = subprocess.run(["curl", "-s", "-m", "60", f"https://huggingface.co/api/{kind}/{repo}"],
                         capture_output=True, check=True).stdout
    d = json.loads(raw)
    lic = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")]
    return {"sha": d["sha"], "last_modified": d["lastModified"], "license": lic[0] if lic else None}


def main():
    out = {"note": "commit at the time of the 2026-10-31 run (repositories unchanged since); logiqa was read "
                   "from refs/convert/parquet", "run_started_utc": RUN_STARTED_UTC, "models": {}, "datasets": {},
           "details": {}}
    for kind, repos in (("models", MODELS), ("datasets", DATASETS)):
        for r in repos:
            d = info(kind, r)
            if d["last_modified"] > RUN_STARTED_UTC:
                raise SystemExit(f"{r} changed after the run ({d['last_modified']}); its commit then is unknown")
            out[kind][r] = d["sha"]
            out["details"][r] = d
    Path("data_gen/revisions.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["details"], indent=1))


if __name__ == "__main__":
    main()
