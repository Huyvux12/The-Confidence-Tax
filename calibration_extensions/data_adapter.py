"""Read/write the canonical JSONL files of spec section 2.3.

Only `evaluator.py` and the M5 oracle may call `load_private_labels`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .numerics import ProbSet, prepare_probs

SPLIT_FILES = {
    "source_train": "source_train.jsonl",
    "source_select": "source_select.jsonl",
    "source_audit": "source_audit.jsonl",
    "target_inputs": "target_inputs.jsonl",
}
PRIVATE_FILE = "target_labels_private.jsonl"
REQUIRED = ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id", "probs", "label_ids")
OPTIONAL = ("label_provenance", "parent_case_id", "question_type", "annotation_cost", "split_id")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rows_to_probset(rows: list[dict], labelled: bool) -> ProbSet:
    seen = set()
    cols = {k: [] for k in ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id")}
    log_p, floored, label_ids, ys = [], [], [], []
    extra = {k: [] for k in OPTIONAL}
    for r in rows:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"row {r.get('sample_id')} missing {missing}")
        if r["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id {r['sample_id']}")
        seen.add(r["sample_id"])
        if labelled and r.get("y") is None:
            raise ValueError(f"labelled row {r['sample_id']} has no y")
        if not labelled and r.get("y") is not None:
            raise ValueError(f"unlabelled input {r['sample_id']} carries y; target labels must stay private")
        lp, fl = prepare_probs(r["probs"])
        if len(r["label_ids"]) != lp.size:
            raise ValueError(f"label_ids/probs length mismatch for {r['sample_id']}")
        for k in cols:
            cols[k].append(r[k] if r[k] is not None else "")
        log_p.append(lp)
        floored.append(fl)
        label_ids.append(list(r["label_ids"]))
        ys.append(r.get("y"))
        for k in OPTIONAL:
            extra[k].append(r.get(k))
    arr = {k: np.array(v, dtype=object) for k, v in cols.items()}
    return ProbSet(**arr, label_ids=label_ids, log_p=log_p,
                   y=np.array(ys, dtype=np.int64) if labelled else None,
                   floored=np.array(floored, dtype=bool),
                   extra={k: v for k, v in extra.items() if any(x is not None for x in v)})


def load_split(data_dir, split: str, where: dict | None = None) -> ProbSet:
    """`where` keeps only rows whose fields equal the given values (e.g. one model x prompt)."""
    rows = []
    with open(Path(data_dir) / SPLIT_FILES[split], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if where is None or all(r.get(k) == v for k, v in where.items()):
                    rows.append(r)
    return rows_to_probset(rows, labelled=split != "target_inputs")


def load_sources(data_dir) -> dict[str, ProbSet]:
    return {s: load_split(data_dir, s) for s in ("source_train", "source_select", "source_audit")}


def load_private_labels(data_dir) -> dict[str, dict]:
    return {r["sample_id"]: r for r in read_jsonl(Path(data_dir) / PRIVATE_FILE)}


def attach_labels(ps: ProbSet, private: dict[str, dict]) -> ProbSet:
    return ps.with_labels([private[str(s)]["y"] for s in ps.sample_id])


def probset_to_rows(ps: ProbSet, include_y: bool) -> list[dict]:
    rows = []
    for i in range(ps.n):
        r = {"sample_id": str(ps.sample_id[i]), "case_id": str(ps.case_id[i]),
             "model_id": str(ps.model_id[i]), "prompt_id": str(ps.prompt_id[i]),
             "source_id": str(ps.source_id[i]) or None, "task_id": str(ps.task_id[i]),
             "probs": np.exp(ps.log_p[i]).tolist(), "label_ids": list(ps.label_ids[i])}
        if include_y:
            r["y"] = int(ps.y[i])
        for k, v in ps.extra.items():
            if v[i] is not None:
                r[k] = v[i]
        rows.append(r)
    return rows


def write_dataset(out_dir, splits: dict[str, ProbSet], target: ProbSet, manifest: dict) -> None:
    out_dir = Path(out_dir)
    for name in ("source_train", "source_select", "source_audit"):
        write_jsonl(out_dir / SPLIT_FILES[name], probset_to_rows(splits[name], include_y=True))
    write_jsonl(out_dir / SPLIT_FILES["target_inputs"], probset_to_rows(target, include_y=False))
    private = [{"sample_id": str(target.sample_id[i]), "case_id": str(target.case_id[i]),
                "y": int(target.y[i]), "label_id": target.label_ids[i][target.y[i]]}
               for i in range(target.n)]
    write_jsonl(out_dir / PRIVATE_FILE, private)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def fingerprint(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_fingerprints(data_dir, include_private: bool = False) -> dict[str, str]:
    names = list(SPLIT_FILES.values()) + ["manifest.json"] + ([PRIVATE_FILE] if include_private else [])
    return {n: fingerprint(Path(data_dir) / n) for n in names if (Path(data_dir) / n).exists()}
