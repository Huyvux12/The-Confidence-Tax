"""Phase 10: derived datasets with label-free per-row fields (no labels are read).

For an input split directory D writes
    D_avg : probs replaced by the mean of the two answer orders' semantic distributions, re-expressed in each
            row's displayed order (both rows of a question carry the same semantic distribution)
    D_aug : original probs (single order)
Both get optional fields
    order_tv_raw : total variation between the two original answer orders (semantic options)
    peer_top     : mean over the 4 other models (order-averaged, same prompt and question) of the probability
                   they give to this row's top option under the row's own base (raw for _aug, averaged for _avg)
    peer_top1    : the same with one pre-chosen peer: Olmo-3-7B-Instruct, or Ministral-3-8B for Olmo itself
Labels and manifests are copied unchanged.

    python data_gen/make_aug.py data/real/dev [data/real/confirm3_sealed ...]
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FILES = ("source_train.jsonl", "source_select.jsonl", "source_audit.jsonl", "target_inputs.jsonl")
PEER1 = {"default": "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Instruct": "mistralai/Ministral-3-8B-Instruct-2512"}


def semantic(probs, label_ids):
    out = np.zeros(len(probs))
    for pos, lab in enumerate(label_ids):
        out[int(lab)] = probs[pos]
    return out


def order_of(sample_id):
    return sample_id.split("::")[-2]


def process(src: Path):
    rows = {}
    for f in FILES:
        if (src / f).exists():
            rows[f] = [json.loads(l) for l in open(src / f, encoding="utf-8") if l.strip()]
    sem = {}                                   # (model, prompt, case, order) -> semantic distribution
    for rs in rows.values():
        for r in rs:
            sem[(r["model_id"], r["prompt_id"], r["case_id"], order_of(r["sample_id"]))] = semantic(r["probs"], r["label_ids"])
    by_q = defaultdict(list)
    for (m, p, c, o), s in sem.items():
        by_q[(m, p, c)].append(s)
    avg = {k: np.mean(v, axis=0) for k, v in by_q.items()}
    models = sorted({k[0] for k in avg})
    unpaired = sum(len(v) < 2 for v in by_q.values())
    for suffix in ("_avg", "_aug"):
        dst = src.parent / (src.name + suffix)
        dst.mkdir(parents=True, exist_ok=True)
        for f, rs in rows.items():
            with open(dst / f, "w", encoding="utf-8") as out:
                for r in rs:
                    m, p, c, o = r["model_id"], r["prompt_id"], r["case_id"], order_of(r["sample_id"])
                    own = sem[(m, p, c, o)]
                    a = avg[(m, p, c)]
                    others = [x for x in by_q[(m, p, c)]]
                    tv = float(np.mean([0.5 * np.abs(own - x).sum() for x in others if x is not own])) if len(others) > 1 else 0.0
                    base = a if suffix == "_avg" else own
                    top = int(np.argmax(base))
                    peers = [avg[(q, p, c)][top] for q in models if q != m and (q, p, c) in avg]
                    p1 = PEER1.get(m, PEER1["default"])
                    r2 = dict(r)
                    if suffix == "_avg":
                        r2["probs"] = [float(a[int(lab)]) for lab in r["label_ids"]]
                    r2["order_tv_raw"] = tv
                    r2["peer_top"] = float(np.mean(peers)) if peers else None
                    r2["peer_top1"] = float(avg[(p1, p, c)][top]) if (p1, p, c) in avg else None
                    out.write(json.dumps(r2) + "\n")
        for extra in ("target_labels_private.jsonl", "manifest.json"):
            if (src / extra).exists():
                shutil.copyfile(src / extra, dst / extra)
        man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        man["derived"] = {"from": str(src), "kind": suffix[1:], "fields": ["order_tv_raw", "peer_top", "peer_top1"],
                          "peer1": PEER1}
        (dst / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(src, {f: len(v) for f, v in rows.items()}, "unpaired questions:", unpaired)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        process(Path(d))
