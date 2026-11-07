"""Item pool of the second sealed confirmation roster (PROTOCOL_LOG 2026-11-06, pre-registered).

Same item format, cap and ordering scheme as build_items.py; every task has role `target_confirm2`.

    python data_gen/build_items_confirm2.py --out data_gen/items_confirm2.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_items import item  # noqa: E402

PARQUET = "refs/convert/parquet"
ROLE = "target_confirm2"


def as_obj(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_qasc():
    for r in load_dataset("allenai/qasc", split="validation"):
        ch = as_obj(r["choices"])
        if r["answerKey"] in ch["label"]:
            yield item("qasc", r["id"], r["question"], ch["text"], ch["label"].index(r["answerKey"]), "qasc")


def load_race_high():
    for r in load_dataset("ehovy/race", "high", split="test"):
        q = f"{r['article']}\n\nQuestion: {r['question']}"
        yield item("race_high", f"{r['example_id']}::{r['question'][:40]}", q, as_obj(r["options"]),
                   "ABCD".index(r["answer"]), "race")


def load_social_iqa():
    for i, r in enumerate(load_dataset("allenai/social_i_qa", split="validation", revision=PARQUET)):
        yield item("social_iqa", i, f"{r['context']}\nQuestion: {r['question']}",
                   [r["answerA"], r["answerB"], r["answerC"]], int(r["label"]) - 1, "social_iqa")


def load_cosmos_qa():
    for r in load_dataset("allenai/cosmos_qa", split="validation", revision=PARQUET):
        yield item("cosmos_qa", r["id"], f"{r['context']}\nQuestion: {r['question']}",
                   [r[f"answer{k}"] for k in range(4)], int(r["label"]), "cosmos_qa")


def load_medqa():
    for i, r in enumerate(load_dataset("GBaker/MedQA-USMLE-4-options", split="test")):
        opts = as_obj(r["options"])
        keys = sorted(opts)
        yield item("medqa_usmle", i, r["question"], [opts[k] for k in keys], keys.index(r["answer_idx"]), "medqa")


def load_belebele():
    for r in load_dataset("facebook/belebele", "eng_Latn", split="test"):
        q = f"{r['flores_passage']}\n\nQuestion: {r['question']}"
        yield item("belebele_eng", f"{r['link']}::{r['question_number']}", q,
                   [r[f"mc_answer{k}"] for k in range(1, 5)], int(r["correct_answer_num"]) - 1, "belebele")


LOADERS = [load_qasc, load_race_high, load_social_iqa, load_cosmos_qa, load_medqa, load_belebele]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_gen/items_confirm2.jsonl")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--n-orders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    by_task: dict[str, list] = {}
    dropped: dict[str, int] = {}
    for loader in LOADERS:
        for it in loader():
            if it is None:
                dropped[loader.__name__] = dropped.get(loader.__name__, 0) + 1
                continue
            by_task.setdefault(it["task_id"], []).append(it)
    out = []
    for task in sorted(by_task):
        pool = sorted({x["item_id"]: x for x in by_task[task]}.values(), key=lambda x: x["item_id"])
        pick = rng.sample(pool, min(args.cap, len(pool)))
        for it in sorted(pick, key=lambda x: x["item_id"]):
            K = len(it["options"])
            orders = [list(range(K))]
            while len(orders) < min(args.n_orders, 2 if K == 2 else args.n_orders):
                p = list(range(K))
                rng.shuffle(p)
                if p not in orders:
                    orders.append(p)
            out.append(it | {"orders": orders, "role": ROLE})
    with open(args.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    summary = {t: {"n": sum(1 for x in out if x["task_id"] == t), "pool": len(by_task[t]), "role": ROLE,
                   "K": sorted({len(x["options"]) for x in out if x["task_id"] == t})} for t in sorted(by_task)}
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(
        {"seed": args.seed, "cap": args.cap, "n_orders": args.n_orders, "dropped_invalid": dropped,
         "tasks": summary}, indent=1), encoding="utf-8")
    for t, s in summary.items():
        print(f"{t:<16} n={s['n']:<4} pool={s['pool']:<5} K={s['K']}")
    print("dropped invalid:", dropped, "| items", len(out), "rows", sum(len(x["orders"]) for x in out))


if __name__ == "__main__":
    main()
