"""Item pool of the third sealed roster (PROTOCOL_LOG 2026-11-06, pre-registered before this file existed).

Same item format, cap and ordering scheme as build_items.py; role `target_confirm3`.

    python data_gen/build_items_confirm3.py --out data_gen/items_confirm3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_items import item  # noqa: E402
from build_items_confirm2 import as_obj  # noqa: E402

ROLE = "target_confirm3"


def load_quartz():
    for r in load_dataset("allenai/quartz", split="validation"):
        ch = as_obj(r["choices"])
        yield item("quartz", r["id"], f"{r['para']}\n\nQuestion: {r['question']}", ch["text"],
                   ch["label"].index(r["answerKey"]), "quartz")


def load_scienceqa():
    for i, r in enumerate(load_dataset("tasksource/ScienceQA_text_only", split="test")):
        q = (f"{r['hint']}\n\n" if r["hint"] else "") + r["question"]
        yield item("scienceqa_text", i, q, as_obj(r["choices"]), int(r["answer"]), "scienceqa")


def load_pubmedqa():
    for r in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
        ctx = "\n".join(as_obj(r["context"])["contexts"])
        yield item("pubmedqa", r["pubid"], f"{ctx}\n\nQuestion: {r['question']}", ["yes", "no", "maybe"],
                   ["yes", "no", "maybe"].index(r["final_decision"]), "pubmedqa")


def load_copa():
    for r in load_dataset("pkavumba/balanced-copa", split="test"):
        ask = "What was the cause of this?" if r["question"] == "cause" else "What happened as a result?"
        yield item("copa", r["id"], f"{r['premise']}\n{ask}", [r["choice1"], r["choice2"]], int(r["label"]), "copa")


def agieval_question(query: str) -> str:
    q = query.split("Answer Choices:")[0] if "Answer Choices:" in query else query.split("(A)")[0]
    q = q.strip()
    q = q[2:].strip() if q.startswith("Q:") else q
    return re.sub(r"\s*Q:\s*", "\n\nQuestion: ", q)


def load_agieval(name, task):
    for i, r in enumerate(load_dataset(f"hails/agieval-{name}", split="test")):
        opts = [re.sub(r"^\([A-E]\)\s*", "", o) for o in as_obj(r["choices"])]
        gold = as_obj(r["gold"])
        if len(gold) == 1:
            yield item(task, i, agieval_question(r["query"]), opts, int(gold[0]), "agieval")


LOADERS = [load_quartz, load_scienceqa, load_pubmedqa, load_copa,
           lambda: load_agieval("lsat-lr", "lsat_lr"), lambda: load_agieval("lsat-rc", "lsat_rc")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_gen/items_confirm3.jsonl")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--n-orders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    by_task: dict[str, list] = {}
    dropped = 0
    for loader in LOADERS:
        for it in loader():
            if it is None:
                dropped += 1
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
    summary = {t: {"n": sum(1 for x in out if x["task_id"] == t), "pool": len(by_task[t]),
                   "K": sorted({len(x["options"]) for x in out if x["task_id"] == t})} for t in sorted(by_task)}
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(
        {"seed": args.seed, "cap": args.cap, "n_orders": args.n_orders, "dropped_invalid": dropped, "role": ROLE,
         "tasks": summary}, indent=1), encoding="utf-8")
    for t, s in summary.items():
        print(f"{t:<16} n={s['n']:<4} pool={s['pool']:<5} K={s['K']}")
    print("dropped invalid:", dropped, "| items", len(out), "rows", sum(len(x["orders"]) for x in out))


if __name__ == "__main__":
    main()
