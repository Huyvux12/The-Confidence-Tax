"""Item pool of the fourth sealed roster (PROTOCOL_LOG 2026-11-06, pre-registered before this file existed).

BIG-bench tasks (tasksource/bigbench, train+validation pooled) and AGIEval sat_en; role `target_confirm4`.
Formatting rules (fixed before any output): drop embedded "choice:" lines, a leading "Q:", a trailing "A:" line and
a short trailing cue line ending with ":"; logical_deduction gets the question "Which statement is true?".

    python data_gen/build_items_confirm4.py --out data_gen/items_confirm4.jsonl
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
from build_items_confirm3 import agieval_question  # noqa: E402

ROLE = "target_confirm4"
BB = ["date_understanding", "disambiguation_qa", "logical_deduction", "temporal_sequences", "strategyqa",
      "implicatures", "tracking_shuffled_objects", "epistemic_reasoning", "navigate"]


def clean(text: str, task: str) -> str:
    lines = [l for l in text.strip().split("\n") if not re.match(r"^\s*choice:", l)]
    while lines and lines[-1].strip() in ("A:", ""):
        lines.pop()
    if lines and lines[-1].strip().endswith(":") and len(lines[-1].split()) <= 4:
        lines.pop()
    t = "\n".join(lines).strip()
    t = re.sub(r"^Q:\s*", "", t)
    if task == "logical_deduction":
        t += "\nWhich statement is true?"
    return t


def load_bigbench(cfg):
    for split in ("train", "validation"):
        for r in load_dataset("tasksource/bigbench", cfg, split=split):
            opts, sc = as_obj(r["multiple_choice_targets"]), as_obj(r["multiple_choice_scores"])
            if sum(sc) != 1:
                continue
            yield item(cfg, f"{split}{r['idx']}", clean(r["inputs"], cfg), opts, sc.index(1), "bigbench")


def load_sat_en():
    for i, r in enumerate(load_dataset("hails/agieval-sat-en", split="test")):
        opts = [re.sub(r"^\([A-E]\)\s*", "", o) for o in as_obj(r["choices"])]
        gold = as_obj(r["gold"])
        if len(gold) == 1:
            yield item("sat_en", i, agieval_question(r["query"]), opts, int(gold[0]), "agieval")


LOADERS = [(lambda c=c: load_bigbench(c)) for c in BB] + [load_sat_en]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_gen/items_confirm4.jsonl")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--n-orders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261108)
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
        print(f"{t:<28} n={s['n']:<4} pool={s['pool']:<6} K={s['K']}")
    print("dropped invalid:", dropped, "| items", len(out), "rows", sum(len(x["orders"]) for x in out))


if __name__ == "__main__":
    main()
