"""Next-token answer-letter probabilities for multiple-choice items (runs on a GPU VM).

For every item x answer ordering x prompt template, the chat prompt ends with an assistant
prefill and we read the next-token logits of the K option letters (both the bare and the
space-prefixed letter token when each is a single token, merged by logsumexp). Probabilities
are renormalised over the K letters; `letter_mass` keeps the share of the full vocabulary
that fell on those letters, as a format-compliance diagnostic.

    python infer_mcq.py --items items.jsonl --model Qwen/Qwen3.5-9B --out probs_qwen.jsonl
Resumable: rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import transformers
from transformers import AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PROMPTS = {
    "letter_colon": {
        "user": lambda q, opts: (q + "\n\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
                                 + "\n\nAnswer with the letter of the correct option."),
        "prefill": "Answer:",
    },
    "paren": {
        "user": lambda q, opts: ("Choose the best answer to the following question.\n\nQuestion: " + q
                                 + "\nOptions:\n" + "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(opts))
                                 + "\n\nRespond only with the letter of the correct option in parentheses."),
        "prefill": "The answer is (",
    },
}


def load_model(name: str, revision: str | None = None):
    kw = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False, revision=revision)
    errors = []
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            return getattr(transformers, cls).from_pretrained(name, **kw).eval(), cls
        except Exception as e:  # architecture not registered for this auto class
            errors.append(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("could not load model\n" + "\n".join(errors))


def letter_token_ids(tok, K: int) -> list[list[int]]:
    out = []
    for L in LETTERS[:K]:
        ids = set()
        for s in (L, " " + L):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"letter {L} has no single-token form")
        out.append(sorted(ids))
    return out


def render(tok, user: str, prefill: str) -> str:
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + prefill


def build_rows(items, prompt_ids, model_id):
    for it in items:
        for oi, order in enumerate(it["orders"]):
            opts = [it["options"][j] for j in order]
            y = order.index(it["answer"])
            for pid in prompt_ids:
                yield {"sample_id": f"{it['item_id']}::o{oi}::{pid}", "case_id": it["item_id"],
                       "model_id": model_id, "prompt_id": pid,
                       "source_id": it["task_id"] if it["role"] == "source" else None,
                       "task_id": it["task_id"], "label_ids": [str(j) for j in order], "y": y,
                       "role": it["role"], "order_index": oi,
                       "_user": PROMPTS[pid]["user"](it["question"], opts), "_prefill": PROMPTS[pid]["prefill"]}


@torch.inference_mode()
def score_batch(model, tok, texts, Ks, letter_ids):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:
        out = model(**enc)
    logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    res = []
    for b, K in enumerate(Ks):
        lp = torch.stack([torch.logsumexp(logp[b, ids], dim=0) for ids in letter_ids[:K]])
        mass = float(torch.logsumexp(lp, dim=0).exp())
        res.append((torch.softmax(lp, dim=0).double().cpu().tolist(), mass))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="letter_colon,paren")
    ap.add_argument("--limit", type=int, default=0, help="items per task (smoke tests)")
    ap.add_argument("--batch-tokens", type=int, default=24000)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--revision", default=None, help="model commit SHA (data_gen/revisions.json); default: latest")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    if args.limit:
        seen = {}
        items = [it for it in items if seen.setdefault(it["task_id"], []).append(1) or len(seen[it["task_id"]]) <= args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["sample_id"] for l in open(args.out, encoding="utf-8")}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, cls = load_model(args.model, args.revision)
    maxK = max(len(it["options"]) for it in items)
    letter_ids = letter_token_ids(tok, maxK)
    print(f"loaded {args.model} via {cls} in {time.time() - t0:.0f}s; transformers {transformers.__version__}; "
          f"letter ids {letter_ids[:4]}", flush=True)

    rows = [r for r in build_rows(items, args.prompts.split(","), args.model) if r["sample_id"] not in done]
    for r in rows:
        r["_text"] = render(tok, r["_user"], r["_prefill"])
        r["_len"] = len(tok(r["_text"], add_special_tokens=False)["input_ids"])
    rows.sort(key=lambda r: r["_len"])
    print(f"{len(rows)} rows to score ({len(done)} already done); first prompt:\n{rows[0]['_text'] if rows else ''}", flush=True)

    n_done, t1 = 0, time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and j - i < args.max_batch and (j - i + 1) * rows[j]["_len"] <= args.batch_tokens:
                j += 1
            j = max(j, i + 1)
            batch = rows[i:j]
            scored = score_batch(model, tok, [r["_text"] for r in batch], [len(r["label_ids"]) for r in batch], letter_ids)
            for r, (probs, mass) in zip(batch, scored):
                rec = {k: v for k, v in r.items() if not k.startswith("_")}
                rec |= {"probs": probs, "letter_mass": mass, "n_tokens": r["_len"]}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done += len(batch)
            i = j
            if n_done % 2000 < len(batch):
                rate = n_done / (time.time() - t1)
                print(f"{n_done}/{len(rows)} rows, {rate:.1f} rows/s, eta {(len(rows) - n_done) / rate / 60:.1f} min", flush=True)
    print(f"JOB DONE {args.model} rows={n_done} seconds={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
