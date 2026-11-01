"""Recheck (todo.md phase 5C): recompute the paper's key numbers independently of evaluator.py and
run_rbcal.py, from the raw probability files, the canonical splits, the frozen unit fits and the labels.

    python scripts/recheck_numbers.py

Writes runs/recheck/numbers.json and prints PASS/FAIL per check.
"""
from __future__ import annotations

import glob
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIMARY, MRTS, POOLED, CONST = "RBCAL_UCB_RBF_e0.01", "MRTS", "POOLED_TS", "RBCAL_UCB_CONST_e0.01"
checks = []


def check(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- data counts quoted in the paper
items = jsonl(ROOT / "data_gen/items.jsonl")
check("questions = 15,037", len(items) == 15037, len(items))
tasks = defaultdict(int)
for it in items:
    tasks[it["task_id"]] += 1
check("31 tasks, at most 500 questions each", len(tasks) == 31 and max(tasks.values()) <= 500,
      {"n_tasks": len(tasks), "max": max(tasks.values())})
raw = {Path(p).stem: jsonl(p) for p in sorted(glob.glob(str(ROOT / "data_gen/raw/probs_*.jsonl")))}
n_rows = {k: len(v) for k, v in raw.items()}
check("60,148 rows per model, 5 models", len(raw) == 5 and set(n_rows.values()) == {60148}, n_rows)
ks = sorted({len(r["probs"]) for rows in raw.values() for r in rows})
check("option count ranges 2..11", ks[0] == 2 and ks[-1] == 11, ks)
qwen = raw["probs_Qwen3_5-9B"]
med = {p: float(np.median([r["letter_mass"] for r in qwen if r["prompt_id"] == p])) for p in ("letter_colon", "paren")}
check("Qwen letter mass median 0.45 (letter_colon) / 0.999 (paren)",
      round(med["letter_colon"], 2) == 0.45 and round(med["paren"], 3) == 0.999, med)

# ---------- source files identical between dev and confirm
same = {s: sha(ROOT / f"data/real/dev/{s}.jsonl") == sha(ROOT / f"data/real/confirm_sealed/{s}.jsonl")
        for s in ("source_train", "source_select", "source_audit")}
check("source splits byte-identical in dev and confirm", all(same.values()), same)

# ---------- confirm endpoints recomputed from unit fits + labels, without evaluator.py
labels = {r["sample_id"]: r["y"] for r in jsonl(ROOT / "data/real/confirm_sealed/target_labels_private.jsonl")}
units = pickle.loads((ROOT / "runs/rbcal_confirm/unit_results.pkl").read_bytes())
regime = {}
per = defaultdict(lambda: defaultdict(list))        # method -> (unit, task) -> per-row delta
case_of = {}
for u in units:
    regime[tuple(u["unit"])] = u["regime"]
    ident = u["methods"]["IDENTITY"]["target_log_q"]
    for m in (PRIMARY, MRTS, POOLED, CONST, "IDENTITY"):
        for sid, lq, l0 in zip(u["target_sample_ids"], u["methods"][m]["target_log_q"], ident):
            y = labels[sid]
            task = sid.split("::")[1]
            per[m][(tuple(u["unit"]), task)].append(float(-lq[y] + l0[y]))
            case_of[sid] = sid.rsplit("::", 2)[0]
summ = {}
for m, d in per.items():
    vals = {k: float(np.mean(v)) for k, v in d.items()}
    summ[m] = {"macro": float(np.mean(list(vals.values()))), "harm": int(sum(v > 0.01 for v in vals.values())),
               "worst": float(max(vals.values())), "n": len(vals)}
rep = json.loads((ROOT / "runs/rbcal_confirm/report.json").read_text(encoding="utf-8"))["summaries"]["all"]
for m in (PRIMARY, MRTS, POOLED, CONST):
    ok = abs(summ[m]["macro"] - rep[m]["macro_delta"]) < 1e-9 and summ[m]["harm"] == rep[m]["harm_001"] \
        and abs(summ[m]["worst"] - rep[m]["worst_delta"]) < 1e-9
    check(f"confirm {m}: macro/harm/worst match report", ok,
          {"recomputed": summ[m], "report": {k: rep[m][k] for k in ("macro_delta", "harm_001", "worst_delta")}})
check("confirm has 30 task-units", summ[PRIMARY]["n"] == 30, summ[PRIMARY]["n"])

# ---------- E1 effect and a fresh case-bootstrap CI (different code path, different seed)
rng = np.random.default_rng(12345)
diffs = []
for key in per[PRIMARY]:
    a, b = np.array(per[PRIMARY][key]), np.array(per[MRTS][key])
    diffs.append(a - b)
point = float(np.mean([d.mean() for d in diffs]))
boots = []
# resample cases within each task-unit: rows of a case are the two answer orders (adjacent ids)
grouped = []
for key in per[PRIMARY]:
    unit, task = key
    sids = [s for u in units if tuple(u["unit"]) == unit for s in u["target_sample_ids"] if s.split("::")[1] == task]
    d = np.array(per[PRIMARY][key]) - np.array(per[MRTS][key])
    cases = np.array([case_of[s] for s in sids])
    uniq, inv = np.unique(cases, return_inverse=True)
    grouped.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
for _ in range(2000):
    vals = []
    for s, c in grouped:
        idx = rng.integers(0, len(s), len(s))
        vals.append(s[idx].sum() / c[idx].sum())
    boots.append(np.mean(vals))
lo, hi = np.percentile(boots, [2.5, 97.5])
e1 = json.loads((ROOT / "runs/rbcal_confirm/report.json").read_text(encoding="utf-8"))["endpoints"]["all"]["E1_vs_MRTS"]["paired"]
check("E1 effect and CI reproduced (CI within 0.002 of report)",
      abs(point - e1["mean_diff"]) < 1e-9 and abs(lo - e1["ci_low"]) < 2e-3 and abs(hi - e1["ci_high"]) < 2e-3,
      {"recomputed": [point, float(lo), float(hi)], "report": [e1["mean_diff"], e1["ci_low"], e1["ci_high"]]})

# ---------- hand-typed numbers in the paper
dev_units = [pickle.loads(Path(p).read_bytes()) for p in sorted(glob.glob(str(ROOT / "runs/rbcal_dev/units/*.pkl")))]
mins = sorted(u["seconds"] / 60 for u in dev_units)
check("full selection 42-286 minutes per unit", 40 <= mins[0] <= 44 and 284 <= mins[-1] <= 287,
      {"min_minutes": round(mins[0], 1), "max_minutes": round(mins[-1], 1)})
reuse = [pickle.loads(Path(p).read_bytes())["seconds"] / 60
         for p in glob.glob(str(ROOT / "runs/rbcal_confirm/units/*.pkl"))]
check("refit with fixed configuration 6-15 minutes", 6 <= min(reuse) and max(reuse) <= 15,
      [round(min(reuse), 1), round(max(reuse), 1)])
loto = json.loads((ROOT / "runs/loto/report.json").read_text(encoding="utf-8"))["summaries"]["conflict"]
h = [loto[m]["harm_001"] for m in ("POOLED_TS", "THERMO_P", "ATS_P")]
check("conflict LOTO: pooled methods harm 13-26 of 75", min(h) == 13 and max(h) == 26 and loto[PRIMARY]["n"] == 75,
      {"harms": h, "n": loto[PRIMARY]["n"]})
check("3 conflict units", sum(v == "conflict" for v in regime.values()) == 3,
      sorted("|".join(k) for k, v in regime.items() if v == "conflict"))
from calibration_extensions.rb_calibration import RBF_MENU  # noqa: E402
check("37 field configurations", len(RBF_MENU) == 37, len(RBF_MENU))
check("prediction hash file matches confirm report",
      (ROOT / "runs/rbcal_confirm/prediction_hash.txt").read_text().strip()
      == json.loads((ROOT / "runs/rbcal_confirm/report.json").read_text(encoding="utf-8"))["prediction_sha256"], "")

out = ROOT / "runs/recheck"
out.mkdir(parents=True, exist_ok=True)
(out / "numbers.json").write_text(json.dumps(checks, indent=1, default=str), encoding="utf-8")
print(f"\n{sum(c['pass'] for c in checks)}/{len(checks)} checks pass")
