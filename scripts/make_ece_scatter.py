"""Calibrated vs uncalibrated ECE per held-out unit, one row of five model panels per method.

    python scripts/make_ece_scatter.py --roster loto        # 25 held-out source tasks x 2 prompts per model
    python scripts/make_ece_scatter.py --roster confirm2    # 6 confirm2 tasks x 2 prompts per model

Each dot is one task x prompt for that model; below the diagonal = lower ECE than the uncalibrated model.
Crosses mark units whose NLL got worse by more than 0.01 nats (the paper's harm threshold).
Writes paper/figures/ece_scatter_<roster>_<method>.{pdf,png} and ece_scatter_<roster>_grid.{pdf,png}.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_study  # noqa: E402

MODELS = [("Qwen/Qwen3.5-9B", "Qwen3.5-9B"), ("google/gemma-4-12B-it", "Gemma-4-12B-it"),
          ("ibm-granite/granite-4.2-8b", "Granite-4.2-8B"), ("mistralai/Ministral-3-8B-Instruct-2512", "Ministral-3-8B"),
          ("allenai/Olmo-3-7B-Instruct", "Olmo-3-7B-Instruct")]
METHODS = [("POOLED_TS", "Pooled TS"), ("THERMO_P", "Thermometer-P"), ("ATS_P", "ATS-P"),
           ("MRTS", "Minimax TS (MRTS)"), ("RBCAL_UCB_RBF_e0.01", "RB-Cal v1"), ("V3", "RB-Cal v3"),
           ("V4B", "RB-Cal v4b"), ("SWITCH", "RB-Switch"), ("EVO", "Evo-RB")]
HARM = 0.01
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
OK, BAD = "#2a78d6", "#eb6834"          # reference palette slots 1 and 2 (validated all-pairs)


def load_loto():
    """rows: (model, prompt, task) -> {method: (ece, delta)} incl. IDENTITY."""
    rows = {}
    for p in glob.glob(str(ROOT / "runs/loto/jobs/*.json")):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        rows[(j["unit"][0], j["unit"][1], j["task"])] = {m: (v["ece_15"], v["delta_nll"]) for m, v in j["methods"].items()}
    for r in gate_study.load_records("loto"):
        k = (r["unit"][0], r["unit"][1], r["task"])
        src = "C1" if gate_study.accept(r["select"].get("C1"), "G-sig") else "IDENTITY"
        m = r["target"][src][r["task"]]
        rows[k]["V3"] = (m["ece_15"], m["delta_nll"])
    v4 = ROOT / "runs/v4_study/loto.jsonl"
    if v4.exists():
        for l in v4.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["variant"] != "v4b":
                continue
            src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
            m = r["target"][src][r["task"]]
            rows[(r["unit"][0], r["unit"][1], r["task"])]["V4B"] = (m["ece_15"], m["delta_nll"])
    evo = ROOT / "runs/evo_study/evo_loto.jsonl"
    if evo.exists():
        for l in evo.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
            m = r["target"][src][r["task"]]
            rows[(r["unit"][0], r["unit"][1], r["task"])].setdefault("EVO", (m["ece_15"], m["delta_nll"]))
    return rows, "LOTO: each dot = one held-out task x prompt  |  25 tasks x 2 prompts per model"


def load_confirm(phase):
    rep = json.loads((ROOT / f"runs/rbcal_{phase}/{phase}_report.json").read_text(encoding="utf-8"))["units"]
    rows = {}
    for m, rs in rep.items():
        key = {"V3_UCB_RBF_GSIG": "V3", "V4B_ORDER_CONSISTENCY": "V4B", "RB_SWITCH": "SWITCH"}.get(m, m)
        for x in rs:
            rows.setdefault((x["model_id"], x["prompt_id"], x["task_id"]), {})[key] = (x["ece_15"], x["delta_nll"])
    return rows, f"{phase.capitalize()}: each dot = one unseen task x prompt  |  {len({k[2] for k in rows})} tasks x 2 prompts per model"


def panel(ax, pts, lim, title, show_y):
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color=MUTED, zorder=1)
    ok = [(u, c) for u, c, d in pts if d <= HARM]
    bad = [(u, c) for u, c, d in pts if d > HARM]
    if ok:
        ax.scatter(*zip(*ok), s=26, marker="o", color=OK, edgecolor="white", linewidth=0.6, zorder=3)
    if bad:
        ax.scatter(*zip(*bad), s=46, marker="X", color=BAD, edgecolor="white", linewidth=0.6, zorder=4)
    dec = sum(c < u for u, c, _ in pts)
    ax.text(0.5, 0.97, f"ECE decreases {dec}/{len(pts)}\nNLL harm (>0.01) {len(bad)}/{len(pts)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=MUTED, lw=0.6))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel("Uncalibrated ECE", fontsize=9, color=INK)
    if show_y:
        ax.set_ylabel("Calibrated ECE", fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.5, ls=":")
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.8)


def method_points(rows, method, model):
    return [(v["IDENTITY"][0], v[method][0], v[method][1]) for (mo, _, _), v in sorted(rows.items())
            if mo == model and method in v]


def legend(fig, y):
    h = [plt.Line2D([], [], ls="", marker="o", color=OK, markeredgecolor="white", markersize=6,
                    label=f"NLL regret <= {HARM}"),
         plt.Line2D([], [], ls="", marker="X", color=BAD, markeredgecolor="white", markersize=7,
                    label=f"NLL regret > {HARM} (harmed unit)"),
         plt.Line2D([], [], ls="--", color=MUTED, label="no change (y = x)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", choices=["loto", "confirm2", "confirm3", "confirm4"], default="loto")
    ap.add_argument("--methods", default="", help="comma list of method keys: only a compact grid of these rows")
    ap.add_argument("--stem", default="", help="output path stem for the compact grid (pdf+png)")
    args = ap.parse_args()
    rows, caption = load_loto() if args.roster == "loto" else load_confirm(args.roster)
    methods = [(k, n) for k, n in METHODS if any(k in v for v in rows.values())]
    if args.methods:
        keep = args.methods.split(",")
        methods = [(k, n) for k in keep for kk, n in methods if kk == k]
        top = max(max(v["IDENTITY"][0], *(v[m][0] for m, _ in methods if m in v)) for v in rows.values())
        lim = 0.6 if top <= 0.6 else round(top + 0.05, 1)
        fig, axes = plt.subplots(len(methods), 5, figsize=(14, 2.9 * len(methods)))
        for r, (key, name) in enumerate(methods):
            for c, (mid, short) in enumerate(MODELS):
                panel(axes[r, c], method_points(rows, key, mid), lim, short if r == 0 else "", c == 0)
                if c == 0:
                    axes[r, c].set_ylabel(name + "\nCalibrated ECE", fontsize=9.5, color=INK)
                if r < len(methods) - 1:
                    axes[r, c].set_xlabel("")
        legend(fig, 0.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(f"{args.stem}.{ext}", dpi=170)
        plt.close(fig)
        return
    top = max(max(v["IDENTITY"][0], *(v[m][0] for m, _ in methods if m in v)) for v in rows.values())
    lim = 0.6 if top <= 0.6 else round(top + 0.05, 1)
    out = ROOT / "paper/figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ece_scatter_{args.roster}"
    for key, name in methods:
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.6))
        for i, (ax, (mid, short)) in enumerate(zip(axes, MODELS)):
            panel(ax, method_points(rows, key, mid), lim, short, i == 0)
        fig.suptitle(name, fontsize=14, fontweight="bold", color=INK, y=0.99)
        fig.text(0.5, 0.008, caption + "  |  below diagonal = lower ECE", ha="center", fontsize=8.5, color=MUTED)
        legend(fig, 0.05)
        fig.tight_layout(rect=(0, 0.13, 1, 0.95))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{stem}_{key.lower()}.{ext}", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(len(methods), 5, figsize=(15, 3.1 * len(methods)))
    for r, (key, name) in enumerate(methods):
        for c, (mid, short) in enumerate(MODELS):
            panel(axes[r, c], method_points(rows, key, mid), lim, short if r == 0 else "", c == 0)
            if c == 0:
                axes[r, c].set_ylabel(f"{name}\nCalibrated ECE", fontsize=9.5, color=INK)
            if r < len(methods) - 1:
                axes[r, c].set_xlabel("")
    fig.text(0.5, 0.004, caption + "  |  below diagonal = lower ECE", ha="center", fontsize=9, color=MUTED)
    legend(fig, 0.014)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_grid.{ext}", dpi=170)
    plt.close(fig)
    for key, name in methods:
        pts = [p for mid, _ in MODELS for p in method_points(rows, key, mid)]
        print(f"{name:20s} ECE decreases {sum(c < u for u, c, _ in pts)}/{len(pts)}  "
              f"NLL harm {sum(d > HARM for _, _, d in pts)}/{len(pts)}")


if __name__ == "__main__":
    main()
