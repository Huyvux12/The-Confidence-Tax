"""Figures for the EC extension: empirical Pareto fronts (CMA-ES archive) with the SLSQP solutions per budget.

    python scripts/make_evo_figures.py      # -> paper/figures/evo_front.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
FRONT, SOLVER = "#2a78d6", "#eb6834"
SHOW = [("google/gemma-4-12B-it", "paren"), ("ibm-granite/granite-4.2-8b", "letter_colon"),
        ("mistralai/Ministral-3-8B-Instruct-2512", "paren"), ("Qwen/Qwen3.5-9B", "letter_colon")]
NAMES = {"google/gemma-4-12B-it": "Gemma-4-12B-it", "ibm-granite/granite-4.2-8b": "Granite-4.2-8B",
         "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B", "Qwen/Qwen3.5-9B": "Qwen3.5-9B"}


def main():
    recs = {tuple(json.loads(l)["unit"]): json.loads(l)
            for l in (ROOT / "runs/evo_study/front.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    fig, axes = plt.subplots(1, len(SHOW), figsize=(15, 3.6))
    for ax, u in zip(axes, SHOW):
        r = recs[u]
        f = sorted(r["front"])
        xs, ys = [b for a, b in f], [a for a, b in f]
        ax.plot(xs, ys, color=FRONT, lw=2, label="CMA-ES non-dominated archive", zorder=2)
        pts = [(row["cma_bound"] if row["eps"] > 0 else None, row["slsqp_mean"], row["eps"]) for row in r["rows"]
               if row["slsqp_valid"]]
        for _, mean, eps in pts:
            ax.axvline(eps, color=GRID, lw=0.6, zorder=0)
        ax.scatter([e for _, _, e in pts], [m for _, m, _ in pts], s=36, color=SOLVER, edgecolor="white", lw=0.6,
                   zorder=3, label="SLSQP optimum at budget eps")
        ax.set_xscale("symlog", linthresh=0.002)
        ax.set_xlim(-0.001, 0.2)
        ax.set_title(f"{NAMES[u[0]]} / {u[1]}", fontsize=10, color=INK)
        ax.set_xlabel("worst-source bound  max_w (D_w + 1.645 SE_w)", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(True, color=GRID, lw=0.5, ls=":")
        for s in ax.spines.values():
            s.set_color(MUTED)
    axes[0].set_ylabel("pooled source regret (nats)", fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "paper/figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"evo_front.{ext}", dpi=200)
    print("saved", out / "evo_front.pdf")


if __name__ == "__main__":
    main()
