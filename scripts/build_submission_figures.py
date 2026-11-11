#!/usr/bin/env python3
"""Regenerate the three main Evo-RB result plots from frozen, committed records.

These plots are data-equivalent replacements, NOT byte/pixel identical to the
author's original manuscript artwork. No model inference or optimization.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_submission"
RUNS = ROOT / "runs"
THRESHOLD = 0.01
METHODS = [
    ("POOLED_TS", "Pooled TS", "#d9a300", "o"),
    ("THERMO_P", "Thermometer-P", "#d37335", "s"),
    ("ATS_P", "ATS-P", "#52739b", "^"),
    ("MRTS", "Minimax TS", "#3b8c57", "p"),
    ("EVO_RB", "Evo-RB", "#563978", "*"),
]
ROSTERS = [("confirm1", "C1"), ("confirm2", "C2"),
           ("confirm3", "C3"), ("confirm4", "Hard"),
           ("LOTO", "LOTO")]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def legend_handles():
    return [Line2D([0], [0], marker=mk, linestyle="None", color=col,
                   markersize=7 if key == "EVO_RB" else 5, label=label)
            for key, label, col, mk in METHODS]


def style_axes(ax):
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def front(ax, rows, n, ylabel="Mean NLL gain", ece=False):
    for key, label, color, marker in METHODS:
        r = rows[key]
        x = 100 * r["harm"] / n
        y = r["ece"] if ece else -r["delta"]
        ax.scatter(x, y, marker=marker, color=color,
                   s=92 if marker == "*" else 37, zorder=5, linewidth=0.35,
                   edgecolors="#333333")
    style_axes(ax)
    ax.set_xlabel("Harmed units (%)")
    ax.set_ylabel(ylabel)


def figure1(payload):
    tot = payload["aggregate"]
    rows = {m: {"harm": tot[m]["harms"], "ece": tot[m]["mean_ece"],
                "delta": tot[m]["mean_delta"]} for m, *_ in METHODS}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    front(axes[0], rows, 500, "Mean NLL gain")
    axes[0].set_title("Gain–harm frontier", fontsize=10)
    front(axes[1], rows, 500, "Mean ECE", ece=True)
    axes[1].set_title("Calibration–harm frontier", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "figure1_gain_calibration.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def figure2(payload):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.3))
    for ax, (roster, short) in zip(axes.flat, ROSTERS + [("all", "All 500 units")]):
        if roster == "all":
            tot = payload["aggregate"]
            rows = {m: {"harm": tot[m]["harms"], "delta": tot[m]["mean_delta"]}
                    for m, *_ in METHODS}
            n = 500
        else:
            rows = payload["per_roster"][roster]
            n = rows["EVO_RB"]["n"]
        front(ax, rows, n, "Mean NLL gain")
        ax.set_title(short, fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.025), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "figure2_roster_frontiers.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def rkey(r):
    return r["unit"][0], r["unit"][1], r["task"]


def gate_passes(stat):
    return bool(stat) and all(float(v["delta"]) - 1.645 * float(v["se"]) <= THRESHOLD + 1e-8
                              for v in stat.values())


def figure3():
    b = {rkey(x): x for x in load_jsonl(RUNS / "v4_study/base_loto_aug.jsonl")}
    e = {rkey(x): x for x in load_jsonl(RUNS / "evo_study/evo_loto.jsonl")}
    assert len(b) == 250 and set(b) == set(e), "Paired LOTO input mismatch"
    colors = {"rescued": "#32c63a", "both harmed": "#e5a42b", "newly harmed": "#b8374b"}
    pts = {key: [] for key in colors}
    for k, baseline in b.items():
        evolved = e[k]
        ed = (float(evolved["target"]["V"][evolved["task"]]["delta_nll"])
              if "V" in evolved["target"] and gate_passes(evolved["select"]["V"]) else 0.0)
        for method in ("POOLED_TS", "THERMO_P", "ATS_P", "MRTS"):
            bd = float(baseline["target"][method][baseline["task"]]["delta_nll"])
            if bd > THRESHOLD >= ed:
                label = "rescued"
            elif bd > THRESHOLD and ed > THRESHOLD:
                label = "both harmed"
            elif bd <= THRESHOLD < ed:
                label = "newly harmed"
            else:
                continue
            pts[label].append((bd, bd - ed))
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    for label, points in pts.items():
        if not points:
            continue
        xx, yy = zip(*points)
        ax.scatter(xx, yy, c=colors[label], s=60, alpha=0.92,
                   edgecolors="#444444", linewidths=0.5, label=label.title(), zorder=4)
    ax.axhline(0, c="#444444", lw=0.9)
    ax.axvline(THRESHOLD, c="#999999", lw=0.9, ls="--")
    xx = [-0.01, 0.17]
    ax.plot(xx, [x - THRESHOLD for x in xx], ls="--", lw=1, color="#999999")
    ax.set_xlabel("Baseline ΔNLL (nats)")
    ax.set_ylabel("Baseline ΔNLL − Evo-RB ΔNLL (nats)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14))
    ax.set_xlim(-0.008, 0.155)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_rescue_cropped.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("Plotted", {label: len(values) for label, values in pts.items()},
          "points over four baseline comparisons (overlap is expected)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = load_json(RUNS / "evo_study/aggregate_500.json")
    figure1(payload)
    figure2(payload)
    figure3()
    print("Three figures written under", OUT)
    print("These are data-derived replacements; original manuscript art is not byte-identical.")


if __name__ == "__main__":
    main()
