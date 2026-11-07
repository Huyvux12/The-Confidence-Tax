"""Numbers, tables and figures of the LNCS paper (paper_lncs/), generated from runs/ only.

    python scripts/make_lncs_assets.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import base_study  # noqa: E402
import gate_study  # noqa: E402
import v4_study  # noqa: E402

OUT = ROOT / "paper_lncs"
GEN, FIG = OUT / "generated", OUT / "figures"
METHODS = [("POOLED_TS", "Pooled TS"), ("THERMO_P", "Thermometer-P"), ("ATS_P", "ATS-P"), ("MRTS", "Minimax TS"),
           ("V4B", "RB-Cal"), ("EVO_RB", "Evo-RB")]
ROSTERS = [("confirm1", "Cone"), ("confirm2", "Ctwo"), ("confirm3", "Cthree"), ("confirm4", "Cfour"), ("LOTO", "Loto")]
TITLES = {"confirm1": "C1", "confirm2": "C2", "confirm3": "C3", "confirm4": "C4", "LOTO": "LOTO"}
macros = {}


def m(name, value, fmt="{:+.3f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def write_macros():
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def unit_metrics():
    """roster -> method -> {(model, prompt, task): metrics} from the result files (labels already applied there)."""
    keep = ("delta_nll", "ece_15", "brier", "nll")

    def from_rows(rows):
        return {(x["model_id"], x["prompt_id"], x["task_id"]): {k: x[k] for k in keep} for x in rows}

    def from_records(path, variant=None):
        out = {}
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if variant and r.get("variant") != variant:
                continue
            src = "V" if gate_study.accept(r["select"]["V"], "G-sig") else "IDENTITY"
            for t, mt in r["target"][src].items():
                out.setdefault((r["unit"][0], r["unit"][1], t), {k: mt[k] for k in keep})   # first record wins
        return out

    R = {}
    c1 = json.loads((ROOT / "runs/extra_baselines/metrics_confirm.json").read_text(encoding="utf-8"))
    R["confirm1"] = {mm: from_rows([x for x in c1 if x["method_id"] == mm and not x["task_id"].startswith("__")])
                     for mm in ("IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS")}
    for ph in ("confirm2", "confirm3", "confirm4"):
        u = json.loads((ROOT / f"runs/rbcal_{ph}/{ph}_report.json").read_text(encoding="utf-8"))["units"]
        R[ph] = {mm: from_rows(u[mm]) for mm in ("IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS")}
    base = v4_study.load("base_loto_aug")       # LOTO baselines on the original probabilities (with Brier)
    R["LOTO"] = {mm: {(x["unit"][0], x["unit"][1], x["task"]): {k: x[k] for k in keep}
                      for x in base_study.rows_of(base, mm)} for mm in ("IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS")}
    v4 = {"confirm1": "final_confirm", "confirm2": "final_confirm2", "confirm3": "final_confirm3",
          "confirm4": "final_confirm4", "LOTO": "loto"}
    ev = {"confirm1": "evo_confirm", "confirm2": "evo_confirm2", "confirm3": "evo_confirm3",
          "confirm4": "evo_confirm4", "LOTO": "evo_loto"}
    for ro in R:
        R[ro]["V4B"] = from_records(ROOT / f"runs/v4_study/{v4[ro]}.jsonl", "v4b")
        R[ro]["EVO_RB"] = from_records(ROOT / f"runs/evo_study/{ev[ro]}.jsonl")
    return R


def aggregate(R):
    per, tot = {}, {}
    for key, _ in METHODS:
        acc = {"units": 0, "harms": 0, "ece_down": 0, "brier_down": 0, "d": 0.0, "e": 0.0, "b": 0.0, "nll": 0.0}
        for ro, _ in ROSTERS:
            ident = R[ro]["IDENTITY"]
            ks = sorted(ident)
            assert set(ks) == set(R[ro][key]), (ro, key)
            d = np.array([R[ro][key][k]["delta_nll"] for k in ks])
            e = np.array([R[ro][key][k]["ece_15"] for k in ks])
            b = np.array([R[ro][key][k]["brier"] for k in ks])
            e0 = np.array([ident[k]["ece_15"] for k in ks])
            b0 = np.array([ident[k]["brier"] for k in ks])
            per.setdefault(ro, {})[key] = {"n": len(ks), "delta": float(d.mean()), "harm": int((d > 0.01).sum()),
                                           "ece": float(e.mean()), "ece_down": int((e < e0).sum()),
                                           "brier": float(b.mean()), "brier_down": int((b < b0).sum())}
            per[ro].setdefault("IDENTITY", {"n": len(ks), "brier": float(b0.mean()), "ece": float(e0.mean())})
            acc["units"] += len(ks)
            acc["harms"] += int((d > 0.01).sum())
            acc["ece_down"] += int((e < e0).sum())
            acc["brier_down"] += int((b < b0).sum())
            acc["d"] += d.sum()
            acc["e"] += e.sum()
            acc["b"] += b.sum()
        n = acc["units"]
        tot[key] = {"units": n, "harms": acc["harms"], "harm_rate": acc["harms"] / n, "mean_delta": acc["d"] / n,
                    "mean_ece": acc["e"] / n, "ece_down_rate": acc["ece_down"] / n, "mean_brier": acc["b"] / n,
                    "brier_down_rate": acc["brier_down"] / n}
    n0 = sum(per[ro]["IDENTITY"]["n"] for ro, _ in ROSTERS)
    tot["IDENTITY"] = {"units": n0, "mean_brier": sum(per[ro]["IDENTITY"]["brier"] * per[ro]["IDENTITY"]["n"]
                                                      for ro, _ in ROSTERS) / n0}
    (ROOT / "runs/evo_study/aggregate_500.json").write_text(json.dumps({"per_roster": per, "aggregate": tot}, indent=1),
                                                            encoding="utf-8")
    return per, tot


def main_table():
    per, tot = aggregate(unit_metrics())
    rows = []
    for key, name in METHODS:
        cells = []
        for ro, _ in ROSTERS:
            r = per[ro][key]
            cells.append(f"{r['delta']:+.3f} & {r['harm']}")
        a = tot[key]
        bold = key == "EVO_RB"
        row = (f"{name} & " + " & ".join(cells) + f" & {a['harms']} ({100 * a['harm_rate']:.1f}\\%) & {a['mean_delta']:+.3f}"
               f" & {a['mean_brier']:.3f} & {100 * a['brier_down_rate']:.0f}\\% & {a['mean_ece']:.3f}"
               f" & {100 * a['ece_down_rate']:.0f}\\%")
        rows.append(("\\textbf{" + name + "}" + row[len(name):]) if bold else row)
    (GEN / "tab_main.tex").write_text(" \\\\\n".join(rows) + " \\\\\n", encoding="utf-8")
    n = {ro: per[ro]["V4B"]["n"] for ro, _ in ROSTERS}
    for ro, short in ROSTERS:
        m(f"N{short}", str(n[ro]), "{}")
    m("NTotal", str(tot["EVO_RB"]["units"]), "{}")
    for key, short in (("EVO_RB", "Evo"), ("V4B", "Rb"), ("THERMO_P", "Thermo"), ("POOLED_TS", "Pooled"),
                       ("ATS_P", "Ats"), ("MRTS", "Mrts")):
        a = tot[key]
        m(f"Agg{short}Harm", str(a["harms"]), "{}")
        m(f"Agg{short}HarmRate", 100 * a["harm_rate"], "{:.1f}")
        m(f"Agg{short}Delta", a["mean_delta"])
        m(f"Agg{short}Ece", a["mean_ece"], "{:.3f}")
        m(f"Agg{short}EceDown", 100 * a["ece_down_rate"], "{:.0f}")
        m(f"Agg{short}Brier", a["mean_brier"], "{:.3f}")
        m(f"Agg{short}BrierDown", 100 * a["brier_down_rate"], "{:.0f}")
        for ro, rs in ROSTERS:
            r = per[ro][key]
            m(f"{rs}{short}Brier", r["brier"], "{:.3f}")
            m(f"{rs}{short}Delta", r["delta"])
            m(f"{rs}{short}Harm", str(r["harm"]), "{}")
            m(f"{rs}{short}Ece", r["ece"], "{:.3f}")
            m(f"{rs}{short}EceDown", str(r["ece_down"]), "{}")
    m("AggIdBrier", tot["IDENTITY"]["mean_brier"], "{:.3f}")
    retain = round(100 * tot["EVO_RB"]["mean_delta"] / tot["THERMO_P"]["mean_delta"])
    m("RetainEvo", str(retain), "{}")
    m("TaxEvo", str(100 - retain), "{}")          # share of the mean gain given up for safety
    m("HarmRatio", tot["THERMO_P"]["harms"] / tot["EVO_RB"]["harms"], "{:.0f}")
    return per, tot


W = 4.8          # LNCS text width in inches
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5, "legend.fontsize": 6, "axes.linewidth": 0.5})
COLORS = {"POOLED_TS": "#8a8986", "THERMO_P": "#eb6834", "ATS_P": "#e8a87c", "MRTS": "#6250d6",
          "V4B": "#7fb0e8", "EVO_RB": "#2a78d6"}
MARKS = {"POOLED_TS": "o", "THERMO_P": "s", "ATS_P": "^", "MRTS": "D", "V4B": "v", "EVO_RB": "*"}


def frontier_figure(per, tot):
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.0))
    for ax, (ro, short) in zip(axes.ravel(), ROSTERS + [("ALL", "All")]):
        for key, name in METHODS:
            if ro == "ALL":
                x, y = 100 * tot[key]["harm_rate"], -tot[key]["mean_delta"]
            else:
                x, y = 100 * per[ro][key]["harm"] / per[ro][key]["n"], -per[ro][key]["delta"]
            ax.scatter(x, y, s=55 if key == "EVO_RB" else 22, marker=MARKS[key], color=COLORS[key],
                       edgecolor="white", lw=0.4, zorder=3, label=name)
        ax.set_title(TITLES.get(ro, f"All {tot['EVO_RB']['units']} units"))
        ax.grid(True, color="#d9d8d4", lw=0.4, ls=":")
        ax.tick_params(length=2, pad=1.5)
    for ax in axes[1]:
        ax.set_xlabel("harmed units (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean NLL gain")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2,
               columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.savefig(FIG / "frontier.pdf")
    plt.close(fig)


def ece_pair_figure():
    import make_ece_scatter as ms
    rows, _ = ms.load_loto()
    meths = [("THERMO_P", "Thermometer-P"), ("EVO", "Evo-RB")]
    fig, axes = plt.subplots(2, 5, figsize=(W, 2.35), sharex=True, sharey=True)
    for r, (key, name) in enumerate(meths):
        for c, (mid, short) in enumerate(ms.MODELS):
            ax = axes[r, c]
            pts = ms.method_points(rows, key, mid)
            ax.plot([0, 0.6], [0, 0.6], ls="--", lw=0.6, color="#52514e")
            ok = [(u, v) for u, v, d in pts if d <= 0.01]
            bad = [(u, v) for u, v, d in pts if d > 0.01]
            if ok:
                ax.scatter(*zip(*ok), s=5, color="#2a78d6", lw=0, zorder=3)
            if bad:
                ax.scatter(*zip(*bad), s=14, marker="x", color="#eb6834", lw=0.8, zorder=4)
            dec = sum(v < u for u, v, _ in pts)
            ax.text(0.03, 0.97, f"ECE down {dec}/{len(pts)}" + "\n" + f"harm {len(bad)}/{len(pts)}",
                    transform=ax.transAxes,
                    va="top", fontsize=5)
            ax.set_xlim(0, 0.6)
            ax.set_ylim(0, 0.6)
            ax.set_xticks([0, 0.3, 0.6])
            ax.set_yticks([0, 0.3, 0.6])
            ax.tick_params(length=1.5, pad=1)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(short, fontsize=6)
            if c == 0:
                ax.set_ylabel(name + "\ncalibrated ECE", fontsize=6)
            if r == 1:
                ax.set_xlabel("uncalibrated ECE", fontsize=5.5)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig(FIG / "ece_pair_loto.pdf")
    plt.close(fig)


def evo_front_figure():
    recs = {tuple(json.loads(l)["unit"]): json.loads(l)
            for l in (ROOT / "runs/evo_study/front.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    show = [("google/gemma-4-12B-it", "paren", "Gemma-4-12B / paren"),
            ("ibm-granite/granite-4.2-8b", "letter_colon", "Granite-4.2-8B / letter"),
            ("mistralai/Ministral-3-8B-Instruct-2512", "paren", "Ministral-3-8B / paren"),
            ("Qwen/Qwen3.5-9B", "letter_colon", "Qwen3.5-9B / letter")]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.55))
    for ax, (mo, pr, title) in zip(axes, show):
        r = recs[(mo, pr)]
        f = sorted(r["front"])
        ax.plot([b for a, b in f], [a for a, b in f], color="#2a78d6", lw=1.1, label="CMA-ES archive (non-dominated)")
        rows = [row for row in r["rows"] if row["slsqp_valid"]]
        ax.scatter([row["eps"] for row in rows], [row["slsqp_mean"] for row in rows], s=9, color="#eb6834",
                   zorder=3, lw=0, label="SLSQP optimum at budget")
        ax.set_xscale("symlog", linthresh=0.002)
        ax.set_xlim(-0.001, 0.2)
        ax.set_xticks([0, 0.01, 0.1])
        ax.set_xticklabels(["0", ".01", ".1"])
        ax.set_title(title, fontsize=6)
        ax.tick_params(length=1.5, pad=1)
        ax.grid(True, color="#d9d8d4", lw=0.4, ls=":")
        ax.set_xlabel("worst-source bound", fontsize=5.5)
    axes[0].set_ylabel("pooled regret", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4)
    fig.savefig(FIG / "evo_front.pdf")
    plt.close(fig)


def evo_numbers():
    recs = [json.loads(l) for l in (ROOT / "runs/evo_study/front.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gaps = np.array([row["cma_mean"] - row["slsqp_mean"] for r in recs for row in r["rows"]
                     if row["slsqp_valid"] and row["cma_valid"]])
    m("FrontPairs", str(len(gaps)), "{}")
    m("FrontBetter", str(int((gaps < -1e-4).sum())), "{}")
    m("FrontWorse", str(int((gaps > 1e-4).sum())), "{}")
    m("FrontMaxWorse", float(gaps.max()), "{:.1e}")
    m("FrontEvals", str(int(np.mean([r["n_evals"] for r in recs]))), "{}")
    m("FrontMinPts", str(min(len(r["front"]) for r in recs)), "{}")
    m("FrontMaxPts", str(max(len(r["front"]) for r in recs)), "{}")


def ablation_numbers():
    """LOTO numbers of the variants that did not work (all design data)."""
    lo = v4_study.load("loto")
    for v, short in (("v5a", "Relax"), ("v6_aug", "Peer")):
        rows = base_study.rows_of([r for r in lo if r["variant"] == v], None, gate_variant=v)
        d = np.array([x["delta_nll"] for x in rows])
        m(f"Loto{short}Delta", float(d.mean()))
        m(f"Loto{short}Harm", str(int((d > 0.01).sum())), "{}")
    avg = v4_study.load("base_loto_avg")
    for meth, short in (("IDENTITY", "AvgId"), ("POOLED_TS", "AvgPooled"), ("THERMO_P", "AvgThermo")):
        rows = base_study.rows_of(avg, meth)
        dr = np.array([x["delta_raw"] for x in rows])
        d = np.array([x["delta_nll"] for x in rows])
        m(f"Loto{short}Raw", float(dr.mean()))
        m(f"Loto{short}Same", float(d.mean()))
        m(f"Loto{short}Harm", str(int((d > 0.01).sum())), "{}")
    for v, short in (("v4b_avg", "AvgRb"), ("v6_avg", "AvgPeer")):
        rows = base_study.rows_of([r for r in lo if r["variant"] == v], None, gate_variant=v)
        m(f"Loto{short}Raw", float(np.mean([x["delta_raw"] for x in rows])))
        m(f"Loto{short}Same", float(np.mean([x["delta_nll"] for x in rows])))
    sw = json.loads((ROOT / "runs/step0/switch_study.json").read_text(encoding="utf-8"))["switch"]
    s = [x for x in sw if x["strong"] == "POOLED_TS" and x["signal"] == "combined" and x["max_harm_train"] == 2][0]
    m("SwitchLotoDelta", s["macro_delta"])
    m("SwitchLotoHarm", str(s["harm"]), "{}")
    c4 = json.loads((ROOT / "runs/rbcal_confirm4/confirm4_report.json").read_text(encoding="utf-8"))["summary"]
    m("SwitchCfourDelta", c4["RB_SWITCH"]["macro_delta"])
    m("SwitchCfourHarm", str(c4["RB_SWITCH"]["harm_001"]), "{}")
    d0 = json.loads((ROOT / "runs/step0/diagnostics.json").read_text(encoding="utf-8"))
    m("OracleDelta", d0["oracle_v4b_or_pooled_safe"])
    m("AucTemplate", d0["auroc"]["pooled_harms"]["template_tv"], "{:.2f}")
    m("AucBeats", max(v for k, v in d0["auroc"]["pooled_beats_v4b_by_0.01"].items() if k != "n_pos"), "{:.2f}")
    # where confirm4 harms concentrate
    u = json.loads((ROOT / "runs/rbcal_confirm4/confirm4_report.json").read_text(encoding="utf-8"))["units"]
    ts = [x for x in u["POOLED_TS"] if x["task_id"] == "temporal_sequences"]
    m("CfourTempPooledHarm", str(sum(x["delta_nll"] > 0.01 for x in ts)), "{}")
    m("CfourTempIdEce", float(np.mean([x["ece_15"] for x in u["IDENTITY"] if x["task_id"] == "temporal_sequences"])), "{:.3f}")


def ece_figure():
    subprocess.check_call([str(ROOT / ".venv/Scripts/python"), str(ROOT / "scripts/make_ece_scatter.py"), "--roster",
                           "loto", "--methods", "THERMO_P,EVO", "--stem", str(FIG / "ece_pair_loto")])


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    per, tot = main_table()
    frontier_figure(per, tot)
    evo_numbers()
    ablation_numbers()
    tm = json.loads((ROOT / "runs/step0/timing_gemma_paren.json").read_text(encoding="utf-8"))
    m("TimeEvo", tm["Evo-RB fit (CMA-ES 300 x SLSQP)"], "{:.0f}")
    m("TimeRbFit", tm["RB-Cal fit, fixed config (SLSQP)"], "{:.1f}")
    m("TimePooled", tm["pooled TS fit (241-pt grid)"], "{:.1f}")
    m("TimeLosoMin", tm["RB-Cal LOSO selection, 37 configs (estimate)"] / 60, "{:.0f}")
    write_macros()
    ece_pair_figure()
    evo_front_figure()
    print(len(macros), "macros; tables, figures written to", OUT)


if __name__ == "__main__":
    main()
