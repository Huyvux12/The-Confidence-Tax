"""Generate every table, figure and number used in paper/main.tex from the frozen runs.

    python scripts/make_paper_assets.py

Writes paper/generated/*.tex and paper/figures/*.pdf. Nothing in the paper is typed by hand.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEN, FIG = ROOT / "paper" / "generated", ROOT / "paper" / "figures"
PRIMARY = "RBCAL_UCB_RBF_e0.01"
NAMES = {
    "IDENTITY": "Identity", "POOLED_TS": "Pooled TS", "MRTS": "MRTS", "SHRINKAGE_LOSO": "Shrinkage-LOSO",
    "THERMO_P": "Thermometer-P", "ATS_P": "ATS-P", "POOLED_RBF__PREGATE": "Pooled RBF field",
    "RBCAL_RBF_e0.01": r"RB-Cal-RBF, $z{=}0$ (pre-reg.)", "RBCAL_CONST_e0.01": r"RB-Cal-const, $z{=}0$ (pre-reg.)",
    "RBCAL_UCB_CONST_e0.01": "RB-Cal-const", "RBCAL_UCB_TOPGAP_e0.01": "RB-Cal-top-gap",
    PRIMARY: r"\textbf{RB-Cal-RBF}",
}


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def f(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def num(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    conf, dev = load("runs/rbcal_confirm/report.json"), load("runs/rbcal_dev/report.json")
    extra, loto = load("runs/extra_baselines/report.json"), load("runs/loto/report.json")
    macros = ""

    # ---------- Table: confirm main results
    def summ(m, reg):
        if m in ("THERMO_P", "ATS_P"):
            return extra["confirm"][reg]["summary"][m]
        return conf["summaries"][reg][m]

    order = ["IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "POOLED_RBF__PREGATE", "MRTS", "SHRINKAGE_LOSO",
             "RBCAL_CONST_e0.01", "RBCAL_RBF_e0.01", "RBCAL_UCB_CONST_e0.01", "RBCAL_UCB_TOPGAP_e0.01", PRIMARY]
    lines = []
    for m in order:
        a, c, s = summ(m, "all"), summ(m, "conflict"), summ(m, "consensus")
        lines.append(f"{NAMES[m]} & {f(a['macro_delta'])} & {a['harm_001']} & {f(a['worst_delta'])} & "
                     f"{a['ece_15']:.3f} & {f(c['macro_delta'])} & {c['harm_001']} & {f(s['macro_delta'])} & "
                     f"{s['harm_001']} \\\\")
        if m in ("POOLED_RBF__PREGATE", "RBCAL_RBF_e0.01"):
            lines.append(r"\midrule")
    (GEN / "tab_confirm.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_conf = conf["summaries"]["all"][PRIMARY]["n_units"]
    n_c = conf["summaries"]["conflict"][PRIMARY]["n_units"]
    macros += macro("NConfirm", n_conf) + macro("NConfirmConflict", n_c) + macro("NConfirmConsensus", n_conf - n_c)

    # ---------- Table: endpoints dev and confirm
    lines = []
    lab = {"E1_vs_MRTS": "E1 vs MRTS", "E2_vs_POOLED_TS": "E2 vs pooled TS", "E3_vs_RBCAL_CONST": "E3 vs RB-Cal-const"}
    for split, rep in (("dev", dev), ("confirm", conf)):
        for reg in ("all", "conflict", "consensus"):
            for e, name in lab.items():
                d = rep["endpoints"][reg][e]
                if "paired" in d:
                    p = d["paired"]
                    eff = f"{f(p['mean_diff'])} [{f(p['ci_low'])}, {f(p['ci_high'])}]"
                    harm = f"{d['harm'][0]} vs {d['harm'][1]}"
                else:
                    eff = f"worst {f(d['worst'][0])} vs {f(d['worst'][1])}"
                    harm = f"{d['harm'][0]} vs {d['harm'][1]}"
                lines.append(f"{split} & {reg} & {name} & {eff} & {harm} & {'pass' if d['pass'] else 'fail'} \\\\")
            if reg != "consensus":
                lines.append(r"\cmidrule(lr){2-6}")
        if split == "dev":
            lines.append(r"\midrule")
    (GEN / "tab_endpoints.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ep = conf["endpoints"]["all"]
    p1, p3 = ep["E1_vs_MRTS"]["paired"], ep["E3_vs_RBCAL_CONST"]["paired"]
    e2 = ep["E2_vs_POOLED_TS"]
    macros += macro("EOneDiff", num(p1["mean_diff"], 3, True)) + macro("EOneLo", num(p1["ci_low"], 3, True)) + \
        macro("EOneHi", num(p1["ci_high"], 3, True)) + macro("EThreeDiff", num(p3["mean_diff"], 3, True)) + \
        macro("EThreeLo", num(p3["ci_low"], 3, True)) + macro("EThreeHi", num(p3["ci_high"], 3, True)) + \
        macro("ETwoHarmP", e2["harm"][0]) + macro("ETwoHarmPooled", e2["harm"][1]) + \
        macro("ETwoWorstP", num(e2["worst"][0], 4, True)) + macro("ETwoWorstPooled", num(e2["worst"][1], 4, True)) + \
        macro("ETwoRetain", f"{100 * e2['gain_retained']:.0f}")
    pd = dev["endpoints"]["all"]["E1_vs_MRTS"]["paired"]
    macros += macro("EOneDevDiff", num(pd["mean_diff"], 3, True))
    for m, key in ((PRIMARY, "Prim"), ("POOLED_TS", "Pooled"), ("MRTS", "Mrts")):
        macros += macro(f"Conf{key}Delta", num(conf["summaries"]["all"][m]["macro_delta"], 3, True)) + \
            macro(f"Conf{key}Harm", conf["summaries"]["all"][m]["harm_001"])
    for m, key in (("THERMO_P", "Thermo"), ("ATS_P", "Ats")):
        s = extra["confirm"]["all"]["summary"][m]
        pv = extra["confirm"]["all"]["primary_vs"][m]["paired_primary_minus_m"]
        macros += macro(f"Conf{key}Delta", num(s["macro_delta"], 3, True)) + macro(f"Conf{key}Harm", s["harm_001"]) + \
            macro(f"Conf{key}Gap", num(pv["mean_diff"], 3, True)) + macro(f"Conf{key}GapLo", num(pv["ci_low"], 3, True)) + \
            macro(f"Conf{key}GapHi", num(pv["ci_high"], 3, True))
    pv = extra["confirm"]["conflict"]["primary_vs"]["THERMO_P"]["paired_primary_minus_m"]
    macros += macro("ConfConflictThermoGap", num(pv["mean_diff"], 4, True)) + \
        macro("ConfConflictThermoGapLo", num(pv["ci_low"], 3, True)) + macro("ConfConflictThermoGapHi", num(pv["ci_high"], 3, True))
    macros += macro("ConfConflictThermoHarm", extra["confirm"]["conflict"]["summary"]["THERMO_P"]["harm_001"])

    # ---------- Table: LOTO
    lo_order = ["IDENTITY", "POOLED_TS", "THERMO_P", "ATS_P", "MRTS", "SHRINKAGE_LOSO", "RBCAL_UCB_CONST_e0.01", PRIMARY]
    lines = []
    for m in lo_order:
        a, c, s = (loto["summaries"][k][m] for k in ("all", "conflict", "consensus"))
        if m == PRIMARY:
            vs = "---"
        else:
            v = loto["primary_vs"]["all"][m]
            vs = f"{f(v['mean_diff'])} [{f(v['ci_low'])}, {f(v['ci_high'])}] & {v['win']}/{v['tie']}/{v['loss']}"
        if m == PRIMARY:
            vs = "--- & ---"
        lines.append(f"{NAMES[m]} & {f(a['macro_delta'])} & {a['harm_001']} & {f(a['worst_delta'])} & "
                     f"{f(c['macro_delta'])} & {c['harm_001']} & {f(s['macro_delta'])} & {s['harm_001']} & {vs} \\\\")
    (GEN / "tab_loto.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ls = loto["summaries"]["all"]
    macros += macro("NLoto", ls[PRIMARY]["n"]) + macro("NLotoConflict", loto["summaries"]["conflict"][PRIMARY]["n"])
    for m, key in ((PRIMARY, "Prim"), ("POOLED_TS", "Pooled"), ("MRTS", "Mrts"), ("THERMO_P", "Thermo"),
                   ("ATS_P", "Ats"), ("RBCAL_UCB_CONST_e0.01", "Const")):
        macros += macro(f"Loto{key}Delta", num(ls[m]["macro_delta"], 3, True)) + macro(f"Loto{key}Harm", ls[m]["harm_001"]) + \
            macro(f"Loto{key}Worst", num(ls[m]["worst_delta"], 3, True))
    for m, key in (("MRTS", "Mrts"), ("RBCAL_UCB_CONST_e0.01", "Const"), ("POOLED_TS", "Pooled")):
        v = loto["primary_vs"]["all"][m]
        macros += macro(f"LotoVs{key}", num(v["mean_diff"], 3, True)) + macro(f"LotoVs{key}Lo", num(v["ci_low"], 3, True)) + \
            macro(f"LotoVs{key}Hi", num(v["ci_high"], 3, True)) + macro(f"LotoVs{key}Win", v["win"]) + \
            macro(f"LotoVs{key}Loss", v["loss"])
    retain = [ls[PRIMARY]["macro_delta"] / ls[m]["macro_delta"] for m in ("POOLED_TS", "THERMO_P", "ATS_P")]
    macros += macro("LotoRetainLo", f"{100 * min(retain):.0f}") + macro("LotoRetainHi", f"{100 * max(retain):.0f}")
    jobs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs/loto/jobs").glob("*.json"))]
    harms = [r for r in jobs if r["methods"][PRIMARY]["harm_001"]]
    macros += macro("LotoPrimHarmMax", num(max(r["methods"][PRIMARY]["delta_nll"] for r in harms), 4, True))

    # ---------- Table: units (base model calibration on confirm targets, regime from sources)
    rows = [r for r in csv.DictReader(open(ROOT / "runs/rbcal_confirm/target_metrics.csv", encoding="utf-8"))
            if r["method_id"] == "IDENTITY" and not r["task_id"].startswith("__")]
    by = defaultdict(list)
    for r in rows:
        by[(r["model_id"], r["prompt_id"])].append(r)
    lines = []
    for (mod, pr), rs in sorted(by.items()):
        reg = conf["regimes"][f"{mod}|{pr}"]
        acc = np.mean([float(r["accuracy"]) for r in rs])
        e = np.mean([float(r["ece_15"]) for r in rs])
        nll = np.mean([float(r["nll"]) for r in rs])
        lines.append(f"\\texttt{{{mod.split('/')[1]}}} & \\texttt{{{pr.replace('_', chr(92) + '_')}}} & {reg} & "
                     f"{acc:.3f} & {nll:.3f} & {e:.3f} \\\\")
    (GEN / "tab_units.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros += macro("NConflictUnits", sum(v == "conflict" for v in conf["regimes"].values()))

    # ---------- Figure 1: epsilon curve on confirm
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    eps0 = [0.0, 0.0025, 0.005, 0.01, 0.02]
    epsu = [0.005, 0.01, 0.02]
    series = [("RBCAL_UCB_RBF_e{}", epsu, "RB-Cal-RBF (UCB)", "#1f4e9c", "o", "-"),
              ("RBCAL_UCB_CONST_e{}", epsu, "RB-Cal-const (UCB)", "#7aa6e0", "s", "-"),
              ("RBCAL_RBF_e{}", eps0, "RB-Cal-RBF ($z{=}0$)", "#b5651d", "o", "--"),
              ("RBCAL_CONST_e{}", eps0, "RB-Cal-const ($z{=}0$)", "#e3a76f", "s", "--")]
    for ax, key in zip(axes, ("macro_delta", "harm_001")):
        for tpl, eps, lab_, col, mk, ls_ in series:
            y = [conf["summaries"]["all"][tpl.format(e)][key] for e in eps]
            ax.plot(eps, y, ls_, marker=mk, color=col, label=lab_, ms=3.5, lw=1.2)
        for m, col, lab_ in (("POOLED_TS", "#c0392b", "Pooled TS"), ("MRTS", "#555555", "MRTS")):
            ax.axhline(conf["summaries"]["all"][m][key], color=col, lw=0.9, ls=":", label=lab_)
        ax.set_xlabel(r"budget $\varepsilon$ (nats)")
    axes[0].set_ylabel(r"macro $\Delta$NLL (lower is better)")
    axes[1].set_ylabel(r"# task-units with $\Delta$NLL $>0.01$")
    axes[0].legend(frameon=False, fontsize=6.5, loc="center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "eps_curve.pdf")
    plt.close(fig)

    # ---------- Figure 2: safety vs efficiency (LOTO and confirm)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    pts = [("POOLED_TS", "#c0392b"), ("THERMO_P", "#8e44ad"), ("ATS_P", "#d35400"), ("MRTS", "#555555"),
           ("SHRINKAGE_LOSO", "#7f8c8d"), ("RBCAL_UCB_CONST_e0.01", "#7aa6e0"), (PRIMARY, "#1f4e9c")]
    OFFSET = {"THERMO_P": (-4, 3), "ATS_P": (4, 3), "POOLED_TS": (4, -9), PRIMARY: (4, 3),
              "RBCAL_UCB_CONST_e0.01": (4, -2), "MRTS": (4, -8)}
    short = {"POOLED_TS": "Pooled TS", "THERMO_P": "Thermometer-P", "ATS_P": "ATS-P", "MRTS": "MRTS",
             "SHRINKAGE_LOSO": "Shrinkage", "RBCAL_UCB_CONST_e0.01": "RB-Cal-const", PRIMARY: "RB-Cal-RBF"}
    for ax, title, getter, n in (
            (axes[0], f"Sealed confirm ({n_conf} target task-units)", lambda m: summ(m, "all"), n_conf),
            (axes[1], f"Leave-one-task-out ({ls[PRIMARY]['n']} target task-units)", lambda m: ls[m], ls[PRIMARY]["n"])):
        for m, col in pts:
            s = getter(m)
            ax.scatter(s["harm_001"], -s["macro_delta"], color=col, s=22, zorder=3)
            ax.annotate(short[m], (s["harm_001"], -s["macro_delta"]), fontsize=6.5, xytext=OFFSET.get(m, (4, 2)),
                        textcoords="offset points", ha="right" if OFFSET.get(m, (4, 2))[0] < 0 else "left")
        ax.set_xlabel(r"# harmed task-units ($\Delta$NLL $>0.01$)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel(r"mean NLL gain (nats)")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.pdf")
    plt.close(fig)

    (GEN / "numbers.tex").write_text(macros, encoding="utf-8")
    print("wrote", sorted(p.name for p in GEN.iterdir()), sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
