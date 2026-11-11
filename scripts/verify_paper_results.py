#!/usr/bin/env python3
"""Check the frozen data behind the final Evo-RB manuscript (no training, no GPU).

Checks table 1, table 2 and the paired LOTO rescue analysis independently
against stored JSON/JSONL. Not a replacement for full methodological audit.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
THRESHOLD = 0.01
METHODS = ("POOLED_TS", "THERMO_P", "ATS_P", "MRTS", "EVO_RB")
ROSTERS = ("confirm1", "confirm2", "confirm3", "confirm4", "LOTO")
EXPECTED = {
    "POOLED_TS": (0.078, 0.431, -0.490, 58),
    "THERMO_P": (0.075, 0.430, -0.493, 49),
    "ATS_P": (0.078, 0.432, -0.487, 74),
    "MRTS": (0.131, 0.462, -0.368, 1),
    "EVO_RB": (0.084, 0.435, -0.473, 8),
}
ROSTER_EXPECTED = {
    "confirm1": (30, ((-0.257, 7), (-0.262, 5), (-0.263, 7), (-0.201, 0), (-0.266, 0))),
    "confirm2": (60, ((-0.319, 9), (-0.323, 7), (-0.315, 12), (-0.265, 0), (-0.321, 0))),
    "confirm3": (60, ((-0.305, 7), (-0.306, 5), (-0.304, 9), (-0.232, 0), (-0.290, 0))),
    "confirm4": (100, ((-0.488, 13), (-0.490, 13), (-0.485, 14), (-0.355, 1), (-0.442, 7))),
    "LOTO": (250, ((-0.605, 22), (-0.607, 19), (-0.600, 32), (-0.451, 0), (-0.591, 1))),
}
RESCUE_EXPECTED = {
    "POOLED_TS": (22, 21, 1, 0),
    "THERMO_P": (19, 18, 1, 0),
    "ATS_P": (32, 31, 1, 0),
    "MRTS": (0, 0, 0, 1),
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def approx_display(value, expected, digits=3):
    # Match a displayed, rounded table entry rather than over-fitting a print format.
    return abs(float(value) - expected) < 0.5 * 10 ** (-digits) + 1e-10


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def record_key(r):
    return (r["unit"][0], r["unit"][1], r["task"])


def gate_passes(stats):
    return bool(stats) and all(
        float(row["delta"]) - 1.645 * float(row["se"]) <= THRESHOLD + 1e-8
        for row in stats.values()
    )


def verify_tables():
    agg = json.loads((RUNS / "evo_study/aggregate_500.json").read_text(encoding="utf-8"))
    total, rosters = agg["aggregate"], agg["per_roster"]
    check(set(rosters) == set(ROSTERS), "wrong set of rosters")
    check(sum(rosters[r]["EVO_RB"]["n"] for r in ROSTERS) == 500, "unit count != 500")
    check(approx_display(rosters["confirm1"]["IDENTITY"]["ece"], 0.140), "unexpected raw C1 ECE")

    for method, (ece, brier, delta, harm) in EXPECTED.items():
        a = total[method]
        check(a["units"] == 500, "%s unit count" % method)
        for name, got, want in (
            ("ECE", a["mean_ece"], ece),
            ("Brier", a["mean_brier"], brier),
            ("DeltaNLL", a["mean_delta"], delta),
        ):
            check(approx_display(got, want), "%s %s %.8f != %.3f" % (method, name, got, want))
        check(a["harms"] == harm, "%s harm count" % method)
        weights = [rosters[r][method]["n"] for r in ROSTERS]
        check(sum(weights) == 500, "%s unit total from rosters" % method)
        for metric, target in (
            ("delta", a["mean_delta"]),
            ("ece", a["mean_ece"]),
            ("brier", a["mean_brier"]),
        ):
            value = sum(rosters[r][method][metric] * rosters[r][method]["n"] for r in ROSTERS) / 500
            check(abs(value - target) < 1e-10, "weighted aggregation: %s %s" % (method, metric))
        check(sum(rosters[r][method]["harm"] for r in ROSTERS) == harm, "harm sum: %s" % method)

    for roster, (n, values) in ROSTER_EXPECTED.items():
        for method, (delta, harm) in zip(METHODS, values):
            row = rosters[roster][method]
            check(row["n"] == n, "%s: n" % roster)
            check(approx_display(row["delta"], delta), "%s/%s: delta" % (roster, method))
            check(row["harm"] == harm, "%s/%s: harm" % (roster, method))

    check(approx_display(total["IDENTITY"]["mean_brier"], 0.507), "uncalibrated Brier")
    print("PASS: Table 1 (500 units) and Table 2 (5 rosters × 5 methods)")


def verify_rescues():
    base = {record_key(r): r for r in rows(RUNS / "v4_study/base_loto_aug.jsonl")}
    evo = {record_key(r): r for r in rows(RUNS / "evo_study/evo_loto.jsonl")}
    check(len(base) == len(evo) == 250, "not exactly 250 LOTO model-prompt-task units")
    check(set(base) == set(evo), "LOTO baseline/evolution key mismatch")
    summary = {}
    for method, expected in RESCUE_EXPECTED.items():
        harmed = rescued = shared = newly = 0
        for key, b in base.items():
            e = evo[key]
            delta_b = float(b["target"][method][b["task"]]["delta_nll"])
            delta_e = (float(e["target"]["V"][e["task"]]["delta_nll"])
                       if gate_passes(e["select"]["V"]) and "V" in e["target"] else 0.0)
            bh, eh = delta_b > THRESHOLD, delta_e > THRESHOLD
            harmed += int(bh)
            rescued += int(bh and not eh)
            shared += int(bh and eh)
            newly += int(not bh and eh)
        got = (harmed, rescued, shared, newly)
        check(got == expected, "rescue %s %s != %s" % (method, got, expected))
        summary[method] = dict(zip(("baseline_harm", "rescued", "shared", "newly"), got))
    print("PASS: Figure 3 paired rescue counts over 250 LOTO units")
    for m, values in summary.items():
        print("  %-10s %s" % (m, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-protocol", action="store_true",
                        help="Fail if reported LOTO hyperparameters used the held-out task during global selection")
    args = parser.parse_args()
    verify_tables()
    verify_rescues()
    msg = ("LOTO protocol caveat: the frozen v4b specification was selected globally on "
           "all 25 source tasks before each fixed-config LOTO refit. This differs "
           "from full nested source-only hyperparameter re-selection claimed in the PDF.")
    print("IMPORTANT:", msg)
    print("NOT VERIFIED: separate raw controlled no-margin ablation output for Table 4.")
    if args.strict_protocol:
        raise AssertionError(msg)
    print("Artifact data checks PASS (not a full independent model refit).")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, FileNotFoundError, ValueError) as err:
        print("FAIL:", err, file=sys.stderr)
        sys.exit(1)
