"""Spec 10.6 - M5."""
import itertools

import numpy as np

from calibration_extensions.calibrators import IdentityCalibrator, ScalarTS, TopGap
from calibration_extensions.m5_active_certificate import BatchBounds, LabelOracle, select_with_oracle
from calibration_extensions.numerics import losses, surprisal
from calibration_extensions.source_splits import cases_of
from calibration_extensions.toy_data import generate_scenario

from .conftest import make_probset


def private_of(ps):
    return {str(ps.sample_id[i]): {"y": int(ps.y[i]), "label_id": ps.label_ids[i][ps.y[i]]} for i in range(ps.n)}


def true_delta(ps, cal):
    return (losses(ps, cal.log_q_groups(ps)) - surprisal(ps)).mean()


def test_bounds_contain_every_labelling():
    rng = np.random.default_rng(1)
    ps = make_probset(rng, n_cases=3, K=3, sources=("s",), rows_per_case=2)
    cands = {"identity": IdentityCalibrator(), "ts": ScalarTS(0.5), "tg": TopGap(1.5, 0.6)}
    known = {0: int(ps.y[0]), 1: int(ps.y[1])}
    bb = BatchBounds(ps.without_labels(), cands)
    bb.reveal(known)
    bounds = bb.bounds()
    unknown = [i for i in range(ps.n) if i not in known]
    for labels in itertools.product(*[range(3) for _ in unknown]):
        y = ps.y.copy()
        y[unknown] = labels
        pl = ps.with_labels(y)
        for a, cal in cands.items():
            d = true_delta(pl, cal)
            assert bounds[a][0] - 1e-12 <= d <= bounds[a][1] + 1e-12


def test_bounds_monotone_and_exact_when_all_known(rng):
    ps = make_probset(rng, n_cases=15, K=4, sources=("s",))
    cands = {"identity": IdentityCalibrator(), "ts": ScalarTS(0.6)}
    bb = BatchBounds(ps.without_labels(), cands)
    prev = bb.bounds()
    for c, rows in cases_of(ps).items():
        bb.reveal({int(i): int(ps.y[i]) for i in rows})
        now = bb.bounds()
        for a in cands:
            assert now[a][0] >= prev[a][0] - 1e-12 and now[a][1] <= prev[a][1] + 1e-12
        prev = now
    for a, cal in cands.items():
        assert abs(prev[a][0] - prev[a][1]) < 1e-12 and abs(prev[a][0] - true_delta(ps, cal)) < 1e-12


def test_certificate_is_never_false():
    _, target = generate_scenario("a_shared_temperature", seed=3, n_target_cases=60, n_target_tasks=1)
    cands = {"identity": IdentityCalibrator(), "ts06": ScalarTS(0.6), "ts2": ScalarTS(2.0)}
    for budget in (0, 5, 20, 60):
        for rule, seed in (("active", 0), ("random", 1), ("random", 2)):
            res = select_with_oracle(cands, target.without_labels(), LabelOracle(target, private_of(target)),
                                     budget, rule=rule, seed=seed)
            if res["certified"]:
                assert true_delta(target, cands[res["selected"]]) <= -1e-6 + 1e-12
            else:
                assert res["selected"] == "identity"
            assert res["total_cost"] <= budget


def test_cost_counted_per_case(rng):
    ps = make_probset(rng, n_cases=5, rows_per_case=3, sources=("s",))
    oracle = LabelOracle(ps, private_of(ps))
    c = str(ps.case_id[0])
    first = oracle.query(c)
    oracle.query(c)
    assert len(first) == 3 and oracle.total_cost == 1.0 and oracle.queried == [c]


def test_semantic_label_mapped_per_row(rng):
    _, target = generate_scenario("a_shared_temperature", seed=0, n_target_cases=10, n_target_tasks=1)
    oracle = LabelOracle(target, private_of(target))
    for c, rows in cases_of(target).items():
        got = oracle.query(c)
        for i in rows:
            assert got[int(i)] == target.y[i]


def test_unqueried_labels_do_not_affect_process():
    _, target = generate_scenario("a_shared_temperature", seed=4, n_target_cases=40, n_target_tasks=1)
    cands = {"identity": IdentityCalibrator(), "ts": ScalarTS(0.6)}
    first = select_with_oracle(cands, target.without_labels(), LabelOracle(target, private_of(target)), 6)
    queried = set(first["queried_cases"])
    cases = cases_of(target)
    y = target.y.copy()
    for c, rows in cases.items():
        if c not in queried:
            for i in rows:
                y[i] = (y[i] + 1) % target.log_p[i].size
    other = target.with_labels(y)
    second = select_with_oracle(cands, target.without_labels(), LabelOracle(other, private_of(other)), 6)
    assert first["queried_cases"] == second["queried_cases"]
    assert [t["bounds"] for t in first["trace"]] == [t["bounds"] for t in second["trace"]]
