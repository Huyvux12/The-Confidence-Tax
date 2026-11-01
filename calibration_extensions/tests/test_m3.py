"""Spec 10.4 - M3."""
import numpy as np
import pytest

from calibration_extensions.calibrators import IdentityCalibrator, ScalarTS, TopGap, top_gap_parts
from calibration_extensions.m3_top_gap import M3Config, fit_m3, nll_decomposition, rank_change_rate
from calibration_extensions.numerics import ProbSet, losses, prepare_probs

from .conftest import make_probset


def max_diff(a, b):
    return max(np.max(np.abs(x - y)) for x, y in zip(a, b))


def test_a_equals_b_is_ts_and_identity(rng):
    ps = make_probset(rng, K=5)
    for beta in (0.4, 1.0, 2.7):
        assert max_diff(TopGap(beta, beta).log_q_groups(ps), ScalarTS(beta).log_q_groups(ps)) < 1e-12
    assert max_diff(TopGap(1, 1).log_q_groups(ps), IdentityCalibrator().log_q_groups(ps)) < 1e-12


def test_binary_depends_only_on_b(rng):
    ps = make_probset(rng, K=2)
    ref = TopGap(1.0, 0.7).log_q_groups(ps)
    for a in (0.3, 2.0, 3.9):
        assert max_diff(TopGap(a, 0.7).log_q_groups(ps), ref) < 1e-12


def test_binary_fit_fixes_a():
    rng = np.random.default_rng(0)
    ps = make_probset(rng, K=2, sources=("x", "y", "z"), n_cases=40)
    fr = fit_m3(ps, ps, M3Config(lambdas=(0.0,)))
    assert fr.parameters["effective_dof"] == 1 and fr.fitted.a == 1.0


def test_rank_preserved_and_ties_kept(rng):
    ps = make_probset(rng, K=6, scale=3.0)
    for a, b in ((0.3, 3.5), (3.9, 0.26), (1.5, 1.5)):
        assert rank_change_rate(ps, TopGap(a, b).log_q_groups(ps)) == 0.0
    lp, _ = prepare_probs([0.35, 0.35, 0.2, 0.1])
    o = np.array(["x"], dtype=object)
    tie = ProbSet(o, o, o, o, o, o, [list("abcd")], [lp], y=np.array([0]))
    q = TopGap(0.5, 3.0).log_q_groups(tie)[0][0]
    assert abs(q[0] - q[1]) < 1e-12 and rank_change_rate(tie, [q[None, :]]) == 0.0


def test_decomposition_sums_to_nll(rng):
    ps = make_probset(rng, K=4, scale=2.5)
    for cal in (IdentityCalibrator(), TopGap(1.4, 0.6)):
        mats = cal.log_q_groups(ps)
        d = nll_decomposition(ps, mats)
        assert d["decomposition_error"] < 1e-12
        assert abs(d["nll"] - losses(ps, mats).mean()) < 1e-12
        assert d["n_wrong"] > 0 and d["tail_nll_wrong_only"] is not None


def test_gradient_and_convexity(rng):
    ps = make_probset(rng, K=5, scale=2.0)

    def L(a, b):
        return losses(ps, TopGap(a, b).log_q_groups(ps)).mean()

    a, b, h = 1.3, 0.7, 1e-6
    ga, gb = 0.0, 0.0
    for g in ps.groups:
        Zp, D, top, d = top_gap_parts(g.logP)
        U = a * Zp + b * D
        q = np.exp(U - np.log(np.exp(U).sum(axis=1, keepdims=True)))
        r = np.arange(len(g.idx))
        ga += ((q * Zp).sum(axis=1) - Zp[r, g.y]).sum()
        gb += (d * (q[r, top] - (g.y == top))).sum()
    ga, gb = ga / ps.n, gb / ps.n
    assert abs((L(a + h, b) - L(a - h, b)) / (2 * h) - ga) < 1e-6
    assert abs((L(a, b + h) - L(a, b - h)) / (2 * h) - gb) < 1e-6
    for _ in range(20):
        p0, p1 = rng.uniform(0.25, 4, size=2), rng.uniform(0.25, 4, size=2)
        for t in (0.25, 0.5, 0.75):
            mid = (1 - t) * p0 + t * p1
            assert L(*mid) <= (1 - t) * L(*p0) + t * L(*p1) + 1e-10
