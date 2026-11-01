"""RB-Cal (PREREGISTRATION.md): oracle gradients, budget feasibility, agreement with M1/M3 solvers."""
import numpy as np
import pytest

from calibration_extensions.m1_functional import M1Config, fit_head
from calibration_extensions.m3_top_gap import M3Config, fit_head as m3_fit_head
from calibration_extensions.features import fit_feature_map
from calibration_extensions.rb_calibration import Spec, build, fit_rbcal, fit_shrinkage_loso, solve
from calibration_extensions.source_objectives import beta_grid, calibrator_deltas, scalar_delta_table
from calibration_extensions.toy_data import generate_scenario

from .conftest import make_probset


@pytest.mark.parametrize("spec", [Spec("constant", 0, None, 0.0), Spec("rbf", 3, 0.5, 0.0),
                                  Spec("top_gap", 0, None, 0.0)])
def test_oracle_gradient(rng, spec):
    ps = make_probset(rng, K=5, sources=("a", "b", "c"))
    oracle, x0, *_ = build(ps, spec)
    x = x0 + rng.normal(size=len(x0)) * 0.1
    d, G = oracle(x)
    e = rng.normal(size=len(x0))
    h = 1e-6
    fd = (oracle(x + h * e)[0] - oracle(x - h * e)[0]) / (2 * h)
    assert np.max(np.abs(fd - G @ e)) < 1e-6


@pytest.mark.parametrize("eps", [0.0, 0.005, 0.02])
@pytest.mark.parametrize("spec", [Spec("constant", 0, None, 0.0), Spec("rbf", 4, 0.5, 0.01),
                                  Spec("top_gap", 0, None, 0.01)])
def test_budget_respected(eps, spec):
    tr = generate_scenario("b_entropy_conflict", seed=2)[0]["source_train"]
    cal, info = solve(tr, spec, eps)
    assert info["valid"] and max(calibrator_deltas(tr, cal).values()) <= eps + 1e-7
    assert info["train_mean"] <= 1e-9          # never worse on average than identity


def test_constant_budget_is_feasible_interval_point_closest_to_pooled():
    tr = generate_scenario("b_entropy_conflict", seed=0)[0]["source_train"]
    grid = beta_grid(n=4001)
    table, pooled, _ = scalar_delta_table(tr, grid)
    feas = table.max(axis=1) <= 0.01
    j = np.argmin(np.where(feas, pooled, np.inf))
    cal, _ = solve(tr, Spec("constant", 0, None, 0.0), 0.01)
    assert abs(np.log(cal.theta[0]) - np.log(grid[j])) < 2 * np.diff(np.log(grid)).max()


def test_unbudgeted_matches_m1_and_m3_pooled():
    tr = generate_scenario("m3_top_gap", seed=1)[0]["source_train"]
    fmap = fit_feature_map(tr, "rbf", 4, 0.5)
    a, _ = solve(tr, Spec("rbf", 4, 0.5, 0.01), None)
    b, _ = fit_head(tr, fmap, "POOLED", 0.01, M1Config(solver="cvxpy"))
    mean = lambda c: np.mean(list(calibrator_deltas(tr, c).values()))
    w = np.array([np.mean([str(s) == k for s in tr.source_id]) for k in tr.sources()])
    pooled = lambda c: w @ np.array([calibrator_deltas(tr, c)[k] for k in tr.sources()])
    assert abs(pooled(a) - pooled(b)) < 1e-5
    c, _ = solve(tr, Spec("top_gap", 0, None, 0.0), None)
    d, _ = m3_fit_head(tr, "POOLED", 0.0, M3Config())
    assert abs(pooled(c) - pooled(d)) < 1e-5


def test_fit_rbcal_gate_and_shrinkage():
    splits, _ = generate_scenario("b_entropy_conflict", seed=0)
    tr, sel = splits["source_train"], splits["source_select"]
    fr = fit_rbcal(tr, sel, [("constant", 0, None, 0.0), ("rbf", 2, 0.5, 0.01)], 0.01, "RBCAL_TEST")
    assert fr.solver_valid
    if fr.source_gate["accept"]:
        assert fr.source_gate["max_delta"] <= 0.01 + 1e-8
    else:
        assert fr.deployed.kind == "identity"
    sh = fit_shrinkage_loso(tr, sel)
    assert 0.0 <= sh.parameters["s"] <= 1.0


def test_ucb_value_and_gradient(rng):
    ps = make_probset(rng, K=4, sources=("a", "b"), rows_per_case=3, n_cases=20)
    oracle, x0, *_ = build(ps, Spec("rbf", 3, 0.5, 0.0))
    x = x0 + rng.normal(size=len(x0)) * 0.2
    z = 1.645
    val, jac = oracle.ucb(x, z)
    loss, _ = oracle.rows(x)
    d = loss - oracle.surprisal
    for w, s in enumerate(oracle.srcs):
        rows = [i for i in range(ps.n) if str(ps.source_id[i]) == s]
        cases = sorted({str(ps.case_id[i]) for i in rows})
        sums = np.array([sum(d[i] for i in rows if str(ps.case_id[i]) == c) for c in cases])
        cnt = np.array([sum(1 for i in rows if str(ps.case_id[i]) == c) for c in cases])
        mean = d[rows].mean()
        se = np.sqrt(len(cases) / (len(cases) - 1) * np.sum((sums - cnt * mean) ** 2) / len(rows) ** 2)
        assert abs(val[w] - (mean + z * se)) < 1e-9
    e = rng.normal(size=len(x0))
    h = 1e-6
    fd = (oracle.ucb(x + h * e, z)[0] - oracle.ucb(x - h * e, z)[0]) / (2 * h)
    assert np.max(np.abs(fd - jac @ e)) < 1e-5


def test_ucb_budget_respected():
    tr = generate_scenario("b_entropy_conflict", seed=3)[0]["source_train"]
    oracle, *_ = build(tr, Spec("rbf", 4, 0.5, 0.01))
    cal, info = solve(tr, Spec("rbf", 4, 0.5, 0.01), 0.01, z=1.645)
    assert info["valid"]
    assert np.max(oracle.ucb(cal.theta, 1.645)[0]) <= 0.01 + 1e-7
