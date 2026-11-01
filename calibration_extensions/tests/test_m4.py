"""Spec 10.5 - M4."""
import itertools

import numpy as np

from calibration_extensions.calibrators import FieldBeta, IdentityCalibrator, ScalarTS
from calibration_extensions.features import fit_feature_map
from calibration_extensions.m4_label_budget import M4Config, fit_m4, robust_risk, state_budget
from calibration_extensions.numerics import losses, surprisal
from calibration_extensions.source_objectives import beta_grid, fit_scalar
from calibration_extensions.toy_data import corrupt_states, generate_scenario

from .conftest import make_probset


def brute_force_worst(ps, cal, B):
    """Enumerate every set of <= B states and every relabelling of their rows."""
    mats = cal.log_q_groups(ps)
    out = {}
    for s in ps.sources():
        rows = [i for i in range(ps.n) if str(ps.source_id[i]) == s]
        cases = sorted({str(ps.case_id[i]) for i in rows})
        sub = ps.subset(rows)
        best = -np.inf
        for k in range(B + 1):
            for chosen in itertools.combinations(cases, k):
                free = [j for j, i in enumerate(rows) if str(ps.case_id[i]) in chosen]
                for labels in itertools.product(*[range(ps.log_p[rows[j]].size) for j in free]):
                    y = sub.y.copy()
                    y[free] = labels
                    sl = sub.with_labels(y)
                    val = (losses(sl, cal.log_q_groups(sl)) - surprisal(sl)).mean()
                    best = max(best, val)
        out[s] = best
    return out


def test_top_b_matches_brute_force():
    rng = np.random.default_rng(7)
    ps = make_probset(rng, n_cases=4, K=3, sources=("a", "b"), rows_per_case=2)
    fmap = fit_feature_map(ps, "rbf", 2, 0.5)
    for cal in (ScalarTS(0.6), ScalarTS(1.8), FieldBeta(fmap, np.array([1.1, 0.3, -0.4]))):
        for B in (0, 1, 2):
            rho = B / 4
            formula = robust_risk(ps, cal, rho)
            brute = brute_force_worst(ps, cal, B)
            for s in ps.sources():
                assert formula[s]["B"] == B
                assert abs(formula[s]["R_rho"] - brute[s]) < 1e-12


def test_identity_has_zero_risk(rng):
    ps = make_probset(rng, K=4)
    for rho in (0.0, 0.1, 0.5, 1.0):
        assert all(abs(v["R_rho"]) < 1e-15 for v in robust_risk(ps, IdentityCalibrator(), rho).values())


def test_risk_monotone_in_budget(rng):
    ps = make_probset(rng, K=4, n_cases=25)
    cal = ScalarTS(0.7)
    prev = None
    for rho in np.linspace(0, 1, 11):
        r = {s: v["R_rho"] for s, v in robust_risk(ps, cal, rho).items()}
        if prev:
            assert all(r[s] >= prev[s] - 1e-15 for s in r)
        prev = r


def test_rho_zero_matches_non_robust():
    tr = generate_scenario("a_shared_temperature", seed=1)[0]["source_train"]
    fr = fit_m4(tr, tr, M4Config(rho=0.0, function_class="scalar"))
    mrts, _ = fit_scalar(tr, beta_grid(), "minimax")
    assert fr.fitted.beta == mrts.beta


def test_budget_counts_states():
    assert state_budget(20, 0.1) == 2           # not 3 from float round-off
    assert state_budget(7, 0.0) == 0 and state_budget(7, 1.0) == 7 and state_budget(7, 0.15) == 2
    rng = np.random.default_rng(0)
    ps = make_probset(rng, n_cases=20, rows_per_case=3)
    _, info = corrupt_states(ps, 0.1, seed=0, by="state")
    assert abs(info["frac_states"] - 0.1) < 1e-12 and info["rows"] == 0.1 * 40 * 3
    _, info = corrupt_states(ps, 0.1, seed=0, by="row")
    assert info["frac_rows"] == 0.1 and info["frac_states"] >= 0.1


def test_field_robust_objective_nonpositive():
    tr = generate_scenario("a_shared_temperature", seed=2)[0]["source_train"]
    fr = fit_m4(tr, tr, M4Config(rho=0.1, function_class="field", J=2, sigma=0.5, lam=0.01))
    assert fr.solver_valid
    assert max(v["R_rho"] for v in fr.diagnostics["train_risk"].values()) <= 1e-7
