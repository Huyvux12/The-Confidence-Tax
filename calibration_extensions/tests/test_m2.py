"""Spec 10.3 - M2."""
import numpy as np
import pytest

from calibration_extensions.m2_target_gain import (M2Config, adapt_unlabeled, choose_beta, fit_m2, mean_lse,
                                                   predict_interval, upper_delta)
from calibration_extensions.numerics import surprisal
from calibration_extensions.source_objectives import beta_grid, scalar_losses
from calibration_extensions.toy_data import generate_scenario

from .conftest import make_probset


@pytest.mark.parametrize("K", [2, 3, 7])
def test_delta_identity_every_label(rng, K):
    ps = make_probset(rng, K=K, n_cases=10)
    for beta in (0.3, 0.9, 1.0, 1.7, 3.5):
        for label in range(K):
            pl = ps.with_labels(np.full(ps.n, label))
            lhs = scalar_losses(pl, beta) - surprisal(pl)
            A = np.array([np.log(np.sum(np.exp(beta * lp))) for lp in pl.log_p])
            assert np.max(np.abs(lhs - ((beta - 1) * surprisal(pl) + A))) < 1e-12


def test_upper_bound_is_worst_endpoint(rng):
    for _ in range(200):
        a = rng.normal()
        L, U = sorted(rng.uniform(0, 3, size=2))
        beta = rng.uniform(0.25, 4.0)
        ub = upper_delta(a, beta, L, U)
        Ms = np.linspace(L, U, 51)
        assert abs(ub - max(a + (beta - 1) * Ms)) < 1e-12


def test_swapped_endpoints_are_detected():
    """Using U below 1 (a wrong implementation) underestimates the worst case."""
    a, L, U, beta = 0.0, 0.5, 2.0, 0.6
    wrong = a + (beta - 1) * U
    true_worst = max(a + (beta - 1) * L, a + (beta - 1) * U)
    assert wrong < true_worst
    assert upper_delta(a, beta, L, U) == true_worst


def test_unsupported_k_falls_back():
    splits, target = generate_scenario("m2_shifted_target", seed=0, K=4, n_target_cases=40)
    model = fit_m2(splits["source_train"], splits["source_select"], M2Config(n_boot=100))
    _, t3 = generate_scenario("m2_shifted_target", seed=1, K=3, n_target_cases=40)
    fr = adapt_unlabeled(model, t3.without_labels())
    assert all(v["status"] == "UNSUPPORTED_K" for v in fr.diagnostics["intervals"].values())
    assert all(b == 1.0 for b in fr.fitted.betas.values())


def test_unsupported_bins_use_raw_bounds():
    splits, target = generate_scenario("m2_shifted_target", seed=0, n_target_cases=40)
    model = fit_m2(splits["source_train"], splits["source_select"], M2Config(n_boot=100, min_cases=10 ** 6))
    env = model.envelopes[4]
    assert not env.supported.any()
    itv = predict_interval(env, target.without_labels(), "bootstrap_envelope")
    assert itv["unsupported_mass"] == 1.0
    assert itv["raw_lower"] - 1e-12 <= itv["m_lower"] <= itv["m_upper"] <= itv["raw_upper"] + 1e-12


def test_choose_beta_keeps_identity_when_no_gain(rng):
    ps = make_probset(rng, K=4, n_cases=20, labelled=False)
    lo = np.mean([-lp.max() for lp in ps.log_p])
    hi = np.mean([-lp.min() for lp in ps.log_p])
    beta, rows = choose_beta(ps, {"m_lower": lo, "m_upper": hi}, beta_grid(), "bootstrap_envelope")
    assert beta == 1.0
    assert all(r["upper_delta"] >= -1e-12 for r in rows)


def test_bound_holds_when_interval_covers():
    splits, target = generate_scenario("m2_shifted_target", seed=5, n_target_cases=80)
    model = fit_m2(splits["source_train"], splits["source_select"], M2Config(n_boot=200))
    fr = adapt_unlabeled(model, target.without_labels())
    for key, itv in fr.diagnostics["intervals"].items():
        idx = np.array([f"{target.model_id[i]}|{target.prompt_id[i]}|{target.task_id[i]}" == key for i in range(target.n)])
        b = target.subset(idx)
        true_m = surprisal(b).mean()
        beta = fr.fitted.betas[key]
        true_delta = mean_lse(b, beta) + (beta - 1) * true_m
        if itv["m_lower"] <= true_m <= itv["m_upper"]:
            assert true_delta <= upper_delta(mean_lse(b, beta), beta, itv["m_lower"], itv["m_upper"]) + 1e-12
