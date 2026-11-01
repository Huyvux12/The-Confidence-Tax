"""Spec 10.2 - M1."""
import numpy as np
import pytest

from calibration_extensions.calibrators import FieldBeta
from calibration_extensions.features import (FeatureMap, beta_bound_slack, fit_feature_map, prob_features,
                                             select_centers)
from calibration_extensions.m1_functional import (M1Config, diagnose_at_identity, fit_head, fit_m1,
                                                  simplex_qp)
from calibration_extensions.numerics import losses, temper
from calibration_extensions.source_objectives import beta_grid, calibrator_deltas, fit_scalar, scalar_losses
from calibration_extensions.toy_data import generate_scenario

from .conftest import make_probset

CFG = M1Config()


def test_dloss_dbeta_matches_finite_difference(rng):
    ps = make_probset(rng, K=5, scale=2.0)
    for beta0 in (1.0, 0.7, 2.3):
        h = 1e-6
        fd = (scalar_losses(ps, beta0 + h) - scalar_losses(ps, beta0 - h)) / (2 * h)
        an = np.empty(ps.n)
        for g in ps.groups:
            q = np.exp(temper(g.logP, beta0))
            an[g.idx] = (q * g.logP).sum(axis=1) - g.logP[np.arange(len(g.idx)), g.y]
        assert np.max(np.abs(fd - an)) < 1e-6


@pytest.mark.parametrize("kind", ["constant", "entropy_affine", "rbf"])
def test_source_gradient_at_identity(rng, kind):
    ps = make_probset(rng, K=4, sources=("a", "b", "c"))
    d = diagnose_at_identity(ps, fit_feature_map(ps, kind, 4, 0.5), CFG)
    assert d["gradient_check_max_abs_error"] < 1e-6


def test_gradient_at_interior_point(rng):
    ps = make_probset(rng, K=4)
    fmap = fit_feature_map(ps, "rbf", 3, 0.5)
    theta = np.array([1.3, 0.2, -0.3, 0.1])
    Phi = fmap.phi(ps)
    beta = Phi @ theta
    dl = np.empty(ps.n)
    for g in ps.groups:
        q = np.exp(temper(g.logP, beta[g.idx]))
        dl[g.idx] = (q * g.logP).sum(axis=1) - g.logP[np.arange(len(g.idx)), g.y]
    grad = (dl[:, None] * Phi).mean(axis=0)
    e = rng.normal(size=4)
    e /= np.linalg.norm(e)
    h = 1e-6
    lp = losses(ps, FieldBeta(fmap, theta + h * e).log_q_groups(ps)).mean()
    lm = losses(ps, FieldBeta(fmap, theta - h * e).log_q_groups(ps)).mean()
    assert abs((lp - lm) / (2 * h) - grad @ e) < 1e-6


def test_norm_bound_is_global(rng):
    centers = rng.uniform(size=(8, 3))
    fmap = FeatureMap("rbf", centers, 0.3)
    z = rng.uniform(size=(20000, 3))
    for _ in range(50):
        b0 = rng.uniform(0.25, 4.0)
        v = rng.normal(size=8)
        v *= rng.uniform(0, 1) * min(b0 - 0.25, 4.0 - b0) / np.linalg.norm(v)
        theta = np.r_[b0, v]
        assert beta_bound_slack(fmap, theta, 0.25, 4.0) >= -1e-12
        beta = fmap.phi_from_z(z) @ theta
        assert beta.min() >= 0.25 - 1e-12 and beta.max() <= 4.0 + 1e-12


def test_centers_deterministic_and_tie_broken_by_sample_id(rng):
    ps = make_probset(rng, K=4)
    z = prob_features(ps)
    a = select_centers(z, ps.sample_id, 5)
    perm = rng.permutation(ps.n)
    b = select_centers(z[perm], ps.sample_id[perm], 5)
    assert np.array_equal(a, b)


def test_simplex_qp_certificate():
    g = np.array([[1.0, 0.0], [-1.0, 0.0]])          # opposite: no common descent
    a, info = simplex_qp(g @ g.T)
    assert np.linalg.norm(a @ g) < 1e-8 and info["sum_alpha_error"] < 1e-12
    g = np.array([[1.0, 0.2], [0.8, -0.3], [1.0, 0.5]])
    a, info = simplex_qp(g @ g.T)
    gbar = a @ g
    d = -gbar / np.linalg.norm(gbar)
    assert np.all(g @ d <= -np.linalg.norm(gbar) + 1e-8)
    assert info["kkt_error"] < 1e-8


def test_constant_class_matches_scalar_ts():
    tr = generate_scenario("a_shared_temperature", seed=4)[0]["source_train"]
    fmap = fit_feature_map(tr, "constant")
    grid = beta_grid(n=2001)
    for obj, rule in (("POOLED", "pooled"), ("MINIMAX", "minimax")):
        cal, _ = fit_head(tr, fmap, obj, 0.0, CFG)
        ts, _ = fit_scalar(tr, grid, rule)
        # continuous optimum vs grid optimum: differences bounded by grid resolution
        assert abs(np.log(cal.theta[0]) - np.log(ts.beta)) < 2 * np.diff(np.log(grid)).max()
        d_cont = calibrator_deltas(tr, cal)
        d_grid = calibrator_deltas(tr, ts)
        f = (lambda d: max(d.values())) if obj == "MINIMAX" else (lambda d: np.mean(list(d.values())))
        assert f(d_cont) <= f(d_grid) + 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_affine_resolvable_toy(seed):
    tr = generate_scenario("b_entropy_conflict", seed=seed)[0]["source_train"]
    const = diagnose_at_identity(tr, fit_feature_map(tr, "constant"), CFG)
    affine = diagnose_at_identity(tr, fit_feature_map(tr, "entropy_affine"), CFG)
    assert not const["common_descent"]
    assert affine["status"] == "COMMON_DESCENT_VERIFIED"
    assert all(x < -1e-8 for x in affine["g_w_dot_d"])
    assert all(v < 0 for v in affine["step_check"]["deltas"].values())


def test_minimax_never_worse_than_identity_on_train():
    tr = generate_scenario("b_entropy_conflict", seed=0)[0]["source_train"]
    for kind in ("constant", "entropy_affine", "rbf"):
        fmap = fit_feature_map(tr, kind, 4, 0.5)
        cal, info = fit_head(tr, fmap, "MINIMAX", 0.01, CFG)
        assert info["valid"] and max(info["train_deltas"].values()) <= 1e-7
        assert info["bound_slack"] >= -1e-12


def test_unidentifiable_toy_reports_reason():
    splits, _ = generate_scenario("d_unidentifiable", seed=0)
    cfg = M1Config(centers=(0, 2), bandwidth=(0.5,), field_l2=(0.01,))
    fr = fit_m1(splits["source_train"], splits["source_select"], cfg, diagnose_classes=True)
    assert fr.solver_valid
    assert not fr.diagnostics["class_constant"]["common_descent"]
    if not fr.source_gate["accept"]:
        assert fr.fallback_reason == "SOURCE_GATE_REJECTED" and fr.deployed.kind == "identity"
    assert fr.diagnostic_scope.startswith("observed_source_loss_geometry")


@pytest.mark.parametrize("kind,J", [("constant", 0), ("entropy_affine", 0), ("rbf", 4)])
@pytest.mark.parametrize("objective", ["MINIMAX", "MEAN_SAFE", "POOLED"])
def test_slsqp_matches_cvxpy(kind, J, objective):
    tr = generate_scenario("b_entropy_conflict", seed=1)[0]["source_train"]
    fmap = fit_feature_map(tr, kind, J, 0.5)
    a, ia = fit_head(tr, fmap, objective, 0.01, M1Config(solver="scipy"))
    b, ib = fit_head(tr, fmap, objective, 0.01, M1Config(solver="cvxpy"))
    assert ia["valid"] and ib["valid"]
    f = (lambda d: max(d.values())) if objective == "MINIMAX" else (lambda d: np.mean(list(d.values())))
    assert abs(f(ia["train_deltas"]) - f(ib["train_deltas"])) < 1e-5
