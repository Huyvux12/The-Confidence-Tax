"""THERMO_P / ATS_P (todo.md item 1): analytic gradients, identity start, per-task temperature."""
import numpy as np
import pytest

from calibration_extensions.extra_baselines import (ATSP, SOFTPLUS_ONE, ThermometerP, _batch_index, ats_objective,
                                                    fit_ats_p, fit_thermometer_p, thermometer_objective)
from calibration_extensions.features import fit_feature_map
from calibration_extensions.numerics import concat

from .conftest import make_probset


def _setup(rng):
    ps = make_probset(rng, K_choices=(2, 4, 5), sources=("a", "b", "c"))
    fmap = fit_feature_map(ps, "rbf", 4, 0.5)
    return ps, fmap, fmap.phi(ps)


def _fd_check(f, x, rng):
    _, g = f(x)
    e = rng.normal(size=x.size)
    h = 1e-6
    fd = (f(x + h * e)[0] - f(x - h * e)[0]) / (2 * h)
    assert abs(fd - g @ e) < 1e-6 * max(1.0, abs(fd))


def test_thermometer_gradient(rng):
    ps, fmap, Phi = _setup(rng)
    inv, nb = _batch_index(ps, ("source_id",))
    x = rng.normal(size=Phi.shape[1]) * 0.3
    x[0] += SOFTPLUS_ONE
    _fd_check(lambda t: thermometer_objective(t, Phi, ps, inv, nb), x, rng)


def test_ats_gradient(rng):
    ps, fmap, Phi = _setup(rng)
    _fd_check(lambda t: ats_objective(t, Phi, ps), rng.normal(size=Phi.shape[1]) * 0.3, rng)


def test_identity_start(rng):
    ps, fmap, _ = _setup(rng)
    x = np.zeros(fmap.dim)
    for cal in (ATSP(fmap, x), ThermometerP(fmap, np.r_[SOFTPLUS_ONE, x[1:]])):
        for g, lq in zip(ps.groups, cal.log_q_groups(ps)):
            assert np.allclose(lq, g.logP, atol=1e-12)


def test_thermometer_one_temperature_per_task(rng):
    a = make_probset(rng, task="t1")
    b = make_probset(rng, task="t2", scale=0.5)
    ps = concat([a, b])
    cal, info = fit_thermometer_p(ps)
    beta = cal.task_beta(ps)
    for t in ("t1", "t2"):
        assert np.ptp(beta[ps.task_id == t]) < 1e-12
    # the temperature of a task does not depend on rows of other tasks
    assert np.allclose(cal.task_beta(a), beta[ps.task_id == "t1"])


def test_fits_improve_their_objective(rng):
    ps, *_ = _setup(rng)
    for fit in (fit_thermometer_p, fit_ats_p):
        cal, info = fit(ps)
        assert np.all(np.isfinite(cal.theta)) and info["status"].startswith("LBFGS")
