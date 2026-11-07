"""Phase 8: answer-order consistency features (features.order_tv, rbf_oc / rbf_oct field)."""
import numpy as np

from calibration_extensions.calibrators import FieldBeta
from calibration_extensions.features import (FeatureMap, batch_key, beta_bound_slack, features_for, fit_feature_map,
                                             order_tv, project_to_bounds, task_mean)
from calibration_extensions.numerics import ProbSet


def make(rows):
    """rows: (sample_id, case_id, source_id, task_id, label_ids, probs)."""
    return ProbSet(sample_id=np.array([r[0] for r in rows]), case_id=np.array([r[1] for r in rows]),
                   model_id=np.array(["m"] * len(rows)), prompt_id=np.array(["p"] * len(rows)),
                   source_id=np.array([r[2] for r in rows]), task_id=np.array([r[3] for r in rows]),
                   label_ids=[r[4] for r in rows], log_p=[np.log(np.asarray(r[5], float)) for r in rows])


def test_tv_zero_when_orders_agree_semantically():
    # order 1 shows options reversed; same semantic distribution -> tv 0
    ps = make([("a::o0", "a", "s", "s", ["0", "1", "2"], [0.7, 0.2, 0.1]),
               ("a::o1", "a", "s", "s", ["2", "1", "0"], [0.1, 0.2, 0.7])])
    tv, unpaired = order_tv(ps)
    assert unpaired == 0 and np.allclose(tv, 0.0, atol=1e-12)


def test_tv_value_and_symmetry():
    ps = make([("a::o0", "a", "s", "s", ["0", "1"], [0.9, 0.1]),
               ("a::o1", "a", "s", "s", ["1", "0"], [0.6, 0.4])])     # semantic: [0.4, 0.6]
    tv, _ = order_tv(ps)
    assert np.allclose(tv, [0.5, 0.5])
    swapped = ps.subset([1, 0])
    assert np.allclose(order_tv(swapped)[0], [0.5, 0.5])


def test_unpaired_rows_get_zero():
    ps = make([("a::o0", "a", "s", "s", ["0", "1"], [0.9, 0.1]),
               ("b::o0", "b", "s", "s", ["0", "1"], [0.5, 0.5]),
               ("b::o1", "b", "s", "s", ["1", "0"], [0.5, 0.5])])
    tv, unpaired = order_tv(ps)
    assert unpaired == 1 and tv[0] == 0.0


def test_task_mean_uses_source_then_task():
    ps = make([("a::o0", "a", "s1", "s1", ["0", "1"], [0.9, 0.1]), ("a::o1", "a", "s1", "s1", ["1", "0"], [0.6, 0.4]),
               ("b::o0", "b", "", "t1", ["0", "1"], [0.5, 0.5]), ("b::o1", "b", "", "t1", ["1", "0"], [0.5, 0.5])])
    keys = batch_key(ps)
    assert list(keys) == ["s1", "s1", "t1", "t1"]
    tv, _ = order_tv(ps)
    assert np.allclose(task_mean(tv, keys), [0.5, 0.5, 0.0, 0.0])
    z = features_for(ps, "rbf_oct")
    assert z.shape == (4, 5) and np.allclose(z[:, 3], tv) and np.allclose(z[:, 4], [0.5, 0.5, 0, 0])
    assert features_for(ps, "rbf").shape == (4, 3)


def random_pairs(n, K, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n):
        p = rng.dirichlet(np.ones(K) * 0.5)
        perm = rng.permutation(K)
        q = np.clip(p[perm] + rng.normal(0, 0.05, K), 1e-6, None)
        q /= q.sum()
        src = f"s{c % 3}"
        rows.append((f"{c}::o0", str(c), src, src, [str(j) for j in range(K)], p))
        rows.append((f"{c}::o1", str(c), src, src, [str(j) for j in perm], q))
    return make(rows)


def test_field_identity_and_bounds():
    ps = random_pairs(60, 4, 0)
    for kind in ("rbf_oc", "rbf_oct"):
        fmap = fit_feature_map(ps, kind, 4, 0.5)
        assert fmap.kind == kind and fmap.centers.shape == (4, 4 if kind == "rbf_oc" else 5)
        cal = FieldBeta(fmap, fmap.identity_theta())
        for g, M in zip(ps.groups, cal.log_q_groups(ps)):
            assert np.allclose(M, g.logP, atol=1e-12)
        rng = np.random.default_rng(1)
        theta = np.r_[2.0, rng.normal(0, 5, 4)]
        theta = project_to_bounds(fmap, theta, 0.25, 4.0)
        assert beta_bound_slack(fmap, theta, 0.25, 4.0) >= -1e-12
        beta = fmap.beta(ps, theta)
        assert beta.min() >= 0.25 - 1e-9 and beta.max() <= 4.0 + 1e-9


def test_old_kinds_unchanged():
    ps = random_pairs(30, 3, 2)
    fmap = fit_feature_map(ps, "rbf", 3, 0.5)
    assert fmap.centers.shape == (3, 3) and FeatureMap("rbf_oc", None, 0.5).kind == "constant"


def test_precomputed_tv_and_peer_fields():
    ps = random_pairs(20, 3, 5)
    tv_computed, _ = order_tv(ps)
    ps.extra["order_tv_raw"] = list(np.full(ps.n, 0.3))
    assert np.allclose(order_tv(ps)[0], 0.3)                 # derived (order-averaged) data use the stored value
    import pytest
    with pytest.raises(ValueError):
        features_for(ps, "rbf_ocp")                          # peer field missing
    ps.extra["peer_top"] = list(np.linspace(0, 1, ps.n))
    z = features_for(ps, "rbf_ocp")
    assert z.shape == (ps.n, 7) and np.allclose(z[:, 5], np.linspace(0, 1, ps.n))
    keys = batch_key(ps)
    assert np.allclose(z[:, 6], task_mean(np.linspace(0, 1, ps.n), keys))
    fmap = fit_feature_map(ps, "rbf_ocp", 3, 0.5)
    assert fmap.centers.shape == (3, 7)
    sub = ps.subset(np.arange(10))
    assert np.allclose(features_for(sub, "rbf_ocp")[:, 5], np.linspace(0, 1, ps.n)[:10])
    assert not np.allclose(tv_computed, 0.3)
