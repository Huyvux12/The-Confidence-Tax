"""Spec 10.1 - whole-system checks."""
import json

import numpy as np
import pytest

from calibration_extensions import data_adapter as da
from calibration_extensions.calibrators import (FieldBeta, IdentityCalibrator, ScalarTS, TopGap,
                                                load_fitted, save_fitted)
from calibration_extensions.evaluator import ece, row_metrics, score
from calibration_extensions.features import fit_feature_map
from calibration_extensions.m2_target_gain import M2Config, adapt_unlabeled, fit_m2
from calibration_extensions.m3_top_gap import M3Config, fit_m3
from calibration_extensions.numerics import (ProbSet, check_log_q, gather, losses, prepare_probs,
                                             source_deltas, surprisal)
from calibration_extensions.toy_data import generate_scenario

from .conftest import make_probset


def all_calibrators(ps):
    fmap = fit_feature_map(ps, "rbf", 3, 0.5)
    return [IdentityCalibrator(), ScalarTS(0.6), ScalarTS(2.5), TopGap(1.7, 0.4),
            FieldBeta(fmap, np.array([1.2, 0.3, -0.2, 0.1]))]


def edge_probset():
    probs = [[1 - 1e-13, 1e-13], [0.5, 0.5], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5],
             list(np.full(16, 1 / 16)), list(np.r_[0.99, np.full(15, 0.01 / 15)]), [0.0] * 15 + [1.0]]
    rows = [{"sample_id": f"e{i}", "case_id": f"c{i}", "model_id": "m", "prompt_id": "p", "source_id": "s",
             "task_id": "t", "probs": p, "label_ids": [str(k) for k in range(len(p))], "y": len(p) - 1}
            for i, p in enumerate(probs)]
    return da.rows_to_probset(rows, labelled=True)


def test_prepare_probs_floor_and_validation():
    lp, floored = prepare_probs([1.0, 0.0, 0.0])
    assert floored and np.all(np.isfinite(lp)) and abs(np.exp(lp).sum() - 1) < 1e-15
    for bad in ([0.5], [np.nan, 1.0], [-0.1, 1.1], [0.0, 0.0]):
        with pytest.raises(ValueError):
            prepare_probs(bad)


def test_identity_is_exact():
    ps = edge_probset()
    mats = IdentityCalibrator().log_q_groups(ps)
    assert all(np.array_equal(M, g.logP) for M, g in zip(mats, ps.groups))
    assert max(abs(v) for v in source_deltas(ps, losses(ps, mats)).values()) == 0.0


@pytest.mark.parametrize("K", [2, 3, 16])
def test_q_normalised_and_finite(rng, K):
    ps = make_probset(rng, K=K, scale=6.0)
    for cal in all_calibrators(ps):
        assert check_log_q(cal.log_q_groups(ps))["ok"], cal.kind


def test_edge_probabilities_all_methods():
    ps = edge_probset()
    for cal in all_calibrators(ps):
        c = check_log_q(cal.log_q_groups(ps))
        assert c["ok"], (cal.kind, c)


def test_permutation_invariance(rng):
    ps = make_probset(rng, K=5, scale=3.0)
    perm = rng.permutation(5)
    inv = np.argsort(perm)
    pp = ps.subset(np.arange(ps.n))
    pp.log_p = [lp[perm] for lp in ps.log_p]
    pp.y = np.array([int(inv[y]) for y in ps.y])
    for cal in all_calibrators(ps):
        a = cal.predict_log_proba(ps)
        b = cal.predict_log_proba(ProbSet(pp.sample_id, pp.case_id, pp.model_id, pp.prompt_id, pp.source_id,
                                          pp.task_id, pp.label_ids, pp.log_p, pp.y))
        assert max(np.max(np.abs(x - y[inv])) for x, y in zip(a, b)) < 1e-12, cal.kind


def test_save_load_roundtrip(rng, tmp_path):
    ps = make_probset(rng, K=4)
    for k, cal in enumerate(all_calibrators(ps)):
        save_fitted(cal, tmp_path / str(k))
        again = load_fitted(tmp_path / str(k))
        a, b = cal.predict_log_proba(ps), again.predict_log_proba(ps)
        assert max(np.max(np.abs(x - y)) for x, y in zip(a, b)) == 0.0


def test_data_adapter_roundtrip_and_private_labels(tmp_path):
    splits, target = generate_scenario("a_shared_temperature", seed=0, n_target_cases=20)
    da.write_dataset(tmp_path, splits, target, {"x": 1})
    tin = da.load_split(tmp_path, "target_inputs")
    assert not tin.labelled
    for line in (tmp_path / "target_inputs.jsonl").read_text().splitlines():
        assert "y" not in json.loads(line)
    back = da.attach_labels(tin, da.load_private_labels(tmp_path))
    assert np.array_equal(back.y, target.y)
    rows = da.read_jsonl(tmp_path / "target_inputs.jsonl")
    rows[0]["y"] = 0
    with pytest.raises(ValueError):
        da.rows_to_probset(rows, labelled=False)


def test_target_labels_do_not_change_fit_or_adapt(rng):
    """Shuffling private target labels cannot change M1-M4 fits (they never receive target) nor M2's beta."""
    splits, target = generate_scenario("m2_shifted_target", seed=3, n_target_cases=60)
    model = fit_m2(splits["source_train"], splits["source_select"], M2Config(n_boot=100))
    shuffled = target.with_labels(rng.permutation(target.y))
    b1 = adapt_unlabeled(model, target.without_labels()).fitted.betas
    b2 = adapt_unlabeled(model, shuffled.without_labels()).fitted.betas
    assert b1 == b2
    with pytest.raises(ValueError):
        adapt_unlabeled(model, target)


def test_seed_reproducible():
    splits, _ = generate_scenario("m3_top_gap", seed=2, n_target_cases=10)
    a = fit_m3(splits["source_train"], splits["source_select"], M3Config(lambdas=(0.0,)))
    b = fit_m3(splits["source_train"], splits["source_select"], M3Config(lambdas=(0.0,)))
    assert abs(a.fitted.a - b.fitted.a) < 1e-9 and abs(a.fitted.b - b.fitted.b) < 1e-9


def test_metrics_definitions():
    conf = np.array([0.95, 0.95, 0.05, 1.0])
    correct = np.array([1.0, 0.0, 0.0, 1.0])
    # bins: 0.95 -> 14, 0.05 -> 0, 1.0 -> 14 (closed); |mean conf - acc| weighted
    expect = (3 * abs((0.95 + 0.95 + 1.0) / 3 - 2 / 3) + 1 * 0.05) / 4
    assert abs(ece(conf, correct) - expect) < 1e-12
    ps = edge_probset()
    m = row_metrics(ps, IdentityCalibrator().predict_log_proba(ps))
    assert np.allclose(m["delta"], 0.0)
    q = np.exp(ps.log_p[3])
    assert abs(m["brier"][3] - ((q - np.eye(3)[2]) ** 2).sum()) < 1e-12
    rows = score(ps, IdentityCalibrator().predict_log_proba(ps), {"method_id": "IDENTITY"})
    assert rows[-2]["task_id"] == "__macro__" and rows[-1]["task_id"] == "__micro__"
    assert all(r["harm_001"] == 0 and r["harm_any"] == 0 for r in rows)
