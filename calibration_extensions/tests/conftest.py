import numpy as np
import pytest

from calibration_extensions.numerics import ProbSet, prepare_probs


def make_probset(rng, n_cases=30, K=4, sources=("s1", "s2"), rows_per_case=2, scale=2.0,
                 labelled=True, K_choices=None, task="t") -> ProbSet:
    cols = {k: [] for k in ("sample_id", "case_id", "model_id", "prompt_id", "source_id", "task_id")}
    log_p, label_ids, ys = [], [], []
    for s in sources:
        for c in range(n_cases):
            k = int(rng.choice(K_choices)) if K_choices else K
            for r in range(rows_per_case):
                lp, _ = prepare_probs(np.exp(rng.normal(size=k) * scale))
                for f, v in zip(cols, (f"{s}_{c}_{r}", f"{s}_{c}", "m", "p", s, task)):
                    cols[f].append(v)
                log_p.append(lp)
                label_ids.append([f"o{j}" for j in range(k)])
                ys.append(int(rng.integers(k)))
    return ProbSet(**{k: np.array(v, dtype=object) for k, v in cols.items()}, label_ids=label_ids,
                   log_p=log_p, y=np.array(ys) if labelled else None)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
