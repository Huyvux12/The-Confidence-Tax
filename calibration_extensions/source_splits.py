"""Case-grouped splits, leave-one-source-out folds and calibrator units (spec 2.4)."""
from __future__ import annotations

import numpy as np

from .numerics import ProbSet


def leave_one_source_out(ps: ProbSet, heldout: str) -> tuple[ProbSet, ProbSet]:
    mask = np.array([str(s) == heldout for s in ps.source_id])
    return ps.subset(~mask), ps.subset(mask)


def loso_folds(ps: ProbSet):
    for s in ps.sources():
        yield s, *leave_one_source_out(ps, s)


def calibrator_units(ps: ProbSet) -> list[tuple[str, str]]:
    """One calibrator per model_id x prompt_id; models are never pooled."""
    return sorted({(str(m), str(p)) for m, p in zip(ps.model_id, ps.prompt_id)})


def unit_subset(ps: ProbSet, unit: tuple[str, str]) -> ProbSet:
    m, p = unit
    return ps.subset(np.array([str(a) == m and str(b) == p for a, b in zip(ps.model_id, ps.prompt_id)]))


def split_by_case(ps: ProbSet, fractions: dict[str, float], seed: int) -> dict[str, ProbSet]:
    """Random split keeping every row of a case on the same side."""
    cases = np.array(sorted({str(c) for c in ps.case_id}))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    names = list(fractions)
    cuts = np.cumsum([fractions[n] for n in names])
    cuts = np.round(cuts / cuts[-1] * len(cases)).astype(int)
    assign, start = {}, 0
    for name, stop in zip(names, cuts):
        for c in cases[start:stop]:
            assign[c] = name
        start = stop
    labels = np.array([assign[str(c)] for c in ps.case_id])
    return {n: ps.subset(labels == n) for n in names}


def assert_case_disjoint(*sets: ProbSet) -> None:
    seen: dict[str, int] = {}
    for k, s in enumerate(sets):
        for c in {str(c) for c in s.case_id}:
            if c in seen and seen[c] != k:
                raise ValueError(f"case {c} appears in more than one split")
            seen[c] = k


def cases_of(ps: ProbSet) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, c in enumerate(ps.case_id):
        out.setdefault(str(c), []).append(i)
    return {c: np.array(v) for c, v in out.items()}
