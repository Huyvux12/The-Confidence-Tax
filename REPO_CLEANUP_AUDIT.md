# Paper-to-repository cleanup audit

Source baseline: `main` at commit `24fbe26099d2207792b766ebbc0f3276e762f95b`; compared with author-provided 15-page paper dated 2026-11-11. This branch is a **non-destructive cleanup proposal**, not a rewrite of experiments or claims.

## Essential code (retain)

| Paper claim/section | Code or evidence |
|---|---|
| Evo-RB bilevel search (Section 4) | `scripts/evo_study.py`, `scripts/v4_study.py` |
| Fixed field and constrained SLSQP inner fit | `calibration_extensions/rb_calibration.py`, `features.py`, `numerics.py`, `source_splits.py`, `source_objectives.py` |
| Baselines Pooled TS, Minimax TS, Thermometer-P, ATS-P | `calibration_extensions/source_objectives.py`, `extra_baselines.py`; `scripts/run_rbcal.py`, `run_extra_baselines.py` |
| Source-only gate / selection | `scripts/gate_study.py`, `runs/gate_study/hypers.json`, `runs/v4_study/chosen.json` |
| LOTO and five rosters, 500 units | `scripts/evo_study.py`, `scripts/v4_study.py`, `runs/evo_study/`, `runs/v4_study/`, `runs/rbcal_confirm*/`, `runs/loto/` |
| Raw model probabilities and inputs | `data_gen/raw*/` (except redundant `*_check` copies), `data/real/dev/`, `data_gen/items*.jsonl`, build/split scripts |
| Prior manuscript/figures | `paper/`, `paper_lncs/` (historic versions; not exact latest PDF) |

`calibration_extensions/evo_calibration.py` implements a separate CMA-ES coefficient/frontier diagnostic used by `scripts/evo_study.py`; do **not** delete it on the assumption that its name indicates the current bilevel Evo-RB implementation. The latter is in `scripts/evo_study.py`.

## Removed from tracking on this branch

- `runs/toy_m1/` ... `runs/toy_m5/`: generated outputs from the five earlier prototype directions; the toy **source code, configs and tests** remain.
- `runs/rbcal_dev_reuse/`: alternative/repeated cached unit fits, not the canonical `runs/rbcal_dev/`. These caches are **not byte-identical**; preserve the original commit for provenance.
- `data_gen/raw_confirm3_check/` and `data_gen/raw_confirm4_check/`: repeated small-sample check files, byte-identical by Git blob SHA to the five corresponding files in `data_gen/raw_confirm2_check/`. Full `raw_confirm{2,3,4}/` files remain.
- Generated `*.log` files. Historical protocol decisions remain in `data_gen/PROTOCOL_LOG.md`, `PREREGISTRATION.md`, and existing results/manifests.

Total: 342 tracked files, 161,900,960 uncompressed bytes (~161.9 MB), removed only from this new branch; originals remain in the base commit and Git history.

## Important outstanding work before public release

1. **Synchronize LaTeX.** The author-provided 15-page paper differs from checked-in `paper_lncs/main.tex`; its exact LaTeX, figures and table generator need adding before claiming one-command reproducibility of the latest PDF.
2. **Build paper-specific artifacts.** Check/generate the latest rescue plot and controlled margin ablation from frozen per-unit results. Current `scripts/make_lncs_assets.py` generates the **earlier** LNCS manuscript, not all current figures.
3. **Add direct regression tests for Evo-RB.** Current `calibration_extensions/tests/` covers core and legacy modules, but no dedicated end-to-end test of the evolutionary outer loop plus SLSQP inner loop and gate is evident.
4. **Then prune old Python methods.** Do not delete M1–M5 modules or `calibration_extensions/run_experiment.py` casually: the latter imports all five, and `scripts/run_rbcal.py` also imports M1 diagnostics.
5. **Verify:** `python -m pytest calibration_extensions/tests -q` and regenerate the existing LNCS assets using the original documented environment. Tests were **not run** in the GitHub-connected audit environment; changes in this branch are limited to tracked research outputs and documentation.

Nothing in the scientific source code, experimental numerical files used for the latest paper, or calculation formulas is changed by this cleanup branch.
