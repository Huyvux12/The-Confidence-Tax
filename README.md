# The Confidence Tax — Evo-RB

**Research artifact for EvoLearn 2027:** *The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models*.

Evo-RB is a source-only, post-hoc calibration method for multiple-choice language models. An outer **CMA-ES** search selects the centres and bandwidth of an input-dependent radial-basis inverse-temperature field; an inner **SLSQP** optimization fits its coefficients subject to **per-source NLL regret budgets** and an empirical uncertainty margin. The base LLM remains frozen.

> **Status, 11 Nov 2026.** The manuscript supplied by the authors contains 15 LNCS pages and 500 held-out model–prompt–task evaluation units (five models × two prompts × 50 task/roster occurrences). This repository contains the frozen records supporting the aggregate/roster comparisons. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for provenance, a LOTO hyperparameter-selection caveat, and components not yet accompanied by standalone ablation output files. Do not interpret the source-side empirical margin as a target-distribution safety guarantee.

## Reported results

| Method | TL-ECE ↓ | Brier ↓ | Mean ΔNLL ↓ | Harm (ΔNLL > 0.01) ↓ |
|:--|--:|--:|--:|--:|
| Pooled TS | 0.078 | 0.431 | −0.490 | 58/500 |
| Thermometer-P | 0.075 | 0.430 | −0.493 | 49/500 |
| ATS-P | 0.078 | 0.432 | −0.487 | 74/500 |
| Minimax TS | 0.131 | 0.462 | −0.368 | 1/500 |
| **Evo-RB** | **0.084** | **0.435** | **−0.473** | **8/500** |

This is a measured **gain–harm trade-off**, not dominance on every metric. Thermometer-P and ATS-P are probability-interface adaptations, **not** exact hidden-state reproductions of their published implementations.

## Repository map — start here

| Path | Purpose |
|:--|:--|
| `scripts/evo_study.py` | **Evo-RB**: outer CMA-ES centre/bandwidth search; inner constrained SLSQP fit; gated target evaluation |
| `calibration_extensions/rb_calibration.py` | Regret budgets, clustered source uncertainty, parameter optimization and model selection |
| `calibration_extensions/features.py` | Label-free entropy, top probability, margin and answer-order consistency features |
| `scripts/v4_study.py`, `scripts/gate_study.py` | Fixed-field ablation and source-side selection/audit variants |
| `calibration_extensions/extra_baselines.py`, `scripts/base_study.py` | Thermometer-P/ATS-P baseline implementations |
| `calibration_extensions/source_objectives.py` | Pooled TS and Minimax TS fitting |
| `runs/evo_study/`, `runs/v4_study/` | Frozen per-unit results underlying the manuscript's principal aggregate, roster and paired-LOTO analyses |
| `data_gen/`, `data/real/dev/` | Pinned benchmark items, inference and source splits; target splits rebuild deterministically |
| `paper_submission/` | Exact-author-source provenance, publication-facing reproduction instructions and figure regeneration |
| `scripts/verify_paper_results.py` | Independent, standard-library-only numerical checks of the paper tables and paired rescues |
| `scripts/build_submission_figures.py` | Rebuild manuscript figures from committed frozen records |

The historical RB-Cal drafts, prototype experiment runs, and duplicate verification files were **removed from the publication branch**, not erased from Git history. The dated decisions in `PREREGISTRATION.md` and `data_gen/PROTOCOL_LOG.md` remain for transparent provenance.

## Quick verification (no GPU, no model weights)

```bash
python scripts/verify_paper_results.py
```

This checks the 500-unit aggregate results, five roster breakdowns and paired LOTO negative-transfer rescues against the author's latest paper. It does **not** prove that every figure/ablation is independently reproducible, and it does not rerun LLM inference or CMA-ES fitting.

## Controlled Margin vs No-margin ablation

The author-supplied script is included at `scripts/run_fixed_margin_control.py`. It fixes the RBF specification (`runs/v4_study/chosen.json`) and compares `z=1.645` with `z=0` in the **inner source-regret constrained fit**, keeping the same `G-sig` deployment gate for both cases:

```bash
python scripts/run_fixed_margin_control.py --workers 4 --out runs/fixed_margin_control.jsonl
```

This is a fixed-field ablation, **not a separate CMA-ES Evo-RB run**. The full 250-unit output was **not provided with the source script** and remains to be verified before linking the manuscript's controlled No-margin numbers to the GitHub artifact.

## Installation and figure regeneration

```bash
python -m venv .venv
# Activate it, then:
python -m pip install -r requirements.txt
python scripts/build_submission_figures.py
python -m pytest calibration_extensions/tests -q
```

The generated figure files appear in `paper_submission/`. Read `paper_submission/README.md` before compiling the paper.

## Large fitted artifacts — Hugging Face

The full large per-unit fitted outputs for the **confirm2, confirm3 and confirm4 rosters** are hosted in the project's Hugging Face dataset:

**[huyvux3005/rbcal-confidence-tax-runs](https://huggingface.co/datasets/huyvux3005/rbcal-confidence-tax-runs)**

These 30 fitted files (about 765 MB as documented by the original repository) are **not meant to be committed to GitHub**. They were produced in the original RB-Cal/confirmation fitting pipeline and retained for reproducibility of the downstream Evo-RB comparison. Download them only when a full refit/artifact audit needs these `runs/rbcal_confirm{2,3,4}/units/` caches:

```bash
python -m pip install huggingface_hub
hf download huyvux3005/rbcal-confidence-tax-runs --repo-type dataset --local-dir .
sha256sum -c runs/HF_SHA256SUMS.txt
```

The existing `runs/HF_SHA256SUMS.txt` contains checksums for all 30 files. **No data were removed from the Hugging Face dataset during this GitHub cleanup**; only the README link was accidentally omitted during the earlier rewrite. The dataset's present remote availability/content has not been independently checked in this cleanup, so use the SHA-256 command to confirm the downloaded files match the original run.

## From-scratch pipeline

1. Rebuild items from pinned dataset revisions: `data_gen/build_items.py` and the `data_gen/build_items_confirm*.py` variants.
2. Infer five open-weight LLMs and both answer orders with `data_gen/infer_mcq.py` (GPU; original runs are already cached in `data_gen/raw*/`).
3. Recreate source and target splits with `data_gen/make_splits.py` and `data_gen/make_confirm2.py` plus the other cohort scripts.
4. Fit the source-side baselines and frozen field configurations with `scripts/run_rbcal.py`, `scripts/run_extra_baselines.py`, `scripts/gate_study.py` and `scripts/v4_study.py`.
5. Run the outer Evo-RB search with `scripts/evo_study.py` through `scripts/modal_v4.py` or a comparable job runner, then verify recorded results.

**Reproducing the original numeric run** means retaining its frozen source-only hyperparameters and seeds; **correcting the strict LOTO hyperparameter-selection issue requires rerunning those experiments** and may change the paper numbers. Do not combine the two protocols.

## Release / access

The repository may currently be **private**. Before including its URL in an anonymous submission, check both the conference anonymity policy and GitHub's repository visibility/access settings. Do not place identifying account names, tokens, private URLs or author metadata in the anonymous paper artifact.

License note: benchmark texts, labels and model terms follow their original licenses. No LLM weights are distributed.
