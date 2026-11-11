# The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models

**Evo-RB — Evolutionary Regret-Budgeted Calibration**  
Research code, frozen model predictions, experiments, and results for *The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models* (EvoLearn 2027 submission).

Evo-RB is a post-hoc calibration method for multiple-choice language models. It uses an **outer CMA-ES** search to choose the structure (RBF centres and bandwidth) of an input-dependent temperature field, and an **inner SLSQP** optimization to fit coefficients under per-source NLL regret constraints. The underlying language model is not retrained.

> **Repository status (2026-11-11):** This is the original `main` branch, with historical research files still present. A publication-oriented cleanup is underway in [PR #1](../../pull/1) on branch `cleanup/evorb-paper-20261111`. The archived `paper/main.tex` is an **older RB-Cal manuscript**, and `paper_lncs/main.tex` is an **earlier Evo-RB draft**—neither should be mistaken for the author's latest 15-page PDF. The final manuscript's exact LaTeX source is not yet synchronized here.

## Main comparison: five methods

The manuscript compares **Pooled TS, Thermometer-P, ATS-P, Minimax TS, and Evo-RB**. Thermometer-P and ATS-P are probability-feature adaptations of the corresponding published methods, not exact implementations using their original hidden-state inputs. Fixed RBF/RB-Cal appears in the code as a development method and ablation rather than a sixth headline baseline.

| Method | Mean ΔNLL ↓ | ECE ↓ | Brier ↓ | Harm (ΔNLL > 0.01) ↓ |
|:--|--:|--:|--:|--:|
| Pooled TS | -0.490 | 0.078 | 0.431 | 58/500 |
| Thermometer-P | -0.493 | 0.075 | 0.430 | 49/500 |
| ATS-P | -0.487 | 0.078 | 0.432 | 74/500 |
| Minimax TS | -0.368 | 0.131 | 0.462 | 1/500 |
| **Evo-RB** | **-0.473** | **0.084** | **0.435** | **8/500** |

These numbers come from the committed frozen results in `runs/evo_study/aggregate_500.json`: five LLMs, two prompts, and 500 model–prompt–task evaluation units over five rosters (C1: 30, C2: 60, C3: 60, Hard: 100, LOTO: 250). Evo-RB trades a small amount of mean NLL gain for substantially fewer harmed units than the unconstrained baselines; **it is not the best method on every metric**.

## Start here: code and results

| File / folder | Role |
|:--|:--|
| [`scripts/evo_study.py`](scripts/evo_study.py) | **Evo-RB implementation**: CMA-ES over RBF centres/bandwidth, inner SLSQP, gating, evaluation and LOTO jobs |
| [`calibration_extensions/rb_calibration.py`](calibration_extensions/rb_calibration.py) | Constrained coefficient optimization and source regret calculations |
| [`calibration_extensions/features.py`](calibration_extensions/features.py) | Label-free feature maps and temperature fields |
| [`calibration_extensions/source_objectives.py`](calibration_extensions/source_objectives.py) | Scalar calibration objectives, Pooled TS and Minimax TS |
| [`calibration_extensions/extra_baselines.py`](calibration_extensions/extra_baselines.py) | Thermometer-P and ATS-P |
| [`scripts/v4_study.py`](scripts/v4_study.py) | Fixed-structure RBF experiments and ablation support |
| [`scripts/gate_study.py`](scripts/gate_study.py) | Source-side selection and deployment gate |
| [`scripts/base_study.py`](scripts/base_study.py) | Consistent baseline evaluation on shared data |
| [`runs/evo_study/`](runs/evo_study) | Frozen Evo-RB per-unit outputs, aggregate results, fronts, LOTO |
| [`runs/v4_study/`](runs/v4_study) | Frozen fixed-field and baseline results |
| [`data_gen/`](data_gen) | Benchmark construction, model probability extraction, pinned revisions |
| [`calibration_extensions/tests/`](calibration_extensions/tests) | Unit tests for core math and calibrators |

The method's **bilevel structure search is implemented in `scripts/evo_study.py::evo_rb_fit`**. The file `calibration_extensions/evo_calibration.py` is a separate CMA-ES coefficient/frontier diagnostic, not the whole Evo-RB method.

## Install and inspect

Python 3.13 was used for the original project. Run commands from the repository root:

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
python -m pytest calibration_extensions/tests -q
```

Existing numbers can be inspected directly:

```bash
python -c "import json; print(json.load(open('runs/evo_study/aggregate_500.json'))['aggregate']['EVO_RB'])"
```

For rebuilding raw model probabilities, source splits and calibrator fits, consult `data_gen/PROTOCOL_LOG.md`, `PREREGISTRATION.md` and the research scripts listed above. Raw probabilities, frozen records, and inference/build scripts are preserved on `main`. Additional large fitted artifacts are linked from the historical run documentation.

## Paper / code alignment notes

- **LOTO configuration caveat:** The frozen runs refit coefficients without the held-out source task, but the discrete RBF specification was selected globally using all 25 sources before the LOTO holdouts. This is *fixed-global-configuration LOTO*, not fully nested per-holdout hyperparameter selection. Do not claim full task exclusion at every selection stage without a new run.
- **Ablation provenance:** The latest controlled **No-margin** ablation results are not yet independently paired with a clearly identified frozen output artifact in this checkout.
- **Original paper sources:** `paper/` and `paper_lncs/` are previous manuscripts; the final submission source and figures must still be synchronized before claiming exact paper reproducibility.
- **Public access / anonymity:** The repository is currently private. Check reviewer access and the conference's double-blind requirements before putting a direct GitHub URL in the submission.

All numerical results shown above reflect **saved experimental records**, not newly rerun GPU experiments. The cleanup PR provides a narrower paper-focused tree and additional artifact verification scripts; do not merge until the remaining reproducibility gaps are addressed.

## Data and model licenses

Original benchmark/data licensing terms remain in force; see `data_gen/revisions.json`. No LLM weights are included in this repository.
