# Reproducibility and paper alignment (12 Nov 2026)

## Manuscript identity

The current **corrected** paper is **The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models**, 15 Springer LNCS pages. Its separate submission source archive is `EvoRB_LOTO_Corrected_Overleaf_Source.zip`; the corrected `main.tex` has SHA-256:

`3070c25bcd5648decc6e1d826208bd211dba9c3911c63756c8ae8b8bd8f16004`.

This public code repository does **not** contain that anonymous submission source archive. Keep the separate Overleaf ZIP as the authority for the submission. An earlier, now superseded Overleaf package was `Confidence_Tax_EvoRB_Table2_TwoTier_Fig3_FIXED_15pages_Overleaf.zip` (SHA-256 of its `main.tex`: `691dfda7772e5e288d3830c547f296866cd74b84742154a93be687499e9ddc2a`). Do not substitute historical `paper_lncs/main.tex` or generated plot replacements for the final submission files.

## Exactly supported by frozen outputs

- Table 1: five baseline/Evo-RB methods plus identity, macro TL-ECE/Brier/ΔNLL and 8/500 harm.
- Table 2: confirmation rosters C1 (30), C2 (60), C3 (60), Hard (100), LOTO (250).
- Figure 1 and Figure 2 operating points: computed from `runs/evo_study/aggregate_500.json`, rechecked against per-unit records by `scripts/build_submission_figures.py`.
- Figure 3 paired rescue: `runs/v4_study/base_loto_aug.jsonl` and `runs/evo_study/evo_loto.jsonl`; counts independently verified by `scripts/verify_paper_results.py`.

Main five-method comparison: Pooled TS, Thermometer-P, ATS-P, Minimax TS, Evo-RB. The fixed regret-budgeted field V4B is **an internal ablation**, not a sixth headline baseline. Neither Thermometer-P nor ATS-P uses the hidden states of the original published method.

## LOTO protocol: globally selected specification and fold-specific refitting

The corrected manuscript (12 Nov 2026) explicitly distinguishes **globally selected hyperparameters** from **fold-specific fitting and auditing**. In earlier wording, the held-out task had incorrectly been described as excluded from all hyperparameter selection. In the committed experimental path, `scripts/evo_study.py::jobs_for("evo_loto")` loads the per-model/prompt `v4b` specification from `runs/v4_study/chosen.json`; `scripts/v4_study.py::jobs_for("loto")` uses that same global choice. That choice was made using all 25 source tasks before each LOTO holdout was designated. The actual LOTO **coefficient fitting** excludes the held-out task, and the held-out task is excluded from the source-selection/audit statistics for those fits, but the discrete specification has prior information from the held-out task.

Therefore the frozen LOTO numbers support **globally selected configuration with held-out task refitting**, **not nested hyperparameter selection**. The corrected manuscript uses that accurate description. A separate, unmerged draft PR (#2) proposes a nested protocol for possible future runs; its code has **not** generated the paper's saved results. The tables/figures remain the original frozen outcomes.

## Additional incomplete items

- The PDF's Table 4 includes scalar, fixed-RBF, matched-feature and a **frozen-structure No-margin** controlled run. The author-supplied script has now been added as `scripts/run_fixed_margin_control.py`, with the existing gate and experimental protocol preserved. The reported aggregate result is now archived as `runs/fixed_margin_control_reported_summary.json`, but **its full 250-unit raw output remains unavailable**. That summary documents previously reported values; it is not a substitute for `runs/fixed_margin_control.jsonl`, and the ablation should not be claimed as independently regenerated from saved per-unit results.
- The exact source ZIP included specific static artwork; the new script creates **data-derived replacement plots**, not byte-identical copies of the provided PNGs. The manuscript wording and results must not be altered to disguise this distinction.
- The code contains exploratory design choices beyond the five final reported methods (for example, previous M1/M3 diagnostics). A small shared dependency remains because the historical fitting and core tests reference it; unused experiments and large generated toy outputs are not part of this publication checkout.
- Tests of code logic and the current artifact validation do **not** certify target-distribution safety, repeated-seed uncertainty or independent third-party reproduction.

## What the checks do

`python scripts/verify_paper_results.py` verifies all headline aggregates, 25 method–roster pairs and rescue counts from **saved records**. It does not refit calibrators or recreate uncommitted ablation experiments. `python scripts/build_submission_figures.py` independently loads saved per-unit data to draw operating points and the rescue scatter.

If new data, different hyperparameters, a changed gate, or nested LOTO selection are introduced, treat them as **new experimental results** and update both the paper and this reference snapshot together.
