# Reproducibility and paper alignment (11 Nov 2026)

## Manuscript identity

The author's latest file is **The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models**, 15 LNCS pages, built from an Overleaf package titled `Confidence_Tax_EvoRB_Table2_TwoTier_Fig3_FIXED_15pages_Overleaf.zip` and verified locally on 2026-11-11 as a 15-page pdfLaTeX document. The extracted source `main.tex` has SHA-256:

`691dfda7772e5e288d3830c547f296866cd74b84742154a93be687499e9ddc2a`.

**Do not use the old `paper_lncs/main.tex` as the submission manuscript**: its text/figures describe a previous Evo-RB draft. The old LaTeX directory was removed from this publication branch, but remains in Git history.

## Exactly supported by frozen outputs

- Table 1: five baseline/Evo-RB methods plus identity, macro TL-ECE/Brier/ΔNLL and 8/500 harm.
- Table 2: confirmation rosters C1 (30), C2 (60), C3 (60), Hard (100), LOTO (250).
- Figure 1 and Figure 2 operating points: computed from `runs/evo_study/aggregate_500.json`, rechecked against per-unit records by `scripts/build_submission_figures.py`.
- Figure 3 paired rescue: `runs/v4_study/base_loto_aug.jsonl` and `runs/evo_study/evo_loto.jsonl`; counts independently verified by `scripts/verify_paper_results.py`.

Main five-method comparison: Pooled TS, Thermometer-P, ATS-P, Minimax TS, Evo-RB. The fixed regret-budgeted field V4B is **an internal ablation**, not a sixth headline baseline. Neither Thermometer-P nor ATS-P uses the hidden states of the original published method.

## Critical LOTO protocol difference — address before claiming strict task isolation

The submitted manuscript says that each held-out source task is removed from **fitting, hyperparameter selection and auditing**. In the committed experimental path, `scripts/evo_study.py::jobs_for("evo_loto")` loads the per-model/prompt `v4b` specification from `runs/v4_study/chosen.json`; `scripts/v4_study.py::jobs_for("loto")` uses that same global choice. That choice was made using all 25 source tasks before each LOTO holdout was designated. The actual LOTO **coefficient fitting** excludes the held-out task, and the held-out task is excluded from the source-selection/audit statistics for those fits, but the discrete specification has prior information from the held-out task.

Therefore the current published LOTO numbers support **fixed global-configuration LOTO with held-out refitting**, **not full nested hyperparameter selection**. Rerunning nested LOTO selection or revising that manuscript sentence is necessary before making a stronger claim. The figures/tables remain the actual frozen outcomes; no re-labelled or invented data have been substituted here.

## Additional incomplete items

- The PDF's Table 4 includes scalar, fixed-RBF, matched-feature and a **frozen-structure No-margin** controlled run. That precise no-margin output is not independently named/checked as a first-class artifact in the present Git tree. The paper's values should not be claimed as regenerated from the repository until the corresponding raw run, selection freeze/config and output hashes are committed.
- The exact source ZIP included specific static artwork; the new script creates **data-derived replacement plots**, not byte-identical copies of the provided PNGs. The manuscript wording and results must not be altered to disguise this distinction.
- The code contains exploratory design choices beyond the five final reported methods (for example, previous M1/M3 diagnostics). A small shared dependency remains because the historical fitting and core tests reference it; unused experiments and large generated toy outputs are not part of this publication checkout.
- Tests of code logic and the current artifact validation do **not** certify target-distribution safety, repeated-seed uncertainty or independent third-party reproduction.

## What the checks do

`python scripts/verify_paper_results.py` verifies all headline aggregates, 25 method–roster pairs and rescue counts from **saved records**. It does not refit calibrators or recreate uncommitted ablation experiments. `python scripts/build_submission_figures.py` independently loads saved per-unit data to draw operating points and the rescue scatter.

If new data, different hyperparameters, a changed gate, or nested LOTO selection are introduced, treat them as **new experimental results** and update both the paper and this reference snapshot together.
