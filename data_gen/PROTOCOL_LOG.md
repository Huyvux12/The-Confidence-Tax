# Protocol log (deviations and decisions, in order)

- 2026-10-31. Items built with seed 20261031 (`build_items.py`): 31 tasks, cap 500, 2 answer orders.
  Task roles drawn once before any model output: target_dev = commonsense_qa, medmcqa,
  truthfulqa_mc1; target_confirm = aqua_rat, piqa, sciq; all other tasks are sources.
- 2026-10-31. Models fixed before inference: Qwen/Qwen3.5-9B, google/gemma-4-12B-it,
  ibm-granite/granite-4.2-8b, mistralai/Ministral-3-8B-Instruct-2512, allenai/Olmo-3-7B-Instruct
  (bf16, A100 40GB, transformers 5.21). Prompts: `letter_colon`, `paren`.
- 2026-10-31. DEVIATION: during a data-quality check of Qwen3.5-9B raw outputs, per-task raw
  accuracy and mean confidence were printed for all tasks, including the three target_confirm
  tasks (aqua_rat, piqa, sciq). No calibration method had been run on them. From this point all
  quality checks exclude target_confirm rows. Recorded so the confirmation run can be judged.
- 2026-10-31. Two HellaSwag items share the dataset `ind` id (2433, 22764); their cases are
  dropped in `make_splits.py` based on the id collision alone.
- 2026-10-31. `letter_colon` puts a median 0.45 of next-token mass on option letters for
  Qwen3.5-9B (`paren`: 0.999). Both prompts are kept unchanged; `letter_mass` is reported and no
  row is filtered on it.
- 2026-10-31. Go/no-go (source_train / source_audit only; no target read), Qwen3.5-9B and
  Gemma-4-12B. With 25 task-level sources no function class (constant, entropy-affine, RBF-4)
  has a verified strict common descent direction for Qwen; with 8 family-level sources RBF-4 has
  one but MINIMAX gains only ~1e-4 nats. Qwen: MRTS = identity; pooled TS mean audit delta
  -0.010/-0.017 but 6/25 and 4/25 sources harmed > 0.01; MEAN_SAFE with per-source tolerance
  eps=0.01 keeps about half of the gain with 0/25 harmed. Gemma: strongly overconfident, every
  source gains (-0.5 to -0.9 nats); no conflict regime, all methods help.
  Fix: the simplex QP of the diagnostic now uses Clarabel + exact KKT polish (SLSQP stopped early).
  Added `source_tolerance` (default 0 = spec) as an exploratory MEAN_SAFE variant.
- 2026-10-31. Dev run (Qwen, Gemma, Granite; target_dev only) of the pre-registered RB-Cal (z=0):
  the select gate rejected every RB-Cal variant on all Gemma/Granite units because the budget is
  binding at exactly epsilon on source_train and noisy per-source select estimates overshoot
  (0.024-0.048). Pre-registered E1 failed overall (passed in the conflict regime only); E3 failed.
  REVISION (made after seeing target_dev, before any target_confirm read): the budget is placed on
  the case-clustered upper confidence bound Delta_w + z*SE_w <= epsilon with z = 1.645
  (RBCAL_UCB_*). The revised primary method is RBCAL_UCB_RBF_e0.01; E1-E3 are computed for it,
  and the original z=0 variants stay in every report. Dev results are exploratory; the sealed
  confirm split remains untouched and decides the claims.
- 2026-11-01. Full dev run, 10 units (target_dev only; fits for 7 units on a Colab CPU VM with
  identical package versions, scoring locally; prediction sha256 df593c535f402b58...). Revised
  primary RBCAL_UCB_RBF_e0.01: E1 vs MRTS pass (-0.066 nats, CI [-0.071,-0.061], harm 0 vs 0),
  E2 vs pooled TS pass overall (harm 0 vs 1, worst +0.0010 vs +0.0169, 89% of pooled gain),
  E3 vs RBCAL_UCB_CONST_e0.01 pass (-0.014, CI [-0.015,-0.012]). Regimes from source_train:
  3 conflict (Qwen x2, Ministral paren), 7 consensus. Gate rejected the primary on Ministral
  letter_colon (select max delta 0.0135). Runner now caches units with as_completed (no effect on
  results). No method or config change after this run.
- 2026-11-01. FREEZE before the confirmation run. Code sha256 (calibration_extensions/*.py +
  scripts/run_rbcal.py, concatenated) b0e2f24c6e5d169d. Two speed changes, no change in method:
  (1) LinearLogitOracle reuses the last per-row evaluation and sums case gradients with reduceat
  (bit-identical on 24/24 test solves); (2) --reuse-selection takes the LOSO menu choice from the
  dev run, allowed only because source_train/select/audit are byte-identical between dev and
  confirm_sealed (checked by sha256 inside the runner). Re-running dev this way reproduced every
  gate decision and every selected spec on all 10 units; target log-probs are bit-identical for
  units fitted on this machine and differ by <= 2.5e-8 for units first fitted on the Colab VM.
  Primary RBCAL_UCB_RBF_e0.01, endpoints E1-E3 as in PREREGISTRATION.md, n_boot 2000. The
  confirm split is scored once with this code.
- 2026-11-01. CONFIRMATION RUN (once; frozen code b0e2f24c6e5d169d; prediction sha256 a0c6b89fdddb549f...;
  targets aqua_rat, piqa, sciq; 30 task-units). Primary RBCAL_UCB_RBF_e0.01:
  E1 vs MRTS pass (-0.029, CI [-0.034,-0.025]; harm 0 vs 0; 20 wins / 10 losses by task-unit);
  E2 vs pooled TS pass (harm 0 vs 7, worst +0.0056 vs +0.0453, 90% of pooled gain);
  E3 vs RBCAL_UCB_CONST_e0.01 pass (-0.011, CI [-0.012,-0.009]; 18 / 7 ties / 5).
  All three also pass within each regime (conflict 9, consensus 21 task-units).
  Pre-registered z=0 variants: RBCAL_RBF_e0.01 -0.085 macro (gate rejections), reported as is.
- 2026-11-01. Secondary analyses after the confirmation run (todo.md items 1 and 3; claims unchanged).
  fit_shrinkage_loso now computes each fold's pooled beta once (same result, checked on a stored fit).
  Item 1: THERMO_P / ATS_P (probability-feature adaptations, fixed hyperparameters, fit on source_train).
  Confirm: -0.262 / -0.263 macro with 5 / 7 harms vs primary -0.230 with 0 harms.
  Item 3: leave-one-task-out, 10 units x 25 source tasks = 250 targets; RB-Cal-RBF spec reused from the
  25-source selection (small leak, stated). Primary -0.524 (2 harms, worst +0.011); MRTS -0.451 (0);
  pooled TS -0.605 (22); THERMO_P -0.607 (19); ATS_P -0.600 (32); RB-Cal const -0.509 (2).
  Task-bootstrap CI primary - MRTS [-0.094,-0.054]; primary - const [-0.021,-0.011].
- 2026-11-01. Recheck (todo.md phase 5, details in paper/RECHECK.md). Fixed three bibliography errors and two
  hand-typed numbers in the paper; added two related works. Recorded model and dataset commits
  (data_gen/revisions.json; all repositories unchanged since the run). Independent recomputation of the
  main numbers: 18/18 match. Full refit of the confirm unit Qwen3.5-9B/paren without --reuse-selection
  (Colab CPU VM): same selected configuration and gate decision for all 33 methods, target log-probabilities
  within 8e-12 of the scored run. No result changed.
