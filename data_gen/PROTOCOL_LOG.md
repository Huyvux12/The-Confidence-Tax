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
- 2026-11-06. Phase 6 (todo.md): deployment-gate study, designed after the confirmation split was unsealed.
  Motivation (seen on both dev and confirm): the z=0 RBF fit before the gate already had zero harms and
  near-pooled gain; the mean gate (max_w D_w^select <= eps over 25 noisy sources) rejected most units.
  Eight variants declared before any result: candidate {C0: z=0, C1: z=1.645} x gate {G-mean: current,
  G-sig: reject only if some source has D_w^select - 1.645 SE_w^select > eps} x fallback {identity, MRTS}.
  Choice rule on dev targets: zero harms (>0.01), then largest macro gain; ties -> identity, G-mean, C1.
  Refits use the spec chosen on all sources in runs/rbcal_dev (Modal CPU, profile trantu7425) and reproduce
  the cached gate statistic and pre-gate target deltas within 4e-16.
  Dev result: C0 + G-sig + MRTS fallback, macro -0.5116, 0 harms (pooled TS -0.5289, 1 harm; current
  primary -0.4718). FROZEN as "RB-Cal v2" before scoring confirm and LOTO with it. Confirm numbers for v2
  are secondary (the split had been seen); the primary method and its confirm result are unchanged.
  Results for the frozen v2 (C0 + G-sig + MRTS fallback), runs/gate_study/:
  confirm: macro -0.2494 (97.2% of pooled TS), 0 harms, worst +0.0070, NLL 0.693, Brier 0.354, ECE 0.062
  (primary: -0.2304, 0 harms, NLL 0.712). LOTO (250): -0.5723 but 7 harms, worst +0.0444 (primary: -0.5241,
  2 harms, worst +0.0108). v2 therefore does NOT keep the primary's harm profile on held-out source tasks;
  dropping the fitting margin (z=0) is what costs safety there. Not adopted.
  Observed afterwards, not frozen: C1 + G-sig (keep the margin, significance gate) gives dev -0.4979 / 0 harms,
  confirm -0.2509 / 0 harms, LOTO -0.5599 / 2 harms (worst +0.0108), i.e. more gain than the primary with the
  same harm counts on all three rosters. Chosen after seeing confirm and LOTO, so it needs a fresh roster.
- 2026-11-06. PRE-REGISTRATION of a second sealed confirmation roster ("confirm2"), written before any item
  file, model output or label of these tasks exists. Purpose: test the gate variant found after unsealing.
  Tasks (6, none used before; each a single family): qasc (allenai/qasc validation), race_high (ehovy/race high
  test), social_iqa (allenai/social_i_qa validation, refs/convert/parquet), cosmos_qa (allenai/cosmos_qa
  validation, refs/convert/parquet), medqa_usmle (GBaker/MedQA-USMLE-4-options test), belebele_eng
  (facebook/belebele eng_Latn test). Reserves in order if one fails to build: quartz, scienceqa_text.
  MathQA excluded (derived from AQuA-RAT, a confirm target). Items: cap 500 per task, 2 answer orders,
  seed 20261106. Inference: same 5 models at the commits in data_gen/revisions.json, same two prompts,
  same infer_mcq.py, bf16 on A100 (Modal, profile acc3). Sources, configurations and code are unchanged;
  confirm2 reuses the source files byte-for-byte and the selection of runs/rbcal_dev.
  Tested method "v3": RBCAL_UCB_RBF_e0.01 candidate (z=1.645) + G-sig gate (reject only if some select source
  has D_w - 1.645 SE_w > 0.01) + identity fallback (phase-6 tie rule). 60 task units (10 x 6); harm = D > 0.01.
  P1 (primary): v3 - MRTS macro D < 0 with the synchronized case-bootstrap 95% CI excluding 0 (questions
     resampled per task, shared by all 10 units; 5000 draws, seed 20261106) and harms(v3) <= harms(MRTS).
  P2: harms(v3) <= harms(pooled TS) and worst(v3) < worst(pooled TS).
  P3: macro D(v3) <= macro D(v1 = current primary) and harms(v3) <= harms(v1).
  Secondary: NLL, Brier, ECE-15, accuracy, h_any, gain retained vs pooled TS; comparators MRTS, pooled TS,
  shrinkage, RB-Cal-const, v1, Thermometer-P, ATS-P. All results reported whatever the outcome.
- 2026-11-06. confirm2 data. Items: data_gen/items_confirm2.jsonl (sha256 b878c7c0...; 6 x 500 items, 6000 rows;
  9 items with duplicate options dropped by the existing item() rule). Dataset commits: data_gen/revisions_confirm2.json.
  Inference on Modal A100-SXM4-40GB (profile acc3), transformers 5.21.0, torch 2.15.0+cu132 (the original Colab
  torch was not recorded), pinned model commits, reference kernels; letter token ids identical to the original
  run. A first full run was lost to a client-side deserialization bug (no outputs kept); rerun wrote to a volume.
  Numerics check (3 items per original task re-scored, 372 rows/model, probabilities only): 4 models show no
  systematic shift (mean change in top confidence <= 0.0005, mean NLL change <= 0.008); gemma-4-12B-it is
  slightly sharper (top confidence +0.010, NLL +0.026, 21/372 rows move > 0.1). All methods see the same target
  probabilities (paired comparisons); a without-gemma sensitivity analysis is added post hoc and labelled so.
  data/real/confirm2_sealed: sources byte-identical to dev; 60000 target rows; sha256 in data/real/confirm2_SHA256SUMS.txt.
- 2026-11-06. CONFIRM2 RESULT (fits on Modal CPU, profile acc3, selection reused from runs/rbcal_dev; labels
  opened once afterwards; runs/rbcal_confirm2/confirm2_report.json). 60 task units.
  v3 (C1 + G-sig + identity): macro -0.3103, 0 harms, worst +0.0086, NLL 0.631, Brier 0.315, ECE 0.068;
  97.2% of pooled TS gain. G-sig accepted all 10 units, so v3 equals the pre-gate C1 fit here.
  Comparators: pooled TS -0.3192 / 9 harms / worst +0.051; Thermometer-P -0.3226 / 7; ATS-P -0.3149 / 12;
  MRTS -0.2655 / 0; v1 (current primary) -0.2961 / 0; RB-Cal-const -0.2940 / 0; shrinkage -0.1586 / 0.
  P1 PASS: v3 - MRTS -0.0449, synchronized 95% CI [-0.0507, -0.0392], harms 0 vs 0.
  P2 PASS: harms 0 vs 9, worst +0.0086 vs +0.0512.   P3 PASS: -0.3103 vs -0.2961 (sync CI of the
  difference [-0.0160, -0.0126]), harms 0 vs 0.
  Post hoc, without gemma (48 units): v3 -0.2564 / 0 harms; pooled TS -0.2533 / 9; Thermometer-P -0.2575 / 7.
- 2026-11-06. Phase 8 (todo.md): answer-order consistency features, declared before any fit. v4a: RBF field on
  z = [h, c, m, tv] (tv = total variation between the two answer orders of a question, semantic options, no labels);
  v4b: z = [h, c, m, tv, task-mean tv] (task batch: source task for sources, target task for targets, unlabelled).
  Candidate otherwise as v3 (eps 0.01, z 1.645, LOSO menu of 37 specs on dev source_train, G-sig gate, identity
  fallback). 740 LOSO jobs + fits on Modal CPU (profile acc3); code: calibration_extensions/features.py
  (rbf_oc, rbf_oct; old kinds unchanged, 90/90 tests), scripts/v4_study.py, scripts/modal_v4.py.
  Dev: v4a -0.5025 / 0 harms / h_any 0; v4b -0.5052 / 0 harms / h_any 0 (worst -0.0010); v3 -0.4979 / 0 / h_any 1.
  Rule (zero harms, then largest gain) -> v4b FROZEN (specs in runs/v4_study/chosen.json). LOTO gate before any
  new roster: v4b advances only if LOTO harms <= 2 and LOTO macro <= -0.5599 (v3).
  LOTO (250, spec reused from the 25-source selection): v4b -0.5731, 0 harms, worst +0.0094, ECE 0.092
  (v3 -0.5599, 2 harms, worst +0.0108, ECE 0.096). LOTO gate PASSED -> v4b advances to confirm3.
- 2026-11-06. PRE-REGISTRATION of a third sealed roster ("confirm3"), written before any item file, output or
  label of these tasks exists. Tasks (6, none used before): quartz (allenai/quartz validation), scienceqa_text
  (tasksource/ScienceQA_text_only test), pubmedqa (qiaojin/PubMedQA pqa_labeled, yes/no/maybe), copa
  (pkavumba/balanced-copa test), lsat_lr (hails/agieval-lsat-lr test), lsat_rc (hails/agieval-lsat-rc test).
  Reserves in order: bigbench_logical_deduction, sat_en. Cap 500, 2 answer orders, seed 20261107; same models,
  commits, prompts, infer_mcq.py, Modal A100-40GB (acc3), transformers 5.21.0, torch 2.15.0 (as confirm2).
  Sources, code and selections unchanged (v4b specs: runs/v4_study/chosen.json; v1/v3 as before).
  Tested method v4b. 60 task units; harm = D > 0.01.
  Q1 (primary): v4b - MRTS macro D < 0, synchronized case-bootstrap 95% CI excluding 0 (5000 draws, seed
     20261107), harms(v4b) <= harms(MRTS).
  Q2: harms(v4b) <= harms(pooled TS) and worst(v4b) < worst(pooled TS).
  Q3: macro D(v4b) <= macro D(v3) and harms(v4b) <= harms(v3).
  Secondary: NLL, Brier, ECE-15, number of units with lower ECE, h_any, gain retained vs pooled TS; comparators
  pooled TS, Thermometer-P, ATS-P, MRTS, shrinkage, RB-Cal-const, v1, v3. All results reported whatever the outcome.
- 2026-11-06. confirm3 data. Items: data_gen/items_confirm3.jsonl (sha256 7d9518fd...; 2652 items, 5304 rows;
  lsat_rc 268 and quartz 384 items = whole pools; 1 item with duplicate options dropped). AGIEval queries were cut
  before "Answer Choices:" and "Q:" rendered as "Question:" (formatting only, before any output). Dataset commits:
  data_gen/revisions_confirm3.json. Inference as confirm2 (Modal A100-40GB, acc3, torch 2.15.0); the 372-row
  numerics check is byte-identical to the confirm2 run's. data/real/confirm3_sealed: sources byte-identical to dev,
  53040 target rows, sha256 in data/real/confirm3_SHA256SUMS.txt.
- 2026-11-06. CONFIRM3 RESULT (60 task units; runs/rbcal_confirm3/confirm3_report.json; fits on Modal CPU, acc3).
  v4b: macro -0.2850, 0 harms, worst +0.0053, NLL 0.512, Brier 0.261, ECE 0.061, ECE lower on 47/60 units;
  93.4% of pooled TS gain. Gates accepted all units (v3 10/10, v4b 10/10).
  Comparators: pooled TS -0.3050 / 7 harms / worst +0.058 / ECE 0.051; Thermometer-P -0.3064 / 5 / ECE 0.049;
  ATS-P -0.3042 / 9 / ECE 0.052; MRTS -0.2320 / 0; v1 -0.2644 / 0; v3 -0.2790 / 0 (ECE lower on 50/60).
  Q1 PASS: v4b - MRTS -0.0530, sync 95% CI [-0.0600, -0.0462], harms 0 vs 0.
  Q2 PASS: harms 0 vs 7, worst +0.0053 vs +0.0577.
  Q3 PASS: -0.2850 vs -0.2790 (sync CI of the difference [-0.0068, -0.0053]), harms 0 vs 0.
  Reading: the order-consistency features add a small, consistent gain over v3 with zero harms on dev, LOTO
  (0/250) and confirm3, but do not close the mean-NLL / ECE gap to the unconstrained baselines.
- 2026-11-06. Phase 9 declared before any fit: relaxed operating point "v5" = v4b features (rbf_oct), no fitting
  margin (z=0), G-sig gate with threshold = eps, identity fallback; v5a eps=0.01, v5b eps=0.02; LOSO menu selection
  on dev source_train (ranking threshold 0.01 as before). Eligible if dev harms <= 1/30 and LOTO harms <= 7/250;
  among eligible, largest LOTO gain; ties -> smaller eps; none eligible -> stop. Purpose: match the unconstrained
  baselines' gain with far fewer harmed units. confirm2/confirm3 are not used for this choice.
- 2026-11-06. Phase 9 result (design data only; confirm2/3 untouched for this choice).
  Dev: v5a -0.4982 / 0 harms, v5b -0.5009 / 0 harms; the G-sig gate rejects Ministral/letter_colon for both
  (openbookqa select regret +0.051 / +0.069, SE 0.020-0.022), so that unit falls back to identity.
  LOTO: v5a -0.5564 / 7 harms / worst +0.038; v5b -0.5602 / 10 harms; v4b -0.5731 / 0; pooled TS -0.6045 / 22.
  By the declared rule v5a is eligible (7 <= 7) and v5b is not. v5a is dominated by v4b on dev and LOTO (less
  gain, more harms), so it cannot deliver the purpose of the phase; STOPPED before confirm4 (no roster built,
  no GPU spent). Reading: on held-out tasks the budgeted family plateaus near -0.57 (95% of pooled TS);
  dropping the fitting margin adds harms without adding gain. The remaining gap to the unconstrained baselines
  is gain they obtain on exactly the tasks they harm.
- 2026-11-06. Phase 10.0 diagnostics (LOTO only; scripts/step0_diagnostics.py; runs/step0/). Oracle that picks, per
  held-out unit, v4b or pooled TS when pooled is safe: -0.6129 (v4b -0.5731, pooled TS -0.6045 with 22 harms) ->
  oracle gain over v4b 0.040 nats. CORRECTION of the phase-9 reading: the v4b-pooled gap does NOT come from units
  pooled harms (their share of the gap is -12%); pooled is better and safe on 160/250 units (+0.062 there).
  AUROC (direction-free, 250 units) for "pooled harms" (22 pos): template TV 0.92, order TV 0.89, peer prob on
  top 0.89, peer argmax agreement 0.89, position-prior entropy 0.85, max prob 0.74, entropy 0.69. Harmed units
  are consistent across orders/templates and agreed by peers (easy, already calibrated tasks). For "pooled beats
  v4b by > 0.01" (134 pos) the best signal reaches 0.69 (peer prob on top). GO criterion met (0.040 >= 0.01 and
  0.92 >= 0.75) -> continue with 10.1.
- 2026-11-06. Phase 10.1/10.2 declared before any fit. Derived data (data_gen/make_aug.py, no labels read):
  base "_avg" = probabilities averaged over the two answer orders; base "_aug" = original probabilities; both with
  order_tv_raw, peer_top (4 other models, order-averaged, prob on this row's top option) and peer_top1 (one peer
  fixed in advance: Olmo-3-7B, Ministral-3-8B for Olmo). Variants: v4b_avg (v4b on _avg), v6_{aug,avg}
  (rbf_ocp = v4b features + peer_top row value and task mean), v6p1_{aug,avg} (peer_top1). Same eps 0.01, z 1.645,
  LOSO menu, G-sig, identity fallback. Fair baselines on each base: pooled TS, MRTS, Thermometer-P, ATS-P, and
  Thermometer-P/ATS-P with the same peer features (rbf_ocp). Harm is reported vs the same-base identity and vs the
  original single-order identity.
  Base rule: base = the one with the larger LOTO gain of v4b vs the original identity (ties -> _avg); v6 is judged
  on that base. LOTO gate for v6 (on the chosen base): harms <= 2/250 and LOTO macro (vs same-base identity) at
  least half-way from v4b to pooled TS on that base. Not met -> stop before confirm4.
- 2026-11-06. Phase 10.1/10.2 RESULT (LOTO 250 + dev 30; Modal CPU, acc3; runs/v4_study/base_*.jsonl, loto.jsonl).
  Averaging the two answer orders alone lowers NLL by 0.290 (LOTO) / 0.211 (dev) vs the original identity, no harms.
  LOTO on _avg (delta vs _avg identity / harms): pooled TS -0.362/22, Thermometer-P -0.364/17, ATS-P -0.351/44,
  Thermometer-P+peer -0.364/18, ATS-P+peer -0.382/29, MRTS -0.236/0, v4b_avg -0.298/0, v6_avg -0.316/0,
  v6p1_avg -0.307/0. Vs the original identity: v6_avg -0.605/0 harms (pooled TS on original probs -0.6045/22).
  LOTO on original probs (_aug): pooled TS -0.6045/22 (reproduces), Thermometer-P -0.607/19, ATS-P -0.600/32,
  Thermometer-P+peer -0.607/18, ATS-P+peer -0.636/19, v4b -0.573/0, v6_aug -0.583/1, v6p1_aug -0.579/1.
  Base rule -> _avg (v4b -0.587 vs -0.573 against the original identity). v6 gate on _avg: needs <= -0.330
  (half-way v4b_avg -> pooled_avg), got -0.316 -> FAILED (also fails on _aug: -0.583 vs -0.589). STOPPED before
  confirm4 and before the selector step (10.3), as declared. Readings: (i) order averaging is the largest free
  improvement found and lifts every method; on that fair base RB-Cal retains 82% of pooled TS gain (vs 95% on
  single-order probabilities); (ii) peer consensus adds +0.010..0.018 nats at 0-1 harms, closing about 30% of the
  gap; (iii) peer features do not make the unconstrained baselines safe (17-29 harms), so the per-source budget,
  not the signal, provides the safety.
- 2026-11-06. Phase 11 (user request: flat ECE like the unconstrained baselines with small or no trade-off).
  Deviation noted: phase 10 stopped at the v6 gate; this resumes report direction 2 (task-level selector).
  Exploratory check on LOTO (scripts/switch_study.py; thresholds chosen nested, on the other 24 tasks): switching
  per target task between pooled TS and v4b on a label-free task signal gave -0.607 / 2 harms / ECE 0.063 / 223 of
  250 units with lower ECE (pooled TS -0.605/22/0.068/204; Thermometer-P -0.607/19/0.065/204; v4b -0.573/0/0.092/224).
  The signal family was chosen after the phase-10.0 AUROC table on the same LOTO data -> needs a fresh roster.
  FROZEN method "RB-Switch" (runs/step0/switch_frozen.json): per target task batch (model x prompt x task) compute
  s = mean(template_tv, order_tv, 1 - peer_prob_on_top) exactly as scripts/step0_diagnostics.task_signals
  (order_tv: TV between the two answer orders; template_tv: TV between the two prompts, same question and order;
  peer_prob_on_top: mean prob the 4 other models, same prompt and order, give to this row's top option);
  deploy pooled TS if s >= 0.19277 else v4b. tau chosen on all 250 LOTO units (largest gain with <= 2 harms).
  Secondary (reported, not tested): no-peer rule s' = mean(template_tv, order_tv), tau 0.12826; Thermometer-P as
  the strong arm, tau 0.28871.
- 2026-11-06. PRE-REGISTRATION confirm4 (before any item file, output or label exists). Tasks (10, none used):
  BIG-bench (tasksource/bigbench, train+validation pooled) date_understanding, disambiguation_qa,
  logical_deduction, temporal_sequences, strategyqa, implicatures, tracking_shuffled_objects, epistemic_reasoning,
  navigate; AGIEval sat_en (hails/agieval-sat-en). Cap 500, 2 orders, seed 20261108; same models, commits,
  prompts, infer_mcq.py, Modal A100-40GB (acc3). 100 task units; harm = D > 0.01.
  S1 (primary): harms(RB-Switch) <= floor(harms(pooled TS) / 3) and <= harms of Thermometer-P and of ATS-P.
  S2: macro D(RB-Switch) <= macro D(pooled TS) + 0.005.
  S3: mean ECE-15(RB-Switch) <= mean ECE-15(pooled TS) + 0.005.
  Secondary: units with lower ECE, ECE slope, worst unit, synchronized bootstrap CI vs pooled TS and vs v4b,
  the two secondary rules, v4b alone. All results reported whatever the outcome.
- 2026-11-06. confirm4 data: items_confirm4.jsonl (sha256 bb403658...; 4325 items, 8650 rows), revisions in
  data_gen/revisions_confirm4.json; inference as before (Modal A100, acc3; numerics check byte-identical to the
  confirm2/3 runs); data/real/confirm4_sealed (sources byte-identical to dev; 86500 target rows; SHA256SUMS).
- 2026-11-06. CONFIRM4 RESULT (100 task units; runs/rbcal_confirm4/confirm4_report.json). FAILED S1 and S2.
  RB-Switch: -0.4475 / 10 harms / worst +0.113 / ECE 0.129 / 83 of 100 units with lower ECE (pooled share 44%).
  Pooled TS -0.4876 / 13 / ECE 0.127 / 76; Thermometer-P -0.4902 / 13 / 0.124 / 78; ATS-P -0.4848 / 14 / 0.124;
  MRTS -0.3554 / 1; v1 -0.4134 / 8; v4b -0.4370 / 7 / worst +0.113 / ECE 0.134 / 85.
  S1 FAIL (10 > floor(13/3)=4), S2 FAIL (-0.4475 vs -0.4876 + 0.005), S3 PASS (0.129 <= 0.132).
  Secondary rules: no-peer -0.4484 / 9; Thermometer arm -0.4453 / 8. Sync CI switch - pooled +0.040 [+0.036,
  +0.045]; switch - v4b -0.010 [-0.013, -0.008].
  Harms concentrate in temporal_sequences (pooled 6, v4b 3), date_understanding (3 / 2), logical_deduction (3 / 1),
  navigate (1 / 1): synthetic BIG-bench reasoning tasks far from the 25 sources; temporal_sequences is already well
  calibrated (identity ECE 0.068). First roster on which v4b harms tasks (7/100); only MRTS stays near zero (1).
  Reading: per-source budgets and label-free switching do not carry their safety to a roster this far from the
  sources (consistent with the TV term of the transfer proposition); the paper must report this roster.
- 2026-11-06. v4b scored on confirm1 and confirm2 (secondary: both rosters were seen before v4b was designed;
  Modal CPU, acc3; runs/v4_study/final_confirm*.jsonl). confirm1: -0.256 / 0 harms (pooled TS -0.257 / 7);
  confirm2: -0.316 / 0 (pooled TS -0.319 / 9).
  Aggregate over five rosters (confirm1-4 + LOTO = 500 task units): harmed units v4b 7 (1.4%), Thermometer-P 49
  (9.8%), pooled TS 58 (11.6%), ATS-P 74 (14.8%), MRTS 1 (0.2%); units with lower ECE v4b 86%, others 75-77%;
  mean D v4b -0.462, Thermometer-P -0.493, pooled TS -0.490, ATS-P -0.487, MRTS -0.368. v4b has the most
  ECE-improved units on every roster. Rosters out-of-sample for v4b (frozen before they existed): confirm3 and
  confirm4 -> 7/160 harms vs Thermometer-P 18/160, pooled TS 20/160.
- 2026-11-06. Target venue EvoLearn 2027 (EC + ML scope, LNCS 14 pages + refs, double-blind, deadline 16 Nov 2026).
  EC extension declared before any run (calibration_extensions/evo_calibration.py, scripts/evo_study.py):
  (a) CMA-ES (pycma 4.4.0, seed 1, sigma0 0.1, 3000 evals, repair projection for beta bounds, penalty 100,
      elitist feasible archive) solves the v4b problem on dev source_train for eps in {0,.0025,.005,.01,.02,.05,.1};
      compared with SLSQP; all CMA-ES evaluations give the empirical front (pooled regret vs worst-source bound).
      Local check on 3 units before declaring: CMA-ES and SLSQP reach the same optimum to 1e-4.
  (b) Evo-RB: outer CMA-ES (seed 1, sigma0 0.05, 300 evals) over the RBF centres and log-bandwidth of the v4b
      field; inner SLSQP for theta under the same budget on source_train; fitness = mean select regret + 100 x
      G-sig violations on source_select; deployment gate = G-sig on source_audit; identity fallback. Same spec
      (kind, J, lambda) as v4b; constant specs unchanged.
  Evaluation paired with v4b on dev, LOTO (250) and confirm1-4 (all already seen: secondary). Decision rule for
  the paper: Evo-RB is reported as an improvement only if LOTO harms <= v4b + 1 and LOTO macro <= v4b - 0.005;
  otherwise reported as a negative EC result.
- 2026-11-06. EC (a) result (runs/evo_study/front.jsonl): over 10 dev units x 7 eps (68 pairs with both feasible),
  CMA-ES (3000 evals each) never improved the SLSQP optimum by more than 1e-4 (mean difference +3.8e-5, worse by
  up to 1.3e-3 in 3 pairs). The margin-constrained problem is non-convex in principle, but the local SLSQP solution
  from the identity is not beaten by a global evolution strategy. Each unit's CMA-ES archive gives 7-646
  non-dominated (pooled regret, worst-source bound) points for the Pareto-front figure.
- 2026-11-06. EC (b) result, Evo-RB vs v4b (paired; runs/evo_study/, aggregate in aggregate_500.json):
  LOTO -0.5906 / 1 harm / ECE 0.081 vs -0.5731 / 0 / 0.092 -> decision rule MET (harms <= v4b+1, gain >= 0.005).
  dev -0.515/0 vs -0.505/0; confirm1 -0.266/0 vs -0.256/0; confirm2 -0.321/0 vs -0.316/0; confirm3 -0.290/0
  vs -0.285/0; confirm4 -0.442/7 vs -0.437/7 (confirm rosters secondary: all seen before Evo-RB was designed).
  Aggregate over 500 units (confirm1-4 + LOTO): Evo-RB 8 harms (1.6%), ECE lower on 87% of units, mean D -0.473,
  mean ECE 0.084; Thermometer-P 49 (9.8%), 77%, -0.493, 0.075; pooled TS 58 (11.6%), 77%, -0.490, 0.078;
  ATS-P 74 (14.8%), 75%, -0.487, 0.078; MRTS 1 (0.2%), 77%, -0.368, 0.131; v4b 7 (1.4%), 86%, -0.462, 0.092.
- 2026-11-07. LNCS paper for EvoLearn 2027 in paper_lncs/ (main.tex, refs.bib, build.sh; every number, table and
  figure generated by scripts/make_lncs_assets.py from runs/). 13 pages including references (limit: 14 + refs).
  Double-blind: no author names, links or repository names; prior minimax work cited as Anonymous. At the user's
  request the AI-use statement required by the EvoStar call is NOT included in the draft (flagged to the user).
