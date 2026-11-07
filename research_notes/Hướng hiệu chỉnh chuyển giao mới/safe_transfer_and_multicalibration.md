# Safe calibration transfer: multicalibration, group-robust calibration, and non-regressive updates (literature map for RB-Cal)

Scope note: searched November 2026 (arXiv, PMLR/ICML 2026, NeurIPS 2022, ICCV 2021, OpenReview hits). About 20 search/fetch calls. Items marked "preprint" have no peer-reviewed venue listed on the page I checked. The CALPORT/MRTS repository that comes up in search belongs to this project's own GitHub account (Huyvux12), so it is listed as prior work by us, not as outside evidence.

## Q1. Multicalibration / multiaccuracy (incl. LLM confidence, low-degree, omniprediction, transfer): do they give per-task guarantees on unseen tasks, and how are groups defined without labels at test time?

### Takeaway
Multicalibration for LLMs (Detommaso et al., ICML 2024) defines groups without labels, using prompt-embedding clusters and LLM self-annotation of the prompt. Its guarantee covers groups from the same distribution as the calibration data, not unseen tasks. The only route to a guarantee on an unseen target is the "universal adaptability" / covariate-shift line. It needs the source-to-target density ratio to lie in, or be well approximated by, the grouping class. That is a different assumption from RB-Cal's empirical, per-source regret budget, and none of these papers gives a "no target is harmed" guarantee.

### Cited Findings
- Detommaso, Bertran, Fogliato, Roth, "Multicalibration for Confidence Scoring in LLMs", ICML 2024 (PMLR 235:10624-10641; arXiv 2404.04689, Apr 2024). It defines multicalibration as calibration "simultaneously across various intersecting groupings of the data" — [PMLR](https://proceedings.mlr.press/v235/detommaso24a.html); [ICML poster](https://icml.cc/virtual/2024/poster/34914)
- How it builds groups without labels: (i) clustering prompt embeddings; (ii) "self-annotation", where the LLM answers yes/no questions about the prompt to create binary features. Building useful grouping features is presented as a main contribution. A secondary contribution is multicalibration algorithms that overfit less (IGLB/IGHB variants). The paper benchmarks on several QA datasets and reports better fine-grained calibration than baselines — [arXiv 2404.04689](https://arxiv.org/abs/2404.04689v1); [ICML poster](https://icml.cc/virtual/2024/poster/34914)
- Low-degree multicalibration (Gopalan, Kim, Singhal, Zhao, COLT 2022) defines a hierarchy of notions running from multiaccuracy to full multicalibration. In the multi-class case, its sample complexity is exponentially better (in the number of classes) than full multicalibration. Multiaccuracy alone does not give the loss-minimization guarantees of multicalibration — [arXiv 2203.01255](https://arxiv.org/abs/2203.01255); [COLT PDF](https://proceedings.mlr.press/v178/gopalan22a/gopalan22a.pdf)
- Omniprediction is implied by multicalibration and related multi-group notions. Known constructions rely on some variant of multicalibration — [arXiv 2302.06726](https://arxiv.org/html/2302.06726v2)
- Transfer under covariate shift: a predictor multicalibrated on the source can achieve "universal adaptability" with an error bound on the target. The bound depends on how well the learned function class captures the relevant density ratios. Multicalibrated predictors are reported to perform on par with or better than propensity-reweighting estimators — "Multicalibration Boosting: Theory, Convergence, and Transferability", preprint, arXiv 2605.24364 (2026) — [arXiv](https://arxiv.org/html/2605.24364)
- Wu, Liu, Cui, Wu, "Bridging Multicalibration and Out-of-distribution Generalization Beyond Covariate Shift" (arXiv 2406.00661, 2024). It extends grouping functions to take covariates and labels jointly, links the result to invariance under concept shift, and proposes post-processing (MC-Pseudolabel). It also shows multicalibration implies Bayes optimality under covariate shift given a rich enough grouping class — [arXiv](https://arxiv.org/pdf/2406.00661)
- Universal adaptability has also been extended to censored survival data (Ye & Li, arXiv 2405.15948) — [arXiv](https://arxiv.org/html/2405.15948v2)

### Inferences
- In multicalibration the guarantee is a bound on calibration error inside each group, for groups that are seen in distribution. It is not a bound on how much a given group's loss can worsen relative to a baseline calibrator. RB-Cal's "regret relative to identity, per source" is a different object: a non-inferiority constraint, not a calibration-error constraint. Multicalibration papers therefore do not claim a "zero harmed groups" result.
- The label-free groups (embedding clusters, self-annotations) could feed RB-Cal directly. Self-annotation features ("is this a math question?", "does it need world knowledge?") could replace or augment the RBF feature space, because they are computable on unseen target tasks.
- The universal-adaptability line could support a target-level guarantee: treat each task as a covariate-shifted mixture of source data and require the density ratio to lie in the RBF-feature class. That would be a new kind of theorem for RB-Cal (a target-harm bound under a stated shift assumption), and I found no paper doing this for temperature fields or for LLM tasks.
- Mechanism to borrow: multicalibration's "patch only where a group shows a significant violation" boosting loop. Patching the pooled-TS field only in feature regions where evidence of improvement is significant across sources would tend to recover the gain that a global budget gives up.

### Gaps
- I did not verify whether Detommaso et al. evaluate on held-out datasets or tasks (their reported gains appear to be within-distribution). I fetched only the abstract and poster pages, not the full text.
- I did not check the original Hebert-Johnson et al. (ICML 2018) or Kim/Ghorbani/Zou multiaccuracy (2019) papers in this session. Treat them as background citations to verify.
- I found no paper applying multicalibration to next-token answer-letter probabilities across many MCQA tasks with a per-task non-degradation guarantee.

## Q2. Group-robust / minimax / domain-generalization calibration (Gong ICCV 2021, Yu MD-TS NeurIPS 2022, Doula ICML 2026, McKenna 2024, Guo ICML 2026, Kobalczyk 2026): what do they guarantee, and is a per-source regret budget with a confidence-bound margin already published?

### Takeaway
None of the works found combines (a) a per-source or per-domain non-degradation (regret) constraint, (b) a statistical upper-confidence margin on that constraint, and (c) transfer of a feature-conditioned temperature to unseen tasks. MD-TS, the closest, regresses per-domain temperatures on features (OLS). Its theory is an average-over-domains domain-adaptation bound, with no per-domain or worst-case guarantee. CALPORT/MRTS (our prior work) gives a minimax-regret one-parameter rule. The per-source budget plus UCB margin in RB-Cal therefore appears unpublished for calibration, but it is close in spirit to slice-wise non-inferiority release gates (Q3).

### Cited Findings
- **Gong et al., ICCV 2021**, "Confidence Calibration for Domain Generalization under Covariate Shift" (Gong, Lin, Yao, Dietterich, Divakaran, Gervasio; pp. 8958-8967).
  - Method: set-level calibration that learns a single temperature across multiple calibration domains, with no target data.
  - Theory: multiple calibration domains reduce the variance of the density ratio.
  - Result: reported ECE drop of 8.86 points versus domain-adaptation calibration baselines on Office-Home. Calibration transfers well to similar domains but less well to very different ones (Art, Clipart).
  - No per-domain guarantee. — [arXiv 2104.00742](https://arxiv.org/abs/2104.00742v2); [mlanthology](https://mlanthology.org/iccv/2021/gong2021iccv-confidence)
- **Yu, Bates, Ma, Jordan, NeurIPS 2022**, "Robust Calibration with Multi-domain Temperature Scaling" (MD-TS).
  - Motivation: pooled TS can be miscalibrated on individual domains.
  - Algorithm: fit TS separately per in-distribution domain, then regress the per-domain temperatures on penultimate-layer features with one OLS model pooled over domains. For a new input, T = <Psi(x), theta>; the test domain index is not needed.
  - Ablations: Ridge, Huber, kernel ridge and kNN gave no significant gain.
  - Theory: a generic DA-style OOD risk bound for calibration maps. It uses a weighted average of in-distribution risks plus an H-divergence term and an unestimated lambda. No per-domain or worst-case guarantee, and the main metric is the average per-domain ECE.
  - Datasets: ImageNet-C (76 domains), WILDS-RxRx1, GLDv2. — [arXiv HTML](https://arxiv.org/html/2206.02757v1); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b054fadf1ccd80b37d465f6082629934-Abstract.html)
- **Doula, ICML 2026**, "Conformal Calibration Transfer" (PMLR 306:26291-26318; arXiv 2609.10737).
  - Setting: labeled calibration data exists in a source space, and conformal prediction sets are needed in a target space linked by unlabeled paired observations (paired modalities, sensor changes).
  - Method (TCC): transport the source calibration data, correct residual mismatch using unlabeled target inputs, and claim finite-sample target coverage.
  - Diagnostics: label-free mismatch severity (delta+) and effective sample size (ESS%) to judge whether a transfer is trustworthy.
  - Scope: about coverage of prediction sets, not probability calibration or NLL regret. Needs paired data. — [PMLR](https://proceedings.mlr.press/v306/doula26a.html); [ICML poster](https://icml.cc/virtual/2026/poster/61667); [code](https://gitlab.com/achref.d/tcc)
- **McKenna & Carse, arXiv 2406.11456 (Jun 2024; also a NeurIPS 2023 page)**, "Calibrating Where It Matters: Constrained Temperature Scaling".
  - "Constrained" means fitting the temperature on the region of the simplex that matters for deployment costs (e.g., the validation subset with p < 0.5 in binary tasks). It is not a safety or regret constraint.
  - Data: ISIC 2019 dermoscopy only.
  - The authors note it may hurt overall expected cost, and that robustness to shift was not studied. — [arXiv](https://arxiv.org/abs/2406.11456v1)
- **Guo, Tao, Luo, Dong, Xu, ICML 2026**, "Sample Margin-Aware Recalibration of Temperature Scaling" (SMART; PMLR 306:38283-38315).
  - Method: a lightweight per-sample mapping from logit margin to temperature, built on the observation that the margin constrains the feasible temperature range for a target confidence.
  - Objective: the ICML version argues that minimizing NLL can worsen calibration and introduces Charbonnier-SoftECE, which provably upper-bounds smooth calibration error.
  - Scope: CNNs/ViTs, including distribution-shifted benchmarks. No per-group or no-harm guarantee. — [PMLR](https://proceedings.mlr.press/v306/guo26s.html); [ICML poster](https://icml.cc/virtual/2026/poster/64903); [arXiv 2506.23492](https://arxiv.org/pdf/2506.23492)
- **Kobalczyk & van der Schaar, arXiv 2605.13484 (May 2026)**, "Discovery of Hidden Miscalibration Regimes" (repo titled "Miscalibration Fields").
  - Method: defines a signed, input-dependent "miscalibration field", estimated by kernel smoothing in a learned calibration-aware representation.
  - Finding: across 4 LLM benchmarks and 12 LLMs, calibration heterogeneity is widespread and the discovered regimes are stable. Local corrections beat TS and isotonic regression in the flagged regions.
  - Scope: discovery and diagnosis. No harm guarantee for transfer to new tasks. — [arXiv](https://arxiv.org/abs/2605.13484); [author page](https://kasia-kobalczyk.com/publications/miscalibration-discovery); [code](https://github.com/kasia-kobalczyk/miscalibration-fields)
- **Thermometer (Shen et al., ICML 2024; arXiv 2403.08819).**
  - Method: a shared auxiliary model trained on multiple tasks predicts a dataset-specific temperature from unlabeled data, so it can calibrate an LLM on an unseen task.
  - Scope: MMLU, BIG-bench, MRQA. Transfers from a smaller to a larger model in the same family.
  - No per-task non-degradation guarantee. — [arXiv](https://arxiv.org/html/2403.08819v2); [mlanthology](https://mlanthology.org/icml/2024/shen2024icml-thermometer)
- **CALPORT / MRTS (our own prior work).** The repo describes a one-parameter, source-only minimax-regret temperature rule. It provably returns T = 1 when two sources disagree on the direction of correction (review draft dated 20 Oct 2026) — [GitHub](https://github.com/Huyvux12/calport-calibration-transfer)
- **Group DRO (Sagawa et al., ICLR 2020).** Worst-group objective. Naive group DRO fails in overparameterized nets without strong regularization, and with it gives 10-40 point worst-group accuracy gains. It targets accuracy, not calibration — [arXiv 1911.08731](https://arxiv.org/abs/1911.08731v2)
- **No combined method found.** A targeted search for "per-domain non-degradation constraint + upper confidence bound + temperature transfer" found no paper combining these. The nearest hits were MD-TS, Gong et al. and PseudoCal — [search-surfaced: MD-TS supplement](https://proceedings.neurips.cc/paper_files/paper/2022/file/b054fadf1ccd80b37d465f6082629934-Supplemental-Conference.pdf); [PseudoCal](https://arxiv.org/pdf/2307.07489)

### Inferences
- **What is already taken:**
  - "multi-domain → feature-conditioned temperature → unseen domains": MD-TS (2022).
  - "predict a task temperature from unlabeled data": Thermometer (2024).
  - "sample-wise temperature from label-free hardness features": SMART (2026).
  - "local miscalibration fields in LLMs": Kobalczyk (2026).
  - "minimax-regret TS returns identity on disagreement": CALPORT.
- **What RB-Cal adds** relative to MD-TS and Thermometer is a per-source regret constraint with a clustered-SE margin, plus a held-out deployment gate. That framing appears new for calibration, but the constraint itself is a standard non-inferiority test (see Q3). A paper should credit Seldonian, LTT and non-inferiority work rather than claim the constraint idea as novel.
- **Free features for the RB-Cal field:** SMART's margin feature and Kobalczyk's calibration-aware representation are cheap, label-free additions. They might close part of the gain gap, since ATS/Thermometer-style gains likely come from per-instance adaptivity.
- **Diagnostic idea:** Doula's label-free diagnostics (delta+, ESS%) suggest a per-target "trust score" that could gate whether the strong calibrator is applied on an unseen task.

### Gaps
- I did not fetch ATS (adaptive temperature scaling for LLMs) in this session, so its exact guarantees are unverified here.
- I did not verify whether CALPORT is under review at a specific venue (the brief says "anonymous under-review").
- I could not confirm whether MD-TS's appendix contains any per-domain worst-case result beyond the averaged bound summarized above.

## Q3. Safe policy improvement / "do no harm" / non-regressive model updates: techniques adaptable to choosing, per target task and label-free, between a strong and a safe calibrator

### Takeaway
The most transferable ideas are:
1. **SPIBB-style per-region fallback.** Use the improved policy where data are dense and copy the baseline where uncertainty is high.
2. **Slice-wise non-inferiority with incumbent fallback** (Zhang et al., Sep 2026). Intersection-union tests avoid a Bonferroni penalty, and a "no detected harm" gate is shown to be badly unsafe compared with a true non-inferiority gate.
3. **Disagreement-support certification** (DISCERN, Jul 2026). The risk difference between two models lives only where they disagree, and small disagreement can be certified with zero labels.
4. **LTT / Pareto Testing** for choosing hyperparameters under multiple risk constraints.

None of these is about probability calibration or unseen tasks.

### Cited Findings
- **SPIBB (Laroche, Trichelair, Tachet des Combes, ICML 2019).**
  - Bootstraps the learned policy with the baseline when uncertainty is high, so that with high confidence it approximately outperforms the baseline.
  - Variants: Pi-SPIBB copies baseline action probabilities on rarely-seen state-action pairs; a soft-constraint variant came from Nadjahi et al.
  - Simão et al. (AAMAS 2020) show that an estimated baseline loses little — [arXiv 1712.06924](https://ar5iv.labs.arxiv.org/html/1712.06924); [arXiv 1909.05236](https://arxiv.org/abs/1909.05236v2)
- **Zhang, Wu, Qian, "Certifying Model Upgrades with Slice-Wise Non-Regression and Incumbent Fallback"** (arXiv 2609.13714, 12 Sep 2026, preprint).
  - Requires no regression on any tracked slice relative to the incumbent, within per-slice tolerances.
  - Separates candidate search from independent paired evaluation, and returns the exact incumbent if certification fails.
  - Finite-sample guarantees for a frozen candidate, a finite candidate library, or a prespecified testing order, built on intersection-union and Learn-then-Test principles. A joint release needs no slice-count Bonferroni penalty, but power drops as slices grow.
  - Key number: in a 32-slice simulation, a "no-detected-harm" gate released a harmful candidate in 99.7% of trials, versus 2.6% for an exact non-inferiority gate.
  - On a digits experiment, certification was underpowered and the incumbent was returned in every run. Not about calibration — [arXiv](https://arxiv.org/abs/2609.13714)
- **Balachandran, "Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates"** (arXiv 2609.17560, Jul 2026, preprint).
  - "Support identity": the risk difference between two models lies on inputs where they disagree, which can be found without labels.
  - Tier 1 certifies benign updates with zero labels when the disagreement rate is below a tolerance. Tier 2 labels sampled disagreements under an anytime-valid confidence sequence.
  - Results over 785 update pairs and more than 14,000 audit streams: miscoverage 0.0002 at a nominal 5%, power 0.986, and 56% of benign updates certified with zero labels — [arXiv](https://arxiv.org/abs/2609.17560)
- **Elmecker-Plakolm et al., "Provably Safe Model Updates"** (arXiv 2512.01899, Dec 2025). Certifies a connected parameter region (a "local invariant domain") in which every parameter provably meets a performance requirement on the prior task. The guarantee is a priori, not a post-hoc check — [arXiv](https://arxiv.org/pdf/2512.01899)
- **Learn-then-Test (Angelopoulos, Bates et al.).**
  - Recasts risk control as multiple testing over candidate hyperparameters, with finite-sample guarantees for bounded risks. It can control several risks at once.
  - Pareto Testing (Laufer-Goldshtein, Fisch, Barzilay, Jaakkola) first finds Pareto-optimal configurations, then tests only along that frontier.
  - Adaptive LTT uses e-processes for sequential testing. A 2025 review covers these extensions — [LTT arXiv 2110.01052](https://arxiv.org/pdf/2110.01052v4); [Pareto Testing arXiv 2210.07913](https://arxiv.org/pdf/2210.07913); [Adaptive LTT](https://arxiv.org/html/2409.15844v2); [review 2502.04206](https://arxiv.org/html/2502.04206v1)
- **Label-free calibration under shift.**
  - LaSCal (NeurIPS 2024) estimates calibration error on unlabeled target data under label shift.
  - IW-GAE uses importance-weighted group accuracy for calibration and model selection in UDA.
  - PseudoCal is source-free and uses unlabeled target data only.
  - None selects between calibrators with a harm guarantee — [LaSCal](https://proceedings.neurips.cc/paper_files/paper/2024/file/783c5986e1d6112cb4688d9b2105609a-Paper-Conference.pdf); [IW-GAE](https://arxiv.org/pdf/2310.10611); [PseudoCal](https://arxiv.org/abs/2307.07489)
- **C3R (arXiv 2607.14157, Jul 2026, preprint, retrieval).** Certifies per-domain budgets with conformal risk control where feasible and otherwise abstains. Domains are inferred at serving time by a frozen router. The guarantee assumes exchangeability and is lost under deployment shift. Shown as an example of "certify-or-abstain per group" — [Pith](https://pith.science/paper/2607.14157)

### Inferences (candidate techniques for RB-Cal v2; these are proposals, not findings)
- **SPIBB-style "bootstrapped temperature field"** (Pi-SPIBB analog).
  - Fit the unconstrained pooled field (maximum gain). Then, at each point in feature space, shrink toward T = 1 or toward the safe field according to source-coverage density or cross-source disagreement in that RBF region.
  - This replaces one global budget, which caps gain everywhere, with region-local safety. Gain is kept in dense, consistent regions, and the method falls back only where targets look unlike any source.
  - I found no calibration paper doing this.
- **Label-free worst-case regret bound for switching calibrators (DISCERN analog).**
  - For temperature scaling, the per-example NLL difference between two temperatures can be bounded over all possible labels, because the label set is the finite answer-letter set. Its expectation can also be bounded with the source-predicted accuracy.
  - This gives a label-free upper bound on target-task harm. Switch to the strong calibrator when the bound is at most eps, otherwise use the safe one. Tier-1 "small disagreement certifies with zero labels" maps directly onto "small temperature difference on this task certifies".
  - This looks like a novel, checkable contribution ("label-free certified switching between calibrators").
- **Intersection-union non-inferiority across sources (Zhang et al. 2609.13714).** This formalizes RB-Cal's "Delta_w + 1.645 * clustered SE <= eps for every source" as a valid joint test without Bonferroni. Their 99.7% vs 2.6% simulation supports using a UCB (non-inferiority) gate rather than "no significant harm". RB-Cal already does this, and the paper should cite this line.
- **Pareto Testing over (eps, bandwidth, ...).** Use LTT/Pareto Testing over the hyperparameters with per-source risks. This turns RB-Cal's LOSO selection into a selection with a stated finite-sample guarantee at source level, and gives a principled frontier to report.
- **Mixture of calibrators with a per-target selector.**
  - Candidates: pooled TS, Thermometer, ATS, RB-Cal, identity.
  - Selector inputs: label-free diagnostics (Doula's delta+/ESS%, the disagreement magnitude, a source-density estimate).
  - Validation: leave-one-source-out, treating each held-out source as a pseudo-target, with an LTT-style test on the selector's harm rate. The guarantee would be at the meta level ("the fraction of harmed tasks is at most alpha, with tasks exchangeable"). That is more honest than a per-task guarantee and closes the gain gap whenever the selector picks the strong calibrator.

### Gaps
- I found no work that adapts SPIBB, slice-wise non-regression, or disagreement certification to probability calibration or NLL regret.
- I found no work giving a per-unseen-task (rather than per-source or exchangeable-task) no-harm guarantee without target labels. This may be impossible without assumptions; I found no impossibility result either.
- The 2026 preprints (2609.13714, 2609.17560, 2607.14157) are not peer-reviewed, and their claims rest on the abstract pages only.

## Q4. Work that explicitly studies "mean gain vs number of harmed tasks/groups" for calibration or post-hoc adaptation, and how the frontier is presented

### Takeaway
I found no calibration paper that reports a frontier of mean improvement against the number of harmed tasks or groups. The nearest are: the model-upgrade literature, which reports harmful-release rates against power or incumbent-fallback rates (Zhang et al. 2609.13714; DISCERN 2609.17560); risk-control work that presents a Pareto frontier of utility under constraints (Pareto Testing); and group-DRO work that reports average against worst-group metrics. A "mean NLL gain vs number of harmed targets" frontier, swept over eps, would be a new presentation for calibration.

### Cited Findings
- **Harmful release vs certification power.**
  - Zhang et al. report harmful-candidate release rates (99.7% for a no-detected-harm gate vs 2.6% for a non-inferiority gate in a 32-slice simulation).
  - They also report the cost side: the incumbent was returned in every digits run because certification was underpowered. This is the safety-utility trade-off in release terms — [arXiv 2609.13714](https://arxiv.org/abs/2609.13714)
- **Label cost vs certification.** DISCERN reports power 0.986, miscoverage 0.0002 (nominal 5%), and the share of benign updates certified with zero labels (56%) — [arXiv 2609.17560](https://arxiv.org/abs/2609.17560)
- **Pareto Testing.** Builds candidate configurations on a Pareto frontier of multiple risks and utility, then tests only frontier points. It is a template for showing gain against harm-constraint satisfaction — [arXiv 2210.07913](https://arxiv.org/pdf/2210.07913)
- **Average vs worst-group trade-off.** Group DRO frames this trade-off through worst-group accuracy improvements (10-40 points with regularization) — [arXiv 1911.08731](https://arxiv.org/abs/1911.08731v2)
- **Negative-flip constraints.** NLP model-update regression work (arXiv 2105.03048) and robustness-congruent training (TPAMI 2025) count negative flips (per-example regressions) as the harm metric, alongside overall accuracy — [arXiv 2105.03048](https://arxiv.org/pdf/2105.03048); [TPAMI](https://doi.org/10.1109/tpami.2025.3573237)
- **Pooled TS harms individual domains.** MD-TS shows pooled TS miscalibrates stand-alone domains, but reports average per-domain ECE, not counts of harmed domains — [arXiv HTML](https://arxiv.org/html/2206.02757v1)

### Inferences
- RB-Cal's headline result (0 harmed of about 370 units vs 5-12 of 60 harmed for pooled TS, Thermometer and ATS, at a cost of about 5-7% of mean NLL gain) can be presented in the style of the model-upgrade literature. Plot mean target NLL gain (y) against the number or fraction of harmed targets (x), sweeping eps and the bandwidth for RB-Cal. Show each baseline as a single point, and draw the frontier.
- A SPIBB-style or label-free-switching variant (Q3) succeeds if it moves the frontier up at x = 0.
- Reporting a "negative-flip"-style per-unit regression count beside mean gain is accepted practice in model-update work, which helps reviewers see the metric as standard.

### Gaps
- I found no paper on LLM calibration transfer that counts harmed tasks as a primary metric. Thermometer, MD-TS and SMART all report average calibration metrics, from the pages I checked.
- I did not search ACL Anthology specifically for "negative transfer" counts in LLM calibration. A follow-up search there could still find a precedent.
