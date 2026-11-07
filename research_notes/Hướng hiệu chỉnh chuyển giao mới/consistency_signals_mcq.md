# Perturbation- and Consistency-Based Confidence Signals for LLM Multiple-Choice Answers, and Their Use for Calibration

Scope note: researched 2026-11-06. Evidence is tagged **[MCQ-letter]** when it comes from next-token answer-letter probabilities on multiple-choice QA, which is our setting. It is tagged **[gen]** when it comes from sampled free-text or CoT generations, and **[cls]** for classification label tokens. Numbers marked "abstract-level" were not checked against full tables.

## Q1. Option-order sensitivity on MCQ: what do the core papers report about its relation to correctness and calibration?

### Takeaway
Order sensitivity is large: best-to-worst accuracy gaps from oracle reordering reach 13-85 pp. It is concentrated on items where the model is uncertain between its top-2 or top-3 options, and its size tracks task error rate across tasks. These papers treat it as a bias to remove and report accuracy, not calibration. None of the core papers (Pezeshkpour & Hruschka, PriDe, Wang et al., Tang et al.) reports ECE or NLL for order-inconsistency as a confidence feature. Our finding that inconsistent rows are strongly overconfident fits their "uncertainty meets positional bias" explanation, but we found no paper that measures it the way we do.

### Cited Findings
- **Pezeshkpour & Hruschka, "LLMs Sensitivity to the Order of Options in MCQ", Findings of NAACL 2024 (arXiv Aug 2023) [MCQ, GPT-4/InstructGPT/Llama-2-13b-chat; mostly generated letters, not logprobs]:**
  - Oracle reordering gives a sensitivity gap of about 13% to 85% across benchmarks, and the gap persists in few-shot settings. — [ACL Anthology](https://aclanthology.org/2024.findings-naacl.130)
  - Table 1 examples: GPT-4 on Professional Law has 66.1 vanilla accuracy with a range of -12.7/+12.1. Llama-2-13b on High School Chemistry has 40.6 with a range of -32.7/+45.6. Even GPT-4 at more than 90% accuracy (Logical Deduction) has a gap of about 13 pp. — [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
  - Conjecture 4.1: sensitivity comes from (1) uncertainty between the top possible choices and (2) positional bias that favours certain placements. — [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
  - Evidence for the conjecture:
    - The sensitivity gap "exhibits a strong correlation with the error rate" across model×task points (Fig. 3).
    - More than 60% of GPT-4's sensitive items are also sensitive for InstructGPT.
    - When asked whether more than one choice could be highly probable, the models say "yes" for more than 94% of sensitive items.
    - [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
  - Majority vote over 10 random reorders raises accuracy by up to 8 pp (InstructGPT Logical Deduction +8.0, Professional Law +5.7; GPT-4 0.0 to +2.0). Multiple Evidence Calibration (explain-then-answer) often lowers accuracy (InstructGPT -1.4 to -6.9). — [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
  - The authors report that fixing only option *position* leaves less than 50% of the reordering gap, so order effects are not just ID-token bias. — [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
- **Zheng et al., "LLMs Are Not Robust Multiple Choice Selectors" (PriDe), ICLR 2024 [MCQ-letter, 20 LLMs, MMLU/ARC-C/CSQA]:**
  - Selection bias comes mainly from *token bias*: an a-priori probability mass on specific option-ID tokens. Position bias is secondary, weaker and less regular. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03882); [ICLR page](https://iclr.cc/virtual/2024/poster/17638)
  - The predictions that PriDe changes usually had low original confidence, and the debiased answer was usually in the original top-2. For gpt-3.5-turbo, PriDe changed 4.0% of predictions: 40.8% went wrong→right and 34.5% right→wrong. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03882)
  - The paper reports no ECE or calibration analysis. It only conjectures that the mismatch between priors and recalls may relate to calibration and leaves this to future work. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03882)
  - Code: [github.com/chujiezheng/LLM-MCQ-Bias](https://github.com/chujiezheng/LLM-MCQ-Bias)
- **Wang et al., "LLMs are not Fair Evaluators" (arXiv May 2023; LLM-as-judge, pairwise) [gen/judge]:**
  - Swapping response order flips verdicts: Vicuna-13B "beats" ChatGPT on 66/80 queries when ChatGPT is the judge.
  - Balanced Position Calibration averages over both orders.
  - "Balanced Position Diversity Entropy" scores per-example difficulty and routes uncertain cases to humans. This is the closest early use of order-disagreement as an uncertainty or difficulty signal.
  - [arXiv](https://arxiv.org/abs/2305.17926); code [github.com/i-Eval/FairEval](https://github.com/i-Eval/FairEval)
- **Tang et al., "Found in the Middle: Permutation Self-Consistency", NAACL 2024 [listwise ranking, gen]:** The method shuffles list order, aggregates to a central ranking, and gains up to 7-18% (GPT-3.5) and 8-16% (LLaMA-2-70B). It is used for accuracy and debiasing, not confidence. — [arXiv](https://arxiv.org/abs/2310.07712)
- **Guda et al., arXiv Nov 2025 (also listed as Findings of IJCNLP 2025) [MCQ-letter, ~3B models]:**
  - The authors propose the label-free Permutation Bias Metric (PBM): the variance of each option's probability across permutations.
  - Baseline PBM was highest on MedMCQA, which also had the lowest accuracy. The authors link bias to difficulty.
  - No ECE or calibration analysis.
  - [arXiv](https://arxiv.org/html/2511.21709); [ACL Anthology PDF](https://aclanthology.org/2025.findings-ijcnlp.127.pdf)
- **Hanna & Feng, arXiv Aug 2026 [MCQ, 6 models]:** Flip rate under cyclic permutation is a different quantity from accuracy. GPT-4.1-mini's MMLU flip rate roughly halved (21.6→11.8) under two-stage prompting while accuracy fell. Flip rate and RStd can also disagree. No calibration metrics. — [arXiv](https://arxiv.org/html/2608.11947v1)

### Inferences
- The literature predicts our row-level pattern: inconsistent rows are the top-2-uncertain rows, so overconfidence concentrates there. It also predicts our task-level pattern: sensitivity rises with error rate, which fits our observation that even consistent rows are overconfident on hard tasks such as MMLU-Pro law. Consistency between two orders is therefore a *row-level* difficulty signal. It does not fully capture *task-level* difficulty, which explains why the task-batch mean adds a separate signal.
- Our per-row TV between 2 orders is close to Guda et al.'s PBM, restricted to 2 permutations, and to the flip-rate statistic. Flip rate and PBM can rank tasks differently (Hanna & Feng), so a continuous divergence (TV, JS, PBM-style variance) is a better feature than a binary flip.

### Gaps
- We found no paper that reports ECE or NLL conditioned on order consistency for MCQ letter probabilities, which is what we measure. Our consistent vs inconsistent conf-acc gap table looks new as a descriptive result.
- Pezeshkpour & Hruschka's appendix on logprob-based uncertainty was not read.

## Q2. Consistency-based calibration and uncertainty (sample consistency, semantic entropy, SPUQ, prompt agreement, prompt ensembles): signals, costs, gains, and cross-task transfer

### Takeaway
The strongest *calibration* evidence for perturbation signals in an MCQ token-probability setting comes from **prompt agreement and prompt ensembles**: Portillo Wightman et al. 2023, CAPE/Jiang et al. 2023, Hua et al. 2023, and BayesPE 2024 on classification. In these papers, averaging letter distributions over paraphrases or permutations improves ECE and Brier, and it works best *combined with* temperature scaling. Sample-consistency (Lyu et al.), SPUQ and semantic entropy are mostly evidence on generation tasks. For MCQ, semantic entropy reduces to entropy over extracted letters. Most of these papers do not study cross-task transfer of a learned calibrator, and none applies per-task safety constraints.

### Cited Findings
- **Portillo Wightman, DeLucia & Dredze, "Strength in Numbers: Estimating Confidence of LLMs by Prompt Agreement", TrustNLP@ACL 2023 [MCQ, 10 datasets; T0, FLAN-T5, GPT-3]:** Confidence is measured as agreement across multiple prompts, including LM-generated prompts. It is "more calibrated" than the log-probability of the answer under a single prompt (abstract-level). — [ACL Anthology](https://aclanthology.org/2023.trustnlp-1.28)
- **Jiang, Ruan, Huang, Liao, Pitis, Grosse, Ba, "Calibrating Language Models via Augmented Prompt Ensembles" (CAPE), ICML 2023 workshop (Challenges in Deployable Generative AI) [MCQ-letter]:**
  - Prompt augmentations include template paraphrasing and *option permutation*. The per-variant answer-token distributions are averaged, with no training. — [ICML virtual](https://icml.cc/virtual/2023/28934)
  - We did not retrieve CAPE's own ECE numbers.
- **Hua, Svegliato, Toyer, Russell, Sinha, "Calibrating Black Box LLMs with Prompt Variations" (workshop paper, 2023) [MCQ-letter, text-davinci-003, 9 QA sets × 100 questions]:**
  - The authors generate LLM-written paraphrases of each question, average the token probability distributions, then apply temperature scaling.
  - ECE results (Table 1, "All" column):
    - none: 0.141
    - variations only: 0.116
    - temperature scaling only: 0.090
    - variations + TS: 0.039
  - The authors summarise this as "decreases ECE by over 70% and Brier by over 34% … minimal impact on accuracy". Accuracy went 0.64→0.62, and some tasks dropped (OpenBookQA 0.71→0.69).
  - [PDF](https://static1.squarespace.com/static/6266e3c48fb0751e74f60eb6/t/6671d62520d991084c200225/1718736422125/HSTSRrelm23.pdf)
  - Caveats: n=100 per task, one model, and the paper does not clearly say how the TS temperature was fit.
- **Tonolini, Massiah, Aletras, Kazai, "Bayesian Prompt Ensembles" (BayesPE), Findings of ACL 2024 [cls; 10 classification tasks; Falcon, MPT, Mistral-7B-Instruct]:**
  - The method uses a weighted ensemble of N=5 semantically equivalent instruction prompts. The weights come from approximate variational inference on a small labeled validation set (50 examples), with an entropy regulariser toward uniform weights.
  - Average relative change vs a single prompt over 100 experiments:
    - BayesPE: ECE -32.1%, AUC +23.7%
    - uniform ensemble: ECE -20.0%
    - best-single-prompt: ECE -26.3%
  - NLL was best or tied in 96/100 experiments for BayesPE vs 37/100 for the uniform ensemble.
  - [PDF](https://aclanthology.org/2024.findings-acl.728.pdf)
- **Lyu et al., "Calibrating LLMs with Sample Consistency", AAAI 2025 (arXiv Feb 2024) [gen/CoT reasoning; 9 datasets incl. GSM8K, StrategyQA, CLUTRR; 11 models]:**
  - Confidence measures: agreement, entropy, and first-second distance (FSD) over n=40 samples at T=0.4. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.13904); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34120)
  - Consistency beats the logit, P(True) and verbalized baselines by a wide margin. Brier: LLaMA-7B logit .474 vs consistency .232-.241; Mistral-7B-instruct .384 vs .215-.220; GPT-4 entropy .116 vs verbalized .154. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.13904)
  - Most of the gain arrives with 3-5 samples and saturates around 15-20. **Instruction tuning makes calibration harder:** Mistral-7B-instruct is worse than base. No post-hoc TS or cross-task calibration over consistency scores is studied. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.13904)
  - Code: [github.com/veronica320/Calibrating-LLMs-with-Consistency](https://github.com/veronica320/Calibrating-LLMs-with-Consistency) (per search result)
- **Gao, Zhang, Mouatadid, Das, "SPUQ", EACL 2024 [mostly gen]:**
  - SPUQ perturbs the prompt (paraphrase, system message, dummy tokens) and the sampling temperature, then aggregates outputs. Perturbation targets epistemic uncertainty and sampling targets aleatoric uncertainty.
  - It reports about a 50% average ECE reduction (abstract-level).
  - [ACL Anthology](https://aclanthology.org/2024.eacl-long.143); [ar5iv](https://ar5iv.labs.arxiv.org/html/2403.02509)
- **Zhang et al., "Calibrating the Confidence of LLMs by Eliciting Fidelity" (UF Calibration), EMNLP 2024 [MCQ via sampling; 6 RLHF LMs, 4 MCQA sets]:**
  - Confidence is split into "uncertainty", the entropy of sampled answers, and "fidelity": the model is re-queried after the chosen option is replaced with "All other options are wrong", and the answer chain is checked.
  - The authors list permutation-based probability calibration as complementary prior work. It needs no logits.
  - [PDF](https://aclanthology.org/2024.emnlp-main.173.pdf)
- **SteeringConf (Zhou, Jin, Shi, Li; arXiv Mar 2025) [verbalized confidence, 7 benchmarks incl. MMLU Law/Ethics]:**
  - The method aggregates 5 "confidence-steering" prompts, from very cautious to very confident.
  - It reports up to 39.8% ECE reduction vs vanilla verbalized confidence, and a CoT GPT-3.5 average ECE of 11.7% vs 18.7% for the best sampling baseline.
  - No TS or logit-based baselines.
  - [arXiv](https://arxiv.org/html/2503.02863v1/)
- **Semantic entropy (Kuhn et al. ICLR 2023; Farquhar et al., Nature 2024) [gen]:**
  - The method clusters sampled generations by meaning (bidirectional NLI) and takes entropy over clusters. It is designed for free-form generation and hallucination detection. — [Nature](https://www.nature.com/articles/s41586-024-07421-0); [arXiv Kuhn](https://arxiv.org/abs/2302.09664)
  - For MCQ, clustering reduces to grouping by the extracted letter. — [arXiv 2606.08098](https://arxiv.org/pdf/2606.08098)
  - A frequency-based "discrete" variant exists for black-box use. — [arXiv 2509.14478](https://arxiv.org/html/2509.14478v2)
  - For black-box MCQA, frequency-based predictive entropy matched or beat logit-based entropy in AUROC across 6 LLMs and 4 datasets. — [arXiv 2508.05544](https://arxiv.org/html/2508.05544)
- **Learned calibrators on consistency features with cross-task transfer [mostly gen]:**
  - Graph-based Confidence Calibration (arXiv Nov 2024): a GNN over self-consistency graphs of multiple responses, with reported cross-domain transfer. — [arXiv](https://arxiv.org/html/2411.02454)
  - Agentic Confidence Calibration (arXiv Jan 2026): a small calibrator on trajectory features including stability and consistency. It transfers zero-shot to GAIA but weaker out of domain (GPQA). — [arXiv](https://arxiv.org/html/2601.15778v1)
  - CCPS (arXiv May 2025): features from representation stability under perturbation, fed to a lightweight classifier. — [arXiv](https://arxiv.org/pdf/2505.21772)
  - These are snippet-level findings; full papers were not read.

### Inferences
- For our setting, the transferable lessons are: (i) average the letter distributions over the perturbation views, then (ii) temperature-scale the average. Hua et al. show that these effects are complementary (ECE .141→.116 with averaging, →.090 with TS, →.039 with both). That is direct [MCQ-letter] evidence, though small-scale.
- Sample consistency with n≥5 CoT samples is a different, much more expensive signal: generation per sample versus one forward pass per view. Lyu et al.'s large Brier gains are against a weak "inverse perplexity of generated answer" baseline on reasoning tasks. They do not transfer directly to letter-probability MCQ, where the logit baseline is much stronger.
- Lyu et al.'s instruction-tuning finding matters because all our models are instruction-tuned. Instruction-tuned models' letter distributions are sharp, so a single forward pass carries little uncertainty information and perturbation views add more.
- BayesPE is a labeled-validation weighting of prompts. In a cross-task setting, its weights would have to be learned on sources and transferred, which is analogous to our temperature transfer.

### Gaps
- No ECE numbers retrieved for CAPE (Jiang et al. 2023). The full SPUQ tables, including which datasets are MCQ, were not checked.
- We found no study that compares, under a matched compute budget, letter-probability TS on a perturbation-averaged distribution against sample-consistency on the same MCQ benchmarks.

## Q3. Does averaging probabilities over answer orders (and prompts) itself improve accuracy and calibration, and by how much? Could it change our "gap is unavoidable" conclusion?

### Takeaway
Yes, for accuracy. Cyclic-permutation averaging gives about +5 pp on average across 20 LLMs (MMLU +4.9, ARC +5.7, 0-shot), and majority voting over permutations gives up to +8 pp (Pezeshkpour & Hruschka), with much larger gains on small models. For calibration, the only direct MCQ-letter evidence we found is Hua et al. 2023: averaging over prompt variations reduces ECE, and averaging plus TS is far better than either alone. PriDe-style prior debiasing is cheap but gives smaller and sometimes negative gains. This changes the *base predictor* for all methods, not specifically for RB-Cal. Whether it narrows our gap to the unconstrained baselines is an empirical question. There is a plausible mechanism by which it could, described under Inferences.

### Cited Findings
- **PriDe paper, accuracy deltas vs default, 0-shot, averaged over 20 LLMs:**
  - MMLU:
    - Cyclic Permutation (×n cost): +4.9
    - PriDe 5% (×1.15 cost): +1.2
    - PriDe 40%: +2.6
    - PriDe 80%: +4.1
  - ARC: Cyclic +5.7; PriDe 5% +1.3; PriDe 40% +2.9; PriDe 80% +4.9.
  - Example, gpt-3.5-turbo MMLU: 67.2 → 68.0 (PriDe 5%) → 69.8 (Cyclic).
  - Cyclic permutation changes more predictions and "flattens the distribution, including high-confidence ones". PriDe priors estimated in one domain transfer reasonably well, but degrade under large domain gaps (e.g. MMLU STEM→ARC).
  - [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03882)
- Majority vote over 10 random reorders: up to +8 pp (InstructGPT); GPT-4 0 to +2 pp. — [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf)
- Guda et al. 2025: full majority voting over up to 24 permutations drives PBM to 0. On QASC it lifts Phi-2 from 0.630 to 0.933 and Llama-3.2-3B from 0.489 to 0.837. KV-cache reuse of the question prefix cuts the time of a 24-permutation vote by up to 84-89%. — [arXiv](https://arxiv.org/html/2511.21709)
- Hanna & Feng 2026: cyclic permutation improved accuracy in 5/6 model pairs on both MMLU and ARC-C. PriDe *reduced* local Llama-3.1-8B accuracy (MMLU 50.2→44.2; ARC 58.0→48.8). — [arXiv](https://arxiv.org/html/2608.11947v1)
- Hua et al. 2023, calibration [MCQ-letter]: averaging over prompt variations cuts ECE from 0.141 to 0.116. With TS it falls to 0.039, vs 0.090 for TS alone. Brier: 0.32 → 0.27 (avg), 0.22 (TS), 0.21 (both). — [PDF](https://static1.squarespace.com/static/6266e3c48fb0751e74f60eb6/t/6671d62520d991084c200225/1718736422125/HSTSRrelm23.pdf)
- BayesPE: a uniform prompt ensemble cuts ECE about 20% relative to a single prompt. Validation-weighted prompts cut it about 32%. — [PDF](https://aclanthology.org/2024.findings-acl.728.pdf)
- Teacher-student permutation debiasing (arXiv Mar 2024) distills permutation-averaged outputs into a model so that inference needs one pass. — [arXiv](https://arxiv.org/pdf/2403.13590)
- PA-GRPO (arXiv Mar 2026) trains for consistency across all 24 arrangements on 4-option MCQ. — [arXiv](https://arxiv.org/html/2603.21016v1)

### Inferences (for RB-Cal; not from sources)
- **We already have 4 views per row (2 orders × 2 templates).** The cheapest strong variant is to make the *4-view averaged letter distribution* the base predictor, either as a probability mean (CAPE/Hua-style) or as a log-prob mean / geometric mean (PriDe-style prior estimation). The temperature field would then be learned on top of it, with no new inference.
- Options for the averaging space and features:
  - Average in probability space, which softens the distribution and acts like a built-in temperature on inconsistent rows.
  - Average in log space and renormalize, which stays sharper.
  - Use the per-view temperatures implied by the disagreement as features.
- **Mechanism by which this could narrow the gap to unconstrained baselines:**
  - Disagreement-weighted averaging is a row-adaptive softening. It is applied *before* any learned parameter, so it cannot "harm" a task by transfer error in the way a learned temperature can.
  - If most of the unconstrained baselines' extra NLL gain comes from softening inconsistent rows on easy tasks, averaging captures part of that gain at zero regret risk. RB-Cal's budget then has to forgo less.
  - Conversely, the hard-task overconfidence we see on consistent rows (MMLU-Pro law, +0.39) will *not* be fixed by averaging. Consistent rows are unchanged by averaging, so a task-level temperature is still needed there.
  - **Prediction:** averaging helps most on easy and medium tasks, and the residual gap on hard tasks remains.
- **Fair comparison requirement:** the unconstrained baselines (pooled TS, Thermometer, ATS) must also be run on the averaged predictor. Otherwise any narrowing is confounded with a base-predictor upgrade. The paper claim would then be "the constrained-unconstrained gap on the best available base predictor", which can legitimately be smaller.
- Two orders is a weak permutation set for 4-10 options. With 2 orders, ID-token bias is only partly cancelled. Cyclic permutation (n views) is the standard debiasing set. If extra compute is allowed, n cyclic shifts with KV-cache reuse (Guda et al.) are cheap for short MCQ prompts.
- PriDe needs no labels: it estimates the per-model letter prior from permutations of a few % of unlabeled *target* rows. That makes it compatible with transfer to unseen tasks. The prior could be (a) divided out to debias the base predictor and (b) used as a feature, for example the KL from a row's distribution to the model prior. Hanna & Feng's negative PriDe result on Llama-3.1-8B is a warning to validate per model.

### Gaps
- We found no large-scale MCQ-letter study reporting ECE or NLL for cyclic or full-permutation averaging specifically; Hua et al. use paraphrases, n=100 per task, and one model. The size of calibration gains from order-averaging on our 5 models is unknown until measured.
- It is unknown whether order-averaging changes the *ranking* of RB-Cal vs pooled TS/Thermometer/ATS. No paper compares constrained vs unconstrained calibrators on averaged predictors.

## Q4. Is there published work combining order/prompt-consistency features with cross-task transfer of calibration, or with per-task safety constraints? Closest matches

### Takeaway
We found no published work that (a) uses answer-order or prompt-consistency features inside (b) a post-hoc temperature model transferred to unseen tasks, under (c) per-source or per-task regret or harm budgets. The pieces exist separately:
- **Cross-task temperature transfer without consistency features:** Thermometer, ICML 2024.
- **Consistency or perturbation features feeding a learned calibrator, mostly for generation, with loose transfer claims:** GNN self-consistency graphs, agentic calibrators, CCPS.
- **Prompt or permutation ensembles weighted or calibrated with labeled data, but per-task, not transferred:** BayesPE, Hua et al.
- **Label-free per-task priors that transfer across domains:** PriDe.

The combination in RB-Cal v4b looks novel. The novelty claim should be phrased carefully because the search was broad but not exhaustive.

### Cited Findings
- **Thermometer (Shen et al., ICML 2024):**
  - It learns an auxiliary model from data across multiple tasks that predicts a temperature for a new task, and it preserves accuracy.
  - The input is the LLM's representations of unlabeled task inputs; this detail comes from our own background, not from the fetched abstract.
  - It has no consistency features and no per-task safety constraint.
  - [arXiv](https://arxiv.org/abs/2403.08819)
- **Graph-based Confidence Calibration (arXiv Nov 2024):** an auxiliary GNN over self-consistency graphs. The authors attribute strong transfer to self-consistency being a broadly applicable signal. Generation tasks; no per-task constraint. — [arXiv](https://arxiv.org/html/2411.02454)
- **Agentic Confidence Calibration (arXiv Jan 2026):** trajectory-level features, including stability and consistency. The pretrained calibrator reaches the best zero-shot ECE on the unseen GAIA task but is weaker out of domain (GPQA). — [arXiv](https://arxiv.org/html/2601.15778v1)
- **Future Confidence Distillation (arXiv Jul 2026):** predictors transfer to unseen datasets within a domain, and transfer is weaker across domains. — [arXiv](https://arxiv.org/pdf/2607.07626)
- **Unsupervised calibration from a single generation (arXiv Apr 2026):** self-consistency on unlabeled questions serves as a proxy target for a distilled one-pass predictor, with reported transfer across task-level shifts. — [arXiv](https://arxiv.org/html/2604.19444v1)
- **BayesPE:** prompt-ensemble weights are fit per task on 50 labeled examples, with no cross-task transfer of weights. — [PDF](https://aclanthology.org/2024.findings-acl.728.pdf)
- **PriDe:** the prior estimated on one domain transfers to others reasonably well, but degrades with large domain gaps. This is the only cross-domain transfer result for a *permutation-derived* MCQ quantity we found. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.03882)
- **Guda et al. 2025:** a LoRA trained to minimise permutation variance on one dataset transfers to others and reduces PBM and flip rate. This is a training-time intervention, not post-hoc calibration. — [arXiv](https://arxiv.org/html/2511.21709)
- **Safety framing:** a 2026 paper argues that strong AUROC does not imply reliable selective prediction at strict risk thresholds, and recommends AURC and risk-coverage reporting. This is related in spirit to "do no harm" constraints but uses no per-task budgets. — [arXiv](https://arxiv.org/pdf/2603.21172)
- **Wang et al. 2023:** uses order-diversity entropy to route hard examples to humans. This is a per-example safety use of order disagreement, but for LLM judges and without calibration transfer. — [arXiv](https://arxiv.org/abs/2305.17926)

### Inferences
- **Positioning:** RB-Cal v4b can claim to be the first to use answer-order inconsistency as a covariate for *cross-task, regret-constrained* temperature transfer. The claim should cite Thermometer for transfer, Portillo Wightman/Hua/BayesPE for prompt-agreement calibration, and Pezeshkpour/PriDe for order sensitivity. It should not claim novelty for order-inconsistency as an uncertainty signal per se, which is implicit in Pezeshkpour & Hruschka and Wang et al.
- **Stronger feature variants to try, ranked by expected value per compute.** None need new inference except where noted.
  1. Cross-template disagreement (TV/JS between the 2 templates), separate from cross-order disagreement. They are different perturbations (paraphrase vs position), as in SPUQ and prompt agreement.
  2. The 4-view averaged distribution as the base predictor (Q3). Its entropy and max-prob would replace the single-view confidence features.
  3. Argmax-agreement count over the 4 views (0-4), plus FSD (first-minus-second vote share), following Lyu et al.'s agreement and FSD measures applied to views instead of samples.
  4. Rank of the single-view argmax in the other views (top-2 flip vs far flip). Pezeshkpour shows flips are mostly within the top-2.
  5. Label-free PriDe-style per-model, per-task letter prior from target unlabeled rows, used as a debiasing step and/or as a KL-to-prior feature.
  6. A task-level feature: the variance of the inconsistency distribution, not just its mean. The literature links task-level sensitivity to error rate, which may help the hard-task, consistent-row overconfidence.
  7. If compute allows, n cyclic orders instead of 2. This needs new inference; KV-cache reuse makes it cheap.

### Gaps
- No exhaustive search of NeurIPS/ICML 2025-2026 proceedings was done. A closely related paper on calibration transfer with consistency features may exist that our searches missed.
- The full Thermometer paper was not re-read to confirm its exact input features and its harm statistics on unseen tasks.
