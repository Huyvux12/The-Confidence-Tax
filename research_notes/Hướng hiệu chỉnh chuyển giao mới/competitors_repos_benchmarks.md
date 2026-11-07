# Competitors, repos and benchmarks for cross-task LLM calibration transfer (state as of Nov 2026)

Scope note: these notes come from about 20 web searches and fetches done on 2026-11-06. Items marked "(unverified)" were not opened during this session. Items in Gaps come from prior knowledge only and still need checking before they are cited.

## Q1. What are the strongest published methods for calibrating an LLM on a task it was not calibrated on? What results, settings and code do they report?

### Takeaway
Thermometer (ICML 2024) is still the reference method for unseen tasks. It is white-box and uses leave-one-task-out within one benchmark family. The 2025-2026 additions take different routes: DACA uses unlabeled target data and the pre-trained model as a teacher, CALM changes the model's weights, calibration-tuning and RLCR fine-tune the model, and Probe-SD distils hidden-state confidence. None of them is a post-hoc, source-only calibrator with a guarantee that it will not harm any task. ATS has no verified public code.

### Cited Findings
**Thermometer: Towards Universal Calibration for LLMs** (Shen, Das, Greenewald, Sattigeri, Wornell, Ghosh; ICML 2024, PMLR v235)
- An auxiliary network maps the LLM's feature-extractor output phi(x), taken on unlabeled prompts, to one temperature per task. It needs access to internal representations and does not use completions at test time — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2); [PMLR](https://proceedings.mlr.press/v235/shen24c.html)
- Models: LLaMA-2-Chat 7B (transferred to 13B and 70B) and FLAN-T5-XL (transferred to XXL and Flan-UL2). Tasks: MMLU (57 subjects), 23 BIG-bench MC tasks, and MRQA free-form QA. Protocol: leave-one-task-out, averaged over 5 trials — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2)
- MMLU results, LLaMA-2-Chat 7B. ECE: vanilla 0.260, TS-CV 0.119, CAPE 0.196, Thermometer 0.078, oracle TS 0.062. NLL: 1.760 / 1.253 (TS-CV) / 1.220 (Thermometer). BIG-bench ECE: 0.233 / 0.087 (TS-CV) / 0.090 (Thermometer), so Thermometer is roughly equal to TS-CV there. MRQA ECE: 0.127 / 0.071 / 0.065 — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2)
- Code: [github.com/maohaos2/Thermometer](https://github.com/maohaos2/Thermometer). The link appears in the paper; the repo contents were not inspected.
- Stated future work: extend to summarization, translation and larger LLMs. The authors note that BIG-bench's diversity makes temperature prediction harder and produces outliers — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2)

**ATS: Calibrating Language Models with Adaptive Temperature Scaling** (Xie, Chen, Lee, Mitchell, Finn; Stanford; arXiv 2409.19817, ICLR 2024 SeT-LLM workshop, EMNLP 2024 per the user's context)
- ATS predicts a temperature per token from features and is fitted on a standard SFT dataset. It reports 10-50% calibration gains on three benchmarks after RLHF — [arXiv 2409.19817](https://arxiv.org/pdf/2409.19817); [MLAnthology](https://mlanthology.org/iclrw/2024/xie2024iclrw-calibrating)
- No public code link was found. SOTAVerified lists "Code Available 0" — [sotaverified](https://sotaverified.org/papers/calibrating-language-models-with-adaptive)

**DACA: Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator** (Luo, Wang, Li, Wei; NeurIPS 2025; arXiv 2505.16690)
- DACA is post-hoc and unsupervised. It fits the temperature of a post-trained LLM by aligning its confidence with a pre-trained model, using only the examples where the two models agree. This needs unlabeled data from the target task plus a reference pre-trained model — [arXiv](https://arxiv.org/abs/2505.16690); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ef6b667990416111e3ebbeeba4a8f283-Abstract-Conference.html)
- Reported: up to 15.08% average ECE improvement. GPT-4o's ECE on MedMCQA falls from 21.23% to 6.99% when Gemma-3-12B is the calibrator — [NeurIPS / arXiv](https://neurips.cc/virtual/2025/poster/118852)
- No GitHub link appears on the arXiv abstract page — [arXiv abs](https://arxiv.org/abs/2505.16690)

**CALM: Beyond Post-Hoc Temperature Scaling: Bilevel Optimization for LLM Calibration** (Jin, Wang, Dai, Xiao, Hou; arXiv 2608.07419, 7 Aug 2026)
- CALM is white-box and fine-tunes with QLoRA. The upper level of the bilevel problem maximizes entropy. The parameters are per-vocabulary logit shifts plus class-wise temperatures. The authors argue that post-hoc temperature scaling is domain-dependent and does not transfer — [arXiv 2608.07419](https://arxiv.org/html/2608.07419)
- Models: Tulu-3.1-8B, Vicuna-7B, OLMo-2-7B, Mistral-7B. The model is trained on Alpaca and tested on MMLU, MedMCQA, OBQA and ARC-C. Out-of-distribution MCQA conf-ECE for Llama-3.1 goes from 0.1784 to 0.1050. In-distribution, plain TS gets the best ECE on most models. Baselines are TS, label smoothing and CFT; Thermometer and ATS are not compared. Code: [github.com/BojianHou/calm](https://github.com/BojianHou/calm) — [arXiv 2608.07419](https://arxiv.org/html/2608.07419)

**Calibration-tuning: Large Language Models Must Be Taught to Know What They Don't Know** (Kapoor et al.; NeurIPS 2024; arXiv 2406.08391)
- LoRA fine-tuning on about 1,000 graded correct and incorrect answers. The resulting uncertainty estimates generalize to new question types and tasks under distribution shift, and they can also estimate other models' uncertainty. Code: [github.com/activatedgeek/calibration-tuning](https://github.com/activatedgeek/calibration-tuning) — [arXiv 2406.08391](https://arxiv.org/html/2406.08391v3); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract.html)

**APRICOT: Calibrating LLMs Using Their Generations Only** (Ulmer, Gubri, Lee, Yun, Oh; ACL 2024)
- Black-box. A DeBERTa-v3 auxiliary model predicts confidence from the question and the generated text. Its targets come from clustering the inputs with an embedding model. Code: [github.com/parameterlab/apricot](https://github.com/parameterlab/apricot). Calibrators are on HF, e.g. [parameterlab/apricot_clustering_trivia_qa_deberta-v3-base_for_gpt-3.5-turbo-0125](https://huggingface.co/parameterlab/apricot_clustering_trivia_qa_deberta-v3-base_for_gpt-3.5-turbo-0125) — [ACL Anthology 2024.acl-long.824](https://aclanthology.org/2024.acl-long.824.pdf)

**Verbalized and RL-trained confidence**
- **RLCR**, "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty" (Damani et al.; arXiv 2507.16806; ICLR 2026). The RL reward is correctness plus a Brier term. The authors report that RLVR worsens out-of-domain calibration while RLCR improves it, beating the base model and a separate confidence-predictor model — [arXiv 2507.16806](https://arxiv.org/pdf/2507.16806); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011036)
- **ConfTuner** (arXiv 2508.18847, Aug 2025). Fine-tunes with a tokenized Brier score. The generalization reported is across confidence formats; out-of-domain task numbers were not verified — [third-party review](https://www.themoonlight.io/review/conftuner-training-large-language-models-to-express-their-confidence-verbally)

**Hidden-state probes and distillation (2025-2026)**
- **Probe-SD**, "Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models" (arXiv 2609.33290, Sep 2026). The authors state that post-hoc rescaling fits one distribution but rarely transfers. Reported Qwen3-14B ECE goes from 0.542 to 0.113 out-of-domain and from 0.178 to 0.024 in-domain. These numbers come from a search snippet and were not checked in the full text — [arXiv 2609.33290](https://arxiv.org/abs/2609.33290v1)
- **CCPS**, "Calibrating LLM Confidence by Probing Perturbed Representation Stability" (arXiv 2505.21772, 2025). A probe on how stable the final hidden states are under perturbation — [arXiv 2505.21772](https://arxiv.org/html/2505.21772)
- **Future Confidence Distillation** (arXiv 2607.07626, 2026). Distils post-solution probe confidence into a predictor that reads pre-solution states, and claims cross-domain generalization (snippet only) — [arXiv 2607.07626](https://arxiv.org/html/2607.07626)
- **Probe-Conditioned Head Intervention** (arXiv 2606.09876, 2026) is an overconfidence-calibration intervention (title only) — [arXiv 2606.09876](https://arxiv.org/pdf/2606.09876)

**Older multi-domain baseline**
- **Multi-domain Temperature Scaling** (Yu et al., NeurIPS 2022). It fits per-domain temperatures and regresses them on features. The authors note that one universal temperature is not enough on each individual domain — [arXiv 2206.02757](https://arxiv.org/pdf/2206.02757)

### Inferences
- The leading methods (Thermometer, ATS, CALM, Probe-SD) are white-box. They need hidden states, weight access, or per-token features. RB-Cal uses only answer-letter probabilities plus label-free features such as order inconsistency, which puts it closer to the black-box-logprob regime. This is a real point of difference, but the paper should state it explicitly.
- Thermometer's leave-one-task-out runs within one benchmark family: one MMLU subject is held out and the other 56 train the model. Sealed cross-benchmark rosters (AQuA-RAT, PIQA, MedQA, LSAT, ...) are a harder and less studied transfer regime.
- DACA uses unlabeled target data and a pre-trained reference model. That is complementary to RB-Cal, not a direct competitor, and its agreement-filtered temperature could become a label-free feature.
- In 2026 the trend is to give up on post-hoc transfer (CALM, Probe-SD both argue that TS does not transfer). A post-hoc method with a certified no-harm property would be a direct counterpoint to that claim.

### Gaps
- ATS official code was not found. The user's context says ATS appeared at EMNLP 2024; this session confirmed only the ICLR 2024 workshop version and arXiv.
- DACA's exact model list, benchmark list and code were not confirmed.
- These works were not re-verified this session and come from prior knowledge only: Batch Calibration (Zhou et al., ICLR 2024, label-free); LitCab (ICLR 2024); Few-shot recalibration (Li et al., 2024); Plaut et al. 2024 (arXiv 2402.13213), which appeared in a search result showing chat-LLM MC probabilities are miscalibrated but still predict correctness. The 2402.13213 URL was seen; its content was not read.

## Q2. Do any works report per-task harm, or impose per-task or per-source constraints?

### Takeaway
Per-task harm is rarely the headline metric. Thermometer reports informally that ECE improved on all 57 MMLU subjects and on 20 of 23 BIG-bench tasks for FLAN-T5. One 2026 audit of LLM judges reports harm rates explicitly, from 0.09 to 0.34 of transfer units. No published LLM calibration method was found that fits under per-source regret or harm constraints, or that adds a significance-based deployment gate.

### Cited Findings
- Thermometer: "thermometer reduces calibration error on all 57 datasets, often substantially" on MMLU. On BIG-bench it "rarely fails to improve calibration". For FLAN-T5-XL its ECE is lower than TS-CV on 20 of 23 tasks. These are ECE scatter plots, not significance-tested harm counts — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2)
- **"When Accuracy Gaps Fail to Certify: Auditing Cross-Domain Recalibration of LLM Judges"** (Afrin, Shihab; arXiv 2609.27954, submitted 23 Aug 2026). Setup: 13 judges, 8 domains of 40 items each, 1,176 transfers, verbalized confidence. Harm rate means the fraction of transfer units whose target Brier is worse than raw. Results: source-fitted Platt 0.34 [0.18, 0.48]; TS 0.13 (dBrier -0.147); multi-domain TS 0.17 (dBrier -0.163); target TS with 16 labels 0.09; target Platt 0.12 — [arXiv 2609.27954](https://arxiv.org/html/2609.27954v1)
- The same paper proposes a one-sided finite-sample certificate that rejects a transferred calibrator. Its power is only 0.13 at m=1024 labels, and the authors say "Non-rejection is not approval". Proposition 5 states that an assumption-free label-free repair is impossible. The paper has no per-source constraint and no no-harm method. Its limitations: English only, small generators, 40 items per domain, binary correctness only — [arXiv 2609.27954](https://arxiv.org/html/2609.27954v1)
- CALM reports per-benchmark loss of general ability, not calibration harm: Llama-3.1 drops 5.8 points out-of-distribution — [arXiv 2608.07419](https://arxiv.org/html/2608.07419)
- Search terms tried without finding an LLM-calibration paper: "minimax regret temperature scaling", "no-harm calibration transfer", "per-source constraint", "safe calibration", "Pareto calibration across tasks", "distributionally robust temperature". The nearest results were COM-BOM (EMNLP 2025), which finds an accuracy-vs-ECE Pareto front over exemplar sets ([ACL 2025.emnlp-main.1027](https://aclanthology.org/2025.emnlp-main.1027.pdf)); "Pareto Optimal Learning for Estimating LLM Errors" (ACL 2024, [2024.acl-long.566](https://aclanthology.org/2024.acl-long.566.pdf), not read); and Thermometer's motivation of avoiding worst-case calibration ([MIT PDF](https://sia.mit.edu/wp-content/uploads/2024/12/2024-shen-das-greenewald-sattigeri-wornell-ghosh-icml.pdf)).
- Search also returned a repo [github.com/Huyvux12/calport-calibration-transfer](https://github.com/Huyvux12/calport-calibration-transfer), titled "Does source calibration transfer across typed decision tasks? ... minimax-regret temperature scaling (MRTS)". The search snippet describes a one-parameter rule that minimizes the worst source-task NLL increase and returns T=1 when sources disagree. The URL returned HTTP 404 when fetched on 2026-11-06, so it may now be private. This is very likely the user's own earlier work (the git user is "Huy Vux", and the local folder is named "mrts"), so it is the RB-Cal predecessor, not an external competitor.

### Inferences
- The "zero harmed tasks over about 370 units, significance-gated" claim has no direct published competitor. The closest is 2609.27954, which measures harm and finds 13-17% for TS and MDTS but offers only a low-power rejection certificate. RB-Cal can present itself as the constructive counterpart to that negative audit result.
- Thermometer's "all 57 improved" is the main precedent a reviewer might raise. RB-Cal's answer: that result is within one family (MMLU subjects), uses ECE without significance testing, and comes from 2023-era models.
- The paper should cite Yu et al. 2022 (MDTS) as the multi-domain TS ancestor. 2609.27954 shows MDTS reaches harm rate 0.17 even on judges.

### Gaps
- No primary source was found for distributionally robust or group-DRO calibration of LLMs over tasks. A paper may still exist under other names such as "worst-group calibration" or "multicalibration for LLMs". Multicalibration work (e.g. Detommaso et al., "Multicalibration for Confidence Scoring in LLMs", ICML 2024) is known from memory only and was not verified this session; it is the most likely place for a reviewer to find a per-group guarantee.
- Thermometer and ATS NLL harm counts per task were not reported in any form found.

## Q3. What calibration benchmarks or released probability datasets for many LLMs on many tasks exist, and could our setting reuse them?

### Takeaway
No public dataset was found that releases answer-letter probabilities for many current (2026) open LLMs on 25+ MC tasks. The existing benchmarks either target UQ for generation (LM-Polygraph), use older models on a few MC sets (LLM-Uncertainty-Bench), or measure verbalized or behavioural calibration (UBench, ConfidenceBench, UncertaintyGym). A released probability set for 5 models across about 40 tasks would itself fill a gap.

### Cited Findings
- **LM-Polygraph** (Vashurin et al., TACL 2025). A UQ benchmark for text generation covering 11 tasks and confidence-normalization methods. Code: [github.com/IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph); pip package `lm-polygraph`. Datasets are on HF under the LM-Polygraph organization — [TACL 2025](https://aclanthology.org/2025.tacl-1.11.pdf); [PyPI](https://pypi.org/project/lm-polygraph)
- **LLM-Uncertainty-Bench**, "Benchmarking LLMs via Uncertainty Quantification" (Ye et al., NeurIPS 2024 D&B). Conformal-prediction UAcc metric on MC tasks; MIT licence; data on HF at `ErikYip/LLM-Uncertainty-Bench` — [GitHub](https://github.com/smartyfh/LLM-Uncertainty-Bench); [NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/hash/1bdcb065d40203a00bd39831153338bb-Abstract-Datasets_and_Benchmarks_Track.html)
- **UBench** (Wang et al., arXiv 2406.12784, 2024/2025). The model picks one of 10 confidence-interval options; metrics are ECE, ACE, MCE and TACE. Reported size differs between versions (3,978 in v1 vs 11,978 in v2). Code at github.com/Cyno2232/UBENCH (from a search snippet, not opened) — [arXiv 2406.12784](https://arxiv.org/pdf/2406.12784)
- **ConfidenceBench** (arXiv 2607.20526, 2026). Evaluates LLM confidence calibration; details not read — [arXiv 2607.20526](https://arxiv.org/html/2607.20526); [site](https://confidencebench.carrd.co)
- **Revisiting Uncertainty Estimation and Calibration of LLMs** (arXiv 2505.23854). MMLU-Pro based; finds that accuracy does not imply calibration (e.g. GPT-4.1) — [arXiv 2505.23854](https://arxiv.org/pdf/2505.23854)
- **CALIBRI** (HF `lavis-nlp/CALIBRI`). Code-generation calibration with token likelihoods from multiple LLMs — [HF dataset](https://huggingface.co/datasets/lavis-nlp/CALIBRI)
- **UncertaintyGym** measures behavioural uncertainty (abstention and false premises) — [HF forum](https://discuss.huggingface.co/t/uncertaintygym-a-benchmark-for-llm-epistemic-calibration-and-uncertainty-expression/178647). **KalshiBench** (arXiv 2512.16030) uses 300 prediction-market questions — [arXiv 2512.16030](https://arxiv.org/pdf/2512.16030)
- **Open LLM Leaderboard**. A maintainer said calibration "is indeed an interesting complement" and that since "we provide all the details you could do a small analysis". This suggests per-sample details datasets exist that could support a calibration analysis — [HF discussion #828](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/828)
- Plaut et al., "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on MC Q&A" (arXiv 2402.13213). Uses 14 LLMs on the 5 Open-LLM-Leaderboard MC datasets (ARC-C, HellaSwag, MMLU, TruthfulQA, WinoGrande) — [arXiv 2402.13213](https://arxiv.org/pdf/2402.13213) (from a search snippet)
- HELM includes calibration (ECE) as an evaluation axis — [ChEF paper citing HELM](https://arxiv.org/pdf/2311.02692). No release of per-instance logprobs across many models was confirmed.

### Inferences
- Item sources are reusable: LLM-Uncertainty-Bench and LM-Polygraph datasets on HF could supply extra sealed target tasks, although they use different prompt formats. The models are not reusable, since none of these covers Qwen3.5, Gemma-4, Granite-4.2, Ministral-3 or Olmo-3.
- Open LLM Leaderboard v1/v2 details hold per-sample loglikelihoods for thousands of models on ARC, HellaSwag, MMLU, MMLU-Pro and others. They could be a large external test of RB-Cal's source-fitted transfer on many more models, but they use lm-eval-harness loglikelihood scoring of full answer strings, not answer-letter next-token probabilities. Not verified in detail.
- Releasing the project's probability tables (5 models x 25 source and about 15 sealed tasks, with answer-order permutations) as an HF dataset would be a distinct contribution. Nothing equivalent was found.

### Gaps
- HELM per-instance raw logprob availability and licence were not checked.
- The exact schema of Open LLM Leaderboard details datasets (whether per-choice logprobs are stored) was not checked this session.
- ConfidenceBench and UBench model and task coverage was not read in full.

## Q4. Is there any work on "regret budget", "per-source constraint", "minimax regret temperature scaling", "no-harm calibration transfer", "safe calibration" or "Pareto calibration across tasks"?

### Takeaway
No external paper with these framings for LLM calibration was found. The only hit for minimax-regret temperature scaling is the user's own (now 404) calport/MRTS repo. The nearest external works are the 2609.27954 harm audit and certificate, MDTS (Yu et al. 2022), and accuracy-vs-calibration Pareto search (COM-BOM).

### Cited Findings
- Minimax-regret TS hit: [github.com/Huyvux12/calport-calibration-transfer](https://github.com/Huyvux12/calport-calibration-transfer) (search snippet; 404 on 2026-11-06; probably the user's own work).
- Harm-rate audit plus rejection-only certificate: [arXiv 2609.27954](https://arxiv.org/html/2609.27954v1)
- Accuracy-ECE Pareto over exemplars: COM-BOM, [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1027.pdf)
- Multi-domain TS (robust calibration across domains): [Yu et al. NeurIPS 2022](https://arxiv.org/pdf/2206.02757)
- Other domain-shift TS work: Consistency-Guided TS for out-of-domain calibration ([arXiv 2402.15019](https://arxiv.org/pdf/2402.15019)); Transferable Calibration in domain adaptation ([NeurIPS 2020 PDF](https://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-calibration-nips20.pdf)). Both are vision or classic domain adaptation, not LLM cross-task work.

### Inferences
- "Regret-budgeted multi-source calibration with a deployment gate" appears to be new for LLMs. The main novelty risk is from multicalibration and group-robust calibration in general ML, which a reviewer may cite. These should be covered in related work.

### Gaps
- Not searched: OpenReview submissions for ICLR 2027, which would be under double-blind review in Oct-Nov 2026 and would not be indexed by the search engine. A manual OpenReview search for "calibration transfer" is advisable.
- Group-DRO calibration or "worst-group ECE" literature was not checked.

## Q5. What open gaps do recent papers state in their limitations or future work that our setting could address?

### Takeaway
The stated gaps are: post-hoc temperatures are claimed not to transfer across domains (CALM, Probe-SD); the harm audit offers only rejection certificates with low power (2609.27954); Thermometer asks for broader tasks, and its predictions are hard to get right on diverse benchmarks; and existing benchmarks do not cover current open models. RB-Cal addresses the first three directly.

### Cited Findings
- 2609.27954 says accuracy gaps are "weak warning signals ... not deployment certificates". Its certificate is "conservative, certifies only failure" and its usefulness at about 10^3 labels per domain is "untested here". It covers English only and binary correctness only, and says thresholds need task-specific validation — [arXiv 2609.27954](https://arxiv.org/html/2609.27954v1)
- CALM argues that post-hoc TS is domain-dependent and that overconfidence is the main failure under shift — [arXiv 2608.07419](https://arxiv.org/html/2608.07419)
- Probe-SD says post-hoc rescaling "fits one distribution but rarely transfers" (search snippet) — [arXiv 2609.33290](https://arxiv.org/abs/2609.33290v1)
- Thermometer finds BIG-bench task diversity makes temperature prediction harder (outliers) and lists extending to summarization, translation and larger LLMs as future work — [arXiv 2403.08819](https://arxiv.org/html/2403.08819v2)
- DACA is motivated by "the scarcity of labeled data for individual downstream tasks" — [arXiv 2505.16690](https://arxiv.org/abs/2505.16690)
- Multi-domain TS: "a universal temperature is not sufficient" for each domain — [arXiv 2206.02757](https://arxiv.org/pdf/2206.02757)

### Inferences (candidate contributions that would not duplicate others)
1. **Constructive no-harm transfer.** 2609.27954 gives only a negative audit and a low-power rejection test. RB-Cal adds per-source regret budgets, a clustered-SE margin and a significance gate, and reports 0 harmed tasks out of about 370. This is the positive counterpart and should be framed against that paper's 0.13-0.17 TS and MDTS harm rates.
2. **Answering "post-hoc TS does not transfer" (CALM, Probe-SD).** RB-Cal keeps about 93-95% of pooled-TS NLL gain with zero harm on sealed cross-benchmark rosters, which is direct evidence against that claim when the method is constrained.
3. **Probability-only, near-black-box features.** Thermometer, ATS and probes need hidden states. Order-inconsistency and logprob features work with any API that returns top-k logprobs. Worth stating, and could be tested on an API model.
4. **Cross-benchmark, sealed, pre-registered evaluation.** Thermometer uses leave-one-subject-out within MMLU or BIG-bench. No competitor found uses sealed unseen-benchmark rosters with a protocol log.
5. **Harm-vs-gain frontier as an evaluation standard.** No paper reports per-task NLL harm counts with significance for Thermometer or ATS. The project's finding of 5-12 harmed out of 60 units for them is new evidence. A "harm count / fraction of oracle gain" Pareto plot would be a new reporting standard.
6. **Released probability dataset** for 2026 open models (Qwen3.5, Gemma-4, Granite-4.2, Ministral-3, Olmo-3) across about 40 MC tasks with permutations. No equivalent was found on HF.
7. **Combining with DACA.** The agreement-based label-free temperature could serve as an extra feature or gate input. DACA has no per-task safety guarantee.

### Gaps
- Limitations sections of ATS, APRICOT, calibration-tuning and DACA were not read in full.
- Novelty against general-ML multicalibration and conformal risk control (e.g. "Conformal Risk Control", Angelopoulos et al. ICLR 2024; "Learn then Test") was not checked this session. Those frameworks give per-task risk guarantees and a reviewer may raise them as the natural tool for "no harm with high probability". The paper should explain why a source-only gate differs from them, since they need target labels or exchangeability.
- Recent 2026 workshop papers and unindexed OpenReview submissions are not covered.
