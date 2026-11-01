# Pre-registration: regret-budgeted multi-source calibration (RB-Cal)

Written 2026-10-31, after the source-only go/no-go and before any calibration method has been
run on a target task. Changes after this point are appended to `data_gen/PROTOCOL_LOG.md` with a
date and a reason; a change made after looking at target_dev results makes the affected analysis
exploratory.

## Question
Strict "no source may get worse" calibration (MRTS, M1-MINIMAX) is stuck at identity once there
are many conflicting sources. Does allowing each source a bounded NLL regret epsilon recover
most of the pooled gain without the pooled method's harm, and do richer label-free calibrator
classes help beyond a budgeted shared temperature, when transferring to unseen tasks?

## Method (RB-Cal)
    min_theta  sum_w pi_w Delta_w(theta) + lambda * Omega(theta)
    s.t.       Delta_w(theta) <= epsilon  for every source w,   theta in class C
Delta_w is the mean NLL change versus identity on source w; pi_w = n_w / n; global beta bounds
[0.25, 4] as in the spec. Classes C:
- constant beta (budgeted shared temperature);
- RBF field beta(p) = b0 + v' psi(z), z = [normalised entropy, top-1, top-1 minus top-2],
  menu J in {0, 1, 2, 4, 8}, sigma in {0.2, 0.5, 1.0}, lambda in {0, 0.01, 0.1} (J = 0 is constant);
- top-gap transform (M3): u = a z + (b - a) d 1[top], lambda (a-b)^2, lambda in {0, 0.01, 0.1}.

Primary method: **RB-Cal-RBF at epsilon = 0.01** (the same 0.01 used by the harm metric; the
full curve epsilon in {0, 0.0025, 0.005, 0.01, 0.02} is reported). Secondary: RB-Cal-TopGap.

Hyperparameters by leave-one-source-out on source_train: rank by (1) number of held-out sources
with Delta > epsilon, (2) mean held-out Delta, (3) fewer centres / larger lambda / fixed order;
ties at 1e-8. Gate on source_select: deploy only if max_w Delta_w <= epsilon + 1e-8, else
identity. Pre- and post-gate results are both reported.

## Baselines (all source-only, same sources and splits)
identity; pooled TS; MRTS (minimax scalar on the 241-point grid); shrinkage-LOSO
(beta = 1 + s (beta_pooled - 1), s in {0, 0.1, ..., 1}, chosen by the same LOSO rule);
RB-Cal-constant (budgeted shrinkage, same epsilon); pooled entropy-affine (HTS-style);
pooled RBF and pooled top-gap (same capacity as RB-Cal, no budget). Pooled TS and MRTS are
also reported with the same select gate.

## Data and units
Models: Qwen3.5-9B, Gemma-4-12B-it, Granite-4.2-8B, Ministral-3-8B-Instruct-2512,
Olmo-3-7B-Instruct. Prompts: `letter_colon`, `paren`. One calibrator per model x prompt
(10 units). Sources: 25 task-level sources (60/20/20 case split into train/select/audit).
Targets: dev = commonsense_qa, medmcqa, truthfulqa_mc1 (development, exploratory);
confirm = aqua_rat, piqa, sciq (sealed; read once, after freezing code and config).

## Endpoints (confirm phase; analysis unit = model x prompt x target task, 30 units)
- **E1 (primary, spec 9.3):** RB-Cal-RBF(0.01) vs MRTS: macro mean Delta NLL lower with the
  paired case-bootstrap 95% CI excluding 0, and harm_001 count not higher.
- **E2 (key secondary):** RB-Cal-RBF(0.01) vs pooled TS: harm_001 count not higher and
  worst-unit Delta NLL lower; the share of pooled TS's mean gain retained is reported.
- **E3 (secondary):** RB-Cal-RBF(0.01) vs RB-Cal-constant(0.01): macro mean Delta NLL lower
  with harm_001 count not higher (value of the richer class over budgeted shrinkage).
- **Regime analysis:** a unit is "conflict" if, on source_train only, the constant class has no
  verified common descent direction; endpoints are also reported within each regime.

Metrics: NLL (nats), Brier (sum over K), ECE-15, accuracy, Delta NLL vs identity, harm_001,
harm_any. E1 is the single primary test; E2, E3 and regime results are labelled secondary.
All wins, ties and losses are reported.
