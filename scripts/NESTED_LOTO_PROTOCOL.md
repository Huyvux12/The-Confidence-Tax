# Nested LOTO source-only hyperparameter selection

## Important: this is a NEW protocol, not the one that produced the paper's frozen 250 LOTO records

Historical LOTO reused the globally selected discrete v4b specification from
`runs/v4_study/chosen.json`. It excluded the held-out task during coefficient
fitting, CMA-ES outer structure search, and source audit, **but not during global
specification selection**. The historical paper results must not be presented
as a completed nested LOTO run.

This branch adds two **opt-in** stages, which do not affect historical jobs:

- `loto_nested`: for fixed-field variants, exclude the outer held-out task,
  use only the remaining 24 source tasks for inner leave-one-source-out (LOSO)
  selection among the 37 discrete field specifications, then fit and evaluate.
- `evo_loto_nested`: perform the same 24-source nested specification selection
  for v4b, then run the existing CMA-ES/SLSQP Evo-RB search using source
  training/selection only. The original source-audit fallback remains unchanged.

For each outer fold the selector checks that the held-out task is absent,
requires exactly 24 remaining sources, and does not read `chosen.json`.
Separate result files prevent accidental overwriting of old records.

### Commands (NOT executed in this change)

```bash
python -m unittest calibration_extensions.tests.test_nested_loto_protocol
modal run scripts/modal_v4.py --stage loto_nested --variants v4b
modal run scripts/modal_v4.py --stage evo_loto_nested
```

New outputs, only if experiments are run:
`runs/v4_study/loto_nested.jsonl`,
`runs/evo_study/evo_loto_nested.jsonl`.

The new stages are significantly more expensive because the per-fold nested
selection requires repeated constrained fits. No inference or calibration
experiments were executed as part of this source change. Paper tables and
figures must only be regenerated after new outputs exist.

## Manuscript implications

For the **existing** 500-unit results, use the accurate statement: global
source-only hyperparameter selection followed by leave-one-source-out
refitting, rather than strict nested task isolation. Code changes do not
retroactively validate the old results or justify unchanged nested-LOTO claims.

For a future nested-LOTO comparison, rerun all relevant methods with a
comparable selection protocol and document deviations transparently.
