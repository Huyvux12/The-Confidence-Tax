# Manuscript — EvoLearn 2027 artifact

The corrected double-anonymous submission is titled **The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models** (15 Springer LNCS pages). The current authority is the separately maintained `EvoRB_LOTO_Corrected_Overleaf_Source.zip`; its `main.tex` SHA-256 is:

`3070c25bcd5648decc6e1d826208bd211dba9c3911c63756c8ae8b8bd8f16004`.

This repository keeps the code, recorded experimental outputs and figure-generation scripts, **not** the final double-blind source archive. The corrected manuscript clarifies that LOTO reuses a globally selected discrete field specification and excludes each held-out task from fold-specific fitting/evolution/audit. Do not use previous `paper_lncs` or historic source ZIPs as the current submission.

Figures can be regenerated from the committed research records:

```bash
python scripts/verify_paper_results.py
python scripts/build_submission_figures.py
```

Outputs in this folder:
- `figure1_gain_calibration.png`: aggregate gain/harm + ECE/harm trade-off.
- `figure2_roster_frontiers.png`: per-roster gain/harm scatter, C1/C2/C3/Hard/LOTO/All.
- `figure3_rescue_cropped.png`: paired baseline-versus-Evo-RB negative-transfer outcomes.

The programmatically regenerated figures **represent the same frozen data but need not be visually identical** to the author's original PNGs. The precise No-margin ablation output for Table 4 remains a documented reproducibility gap. See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

The corrected Overleaf ZIP references the three figure filenames above and includes the official LNCS class and original artwork. Compile that ZIP separately with pdfLaTeX, preserving the conference layout. The reported Margin vs No-margin aggregate is archived in `runs/fixed_margin_control_reported_summary.json`, but the per-unit raw output remains unrecovered; do not claim independent reproduction of Table 4 from that summary.
