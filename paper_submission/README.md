# Manuscript — EvoLearn 2027 artifact

The author-provided PDF is titled **The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models** (15 pages, Springer LNCS, double-anonymous). Its original Overleaf ZIP, preserved in the author's source files, was compiled locally with pdfLaTeX to 15 pages with matching paragraphs and tables.

The archived main-source SHA-256 is `691dfda7772e5e288d3830c547f296866cd74b84742154a93be687499e9ddc2a`. The source is the authority for exact manuscript wording and typesetting; do **not** substitute the older `paper_lncs` draft.

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

The original Overleaf `main.tex` references the three names above. Once the verified source is placed in this folder, use pdfLaTeX (and the official LNCS class) to compile. Preserve the conference-specified layout. Never substitute regenerated Table 4 numbers without their audited source run.
