#!/bin/sh
# Regenerate every number/table/figure from runs/, then build main.pdf.
set -e
cd "$(dirname "$0")"
../.venv/Scripts/python ../scripts/make_paper_assets.py
pdflatex -interaction=nonstopmode main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode main.tex >> build.log
pdflatex -interaction=nonstopmode main.tex >> build.log
grep -E "^!|undefined" main.log || echo "build ok: $(grep -o 'Output written.*' build.log | tail -1)"
