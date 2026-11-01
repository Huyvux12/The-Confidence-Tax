# RB-Cal: regret-budgeted multi-source calibration

Code, data and paper for *Regret-Budgeted Calibration: Transferring Temperature Scaling Across Tasks
While Bounding the Harm to Each Source* (`paper/main.pdf`).

Progress and scope: `todo.md`. Pre-registration: `PREREGISTRATION.md`. Dated decisions and deviations:
`data_gen/PROTOCOL_LOG.md`.

## Layout

| Path | Content |
|---|---|
| `calibration_extensions/` | calibrators, objectives, RB-Cal (`rb_calibration.py`), baselines, evaluator, tests |
| `data_gen/` | question set builder, GPU inference script, split builder, raw per-model probabilities (`raw/`), pinned revisions |
| `data/real/dev`, `data/real/confirm_sealed` | canonical splits (sources identical in both; targets differ) |
| `scripts/` | experiment runners, secondary analyses, paper assets, rechecks |
| `runs/` | frozen outputs: per-unit fits, predictions, hashes, reports |
| `paper/` | LaTeX source; every number, table and figure is generated into `paper/generated/` and `paper/figures/` |

## Reproduce

Python 3.13. All commands from this directory.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt    # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest calibration_extensions/tests -q
```

**From the shipped probabilities (CPU only).** The repository ships the raw probabilities
(`data_gen/raw/`) and the per-unit fits (`runs/*/units/`), not the derived splits or the concatenated
`unit_results.pkl` files. Rebuild them first (both steps are deterministic):

```bash
python data_gen/make_splits.py --raw data_gen/raw --out data/real
(cd data/real && sha256sum -c ../real_SHA256SUMS.txt)                 # must print OK for all 12 files
PYTHONPATH=. python scripts/run_rbcal.py --data data/real/dev --out runs/rbcal_dev        # reads the cached units: minutes
PYTHONPATH=. python scripts/run_rbcal.py --data data/real/confirm_sealed --out runs/rbcal_confirm
```

To refit everything instead of reading the cached units, delete `runs/*/units/` first:

```bash
PYTHONPATH=. python scripts/run_rbcal.py --data data/real/dev --out runs/rbcal_dev               # 42-285 min per unit
PYTHONPATH=. python scripts/run_rbcal.py --data data/real/confirm_sealed --out runs/rbcal_confirm \
    --reuse-selection data/real/dev runs/rbcal_dev/units                                          # 6-15 min per unit
PYTHONPATH=. python scripts/run_extra_baselines.py --out runs/extra_baselines
PYTHONPATH=. python scripts/run_loto.py --out runs/loto
sh paper/build.sh                                                                                 # needs pdflatex + bibtex
```

`--reuse-selection` is valid only because the source files of the two phases are byte-identical; the runner
checks their SHA-256 and refuses otherwise. `scripts/recheck_full_refit.py` refits a confirm unit without it.

**Checks.** `scripts/recheck_numbers.py` recomputes the paper's main numbers without the evaluator;
`runs/*/prediction_hash.txt` holds the SHA-256 of all target predictions of each run.

**From scratch (GPU).** `data_gen/build_items.py` rebuilds `data_gen/items.jsonl` from Hugging Face datasets;
`data_gen/run_all.py` with `data_gen/infer_mcq.py` scores the five models (one A100 40GB, about 2.6 hours;
`data_gen/requirements-inference.txt`); `data_gen/make_splits.py` writes `data/real/`. The model and dataset
commits used are in `data_gen/revisions.json` (`run_all.py` pins them when the file is copied next to it).
GPU numerics can differ in the last bits between hardware and library versions.

## Data licences

Benchmark texts and labels keep their original licences (`data_gen/revisions.json` lists the Hugging Face
licence tags); SciQ is CC BY-NC 3.0. The five models are Apache-2.0. No model weights are included.
