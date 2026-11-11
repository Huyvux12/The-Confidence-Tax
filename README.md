# The Confidence Tax — Evo-RB research repository

> **Manuscript alignment (11 November 2026).** The latest author-provided 15-page manuscript is titled *The Confidence Tax: Evolutionary Regret-Budgeted Calibration for Language Models*. The checked-in `paper_lncs/main.tex` remains an **earlier EvoLearn draft**, and `paper/main.tex` is the **historical RB-Cal draft**. Neither should be represented as the exact source of that latest manuscript until synchronized. In particular, the current paper version includes a new paired negative-transfer rescue figure and controlled ablations. See [REPO_CLEANUP_AUDIT.md](REPO_CLEANUP_AUDIT.md) for the source-to-paper map and cleanup scope.

## Core research path

- **Method:** `scripts/evo_study.py` (bilevel evolutionary centre/bandwidth search and SLSQP inner fit), `calibration_extensions/rb_calibration.py` (coefficient fitting and regret constraints), `calibration_extensions/features.py` (label-free features).
- **Baselines:** `calibration_extensions/extra_baselines.py`, `calibration_extensions/source_objectives.py`, and `scripts/run_extra_baselines.py`.
- **Experiments:** `scripts/v4_study.py`, `scripts/gate_study.py`, `runs/evo_study/`, `runs/v4_study/`, `runs/gate_study/`, and the source/target probability files.
- **Historical manuscripts, M1–M5 prototype code and tests** are retained for traceability; generated toy-run outputs and redundant caches are not tracked on this cleanup branch.

---

### Historical README (RB-Cal)

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
(`data_gen/raw/`), the three source splits (`data/real/dev/source_{train,select,audit}.jsonl`), and the
per-unit fits (`runs/*/units/`). The remaining derived splits and the concatenated `unit_results.pkl`
files are not included. Rebuild them first (both steps are deterministic):

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

## EvoLearn 2027 version (`paper_lncs/`)

LNCS paper *The Confidence Tax: Evolving Regret-Budgeted Temperature Fields for Safe Calibration Transfer in
Language Models*. Build: `sh paper_lncs/build.sh` (every number, table and figure comes from `runs/` through
`scripts/make_lncs_assets.py`). Later phases are logged in `data_gen/PROTOCOL_LOG.md` and `todo.md` (phases 6-12):

| Path | Content |
|---|---|
| `data_gen/build_items_confirm{2,3,4}.py`, `data_gen/raw_confirm{2,3,4}/` | three further sealed rosters (items, raw probabilities) |
| `data/real/confirm{2,3,4}_SHA256SUMS.txt` | rebuild with `data_gen/make_confirm2.py --raw ... --out ... --role target_confirmN` |
| `calibration_extensions/features.py` | order-consistency (`rbf_oc`, `rbf_oct`) and peer (`rbf_ocp*`) feature maps |
| `calibration_extensions/evo_calibration.py`, `scripts/evo_study.py` | CMA-ES solver, Pareto fronts, Evo-RB |
| `scripts/v4_study.py`, `base_study.py`, `gate_study.py`, `modal_*.py` | experiment drivers (Modal CPU/GPU) |
| `runs/v4_study`, `runs/evo_study`, `runs/rbcal_confirm{2,3,4}`, `runs/step0` | outputs and reports |

## Large artifacts (Hugging Face)

The per-unit fits of the confirm2-4 rosters (`runs/rbcal_confirm{2,3,4}/units/`, 765 MB) are in the public
Hugging Face dataset [huyvux3005/rbcal-confidence-tax-runs](https://huggingface.co/datasets/huyvux3005/rbcal-confidence-tax-runs), not in git:

```bash
hf download huyvux3005/rbcal-confidence-tax-runs --repo-type dataset --local-dir .
sha256sum -c runs/HF_SHA256SUMS.txt                                                   # 30 files must print OK
```

**From scratch (GPU).** `data_gen/build_items.py` rebuilds `data_gen/items.jsonl` from Hugging Face datasets;
`data_gen/run_all.py` with `data_gen/infer_mcq.py` scores the five models (one A100 40GB, about 2.6 hours;
`data_gen/requirements-inference.txt`); `data_gen/make_splits.py` writes `data/real/`. The model and dataset
commits used are in `data_gen/revisions.json` (`run_all.py` pins them when the file is copied next to it).
GPU numerics can differ in the last bits between hardware and library versions.

## Data licences

Benchmark texts and labels keep their original licences (`data_gen/revisions.json` lists the Hugging Face
licence tags); SciQ is CC BY-NC 3.0. The five models are Apache-2.0. No model weights are included.
