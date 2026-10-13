# Reproducing GLWM-2

Everything below was run on Windows 11, an RTX 5070 (12 GB, compute capability 12.0), an
i5-12400 and 16 GB RAM, with the pinned environment in `requirements-lock.txt`
(Python 3.12.15, torch 2.11.0+cu128).

## One command

```powershell
.\scripts\reproduce_all.ps1 -Workers 3     # setup, tests, every sweep, figures, tables, bib check
.\scripts\reproduce_all.ps1 -FiguresOnly   # only rebuild figures/tables from results/runs
```

`-Workers` sets how many sweep processes share the GPU. Each run is locked
(`checkpoints/<run_id>/sweep.lock`), so workers never train the same run, and an interrupted run
resumes from `checkpoints/<run_id>/last.pt`.

## Step by step

| Step | Command | Output |
| --- | --- | --- |
| Environment | see `README.md` (uv, cu128 wheel, `requirements-lock.txt`) | `.venv` |
| Tests (theory, data, rules, FLOPs, determinism, smoke) | `.venv\Scripts\python.exe -m pytest -q` | 43 tests |
| Environment check, v1 reproduction, frozen splits | `python scripts\build_notebooks.py --run 00 01 02` | `results/env.json`, `results/v1/`, `data/frozen/` |
| Backbone pilot (G1) | `python scripts\run_sweep.py configs\pilot_addition.yaml`, then notebook 03 | `results/pilot_summary.json` |
| Gate tuning (equal budget) | `python scripts\run_sweep.py configs\tune_gates.yaml` | chosen settings in notebook 04 |
| Main grid | `python scripts\run_sweep.py configs\main_addition.yaml configs\main_parity.yaml configs\main_gates.yaml configs\headline_addition.yaml` | `results/runs/<run_id>/` |
| Mazes | `python scripts\run_sweep.py configs\main_mazes.yaml` | maze runs |
| Ablations | `python scripts\run_sweep.py configs\ablations.yaml` | ablation runs |
| Analysis, figures, tables | `python scripts\make_figures.py` (notebooks 02-05, 07-09) | `paper/figures/*.pdf`, `paper/tables/*.tex`, `results/paper_numbers.json` |
| Bibliography | `python scripts\fetch_bib.py` then `python scripts\check_bib.py` | `paper/references.bib` |
| Paper | `cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main` | `paper/main.pdf` |

Every run directory contains `config.json`, `train_log.jsonl`, `train_summary.json` (seconds,
git commit, environment), `metrics.json` (every stopping rule on every split at the threshold
selected on ID-val), `curves.json` (every threshold), `step_acc.json`, `calibration.json` and
`per_instance.pt` (per-instance halting step and correctness, used for all bootstrap statistics).

## Determinism and data integrity

- Seeds: every run seeds Python, NumPy and torch from `RunConfig.seed`; data streams use their
  own `torch.Generator`s.
- Default mode allows TF32 and non-deterministic cuDNN kernels for speed. `train(...,
  deterministic=True)` turns both off and enables `torch.use_deterministic_algorithms`
  (`warn_only=True`, so an op without a deterministic kernel warns instead of failing);
  `tests/test_determinism.py` checks bitwise-equal weights on CPU, and notebook 00 checks it on GPU.
- Frozen test sets: `data/frozen/<task>_<split>.pt` are regenerated from fixed seeds if missing
  and verified against the SHA-256 hashes in `data/frozen/HASHES.json` (committed). The CPU CI job
  regenerates them and fails if a hash differs.
- The training sampler drops any instance whose row hash matches an ID-val or ID-test instance.

## Compute

GPU-hours per sweep are summed from `train_summary.json` and `eval_summary.json` by notebook 06
(`results/grid_status.json`) and reported in the paper.

## What is not automated

- Uploading checkpoints and logs (planned: Hugging Face Hub) and posting to arXiv / submitting to
  the venue need the authors' accounts.
- DBLP serves a bot check to scripts, so `fetch_bib.py` falls back to arXiv API metadata with the
  venue from Semantic Scholar; `check_bib.py` verifies every entry against arXiv or Crossref.
