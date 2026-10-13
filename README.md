# GLWM-2: knowing when you're done

Code, benchmark and paper for **"Knowing When You're Done: Termination Detection for Recurrent
Latent Reasoners under Depth Shift"**. This is the v2 successor of the GLWM paper; the v1 paper
(`main.tex` and the `fig*.png` files in the project root) is kept untouched as an archive.

A recurrent latent reasoner keeps one latent vector per input position and updates all of
them with the same local step. It can only "think longer on harder problems" if it can tell
when its computation is finished. This repository contains:

- **Theory** (`paper/sections/theory.tex`, checked by `tests/test_theory.py`): a light-cone
  lower bound on the number of steps, an unsoundness result for rules that halt earlier, and a
  local, length-independent completion automaton that halts exactly at the bound.
- **DepthShift benchmark** (`src/glwm2/data`): addition with controlled carry chains,
  segmented parity, and mazes, each with a per-instance depth oracle and a 2x2 split design
  that separates *longer* inputs from *deeper* inputs.
- **Stopping-rule library** (`src/glwm2/readouts`, `src/glwm2/models/halting_units.py`): fixed
  depth, oracles, max-softmax, entropy, stability, KL, hidden-state change, max-confidence,
  correctness heads, ACT, PonderNet, the v1 GLWM gate, and our local completion heads.

## Setup (Windows, RTX 50-series)

```powershell
winget install --id=astral-sh.uv -e
uv venv --python 3.12 --managed-python .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements-lock.txt
uv pip install --python .venv -e .
.venv\Scripts\python.exe -m pytest -q
```

`easy-to-hard-data` pins an old `reportlab` that needs a C++ compiler on Windows. It is
optional (only used to cross-check our maze generator) and can be installed with
`uv pip install --python .venv --no-deps easy-to-hard-data`.

## Layout

| Path | Contents |
| --- | --- |
| `src/glwm2/data` | generators, depth oracles, splits, frozen test sets |
| `src/glwm2/models` | recurrent backbone, output/completion heads, halting units |
| `src/glwm2/train` | objectives (fixed-prior, progressive, ACT, PonderNet, ...) and trainer |
| `src/glwm2/readouts` | stopping rules evaluated in one pass over a trajectory, online halting |
| `src/glwm2/eval` | metrics, FLOPs, latency, statistics |
| `configs/` | one YAML per experiment family |
| `scripts/` | sweeps, figures, bibliography check, full reproduction |
| `notebooks/` | `00`-`09`, one per stage of the study |
| `paper/` | the new paper (Overleaf-ready) |

See `REPRODUCE.md` for the exact commands and `PREREGISTRATION.md` for the frozen hypotheses.
