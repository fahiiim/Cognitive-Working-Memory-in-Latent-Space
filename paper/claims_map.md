# Claims map

Every quantitative claim in the paper, where it appears, and the artefact that produces it.
Notebooks are built by `scripts/build_notebooks.py` and executed by `scripts/make_figures.py`;
all of them read `results/runs/<run_id>/` and never retrain.

| Claim | Section | Source |
| --- | --- | --- |
| Light-cone lemma, Theorem 1, Corollaries 1-2, Propositions 1-2 | 2, App. A | `tests/test_theory.py` (flip test, pair construction, brute-force oracles, automata to length 512) |
| Split sizes, lengths, depths, SHA-256 hashes | 3, Tab. 1, Fig. 2 | notebook 02, `data/frozen/HASHES.json` |
| Objective choice (tie on validation, fixed-prior) | 5 | notebook 03, `results/pilot_summary.json` |
| G1: addition stable oracle 99.6% on OOD-both; parity 85% | 5, App. C (`fig:oracle`, `fig:stepacc`) | notebook 03 (`g1`), `paper/tables/main_*.tex` (Oracle (stable) row) |
| Fixed T: 71% (addition) / 31% (parity) on OOD-depth | 5 | `paper/tables/main_addition.tex`, `main_parity.tex` |
| ACT 99.8% OOD-depth (2.4 steps), 43.2% OOD-length, 35.9% OOD-both | Abstract, 1, 5 | `main_addition.tex` (ACT row); notebook 04 |
| ACT step-1 / step-5 shortcut | 5 | `PREREGISTRATION.md` deviation; per-instance steps in `results/runs/addition-act-own-s0-acbfa856/per_instance.pt` |
| PonderNet 47.6%, v1 gate 69.7%, pooled heads 50.0% (post hoc) / 31.9% (joint) on addition OOD-both | 5 | `main_addition.tex` |
| Parity ACT 98.9%, PonderNet 96.7% on OOD-depth, < 1% on OOD-both; v1 gate 3% ID | 5 | `main_parity.tex` |
| Premature-error rates (stability/KL 0.1%, pooled gates 18-23%) | 5 | notebook 04, `results/gates/g2.json` (`premature_err`) |
| H4: joint heads < 0.02 points; ACT -34, v1 gate -17 (addition) / -86 (parity) | 5 | `results/gates/g3.json` (`h4_auc_diff_points`) |
| Ablations: no recall 0.6%, width 64 95.2% / 48.0%, r = 2 100% | 5 | `probes_ablations.tex`, `results/compute_probes_ablations.json` |
| Latency 19.8 / 23.8 / 23.8 / 28.3 ms | 5 | `probes_ablations.tex` |
| Pooled gates flat in depth | 5, Fig. 4 | notebook 04 (`fig4_steps_vs_depth_length`), `results/gates/g2.json` (`rho_depth`) |
| Stability/KL premature on 75% (OOD-depth) / 50% (OOD-both), 99.8% accurate | 5, Tab. 4 | `paper/tables/premature.tex`, `main_addition.tex` |
| Hidden-state change about 4 steps, 42.7% on OOD-both | 5 | `main_addition.tex` |
| Parity max-softmax: 91% premature, 51% / 0.8% accurate | 5 | `premature.tex`, `main_parity.tex` |
| Parity KL 94% / 57% | 5 | `main_parity.tex` |
| G2 passes | 5 | notebook 04, `results/gates/g2.json` |
| Per-position rules halt exactly at LB on addition (13.5 / 75.4 steps) | Abstract, 1, 5 | `main_addition.tex` (steps equal to the oracle LB; `oracle_lb` rows in `metrics.json`) |
| Probe AUC > 0.999 on every split | 5, App. C | notebook 08, `results/compute_probes_ablations.json`, `paper/tables/probes_ablations.tex` |
| Parity completion heads halt at about 9 steps; 31% / 0% accurate | Abstract, 1, 5 | `main_parity.tex` (completion-head rows) |
| G3 fails | 5 | notebook 05, `results/gates/g3.json` |
| FLOPs dominated by steps | 5, App. C | notebook 08, `fig8_compute` |
| v1 model: accuracy 1.0, 2.6 mean steps, 84% correct at step 1 | App. C, Tab. 5 | notebook 01, `paper/tables/v1_corrected.tex` |
| GPU-hours | App. B | notebook 09, `paper/tables/compute.tex` |
| Wilcoxon + Holm comparisons | 5 | notebook 09, `results/paper_numbers.json` (`holm`) |

Numbers in the text were checked against the tables after the final run of `scripts/make_figures.py`.
