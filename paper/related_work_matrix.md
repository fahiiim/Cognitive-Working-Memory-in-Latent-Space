# Related-work matrix (novelty audit)

Search protocol: `scripts/lit_search.py` runs 20 fixed arXiv queries (listed in the script)
and stores the raw hits in `paper/lit_search/<date>.json` and `.csv`. Run on **2026-10-13**: 315
unique papers, screened by title and abstract. Every arXiv ID below was checked against the arXiv
API on the same day. Semantic Scholar, OpenReview and Google Scholar still need a manual pass
with the same queries (see "Open items"). Re-run monthly and diff the CSVs.

Columns (the six properties our contributions rely on):

- **PD**: defines a per-instance minimum number of steps (depth), not just a size knob.
- **DL**: separates deeper inputs from longer inputs in the evaluation.
- **OOD-H**: tests a halting / stopping rule on inputs deeper or harder than training.
- **Loc**: halting signal is local (per position / per token) rather than pooled.
- **Th**: gives a formal result about when stopping must fail or can succeed.
- **Dec**: trains the halting signal decoupled from the task-loss weighting.

Marks: `Y` yes, `~` partial, `-` no. **Basis**: `F` = full paper read, `A` = abstract only (verify
before citing a specific claim).

## Closest works (property table candidates for the paper)

| # | Work | arXiv | PD | DL | OOD-H | Loc | Th | Dec | Basis | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Graves 2016, ACT | 1603.08983 | - | - | - | ~ | - | - | F | per-timestep ponder, pooled within step, ponder cost on task loss |
| 2 | Banino et al. 2021, PonderNet | 2107.05407 | - | - | ~ | - | - | - | F | parity extrapolation is in length; exit distribution weights the task loss |
| 3 | Dehghani et al. 2018, Universal Transformers | 1807.03819 | - | - | - | Y | - | - | F | per-token ACT |
| 4 | Figurnov et al. 2016, SACT | 1612.02297 | - | - | - | Y | - | - | F | spatially local halting in ResNets; image classification, no depth notion |
| 5 | Price, Zaremba, Sutskever 2016, Neural GPU limits | 1611.00736 | ~ | ~ | - | - | - | - | F | long carry chains break generalisation; random inputs rarely contain them |
| 6 | Schwarzschild et al. 2021, easy-to-hard | 2106.04537 | - | - | - | - | - | - | F | test-time iterations chosen by experimenter |
| 7 | Bansal et al. 2022, DT-recall + progressive loss | 2202.05826 | - | - | - | - | - | - | F | extrapolation by thinking longer; stops at fixed T |
| 8 | Anil et al. 2022, path independence | 2211.09961 | - | - | ~ | - | ~ | - | A | convergence-based use of test-time compute in equilibrium models |
| 9 | Veerabadran et al. 2023, AdRNNs | 2311.06964 | ~ | - | Y | - | - | - | A | ACT-style halting in ConvRNNs zero-shot scales to harder mazes/PathFinder; pooled gate, difficulty is a size/length knob |
| 10 | Fan et al. 2024, looped Transformers | 2409.15647 | ~ | - | ~ | - | - | - | F | needs ground-truth step counts; test-time max-confidence or oracle stopping |
| 11 | Bao et al. 2025, stop overthinking | 2502.10954 | - | - | ~ | - | - | Y | A | test-time training to choose per-sample computation for DT models |
| 12 | Geiping et al. 2025, recurrent depth | 2502.05171 | - | - | - | Y | - | Y | F | per-token KL-based zero-shot exit |
| 13 | Zhu et al. 2025, Ouro looped LMs | 2510.25741 | - | - | - | Y | - | - | A | learned exit gate with entropy regulariser |
| 14 | Wang et al. 2025, HRM | 2506.21734 | - | - | - | - | - | ~ | F | Q-learning halting head |
| 15 | Jolicoeur-Martineau 2025, TRM | 2510.04871 | - | - | - | - | - | ~ | F | correctness head used in training only |
| 16 | Moosa et al. 2026, ANIRA | 2602.08864 | ~ | - | ~ | - | - | - | A | compute tracks difficulty, difficulty = size; fails on unseen sizes |
| 17 | Nagle et al. 2026, TERMINATOR | 2603.12529 | - | - | - | - | - | Y | A | learns that an LLM CoT answer is already final |
| 18 | Kuo et al. 2026, stochastic stopping | 2606.29983 | - | ~ | ~ | - | - | - | A | traces OOD variance to length-loop correlation; no per-instance depth |
| 19 | Popescu et al. 2026, halting-gate diagnosis | 2607.20519 | - | - | - | - | - | Y | A | post-hoc confidence matches learned gates; gate-trajectory coupling |
| 20 | Krishna et al. 2026, when recursive models finish | 2609.26487 | - | - | ~ | - | - | Y | A | latent motion drops after the first exact solution (TRM on Sudoku) |
| 21 | Liang et al. 2026, beyond the training horizon | 2609.33144 | ~ | - | - | - | - | - | A | states encode "computational status"; reliability declines with depth |
| 22 | Brown et al. 2026, Settle | 2609.38997 | - | - | - | - | - | Y | A | stopping score predicts if a correct answer stays correct (LLM traces) |
| **-** | **This work (GLWM-2)** | - | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | - | per-instance light-cone depth, 2x2 splits, local completion heads |

## Wider context (cited, not in the property table)

| # | Work | arXiv / venue | Line of work |
|---|---|---|---|
| 23 | Kaiser and Sutskever 2016, Neural GPUs | 1511.08228 | recurrent/local models |
| 24 | Bai et al. 2019, Deep Equilibrium Models | 1909.01377 | implicit models |
| 25 | Giannou et al. 2023, looped Transformers as programmable computers | 2301.13196 | looped models |
| 26 | Yang et al. 2023, looped Transformers learn learning algorithms | 2311.12424 | looped models |
| 27 | Saunshi et al. 2025, reasoning with latent thoughts | 2502.17416 | looped models / theory |
| 28 | Hao et al. 2024, Coconut | 2412.06769 | latent reasoning |
| 29 | Ge et al. 2025, HRM perspectives and misconceptions | 2510.00355 | HRM/TRM |
| 30 | Teerapittayanon et al. 2017, BranchyNet | 1709.01686 | early exit |
| 31 | Elbayad et al. 2019, Depth-Adaptive Transformer | 1910.10073 | early exit |
| 32 | Xin et al. 2020, DeeBERT | 2004.12993 | early exit |
| 33 | Zhou et al. 2020, PABEE (patience) | 2006.04152 | early exit; k-step stability baseline |
| 34 | Schuster et al. 2022, CALM | 2207.07061 | early exit with calibrated confidence |
| 35 | Kadavath et al. 2022, LMs (mostly) know what they know | 2207.05221 | correctness prediction |
| 36 | Yang et al. 2025, dynamic early exit in reasoning models | 2504.15895 | LLM adaptive thinking |
| 37 | Liu et al. 2025, answer convergence | 2506.02536 | LLM adaptive thinking |
| 38 | Nogueira et al. 2021, limits of Transformers on arithmetic | 2102.13019 | arithmetic generalisation |
| 39 | Anil et al. 2022, length generalisation in LLMs | 2207.04901 | length generalisation |
| 40 | Deletang et al. 2022, Chomsky hierarchy | 2207.02098 | length generalisation |
| 41 | Liu et al. 2022, shortcuts to automata | 2210.10749 | depth vs length theory |
| 42 | Zhou et al. 2023, RASP-L length generalisation | 2310.16028 | length generalisation |
| 43 | McLeish et al. 2024, arithmetic with the right embeddings | 2405.17399 | looped models, addition |
| 44 | Dijkstra and Scholten 1980, termination detection for diffusing computations | Inf. Process. Lett. 11(1):1-4, doi:10.1016/0020-0190(80)90021-6 | distributed termination detection |

## Verdict on the four contributions (2026-10-13)

- **Per-instance depth + depth/length separation**: not found in any screened paper. Price et al.
  (row 5) observe the carry-chain issue but do not build a split or a halting study on it. Kuo et
  al. (row 18) identify the length-loop correlation but use length, not per-instance depth.
- **Halting tested deeper than training**: Veerabadran et al. (row 9) and ANIRA (row 16) test
  halting on harder/larger instances, but difficulty is tied to size, so deeper and longer are
  confounded. This is the main prior work to position against.
- **Light-cone theory**: no screened paper gives a lower bound or unsoundness result for stopping
  rules of local recurrent models.
- **Local, decoupled completion heads**: per-token halting exists (rows 3, 4, 12, 13), and decoupled
  correctness/stability predictors exist (rows 17, 19, 20, 22), but none combine a local
  "my part is finished" target with an evaluation beyond the training depth.

## Open items

- Manual pass on Semantic Scholar, OpenReview (ICLR 2027 submissions appear in early October) and
  Google Scholar with the same queries.
- Read rows 9, 16, 18, 19, 20, 21 in full (currently abstract-level) before the related-work section is final.
