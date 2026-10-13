# Pre-registration: GLWM-2 termination detection under depth shift

Frozen on **2026-10-13**, before any main-grid run. Changes after this date go in the
"Deviations" log at the bottom with a date and a reason. Out-of-distribution (OOD) test sets are
evaluated once per trained model, at the end of its run; nothing is tuned on them.

## 1. Hypotheses

- **H1 (length leakage).** For pooled, jointly trained gates (ACT, PonderNet, joint TRM-style head,
  v1 GLWM gate), the number of steps used correlates with input length at fixed depth d\*
  (partial Spearman rho(steps, length | d\*) > 0 on ID + OOD-length). These gates also fail to use
  more steps when d\* exceeds the training range: mean steps on OOD-depth < mean lower bound.
- **H2 (premature stopping).** Post-hoc confidence and convergence rules (max-softmax, entropy,
  k-step stability, KL, hidden-state change) halt before the lower bound ceil(d\*/r) on a
  substantial share of OOD-depth and OOD-both instances. This happens while oracle-over-iterations
  accuracy on those instances stays high.
- **H3 (extrapolation).** Local completion heads keep the premature-stop rate near zero on every
  split. They halt within a few steps of the lower bound and match oracle-over-iterations accuracy.
- **H4 (decoupling).** Training a completion head as an auxiliary task changes the per-step
  accuracy curve of the backbone by less than 1 point (area under the curve, ID validation).
  Jointly learned gates that weight the task loss (ACT, PonderNet, v1 gate) change it by more.

## 2. Tasks, depth oracles and splits

Depth is the light-cone depth d\*(x): the smallest number of positions information must travel so
that every output is determined, counting the left boundary as a virtual position -1 (see
`paper/sections/theory.tex`). For a local model with receptive-field growth r per step, the lower
bound is LB(x) = ceil(d\*(x)/r).

- **T1 addition.** L-digit operands, least-significant digit first, L+1 output digits. d\* is the
  largest gap between consecutive "stops" (positions with a_j + b_j != 9, the virtual position -1,
  and the final position L). 1 <= d\* <= L+1.
- **T2 segmented parity.** Bits plus a delimiter token; output = parity of the current segment so
  far. d\* = the longest run of non-delimiter tokens. 1 <= d\* <= L.
- **T3 mazes.** Perfect mazes (randomised DFS) on an n x n cell grid rendered as (2n+1)^2 pixels;
  output = pixels on the start-goal path. Two depth proxies: the path length and the dead-end
  filling depth. Neither is a tight lower bound, so T3 is empirical only.

| Split | Length L | Depth d\* | Size (per task) | Use |
|---|---|---|---|---|
| train | 4-20 | 1-10 | sampled online, test inputs filtered out by hash | training |
| ID-val | 4-20 | 1-10 | 2,000 | thresholds, early stopping, completion-head fitting checks |
| ID-test | 4-20 | 1-10 | 2,000 | final |
| OOD-val | 24-32 | 12-16 | 1,000 | backbone hyperparameters only |
| OOD-depth | 10-20 (T1), 11-20 (T2) | 11-20 | 2,000 | final |
| OOD-length | 64-256 | 1-10 | 1,000 | final |
| OOD-both | 64-256 | 32-128 | 1,000 | final |

Length is sampled uniformly in the range, then depth uniformly in the allowed range given the
length. Mazes: train and ID-val at n = 9; OOD tests at n in {13, 17, 21, 25, 33} (500 each).

Frozen test sets are generated with fixed seeds by `glwm2.data.splits.build_frozen_splits`, saved
under `data/frozen/`, and their SHA-256 hashes are committed in `data/frozen/HASHES.json`.

## 3. Models and training

- **Backbone**: one latent per position; pointwise encoder; shared step
  h <- LayerNorm(h + MLP(Conv_k3([h; e(x)]))); pointwise decoder. One kernel-3 convolution per step
  for T1/T2 (r = 1), two for mazes (r = 2).
- **Objectives**: fixed-prior per-step cross-entropy (uniform prior over steps 1..T_train) and
  progressive loss (Bansal et al. 2022, alpha = 0.5). The pilot picks one per task on OOD-val and
  ID-val only. T_train = 20 steps for T1/T2.
- **Seeds**: 5 per configuration (0-4); 10 (0-9) for the headline comparison on T1 OOD-depth.
- **Tuning budget**: at most 16 configurations per method, selected on validation data only.

## 4. Stopping rules

References: fixed T, oracle lower bound LB(x), oracle-over-iterations (first step at which the
answer is exactly right).
Post-hoc: max-softmax (min over positions), entropy (max over positions), k-step stability
(k in {2, 3, 5}), KL between consecutive steps (max over positions), relative hidden-state change,
Fan et al. maximum confidence (runs to T_max), post-hoc TRM-style correctness head.
Joint: ACT, PonderNet (KL to a truncated geometric prior), joint TRM-style head, v1 GLWM gate
(initialisation bug fixed).
Ours: local completion head fitted post hoc, trained jointly as an auxiliary task, and the
dependency-propagating variant.

**Threshold selection (all threshold rules).** On ID-val, among thresholds whose accuracy is within
0.5 points of the best accuracy that rule reaches on ID-val, pick the one with the fewest mean
steps. Full accuracy-versus-steps curves are reported as well.

T_max for evaluation = max LB in the split + 16.

## 5. Metrics

Exact-match accuracy; mean steps; premature-stop rate (share halted before LB);
overthinking (steps beyond the first step at which the answer becomes right and stays right);
depth tracking (Spearman rho(steps, d\*)); length leakage (partial Spearman rho(steps, L | d\*));
regret versus oracle-over-iterations (accuracy gap and step gap); completion-head calibration
(ECE, ID vs OOD); measured FLOPs (`torch.utils.flop_counter.FlopCounterMode`) and CUDA-event latency.

Statistics: per-instance paired bootstrap 95% CIs (10,000 resamples) for differences between
rules on the same model; Wilcoxon signed-rank across seeds; Holm correction over all pairwise
comparisons reported in one table.

## 6. Decision gates

- **G1 (backbone).** Oracle-over-iterations accuracy >= 90% on instances with d\* >= 40 (4x the
  training maximum) in OOD-both, on both T1 and T2, averaged over seeds. If not met, the claims are
  narrowed to the depth range that does extrapolate.
- **G2 (baselines fail).** At least one baseline family shows premature-stop rate >= 5% on OOD-depth
  or OOD-both, or partial rho(steps, L | d\*) >= 0.1, with the 95% CI excluding zero.
- **G3 (our method).** On OOD-depth, the best completion-head variant beats the best baseline at the
  val-selected operating point by >= 10 accuracy points or >= 50% fewer premature stops, with the
  95% CI of the difference excluding zero across 5 seeds.
- **Pivot.** If G2 or G3 fails, the paper becomes a theory-plus-benchmark analysis of when simple
  stopping signals suffice.

## Deviations

- **2026-10-13, before any main-grid run (found in a 2,000-iteration smoke run).** "Oracle over
  iterations" defined as "the first step at which the answer is exactly right" counts lucky
  guesses: at unresolved carry positions the model's default guess is right half the time, so this
  oracle reached 99.6% on OOD-both while the model's accuracy at a fixed step was 43%. We add the
  stricter **stable oracle** (halt at the first step from which the answer stays right until
  T_max; accuracy = share of instances that are stably right by T_max). G1 and the regret metrics
  use the stable oracle; the any-step oracle is still reported. This makes G1 harder to pass.
- **2026-10-13, after the first pilot run (addition, fixed-prior, seed 0), before the main grid.**
  Global convergence rules (k-step stability, KL between steps) and the stable oracle halt
  before the light-cone bound on ~75% of OOD-depth instances while staying 100% accurate. This
  is consistent with Theorem 1 (an answer before the bound is uncertified, not necessarily
  wrong): when no carry enters a nine-run, the model's default output is already right, and when
  a carry does enter, a visible wave of output changes keeps global stability rules from firing.
  We therefore add a secondary metric, **premature-error rate** = share of instances halted
  before the bound *and* answered wrongly, reported next to the premature-stop rate. G2 is still
  decided by the pre-registered premature-stop definition, and the paper reports both, so a G2
  pass that rests only on premature-but-correct halting is described as such.
- **2026-10-13, pilot objective selection.** Both objectives reach 100% stable-oracle accuracy on
  OOD-val and ID-val for both tasks, so the pre-registered selection is a tie. No tie-break was
  pre-registered; we keep the first objective in the pre-registered config order
  (`fixed_prior`) for both tasks and did not consult any test split for this choice. (For the
  record, OOD-both, which was not used, favours fixed-prior on addition and progressive on parity.)
- **2026-10-13, G1.** Addition passes (stable-oracle accuracy 99.7% on OOD-both instances with
  d\* >= 40). Parity fails (about 78%); as pre-registered, parity claims are restricted to the depth
  range where the backbone extrapolates, and accuracy is reported as a function of depth.
- **2026-10-13, reduced scale (compute).** One RTX 5070 shared by up to five runs gives about
  20 minutes per T1/T2 run. We run 3 seeds (0-2) for every method, 5 seeds (0-4) for the base model
  on both tasks and 10 (0-9) for the addition base model (which carries every post-hoc rule,
  including our post-hoc completion heads), 3 seeds for the pooled gates, 1 seed for ablations and
  2 for mazes. The gate hyperparameters get an equal budget of 3 settings each
  (`configs/tune_gates.yaml`). Every reported CI states its number of seeds.
- **2026-10-13, mazes.** The first maze configuration (fixed-prior, width 64, T_train 32, 5000
  iterations) reached 0% exact match on training-size mazes; a width-128 retry ran at 3.5 s per
  iteration and was stopped. The maze backbone is re-piloted with the progressive loss
  (alpha = 0.01, as in Bansal et al. 2022 for mazes) and T_train 40 (`configs/pilot_mazes.yaml`).
- **2026-10-13, stability of global rules (observation, no change to analysis).** The ACT model
  with tau = 0.01 halts at step 1 or 5 depending on whether any carry enters the long nine-run: the
  pooled gate reads this from the whole sequence, and the backbone's default output switches from
  "no carry" to "carry" around step 4. This beats the light-cone bound on OOD-depth (one long run)
  but fails on OOD-length and OOD-both (many runs). The analysis reports it as a shortcut.
- **2026-10-13, 5-hour compute limit (supersedes the reduced-scale entry above).** Under six
  concurrent runs an 8000-iteration run took 1.5-3 h, so the remaining grid (about 83 runs) was cut
  to fit a 5-hour budget, before any main-grid result was analysed: 2 seeds (0-1) per method on T1
  and T2, pooled gates with 2 seeds on both tasks, 3 ablation runs (no recall, r = 2, width 64; seed
  0), no extra base seeds. Training length stays at 8000 iterations so all runs are comparable.
  Gate tuning was stopped after ACT: ACT uses tau = 0.01 (the default 1e-3 never halts); PonderNet
  and the v1 gate use their defaults, which is a weaker tuning budget than pre-registered.
- **2026-10-13, mazes dropped.** The progressive re-pilot reached 3% exact match on training-size
  mazes after 1000 iterations (loss not decreasing) and was stopped under the compute limit. Mazes
  are reported as a negative result: we could not train a maze backbone that passes G1 within the
  budget, so no maze stopping-rule claims are made.
