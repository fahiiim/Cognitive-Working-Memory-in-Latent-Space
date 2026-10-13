"""Generate the notebooks in notebooks/ from the cell lists below.

    .venv\\Scripts\\python.exe scripts/build_notebooks.py            # write all notebooks
    .venv\\Scripts\\python.exe scripts/build_notebooks.py --run 02   # write, then execute 02*.ipynb

Each notebook starts with a parameters cell, imports its heavy logic from src/glwm2, and ends by
listing the files it wrote. Executing a notebook here is the "Restart and Run All" check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
%matplotlib inline
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from glwm2 import PROJECT_ROOT, RESULTS_DIR, PAPER_DIR
from glwm2.eval.plotting import set_style
set_style()
WRITTEN = []  # files this notebook writes
"""

FOOTER = """\
print("Files written by this notebook:")
for p in WRITTEN:
    print(" ", Path(p).relative_to(PROJECT_ROOT))
"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS: dict[str, list] = {}

# --------------------------------------------------------------------------------------------
NOTEBOOKS["00_env_check"] = [
    md("""
# 00 - Environment check
Checks the Python/torch/CUDA versions, that the RTX 5070 (compute capability 12.0, `sm_120`) is
supported, bf16 matmul throughput, and run-to-run determinism. Writes `results/env.json`.
"""),
    code("""
# parameters
MATMUL_N = 4096
REPEATS = 50
"""),
    code(SETUP),
    code("""
from glwm2.utils.io import env_info, write_json
info = env_info()
for k, v in info.items():
    print(f"{k:22s} {v}")
assert torch.cuda.is_available(), "CUDA not available"
assert tuple(info["capability"]) == (12, 0) and info["sm_120_in_arch_list"]
"""),
    code("""
a = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    a @ a
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(REPEATS):
    a @ a
end.record(); torch.cuda.synchronize()
sec = start.elapsed_time(end) / 1e3
info["bf16_matmul_tflops"] = round(REPEATS * 2 * MATMUL_N**3 / sec / 1e12, 1)
print("bf16 TFLOPS:", info["bf16_matmul_tflops"])
"""),
    code("""
# determinism: two identically seeded short training runs give identical weights
from glwm2.utils.config import RunConfig
from glwm2.train.trainer import train
def tiny_run():
    cfg = RunConfig(task="addition", iters=30, batch_size=64, width=32, t_train=8, eval_n=0, warmup=5)
    return train(cfg, save=False, deterministic=True)["model"]
m1, m2 = tiny_run(), tiny_run()
same = all(torch.equal(p, q) for p, q in zip(m1.state_dict().values(), m2.state_dict().values()))
info["deterministic_training"] = bool(same)
print("deterministic:", same)
"""),
    code("""
path = write_json(RESULTS_DIR / "env.json", info); WRITTEN.append(path)
"""),
    code(FOOTER),
]

# --------------------------------------------------------------------------------------------
NOTEBOOKS["01_v1_corrected"] = [
    md("""
# 01 - v1 GLWM, corrected re-run (motivation and tooling check)
Re-implements the v1 GLWM (GRU encoder, residual latent loop, PonderNet-style gate, halting at
gate > 0.7, T_max = 8) on 1-4 digit addition, with the fixes the review asked for:
the gate initialisation as published (`v1`: both gate biases -2) and as intended (`fixed`),
a validation split for model selection, a test set de-duplicated against training, 5 seeds,
and FLOPs measured with `FlopCounterMode` instead of hand-derived formulas.
"""),
    code("""
# parameters
SEEDS = [0, 1, 2, 3, 4]
INITS = ["v1", "fixed"]
EPOCHS = 30
"""),
    code(SETUP),
    code("""
from glwm2.models.v1_glwm import train_v1
from glwm2.utils.io import read_json, write_json
OUT = RESULTS_DIR / "v1"
res = []
for init in INITS:
    for seed in SEEDS:
        p = OUT / f"{init}_s{seed}.json"
        if not p.exists():
            r, _ = train_v1(init, seed, epochs=EPOCHS)
            write_json(p, r)
        res.append(read_json(p)); WRITTEN.append(p)
print(len(res), "runs")
"""),
    code("""
rows = []
for r in res:
    for proto in ["test_v1_protocol", "test_dedup"]:
        t = r[proto]
        rows.append(dict(init=r["gate_init"], seed=r["seed"], protocol=proto, acc=t["acc"], mean_steps=t["mean_steps"],
                         steps_4digit=t["steps_by_digits"]["4"] if "4" in t["steps_by_digits"] else t["steps_by_digits"][4],
                         acc_step1=t["acc_at_step"][0], gflops_per_sample=r["flops"]["per_sample_at_mean_steps"] / 1e9))
df = pd.DataFrame(rows)
summary = df.groupby(["init", "protocol"]).agg(["mean", "std"]).drop(columns="seed")
summary
"""),
    code("""
dup = pd.DataFrame([r["test_dup_share_by_digits"] for r in res]).mean()
print("share of v1-protocol test pairs that also appear in train/val, by digit count:")
print(dup.round(3).to_string())
"""),
    code("""
from glwm2.eval.plotting import savefig
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
for init, c in zip(INITS, ["#D55E00", "#009E73"]):
    H = np.mean([r["test_dedup"]["steps_hist"] for r in res if r["gate_init"] == init], 0)
    axes[0].bar(np.arange(1, 9) + (0.2 if init == "fixed" else -0.2), H / H.sum(), width=0.4, color=c, label=f"gate init: {init}")
    A = np.mean([r["test_dedup"]["acc_at_step"] for r in res if r["gate_init"] == init], 0)
    axes[1].plot(np.arange(1, 9), A, "o-", color=c, label=init)
axes[0].set_xlabel("halting step"); axes[0].set_ylabel("share of test instances"); axes[0].legend(frameon=False)
axes[1].set_xlabel("latent step t"); axes[1].set_ylabel("exact-match accuracy at step t"); axes[1].set_ylim(0.9, 1.001)
WRITTEN.append(savefig(fig, "figA_v1_corrected"))
"""),
    code("""
tab = df[df.protocol == "test_dedup"].groupby("init").agg(
    acc=("acc", "mean"), acc_sd=("acc", "std"), steps=("mean_steps", "mean"), steps4=("steps_4digit", "mean"),
    acc1=("acc_step1", "mean"), gflops=("gflops_per_sample", "mean"))
lines = [r"\\begin{tabular}{lrrrrr}", r"\\toprule",
         r"Gate init & Accuracy & Mean steps & Steps (4-digit) & Acc.\\ at step 1 & MFLOPs/sample \\\\", r"\\midrule"]
for init, r in tab.iterrows():
    lines.append(f"{init} & {r.acc:.4f} $\\\\pm$ {r.acc_sd:.4f} & {r.steps:.2f} & {r.steps4:.2f} & {r.acc1:.4f} & {r.gflops*1e3:.2f} \\\\\\\\")
lines += [r"\\bottomrule", r"\\end{tabular}"]
p = PAPER_DIR / "tables" / "v1_corrected.tex"; p.write_text("\\n".join(lines)); WRITTEN.append(p)
print("\\n".join(lines))
WRITTEN.append(write_json(RESULTS_DIR / "v1" / "summary.json", tab.reset_index().to_dict(orient="records")))
"""),
    code(FOOTER),
]

# --------------------------------------------------------------------------------------------
NOTEBOOKS["02_tasks_and_depth"] = [
    md("""
# 02 - Tasks, depth oracles and the 2x2 splits
Builds the generators and depth oracles (`src/glwm2/data`), runs the flip checks, builds the
frozen splits with their SHA-256 hashes, and plots depth and length histograms per split.
"""),
    code("""
# parameters
N_CHECK = 300
"""),
    code(SETUP),
    code("""
from glwm2.data.addition import sample_addition
from glwm2.data.parity import sample_parity
from glwm2.data.depth import carry_depth, segment_depth, addition_rho
from glwm2.data.splits import build_frozen_splits, load_split, SEQ_SPLITS, MAZE_SPLITS, split_names
g = torch.Generator().manual_seed(0)
b = sample_addition(4, (6, 8), (3, 6), g)
for row, L, d in zip(b["x"].tolist(), b["length"].tolist(), b["depth"].tolist()):
    a = [t // 10 for t in row[:L]]; bb = [t % 10 for t in row[:L]]
    A = int("".join(map(str, a[::-1]))); B = int("".join(map(str, bb[::-1])))
    print(f"{A:>9d} + {B:>9d} = {A+B:>10d}   d* = {d}  (reference {carry_depth(a, bb)})")
"""),
    code("""
# flip checks (the full versions are tests/test_depth.py and tests/test_theory.py)
import subprocess
out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(PROJECT_ROOT / "tests" / "test_depth.py"),
                      str(PROJECT_ROOT / "tests" / "test_data.py")], capture_output=True, text=True, cwd=PROJECT_ROOT)
print(out.stdout[-400:])
assert out.returncode == 0
"""),
    code("""
hashes = {t: build_frozen_splits(t) for t in ["addition", "parity", "mazes"]}
rows = []
for task in ["addition", "parity"]:
    for s in split_names(task):
        d = load_split(task, s)
        rows.append(dict(task=task, split=s, n=len(d["x"]), L_min=int(d["length"].min()), L_max=int(d["length"].max()),
                         d_min=int(d["depth"].min()), d_max=int(d["depth"].max()), sha256=hashes[task][s][:16]))
for s in split_names("mazes"):
    d = load_split("mazes", s)
    rows.append(dict(task="mazes", split=s, n=len(d["x"]), L_min=int(d["size"][0]), L_max=int(d["size"][0]),
                     d_min=int(d["depth"].min()), d_max=int(d["depth"].max()), sha256=hashes["mazes"][s][:16]))
splits_df = pd.DataFrame(rows); splits_df
"""),
    code("""
from glwm2.eval.plotting import savefig, SPLIT_COLORS
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, task in zip(axes, ["addition", "parity"]):
    for s in ["id_test", "ood_depth", "ood_length", "ood_both"]:
        d = load_split(task, s)
        ax.scatter(d["length"].numpy(), d["depth"].numpy(), s=3, alpha=0.4, color=SPLIT_COLORS[s], label=s)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("length L"); ax.set_ylabel("depth $d^*$"); ax.set_title(task)
axes[0].legend(markerscale=4, fontsize=7, frameon=False)
WRITTEN.append(savefig(fig, "fig2_splits"))
"""),
    code("""
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for s in split_names("mazes"):
    d = load_split("mazes", s)
    ax.scatter(d["path_len"].numpy(), d["depth"].numpy(), s=3, alpha=0.4, label=s)
ax.set_xlabel("path length"); ax.set_ylabel("dead-end filling depth"); ax.legend(fontsize=6, markerscale=3, frameon=False)
WRITTEN.append(savefig(fig, "figA_maze_depths"))
from glwm2.utils.io import write_json
WRITTEN.append(write_json(RESULTS_DIR / "splits_summary.json", rows))
WRITTEN.append(PROJECT_ROOT / "data" / "frozen" / "HASHES.json")
"""),
    code(FOOTER),
]


# --------------------------------------------------------------------------------------------
NOTEBOOKS["03_backbone_pilot"] = [
    md("""
# 03 - Backbone pilot (decision gate G1)
Compares the fixed-prior per-step objective with the progressive loss (Bansal et al. 2022) for
the GLWM-2 backbone on addition (T1) and segmented parity (T2). The runs themselves are launched
from a terminal with `scripts/run_sweep.py configs/pilot_addition.yaml`; this notebook only reads
their results. The objective for the main grid is chosen per task on OOD-val and ID-val only.

**G1 (pre-registered):** stable-oracle accuracy >= 90% on OOD-both instances with d* >= 40
(4x the training maximum), on both tasks, averaged over seeds.
"""),
    code("""
# parameters
CONFIG = "configs/pilot_addition.yaml"
G1_DEPTH, G1_ACC = 40, 0.90
"""),
    code(SETUP),
    code("""
from glwm2.eval.results import configs_from_yaml, finished, metric_rows, instance_frame, step_acc
from glwm2.utils.io import write_json
cfgs = configs_from_yaml(PROJECT_ROOT / CONFIG)
done = finished(cfgs)
print(f"{len(done)}/{len(cfgs)} pilot runs finished")
df = metric_rows(done)
ref = df[df.rule.isin(["oracle_stable", "oracle_iter", "oracle_lb", "fixed_T"])]
tab = ref.pivot_table(index=["task", "objective", "rule"], columns="split", values="acc", aggfunc="mean")
tab.round(3)
"""),
    code("""
# G1: stable-oracle accuracy on deep OOD-both instances, per run
g1_rows = []
for c in done:
    inst = instance_frame(c, "ood_both", ["oracle_stable"])
    deep = inst[inst.depth >= G1_DEPTH]
    g1_rows.append(dict(task=c.task, objective=c.objective, seed=c.seed, n_deep=len(deep), acc_deep=deep.correct.mean()))
g1 = pd.DataFrame(g1_rows)
g1_summary = g1.groupby(["task", "objective"]).acc_deep.agg(["mean", "std", "count"])
g1_summary
"""),
    code("""
# objective selection on validation splits only (OOD-val, then ID-val as tie-break)
sel = (df[df.rule == "oracle_stable"].pivot_table(index=["task", "objective"], columns="split", values="acc", aggfunc="mean")
       [["ood_val", "id_val"]])
choice = {}
for task in sel.index.get_level_values(0).unique():
    s = sel.loc[task].sort_values(["ood_val", "id_val"], ascending=False)
    choice[task] = s.index[0]
print("selected objective:", choice)
g1_pass = {task: bool(g1_summary.loc[(task, obj), "mean"] >= G1_ACC) for task, obj in choice.items()}
print("G1 passed per task:", g1_pass, "-> overall", all(g1_pass.values()))
"""),
    code("""
from glwm2.eval.plotting import savefig, SPLIT_COLORS
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, task in zip(axes, ["addition", "parity"]):
    for obj, ls in [("fixed_prior", "-"), ("progressive", "--")]:
        runs = [c for c in done if c.task == task and c.objective == obj]
        if not runs:
            continue
        for split in ["id_test", "ood_depth", "ood_length", "ood_both"]:
            curves = [step_acc(c)[split] for c in runs]
            T = min(len(v) for v in curves)
            ax.plot(np.arange(1, T + 1), np.mean([v[:T] for v in curves], 0), ls, color=SPLIT_COLORS[split],
                    label=f"{split} ({obj})" if task == "addition" else None)
    ax.set_xscale("log"); ax.set_xlabel("step t"); ax.set_ylabel("exact-match accuracy at step t"); ax.set_title(task)
axes[0].legend(fontsize=6, frameon=False, ncol=1)
WRITTEN.append(savefig(fig, "figA_step_accuracy"))
"""),
    code("""
# Fig. 3: stable-oracle accuracy against depth and against length (selected objective, all splits pooled)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for task, color in [("addition", "#D55E00"), ("parity", "#0072B2")]:
    runs = [c for c in done if c.task == task and c.objective == choice.get(task)]
    frames = [instance_frame(c, s, ["oracle_stable"]) for c in runs for s in ["id_test", "ood_depth", "ood_length", "ood_both"]]
    if not frames:
        continue
    inst = pd.concat(frames)
    for ax, col, bins in [(axes[0], "depth", [0, 2, 4, 8, 12, 16, 20, 32, 48, 64, 96, 128]), (axes[1], "length", [4, 8, 12, 16, 20, 64, 96, 128, 192, 256])]:
        inst["bin"] = pd.cut(inst[col], bins, include_lowest=True)
        g = inst.groupby("bin", observed=True).correct.mean()
        ax.plot([b.right for b in g.index], g.values, "o-", color=color, label=task)
        ax.set_xscale("log", base=2); ax.set_xlabel(col if col == "length" else "depth $d^*$"); ax.set_ylim(0, 1.02)
axes[0].axvline(10, color="grey", lw=0.8, ls=":"); axes[1].axvline(20, color="grey", lw=0.8, ls=":")
axes[0].set_ylabel("stable-oracle accuracy"); axes[0].legend(frameon=False)
WRITTEN.append(savefig(fig, "fig3_oracle_vs_depth_length"))
"""),
    code("""
# position-by-step correctness for one deep OOD-both addition instance
from glwm2.train.trainer import load_model
from glwm2.data.splits import load_split
from glwm2.eval.trajectory import per_position_rho
c = next(c for c in done if c.task == "addition" and c.objective == choice["addition"])
model = load_model(c)
data = load_split("addition", "ood_both")
k = int(torch.argsort(data["depth"])[len(data["depth"]) // 2])
x, y = data["x"][k:k+1].cuda(), data["y"][k:k+1].cuda()
L = int(data["length"][k]); rho = per_position_rho("addition", {k_: v[k:k+1] for k_, v in data.items()})[0, :L+1].numpy()
with torch.no_grad():
    T = int(data["depth"][k]) + 16
    pc = torch.stack([(o["logits"].argmax(-1) == y)[0, :L+1] for o in model.rollout(x, T)]).cpu().numpy()
fig, ax = plt.subplots(figsize=(7.0, 2.4))
ax.imshow(pc.T, aspect="auto", origin="lower", cmap="Greys_r", interpolation="nearest", extent=[0.5, T + 0.5, -0.5, L + 0.5])
ax.plot(rho, np.arange(L + 1), color="#D55E00", lw=1, label=r"lower bound $\\rho_i$")
ax.set_xlabel("step t"); ax.set_ylabel("output position i"); ax.legend(frameon=False, loc="upper left")
ax.set_title(f"addition, L={L}, d*={int(data['depth'][k])}: white = correct")
WRITTEN.append(savefig(fig, "figA_position_by_step"))
"""),
    code("""
WRITTEN.append(write_json(RESULTS_DIR / "pilot_summary.json", {
    "objective": choice, "g1": g1_summary.reset_index().to_dict(orient="records"), "g1_pass": g1_pass,
    "table": tab.reset_index().to_dict(orient="records")}))
"""),
    code(FOOTER),
]


MAIN_LOAD = """\
from glwm2.eval.results import configs_from_yaml, finished, curve_rows, step_acc
from glwm2.eval.analysis import (ENTRIES, BASELINES, OURS, POOLED_JOINT, CONF_RULES, entry_rows, entry_instances,
                                 summary_table, g2_check, g3_check, binned, premature_ci, length_leakage_ci)
from glwm2.eval.plotting import savefig, FAMILY_COLORS
from glwm2.utils.io import write_json
cfgs = {t: [c for f in CONFIGS for c in configs_from_yaml(PROJECT_ROOT / f) if c.task == t] for t in TASKS}
for t in TASKS:
    print(t, f"{len(finished(cfgs[t]))}/{len(cfgs[t])} runs finished, seeds:",
          sorted({c.seed for c in finished(cfgs[t])}))
rows = {t: entry_rows(cfgs[t]) for t in TASKS}
"""

NOTEBOOKS["04_stopping_rule_zoo"] = [
    md("""
# 04 - Stopping-rule zoo (decision gate G2)
Every post-hoc rule is evaluated at every threshold in one pass over each test set (thresholds
chosen on ID-val, `src/glwm2/readouts/rules.py`); the jointly trained gates (ACT, PonderNet,
joint correctness head, v1 GLWM gate) come from their own runs. Checks **H1** (length leakage of
pooled gates) and **H2** (premature stopping of confidence/convergence rules), then **G2**.
"""),
    code("""
# parameters
CONFIGS = ["configs/main_addition.yaml", "configs/main_parity.yaml", "configs/main_gates.yaml"]
TASKS = ["addition", "parity"]
SPLITS = ["id_test", "ood_depth", "ood_length", "ood_both"]
"""),
    code(SETUP),
    code("""
# Gate tuning (equal budget, addition seed 0): pick per method the setting with the fewest ID-val
# steps among those within 0.5 points of the best ID-val accuracy. The choice is copied into
# configs/main_gates.yaml before the main gate runs.
from glwm2.eval.results import configs_from_yaml, finished, metric_rows
from glwm2.utils.io import write_json
GATE_RULE = {"act": "act", "ponder": "ponder", "v1gate": "gate"}
HP = {"act": ["act_tau"], "ponder": ["ponder_beta", "ponder_lambda_p"], "v1gate": ["v1_lambda"]}
tune = [c for f in ["configs/tune_gates.yaml", "configs/main_gates.yaml"] for c in configs_from_yaml(PROJECT_ROOT / f)
        if c.task == "addition" and c.seed == 0]
tdf = metric_rows(finished(tune))
tuning = {}
if len(tdf):
    tdf = tdf[np.array([GATE_RULE.get(m) == r for m, r in zip(tdf.method, tdf.rule)]) & (tdf.split == "id_val").to_numpy()]
    for m, sub in tdf.groupby("method"):
        ok = sub[sub.acc >= sub.acc.max() - 0.005].sort_values("mean_steps")
        best = ok.iloc[0]
        tuning[m] = {k: float(best[k]) for k in HP[m]}
        display(sub[HP[m] + ["acc", "mean_steps"]].round(4))
print("selected gate settings:", tuning)
WRITTEN.append(write_json(RESULTS_DIR / "gates" / "tuning.json", tuning))
"""),
    code(MAIN_LOAD),
    code("""
for t in TASKS:
    print(f"== {t}: accuracy (mean over seeds)")
    display(summary_table(rows[t], SPLITS, "acc")["mean"].round(3))
"""),
    code("""
for t in TASKS:
    print(f"== {t}: premature-stop rate and mean steps on OOD-depth / OOD-both")
    sub = rows[t][rows[t].split.isin(["ood_depth", "ood_both"])]
    display(sub.pivot_table(index="entry", columns="split", values=["premature", "mean_steps", "mean_lb"], aggfunc="mean")
            .reindex([e for e in ENTRIES if e in set(sub.entry)]).round(3))
"""),
    code("""
# H1: length leakage (partial Spearman of steps with length given depth) on ID-test + OOD-length,
# and whether pooled gates add steps when depth exceeds training (OOD-depth mean steps vs mean lower bound)
h1, g2 = {}, {}
for t in TASKS:
    inst_len = pd.concat([entry_instances(cfgs[t], s) for s in ["id_test", "ood_length"]])
    inst_deep = pd.concat([entry_instances(cfgs[t], s) for s in ["ood_depth", "ood_both"]])
    res = g2_check(inst_deep, inst_len)
    od = rows[t][rows[t].split == "ood_depth"].groupby("entry")[["mean_steps", "mean_lb"]].mean()
    res = res.merge(od, left_on="entry", right_index=True, how="left")
    h1[t] = res; display(res.round(3))
    g2[t] = bool(res.shows_premature.any() or res.shows_leak.any())
print("G2 (some baseline family shows premature stopping or length leakage, CI excluding 0):", g2)
"""),
    code("""
# Fig. 4: steps against depth (OOD-depth + ID) and against length (ID + OOD-length), addition
show = ["oracle_lb", "max_softmax", "stability", "kl", "act", "ponder", "v1gate", "trm_joint", "done_prop_post", "done_joint"]
fig, axes = plt.subplots(len(TASKS), 2, figsize=(7.0, 2.4 * len(TASKS)), squeeze=False)
for i, t in enumerate(TASKS):
    deep = pd.concat([entry_instances(cfgs[t], s, show) for s in ["id_test", "ood_depth", "ood_both"]])
    longi = pd.concat([entry_instances(cfgs[t], s, show) for s in ["id_test", "ood_length"]])
    for ax, inst, col, bins in [(axes[i, 0], deep, "depth", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 48, 64, 96, 128]),
                                (axes[i, 1], longi, "length", [4, 8, 12, 16, 20, 64, 96, 128, 192, 256])]:
        g = binned(inst, col, bins)
        for e in show:
            ge = g[g.entry == e]
            if ge.empty: continue
            fam = ENTRIES[e][2]
            ax.plot(ge.x, ge.steps, marker=".", lw=1.2 if fam == "ours" else 0.8, ls="-" if fam != "reference" else ":",
                    color=FAMILY_COLORS[fam], alpha=1.0 if fam in ("ours", "reference") else 0.6, label=ENTRIES[e][3])
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("depth $d^*$" if col == "depth" else "length L"); ax.set_ylabel("steps used"); ax.set_title(t)
axes[0, 0].legend(fontsize=5.5, frameon=False, ncol=2)
WRITTEN.append(savefig(fig, "fig4_steps_vs_depth_length"))
"""),
    code("""
# Fig. 5: accuracy-versus-compute curves on OOD-depth (every threshold of every rule)
fig, axes = plt.subplots(1, len(TASKS), figsize=(7.0, 2.6), squeeze=False)
for ax, t in zip(axes[0], TASKS):
    cur = curve_rows(finished(cfgs[t]))
    cur = cur[cur.split == "ood_depth"]
    for e, (method, rule, fam, label) in ENTRIES.items():
        sub = cur[(cur.method == method) & (cur.rule == rule)]
        if sub.empty or e in ("oracle_iter",): continue
        g = sub.groupby("threshold", dropna=False)[["acc", "mean_steps"]].mean().sort_values("mean_steps")
        ax.plot(g.mean_steps, g.acc, marker="." if len(g) > 1 else "*", ms=4 if len(g) > 1 else 8,
                lw=1.2 if fam == "ours" else 0.7, color=FAMILY_COLORS[fam], alpha=0.9 if fam != "posthoc" else 0.6,
                label=label if fam in ("ours", "reference") or e in ("max_softmax", "act", "ponder") else None)
    ax.set_xlabel("mean steps"); ax.set_ylabel("accuracy"); ax.set_title(f"{t}, OOD-depth"); ax.set_xscale("log")
axes[0, 0].legend(fontsize=5.5, frameon=False)
WRITTEN.append(savefig(fig, "fig5_tradeoff"))
"""),
    code("""
# Fig. 6: premature-stop rate against depth
fig, axes = plt.subplots(1, len(TASKS), figsize=(7.0, 2.4), squeeze=False)
for ax, t in zip(axes[0], TASKS):
    inst = pd.concat([entry_instances(cfgs[t], s, show) for s in ["id_test", "ood_depth", "ood_both"]])
    inst["premature"] = (inst.steps < inst.lb).astype(float)
    g = binned(inst, "depth", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 48, 64, 96, 128], value="premature")
    for e in show:
        ge = g[g.entry == e]
        if ge.empty or e == "oracle_lb": continue
        fam = ENTRIES[e][2]
        ax.plot(ge.x, ge.premature, marker=".", color=FAMILY_COLORS[fam], lw=1.2 if fam == "ours" else 0.7, label=ENTRIES[e][3])
    ax.axvline(10, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xlabel("depth $d^*$"); ax.set_ylabel("premature-stop rate"); ax.set_title(t)
axes[0, 0].legend(fontsize=5.5, frameon=False, ncol=2)
WRITTEN.append(savefig(fig, "fig6_premature_vs_depth"))
"""),
    code("""
WRITTEN.append(write_json(RESULTS_DIR / "gates" / "g2.json", {"G2": g2, **{t: h1[t].to_dict(orient="records") for t in TASKS}}))
"""),
    code(FOOTER),
]

NOTEBOOKS["05_completion_heads"] = [
    md("""
# 05 - Local completion heads (decision gate G3)
Three variants: a pointwise head fitted post hoc on the frozen backbone, the dependency-propagating
head (reads its neighbours' previous done probabilities, like the automaton of Proposition 3), and
both trained jointly as an auxiliary task. Target: position i is done at step t if it is correct
at t and at every later step. Halting: every position reports done with probability > tau.

**G3 (pre-registered):** on OOD-depth the best completion-head variant beats the best baseline by
>= 10 accuracy points or >= 50% fewer premature stops, CI excluding zero across seeds.
"""),
    code("""
# parameters
CONFIGS = ["configs/main_addition.yaml", "configs/main_parity.yaml", "configs/main_gates.yaml"]
TASKS = ["addition", "parity"]
SPLITS = ["id_test", "ood_depth", "ood_length", "ood_both"]
"""),
    code(SETUP),
    code(MAIN_LOAD),
    code("""
g3 = {}
for t in TASKS:
    for split in ["ood_depth", "ood_both"]:
        inst = entry_instances(cfgs[t], split)
        r = g3_check(inst)
        g3[f"{t}/{split}"] = r
        print(t, split, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("acc_diff", "prem_diff")})
        if r:
            print("   acc diff (point, lo, hi):", np.round(r["acc_diff"], 3), "  premature diff:", np.round(r["prem_diff"], 3))
G3 = all(g3[f"{t}/ood_depth"].get("G3", False) for t in TASKS)
print("G3 on OOD-depth (pre-registered):", G3)
"""),
    code("""
for t in TASKS:
    keep = OURS + ["oracle_stable", "max_softmax", "entropy", "stability", "trm_post", "act", "ponder", "trm_joint"]
    sub = rows[t][rows[t].entry.isin(keep)]
    print(f"== {t}"); display(sub.pivot_table(index="entry", columns="split", values=["acc", "premature", "mean_steps"], aggfunc="mean")
                              .reindex([e for e in ENTRIES if e in keep]).round(3))
"""),
    code("""
# completion-head calibration (ECE of per-position done probabilities against done targets)
from glwm2.train.trainer import run_dirs
from glwm2.utils.io import read_json
cal = []
for t in TASKS:
    for c in finished(cfgs[t]):
        if c.method not in ("base", "done_joint", "done_prop_joint"): continue
        for split, d in read_json(run_dirs(c)[1] / "calibration.json").items():
            for head, ece in d.items():
                cal.append(dict(task=t, method=c.method, head=head, split=split, seed=c.seed, ece=ece))
cal = pd.DataFrame(cal)
display(cal.pivot_table(index=["task", "method", "head"], columns="split", values="ece", aggfunc="mean").round(4))
"""),
    code("""
# H4: does joint training distort the per-step accuracy curve? AUC of ID-val accuracy over steps 1..T_max
auc = []
for t in TASKS:
    for c in finished(cfgs[t]):
        a = step_acc(c)["id_val"]
        auc.append(dict(task=t, method=c.method, seed=c.seed, auc=float(np.mean(a))))
auc = pd.DataFrame(auc)
h4 = auc.pivot_table(index="method", columns="task", values="auc", aggfunc="mean")
h4_diff = (h4 - h4.loc["base"]) * 100
display(h4.round(4)); print("difference from base, accuracy points:"); display(h4_diff.round(2))
"""),
    code("""
# how the done signal spreads over positions and steps (deep OOD-both addition instance)
from glwm2.eval.results import per_instance
from glwm2.data.splits import load_split
from glwm2.eval.trajectory import per_position_rho
c = next(c for c in finished(cfgs["addition"]) if c.method == "base")
pi = per_instance(c)
data = load_split("addition", "ood_both")
heads = [h for h in ["done_post", "done_prop_post"] if h in pi["pos_done"]["ood_both"]]
k = int(torch.argmax(data["depth"][:64]))
L = int(data["length"][k]); rho = per_position_rho("addition", {kk: v[k:k+1] for kk, v in data.items()})[0, :L+1].numpy()
fig, axes = plt.subplots(1, len(heads), figsize=(7.0, 2.4), squeeze=False)
for ax, h in zip(axes[0], heads):
    P = pi["pos_done"]["ood_both"][h][:, k, :L+1].float().numpy()
    im = ax.imshow(P.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[0.5, P.shape[0] + 0.5, -0.5, L + 0.5])
    ax.plot(rho, np.arange(L + 1), color="#D55E00", lw=1)
    ax.set_title(f"{h}: done probability"); ax.set_xlabel("step t"); ax.set_ylabel("position i")
fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
WRITTEN.append(savefig(fig, "figA_done_spread"))
"""),
    code("""
WRITTEN.append(write_json(RESULTS_DIR / "gates" / "g3.json", {"G3": G3, "detail": g3, "h4_auc_diff_points": h4_diff.to_dict()}))
"""),
    code(FOOTER),
]


NOTEBOOKS["06_launch_main_grid"] = [
    md("""
# 06 - Launch and monitor the main grid
Long runs go through `scripts/run_sweep.py` in a terminal, never in this kernel. Each run resumes
from `checkpoints/<run_id>/last.pt` after an interruption, and finished runs are skipped.
Set `LAUNCH = True` to start a sweep in a new console window; otherwise this notebook only
reports progress and GPU-hours from the JSON logs.
"""),
    code("""
# parameters
CONFIGS = ["configs/pilot_addition.yaml", "configs/tune_gates.yaml", "configs/main_addition.yaml",
           "configs/main_parity.yaml", "configs/main_gates.yaml", "configs/ablations.yaml"]
LAUNCH = False
"""),
    code(SETUP),
    code("""
import subprocess
from glwm2.eval.results import configs_from_yaml, finished
from glwm2.train.trainer import run_dirs
from glwm2.utils.io import read_json, write_json
status = []
for f in CONFIGS:
    cs = configs_from_yaml(PROJECT_ROOT / f)
    fin = finished(cs)
    secs = 0.0
    for c in fin:
        d = run_dirs(c)[1]
        secs += read_json(d / "train_summary.json")["train_seconds"] if (d / "train_summary.json").exists() else 0
        secs += read_json(d / "eval_summary.json")["eval_seconds"] if (d / "eval_summary.json").exists() else 0
    status.append(dict(config=f, runs=len(cs), finished=len(fin), gpu_hours=round(secs / 3600, 2)))
status = pd.DataFrame(status); status
"""),
    code("""
if LAUNCH:
    todo = [f for f, r in zip(status.config, status.itertuples()) if r.finished < r.runs]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_sweep.py"), *todo]
    subprocess.Popen(cmd, cwd=PROJECT_ROOT, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    print("launched:", " ".join(cmd))
else:
    print("to launch from a terminal:")
    print(r"  .venv\\Scripts\\python.exe scripts\\run_sweep.py " + " ".join(CONFIGS))
"""),
    code("""
from glwm2.utils.io import read_jsonl
logs = []
for f in CONFIGS:
    for r in read_jsonl(RESULTS_DIR / "sweeps" / f"{Path(f).stem}.jsonl"):
        logs.append(dict(config=Path(f).stem, run_id=r["run_id"], status=r["status"], minutes=round(r["sec"] / 60, 1)))
logs = pd.DataFrame(logs)
print(logs.status.value_counts() if len(logs) else "no sweep logs yet")
logs.tail(10)
"""),
    code("""
WRITTEN.append(write_json(RESULTS_DIR / "grid_status.json", {"status": status.to_dict(orient="records"),
                                                             "total_gpu_hours": float(status.gpu_hours.sum())}))
print("total GPU-hours so far:", round(float(status.gpu_hours.sum()), 2))
"""),
    code(FOOTER),
]

NOTEBOOKS["07_mazes"] = [
    md("""
# 07 - Mazes (T3)
2D backbone (r = 4 kernel-3 convolutions per step), trained on 9x9 mazes, tested on 13x13 to
33x33. Depth proxies: dead-end-filling depth (used for the lower-bound column, so "premature" here
means halting before a local dead-end filler could finish) and optimal path length. There is no
tight lower bound for mazes, so this task is empirical only.
"""),
    code("""
# parameters
CONFIGS = ["configs/main_mazes.yaml"]
TASKS = ["mazes"]
SPLITS = ["id_test", "ood_13", "ood_17", "ood_21", "ood_25", "ood_33"]
"""),
    code(SETUP),
    code(MAIN_LOAD),
    code("""
r = rows["mazes"]
if len(r):
    display(summary_table(r, SPLITS, "acc")["mean"].round(3))
    display(summary_table(r, SPLITS, "mean_steps")["mean"].round(1))
    display(summary_table(r, SPLITS, "premature")["mean"].round(3))
"""),
    code("""
show = ["oracle_stable", "fixed_T", "max_softmax", "stability", "kl", "act", "ponder", "trm_joint", "done_post", "done_prop_post", "done_joint"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
if len(r):
    sizes = {"id_test": 9, "ood_13": 13, "ood_17": 17, "ood_21": 21, "ood_25": 25, "ood_33": 33}
    for e in show:
        sub = r[(r.entry == e) & r.split.isin(SPLITS)].groupby("split")[["acc", "mean_steps"]].mean()
        if sub.empty: continue
        xs = [sizes[s] for s in sub.index]
        fam = ENTRIES[e][2]
        kw = dict(marker=".", color=FAMILY_COLORS[fam], lw=1.2 if fam == "ours" else 0.7, label=ENTRIES[e][3])
        axes[0].plot(xs, sub.acc.values, **kw); axes[1].plot(xs, sub.mean_steps.values, **kw)
axes[0].set_xlabel("maze size n"); axes[0].set_ylabel("accuracy"); axes[1].set_xlabel("maze size n"); axes[1].set_ylabel("mean steps")
axes[1].set_yscale("log"); axes[0].legend(fontsize=5.5, frameon=False, ncol=2)
WRITTEN.append(savefig(fig, "fig7_mazes"))
"""),
    code("""
if len(r):
    inst = pd.concat([entry_instances(cfgs["mazes"], s, show) for s in SPLITS])
    print("Spearman(steps, dead-end depth) per entry:")
    print(inst.groupby("entry").apply(lambda d: d[["steps", "depth"]].corr(method="spearman").iloc[0, 1], include_groups=False).round(3).to_string())
    WRITTEN.append(write_json(RESULTS_DIR / "mazes_summary.json", r.groupby(["entry", "split"])[["acc", "mean_steps", "premature"]].mean().reset_index().to_dict(orient="records")))
"""),
    code(FOOTER),
]

NOTEBOOKS["08_ablations_and_probes"] = [
    md("""
# 08 - Ablations, probes and measured compute
1. Ablations on addition (`configs/ablations.yaml`, seed 0): no recall, r = 2, width 64.
2. Linear probes: is "position i is still pending" (t < rho_i) linearly decodable from h_i(t), and
   does a probe fitted on ID transfer to OOD-depth and OOD-both?
3. Measured FLOPs (`torch.utils.flop_counter.FlopCounterMode`) and CUDA-event latency for every
   stopping rule, using online early exit (halted instances leave the batch).
"""),
    code("""
# parameters
ABL_CONFIG = "configs/ablations.yaml"
MAIN_CONFIG = "configs/main_addition.yaml"
GATES_CONFIG = "configs/main_gates.yaml"
PROBE_ROWS = 150_000
LATENCY_N = 512
"""),
    code(SETUP),
    code("""
from glwm2.eval.results import configs_from_yaml, finished, metric_rows
from glwm2.eval.analysis import ENTRIES, entry_rows
from glwm2.eval.plotting import savefig
from glwm2.utils.io import write_json
abl = configs_from_yaml(PROJECT_ROOT / ABL_CONFIG)
main = [c for c in configs_from_yaml(PROJECT_ROOT / MAIN_CONFIG)]
print(f"ablations: {len(finished(abl))}/{len(abl)} finished")
ref = [c for c in finished(main) if c.method in ("base", "done_joint") and c.seed in (0, 1)]
for c in ref:
    c.tags = {"ablation": "main"}
A = metric_rows(finished(abl) + ref)
if len(A):
    tag = {c.run_id(): c.tags.get("ablation") for c in finished(abl) + ref}
    A["ablation"] = A.run_id.map(tag)
    rule_of = {"base": ["oracle_stable", "max_softmax", "done_prop_post"], "done_joint": ["done_joint"]}
    A = A[[r in rule_of.get(m, []) for m, r in zip(A.method, A.rule)]]
    abl_tab = A.pivot_table(index=["ablation", "method", "rule"], columns="split", values="acc", aggfunc="mean")[["id_test", "ood_depth", "ood_length", "ood_both"]]
    display(abl_tab.round(3))
"""),
    code("""
# linear probes for the pending state (seed-0 base model)
from glwm2.train.trainer import load_model
from glwm2.data.splits import load_split
from glwm2.eval.probes import collect_probe_data, fit_linear_probe, probe_scores
c0 = next(c for c in finished(main) if c.method == "base" and c.seed == 0)
model = load_model(c0)
probe_res = {}
tr = collect_probe_data(model, "addition", load_split("addition", "id_val"), t_max=26, max_rows=PROBE_ROWS)
probe = fit_linear_probe(tr["h"], tr["pending"])
for split, T in [("id_test", 26), ("ood_depth", 36), ("ood_both", 144)]:
    te = collect_probe_data(model, "addition", load_split("addition", split), t_max=T, max_rows=PROBE_ROWS, seed=1)
    probe_res[split] = probe_scores(probe, te["h"], te["pending"])
pd.DataFrame(probe_res).T.round(3)
"""),
    code("""
# measured FLOPs per instance for every rule: encode + steps x (step + decode + head) at the mean length of each split
from glwm2.eval.flops import backbone_flops, module_flops
from glwm2.train.pipeline import get_posthoc_heads
heads = get_posthoc_heads(model, c0, "cuda")
rows_main = entry_rows(finished(main) + [c for c in finished(configs_from_yaml(PROJECT_ROOT / GATES_CONFIG)) if c.task == "addition"])
flop_rows = []
cpu_model = load_model(c0, "cpu")
for split in ["id_test", "ood_depth", "ood_length", "ood_both"]:
    d = load_split("addition", split)
    P = int(d["length"].float().mean().round()) + 1
    x = torch.randint(0, 100, (1, P))
    f = backbone_flops(cpu_model.backbone, x)
    h = cpu_model.backbone.init_state(cpu_model.backbone.encode(x))
    head_f = {"done_post": module_flops(heads["done_post"].cpu(), h), "done_prop_post": module_flops(heads["done_prop_post"].cpu(), h),
              "trm_post": module_flops(heads["trm_post"].cpu(), h, None)}
    for e, (method, rule, fam, label) in ENTRIES.items():
        sub = rows_main[(rows_main.entry == e) & (rows_main.split == split)]
        if sub.empty: continue
        steps = sub.mean_steps.mean()
        per_step = f["step"] + f["decode"] + head_f.get(e, 0)
        flop_rows.append(dict(split=split, entry=e, family=fam, mean_steps=steps, kflops=(f["encode"] + steps * per_step) / 1e3,
                              acc=sub.acc.mean()))
for hh in heads.values():
    hh.cuda()
flops_df = pd.DataFrame(flop_rows)
display(flops_df.pivot_table(index="entry", columns="split", values="kflops").reindex([e for e in ENTRIES if e in set(flops_df.entry)]).round(1))
"""),
    code("""
# CUDA-event latency with online early exit on OOD-depth (batch of LATENCY_N instances)
from glwm2.eval.latency import cuda_time_ms
from glwm2.readouts.online import run_online
from glwm2.data.splits import load_split
d = load_split("addition", "ood_depth", "cuda")
x = d["x"][:LATENCY_N]
thr = {r.rule: r.threshold for r in metric_rows([c0]).itertuples() if r.split == "id_val"}
T_MAX = int(d["depth"].max()) + 16
def conf_rule(tau):
    def f(o):
        p = torch.softmax(o["logits"].float(), -1).max(-1).values
        return torch.where(o["mask"], p, torch.ones_like(p)).min(1).values > tau
    return f
lat = {
    "fixed_T (val-selected)": cuda_time_ms(lambda: run_online(model, x, int(thr["fixed_T"]), None)),
    "fixed_T = T_max": cuda_time_ms(lambda: run_online(model, x, T_MAX, None)),
    "max_softmax": cuda_time_ms(lambda: run_online(model, x, T_MAX, conf_rule(float(thr["max_softmax"])))),
    "done_post": cuda_time_ms(lambda: run_online(model, x, T_MAX, None, head=heads["done_post"], head_thr=float(thr["done_post"]))),
    "done_prop_post": cuda_time_ms(lambda: run_online(model, x, T_MAX, None, head=heads["done_prop_post"], head_thr=float(thr["done_prop_post"]))),
}
lat_df = pd.DataFrame(lat).T; lat_df.round(2)
"""),
    code("""
fig, ax = plt.subplots(figsize=(3.4, 2.4))
sub = flops_df[flops_df.split == "ood_depth"]
from glwm2.eval.plotting import FAMILY_COLORS
for r_ in sub.itertuples():
    ax.scatter(r_.kflops, r_.acc, color=FAMILY_COLORS[r_.family], s=14)
    ax.annotate(r_.entry, (r_.kflops, r_.acc), fontsize=4.5)
ax.set_xscale("log"); ax.set_xlabel("measured kFLOPs per instance"); ax.set_ylabel("accuracy (OOD-depth)")
WRITTEN.append(savefig(fig, "fig8_compute"))

# LaTeX: probes, ablations (OOD accuracy of the stable oracle and max-softmax) and latency
lines = [r"\\begin{table}[h]", r"\\centering",
         r"\\caption{Left: linear probe for the pending state (fitted on ID-val, seed-0 base model on addition). "
         r"Middle: ablations on addition (seed 0; accuracy in \\%). Right: CUDA-event latency for 512 OOD-depth "
         r"instances with online early exit (median ms).}",
         r"\\label{tab:probes}", r"\\small",
         r"\\begin{tabular}{lrr}", r"\\toprule", r"Split & Acc. & AUC \\\\", r"\\midrule"]
for s, v in probe_res.items():
    lines.append(f"{s.replace('_', '-')} & {100 * v['acc']:.1f} & {v['auc']:.3f} \\\\\\\\")
lines += [r"\\bottomrule", r"\\end{tabular}\\quad"]
if len(A):
    t = abl_tab.reset_index()
    t = t[t.rule.isin(["oracle_stable", "max_softmax"])]
    lines += [r"\\begin{tabular}{llrr}", r"\\toprule", r"Ablation & Rule & OOD-depth & OOD-both \\\\", r"\\midrule"]
    for r_ in t.itertuples():
        lines.append(f"{str(r_.ablation).replace('_', ' ')} & {r_.rule.replace('_', ' ')} & {100 * r_.ood_depth:.1f} & {100 * r_.ood_both:.1f} \\\\\\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}\\quad"]
lines += [r"\\begin{tabular}{lr}", r"\\toprule", r"Rule & ms \\\\", r"\\midrule"]
for k_, v in lat.items():
    lines.append(f"{k_.replace('_', ' ')} & {v['median_ms']:.1f} \\\\\\\\")
lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}"]
p = PROJECT_ROOT / "paper" / "tables" / "probes_ablations.tex"
p.write_text("\\n".join(lines) + "\\n", encoding="utf-8"); WRITTEN.append(p)

WRITTEN.append(write_json(RESULTS_DIR / "compute_probes_ablations.json", {
    "probes": probe_res, "flops": flops_df.to_dict(orient="records"), "latency_ms": lat_df.reset_index().to_dict(orient="records"),
    "ablations": (abl_tab.reset_index().to_dict(orient="records") if len(A) else [])}))
"""),
    code(FOOTER),
]

NOTEBOOKS["09_paper_figures_tables"] = [
    md("""
# 09 - Statistics, figures and tables for the paper
Per-instance paired bootstrap 95% CIs, Wilcoxon signed-rank tests across seeds with Holm
correction, the teaser figure, and LaTeX tables written to `paper/tables/`. Every number in the
paper is produced here or in notebooks 01-08; `paper/claims_map.md` lists which.
"""),
    code("""
# parameters
CONFIGS = ["configs/main_addition.yaml", "configs/main_parity.yaml", "configs/main_gates.yaml"]
TASKS = ["addition", "parity"]
SPLITS = ["id_test", "ood_depth", "ood_length", "ood_both"]
MAIN_ENTRIES = ["fixed_T", "oracle_stable", "max_softmax", "entropy", "stability", "kl", "hidden_change", "max_conf_fan",
                "trm_post", "act", "ponder", "trm_joint", "v1gate", "done_post", "done_prop_post", "done_joint", "done_prop_joint"]
"""),
    code(SETUP),
    code(MAIN_LOAD),
    code("""
from glwm2.eval.stats import bootstrap_ci, wilcoxon_seeds, holm
def fmt(m, lo, hi, pct=True):
    s = 100 if pct else 1
    return f"{m*s:.1f} \\\\tiny{{[{lo*s:.1f}, {hi*s:.1f}]}}"
tables = {}
for t in TASKS:
    r = rows[t]
    lines = [r"\\begin{tabular}{l" + "rr" * len(SPLITS) + "}", r"\\toprule",
             " & " + " & ".join([rf"\\multicolumn{{2}}{{c}}{{{s.replace('_', '-')}}}" for s in SPLITS]) + r" \\\\",
             " & " + " & ".join(["Acc. (\\\\%)", "Steps"] * len(SPLITS)) + r" \\\\", r"\\midrule"]
    fam_prev = None
    for e in MAIN_ENTRIES:
        if e not in set(r.entry): continue
        fam = ENTRIES[e][2]
        if fam_prev is not None and fam != fam_prev: lines.append(r"\\midrule")
        fam_prev = fam
        cells = []
        for s in SPLITS:
            v = r[(r.entry == e) & (r.split == s)]
            if v.empty: cells += ["--", "--"]; continue
            m, lo, hi = bootstrap_ci(v.acc.values, n_boot=2000)
            cells += [fmt(m, lo, hi), f"{v.mean_steps.mean():.1f}"]
        lines.append(ENTRIES[e][3] + " & " + " & ".join(cells) + r" \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}"]
    tables[t] = "\\n".join(lines)
    p = PAPER_DIR / "tables" / f"main_{t}.tex"; p.write_text(tables[t]); WRITTEN.append(p)
print(tables["addition"][:1500])
"""),
    code("""
# paired tests: each completion-head variant vs each baseline, OOD-depth accuracy across seeds, Holm-corrected
tests = {}
for t in TASKS:
    r = rows[t][rows[t].split == "ood_depth"]
    pv = {}
    for o in [e for e in ["done_post", "done_prop_post", "done_joint", "done_prop_joint"] if e in set(r.entry)]:
        for b in [e for e in MAIN_ENTRIES if ENTRIES[e][2] in ("posthoc", "joint") and e in set(r.entry)]:
            A = r[r.entry == o].set_index("seed").acc
            B = r[r.entry == b].set_index("seed").acc
            common = A.index.intersection(B.index)
            if len(common) >= 2:
                pv[f"{o} vs {b}"] = wilcoxon_seeds(A[common].values, B[common].values)
    tests[t] = holm(pv)
    df_t = pd.DataFrame(tests[t]).T.sort_values("p")
    print(t, f"{int(df_t.reject.sum())}/{len(df_t)} comparisons significant after Holm (alpha 0.05)")
    display(df_t.head(12))
"""),
    code("""
# premature-stop table (OOD-depth and OOD-both), with per-instance bootstrap CIs pooled over seeds
lines = [r"\\begin{tabular}{l" + "rr" * len(TASKS) + "}", r"\\toprule",
         " & " + " & ".join([rf"\\multicolumn{{2}}{{c}}{{{t}}}" for t in TASKS]) + r" \\\\",
         " & " + " & ".join(["OOD-depth", "OOD-both"] * len(TASKS)) + r" \\\\", r"\\midrule"]
inst = {(t, s): entry_instances(cfgs[t], s) for t in TASKS for s in ["ood_depth", "ood_both"]}
for e in MAIN_ENTRIES:
    if e in ("oracle_stable",): continue
    cells = []
    for t in TASKS:
        for s in ["ood_depth", "ood_both"]:
            d = inst[(t, s)]
            if e not in set(d.entry): cells.append("--"); continue
            m, lo, hi = premature_ci(d, e)
            cells.append(fmt(m, lo, hi))
    lines.append(ENTRIES[e][3] + " & " + " & ".join(cells) + r" \\\\")
lines += [r"\\bottomrule", r"\\end{tabular}"]
p = PAPER_DIR / "tables" / "premature.tex"; p.write_text("\\n".join(lines)); WRITTEN.append(p)
"""),
    code("""
# Fig. 1 (teaser): one deep and long addition instance; where two global convergence rules halt
# (hidden-state change: wrong; k-step stability: before the bound) vs the propagating completion head
from glwm2.train.trainer import load_model
from glwm2.data.splits import load_split
from glwm2.eval.trajectory import per_position_rho
from glwm2.eval.results import per_instance
c = next(c for c in finished(cfgs["addition"]) if c.method == "base" and c.seed == 0)
pi = per_instance(c)["per_instance"]["ood_both"]
data = load_split("addition", "ood_both")
short = data["length"].numpy() <= 96
cand = np.where((pi["hidden_change"]["correct"] == 0) & (pi["done_prop_post"]["correct"] == 1) & short)[0]
k = int(cand[np.argmax(data["depth"].numpy()[cand])]) if len(cand) else int(torch.argmax(data["depth"]))
model = load_model(c)
L = int(data["length"][k]); rho = per_position_rho("addition", {kk: v[k:k+1] for kk, v in data.items()})[0, :L+1].numpy()
T = int(data["depth"][k]) + 8
x, y = data["x"][k:k+1].cuda(), data["y"][k:k+1].cuda()
with torch.no_grad():
    pc = torch.stack([(o["logits"].argmax(-1) == y)[0, :L+1] for o in model.rollout(x, T)]).cpu().numpy()
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.imshow(pc.T, aspect="auto", origin="lower", cmap="Greys_r", interpolation="nearest", extent=[0.5, T + 0.5, -0.5, L + 0.5], alpha=0.85)
ax.plot(rho, np.arange(L + 1), color="#D55E00", lw=1.2, label=r"carry arrives ($\\rho_i$)")
ax.axvline(pi["hidden_change"]["steps"][k], color="#CC79A7", lw=1.5, ls="--", label="hidden-change halts (wrong)")
ax.axvline(pi["stability"]["steps"][k], color="#56B4E9", lw=1.5, ls=":", label="stability halts")
ax.axvline(pi["done_prop_post"]["steps"][k], color="#009E73", lw=1.5, label="completion head halts")
ax.set_xlabel("step t"); ax.set_ylabel("output digit i"); ax.legend(fontsize=5.5, frameon=True, loc="lower right")
ax.set_title(f"{L}-digit addition, $d^*$ = {int(data['depth'][k])} (white = correct)", fontsize=7)
WRITTEN.append(savefig(fig, "fig1_teaser"))
"""),
    code("""
from glwm2.train.trainer import run_dirs
from glwm2.utils.io import read_json
comp = []
for f in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
    if f.stem.startswith("_"): continue
    cs = finished(configs_from_yaml(f))
    tr = sum(read_json(run_dirs(c)[1] / "train_summary.json").get("train_seconds", 0) for c in cs if (run_dirs(c)[1] / "train_summary.json").exists())
    ev = sum(read_json(run_dirs(c)[1] / "eval_summary.json").get("eval_seconds", 0) for c in cs if (run_dirs(c)[1] / "eval_summary.json").exists())
    if cs: comp.append(dict(sweep=f.stem, runs=len(cs), train_h=tr / 3600, eval_h=ev / 3600))
comp = pd.DataFrame(comp)
lines = [r"\\begin{table}[h]", r"\\centering", r"\\caption{Measured GPU time per sweep (RTX 5070; several runs shared the GPU, so wall-clock per run is higher than in isolation). Runs listed in more than one sweep are counted once per sweep.}",
         r"\\label{tab:compute}", r"\\small", r"\\begin{tabular}{lrrr}", r"\\toprule", r"Sweep & Runs & Train (h) & Eval (h) \\\\", r"\\midrule"]
for r_ in comp.itertuples():
    lines.append(f"\\\\texttt{{{r_.sweep.replace('_', '\\\\_')}}} & {r_.runs} & {r_.train_h:.1f} & {r_.eval_h:.1f} \\\\\\\\")
lines += [r"\\midrule", f"Total & {int(comp.runs.sum())} & {comp.train_h.sum():.1f} & {comp.eval_h.sum():.1f} \\\\\\\\", r"\\bottomrule", r"\\end{tabular}", r"\\end{table}"]
p = PAPER_DIR / "tables" / "compute.tex"; p.write_text("\\n".join(lines)); WRITTEN.append(p)
display(comp.round(2))
summary = {t: rows[t].groupby(["entry", "split"])[["acc", "mean_steps", "premature", "rho_depth", "rho_length_given_depth"]]
           .mean().reset_index().to_dict(orient="records") for t in TASKS}
WRITTEN.append(write_json(RESULTS_DIR / "paper_numbers.json", {"summary": summary,
               "holm": {t: tests[t] for t in TASKS}}))
"""),
    code(FOOTER),
]


def build(names=None):
    NB_DIR.mkdir(exist_ok=True)
    written = []
    for name, cells in NOTEBOOKS.items():
        if names and not any(name.startswith(n) for n in names):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "glwm2", "display_name": "GLWM-2 (py3.12, cu128)", "language": "python"}
        path = NB_DIR / f"{name}.ipynb"
        nbf.write(nb, path)
        written.append(path)
    return written


def run(paths, timeout=None):
    from nbclient import NotebookClient

    for path in paths:
        nb = nbf.read(path, as_version=4)
        t0 = time.time()
        client = NotebookClient(nb, timeout=timeout, kernel_name="glwm2", resources={"metadata": {"path": str(NB_DIR)}})
        client.execute()
        nbf.write(nb, path)
        print(f"executed {path.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=None, help="notebook prefixes to execute after building")
    ap.add_argument("--only", nargs="*", default=None, help="only (re)write these notebook prefixes")
    args = ap.parse_args()
    paths = build(args.only or args.run)
    print("wrote", [p.name for p in paths])
    if args.run is not None:
        run(paths)
        sys.exit(0)
