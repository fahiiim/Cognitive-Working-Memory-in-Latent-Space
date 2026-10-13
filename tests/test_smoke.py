"""End-to-end CPU smoke test (run in CI): train a tiny model, fit post-hoc heads, roll out on two
small splits, evaluate every stopping rule, and run online early exit."""

import math

import torch

from glwm2.data.addition import sample_addition
from glwm2.eval.trajectory import collect_trajectory
from glwm2.readouts.online import run_online
from glwm2.readouts.rules import evaluate_rules, tradeoff_curves
from glwm2.train.pipeline import split_meta
from glwm2.train.posthoc import fit_posthoc_heads
from glwm2.train.trainer import train
from glwm2.utils.config import RunConfig


def test_end_to_end_cpu():
    cfg = RunConfig(task="addition", method="done_prop_joint", iters=20, batch_size=32, width=16, t_train=6, warmup=2)
    model = train(cfg, save=False, deterministic=True, device="cpu")["model"].eval()
    heads, _ = fit_posthoc_heads(model, cfg, iters=5, batch_size=16, device="cpu")
    g = torch.Generator().manual_seed(0)
    splits = {"id_val": sample_addition(64, (4, 10), (1, 5), g), "ood_depth": sample_addition(64, (8, 12), (6, 10), g)}
    trajs, metas = {}, {}
    for s, d in splits.items():
        trajs[s] = collect_trajectory(model, d, t_max=12, posthoc=heads, task="addition")
        metas[s] = split_meta("addition", d, cfg.convs_per_step)
    rows, per_inst = evaluate_rules(trajs, metas)
    rules = {r["rule"] for r in rows}
    assert {"fixed_T", "oracle_stable", "max_softmax", "done_post", "done_prop_post", "done_joint"} <= rules
    for r in rows:
        assert 0.0 <= r["acc"] <= 1.0 and 1 <= r["mean_steps"] <= 12 and not math.isnan(r["premature"])
    assert len(tradeoff_curves(trajs, metas)) > 0
    out = run_online(model, splits["ood_depth"]["x"], 12, None, head=heads["done_prop_post"], head_thr=0.5)
    assert out["steps"].min() >= 1 and out["steps"].max() <= 12
