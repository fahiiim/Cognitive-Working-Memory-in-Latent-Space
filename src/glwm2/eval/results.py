"""Load run outputs from results/runs/<run_id>/ into DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..train.trainer import run_dirs
from ..utils.config import RunConfig, expand_grid, load_yaml
from ..utils.io import read_json


def configs_from_yaml(path: str | Path) -> list[RunConfig]:
    return expand_grid(load_yaml(path))


def finished(cfgs: list[RunConfig]) -> list[RunConfig]:
    return [c for c in cfgs if (run_dirs(c)[1] / "metrics.json").exists()]


def metric_rows(cfgs: list[RunConfig]) -> pd.DataFrame:
    rows = []
    for c in finished(cfgs):
        rows.extend(read_json(run_dirs(c)[1] / "metrics.json"))
    return pd.DataFrame(rows)


def curve_rows(cfgs: list[RunConfig]) -> pd.DataFrame:
    rows = []
    for c in finished(cfgs):
        for r in read_json(run_dirs(c)[1] / "curves.json"):
            rows.append({"run_id": c.run_id(), "task": c.task, "method": c.method, "objective": c.objective, "seed": c.seed, **r})
    return pd.DataFrame(rows)


def step_acc(cfg: RunConfig) -> dict:
    return read_json(run_dirs(cfg)[1] / "step_acc.json")


def per_instance(cfg: RunConfig) -> dict:
    return torch.load(run_dirs(cfg)[1] / "per_instance.pt", weights_only=False)


def instance_frame(cfg: RunConfig, split: str, rules: list[str] | None = None) -> pd.DataFrame:
    """One row per (instance, rule): depth, length, lb, steps, correct."""
    pi = per_instance(cfg)
    meta = pi["metas"][split]
    out = []
    for rule, v in pi["per_instance"][split].items():
        if rules and rule not in rules:
            continue
        out.append(pd.DataFrame({"rule": rule, "idx": np.arange(len(v["steps"])), "depth": meta["depth"].numpy(),
                                 "length": meta["length"].numpy(), "lb": meta["lb"].numpy(),
                                 "steps": v["steps"], "correct": v["correct"]}))
    df = pd.concat(out, ignore_index=True)
    df["run_id"], df["seed"], df["method"], df["objective"], df["task"] = cfg.run_id(), cfg.seed, cfg.method, cfg.objective, cfg.task
    return df
