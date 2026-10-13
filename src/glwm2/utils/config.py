"""Run configuration. Every run is fully described by one RunConfig, stored next to its results."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    task: str = "addition"  # addition | parity | mazes
    method: str = "base"  # base | act | ponder | trm_joint | v1gate | done_joint | done_prop_joint
    objective: str = "fixed_prior"  # fixed_prior | progressive
    seed: int = 0

    # backbone
    width: int = 128
    convs_per_step: int = 1  # receptive-field growth per step r = convs_per_step (kernel 3)
    recall: bool = True
    norm: str = "layer"  # layer | none

    # training
    iters: int = 8000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 300
    grad_clip: float = 1.0
    t_train: int = 20  # recurrent steps unrolled during training
    progressive_alpha: float = 0.5  # weight of the progressive term (Bansal et al. 2022)
    bf16: bool = False  # this model is launch-bound; fp32 with TF32 measured ~25% faster than bf16 autocast

    # halting-unit hyperparameters (joint methods)
    act_tau: float = 1e-3  # ACT ponder cost
    ponder_beta: float = 0.01  # PonderNet KL weight
    ponder_lambda_p: float = 0.1  # PonderNet geometric prior
    v1_lambda: float = 0.01  # v1 GLWM expected-steps penalty
    aux_weight: float = 1.0  # completion / correctness head loss weight

    # data (ranges are inclusive)
    train_len: tuple[int, int] = (4, 20)
    train_depth: tuple[int, int] = (1, 10)
    maze_train_size: int = 9

    # evaluation
    eval_n: int = 2000
    t_max_extra: int = 16  # T_max = max lower bound in the split + t_max_extra

    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["train_len"] = list(self.train_len)
        d["train_depth"] = list(self.train_depth)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        for k in ("train_len", "train_depth"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def run_id(self) -> str:
        d = self.to_dict()
        d.pop("tags", None)
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.task}-{self.method}-{self.objective}-s{self.seed}-{h}"

    def replace(self, **kw) -> "RunConfig":
        return dataclasses.replace(copy.deepcopy(self), **kw)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def expand_grid(spec: dict) -> list[RunConfig]:
    """YAML spec: {base: {...}, grid: {key: [values...]}, runs: [{...}, ...] (optional)}.

    The grid is the Cartesian product of the listed values applied on top of `base`.
    Entries of `runs` are applied on top of `base` and then crossed with the grid.
    """
    base = spec.get("base", {})
    grid = spec.get("grid", {})
    runs = spec.get("runs", [{}])
    keys = list(grid)
    out = []
    for run in runs:
        for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            d = {**base, **run, **dict(zip(keys, values))}
            out.append(RunConfig.from_dict(d))
    return out
