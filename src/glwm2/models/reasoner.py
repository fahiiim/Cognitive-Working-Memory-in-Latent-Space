"""Backbone plus the method-specific heads that are trained jointly with it."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.splits import TASK_META
from ..utils.config import RunConfig
from .backbone import RecurrentBackbone
from .heads import CompletionHead, PooledHead, PropCompletionHead

POOLED_METHODS = {"act", "ponder", "v1gate", "trm_joint"}
DONE_METHODS = {"done_joint", "done_prop_joint"}
METHODS = {"base"} | POOLED_METHODS | DONE_METHODS


class Reasoner(nn.Module):
    def __init__(self, cfg: RunConfig):
        super().__init__()
        if cfg.method not in METHODS:
            raise ValueError(f"unknown method {cfg.method}")
        meta = TASK_META[cfg.task]
        self.task, self.method = cfg.task, cfg.method
        self.dim = 2 if cfg.task == "mazes" else 1
        self.pad = meta.get("pad")
        self.backbone = RecurrentBackbone(
            n_out=meta["n_out"],
            width=cfg.width,
            vocab=meta.get("vocab"),
            in_channels=meta.get("in_channels"),
            dim=self.dim,
            convs_per_step=cfg.convs_per_step,
            recall=cfg.recall,
            norm=cfg.norm,
        )
        self.r = cfg.convs_per_step
        self.gate = None
        self.done = None
        if cfg.method == "v1gate":
            self.gate = PooledHead(cfg.width, act="relu", out_bias=-2.0)
        elif cfg.method in POOLED_METHODS:
            self.gate = PooledHead(cfg.width)
        elif cfg.method == "done_joint":
            self.done = CompletionHead(cfg.width, dim=self.dim)
        elif cfg.method == "done_prop_joint":
            self.done = PropCompletionHead(cfg.width, dim=self.dim)

    def valid_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.dim == 2:
            return None
        return x != self.pad

    def initial(self, x: torch.Tensor):
        e = self.backbone.encode(x)
        h0 = self.backbone.init_state(e)
        return e, h0, self.backbone.decode(h0)

    def rollout(self, x: torch.Tensor, T: int, mask: torch.Tensor | None = None, h0=None, e=None, start: int = 0,
                done_prev: torch.Tensor | None = None):
        """Yield dicts with t, h, logits and (if present) gate logit [B] / done logits [B, *S]."""
        if e is None:
            e = self.backbone.encode(x)
        h = self.backbone.init_state(e) if h0 is None else h0
        for t in range(start + 1, start + T + 1):
            h = self.backbone.step(h, e)
            out = {"t": t, "h": h, "logits": self.backbone.decode(h)}
            if self.gate is not None:
                out["gate"] = self.gate(h, mask)
            if self.done is not None:
                out["done"] = self.done(h, done_prev)
                done_prev = torch.sigmoid(out["done"].float())
            yield out
