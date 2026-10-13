"""Per-step losses and trajectory-derived targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.addition import IGNORE


def position_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logits [T, B, *S, C], y [B, *S] -> CE [T, B, *S] (0 where y == IGNORE)."""
    T = logits.shape[0]
    C = logits.shape[-1]
    yy = y.unsqueeze(0).expand(T, *y.shape)
    ce = F.cross_entropy(logits.reshape(-1, C).float(), yy.reshape(-1), ignore_index=IGNORE, reduction="none")
    return ce.view(yy.shape)


def masked_instance_mean(v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """v [T, B, *S], valid [B, *S] -> [T, B] mean over valid positions."""
    m = valid.unsqueeze(0).to(v.dtype)
    return (v * m).flatten(2).sum(-1) / m.flatten(2).sum(-1).clamp_min(1.0)


def position_correct(logits_or_preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[T, B, *S] bool; invalid positions count as correct."""
    preds = logits_or_preds.argmax(-1) if logits_or_preds.is_floating_point() else logits_or_preds
    valid = y != IGNORE
    return (preds == y.unsqueeze(0)) | ~valid.unsqueeze(0)


def exact_match(pos_correct: torch.Tensor) -> torch.Tensor:
    return pos_correct.flatten(2).all(-1)  # [T, B]


def done_targets(pos_correct: torch.Tensor) -> torch.Tensor:
    """Position i is done at step t if it is correct at t and at every later step of the rollout."""
    return pos_correct.flip(0).int().cummin(0).values.flip(0).bool()


def fixed_prior_loss(ce_inst: torch.Tensor) -> torch.Tensor:
    """Uniform prior over steps 1..T (Popescu et al. 2026 'fixed-prior depth supervision')."""
    return ce_inst.mean()
