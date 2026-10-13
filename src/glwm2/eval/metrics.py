"""Metrics computed from per-instance halting steps and correctness."""

from __future__ import annotations

import numpy as np
import torch

from .stats import partial_spearman, spearman


def ece_bins(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15):
    """Returns per-bin [sum of probs, sum of targets] (2 x n_bins) and counts, for later aggregation."""
    idx = torch.clamp((probs * n_bins).long(), max=n_bins - 1)
    sums = torch.zeros(2, n_bins, dtype=torch.float64, device=probs.device)
    sums[0].index_add_(0, idx, probs.double())
    sums[1].index_add_(0, idx, targets.double())
    counts = torch.bincount(idx, minlength=n_bins).double()
    return sums, counts


def ece_from_bins(sums: torch.Tensor, counts: torch.Tensor) -> float:
    n = counts.sum()
    if n == 0:
        return float("nan")
    nz = counts > 0
    return float((counts[nz] / n * (sums[0][nz] / counts[nz] - sums[1][nz] / counts[nz]).abs()).sum())


def first_stable_correct(em: torch.Tensor) -> torch.Tensor:
    """em [T, B] bool -> first step (1-indexed) from which the answer is right until T_max (0 if never)."""
    stable = em.flip(0).int().cummin(0).values.flip(0).bool()
    any_ = stable.any(0)
    first = stable.float().argmax(0) + 1
    return torch.where(any_, first, torch.zeros_like(first))


def halting_metrics(steps: np.ndarray, correct: np.ndarray, lb: np.ndarray, depth: np.ndarray, length: np.ndarray,
                    fsc: np.ndarray, oracle_steps: np.ndarray, oracle_correct: np.ndarray) -> dict:
    steps = steps.astype(float)
    over = (steps - fsc)[(correct > 0) & (fsc > 0)]
    both = oracle_correct > 0
    return {
        "acc": float(correct.mean()),
        "mean_steps": float(steps.mean()),
        "premature": float((steps < lb).mean()),
        "overthink": float(over.mean()) if over.size else float("nan"),
        "rho_depth": spearman(steps, depth),
        "rho_length_given_depth": partial_spearman(steps, length, depth),
        "regret_acc": float(oracle_correct.mean() - correct.mean()),
        "regret_steps": float((steps[both] - oracle_steps[both]).mean()) if both.any() else float("nan"),
        "mean_lb": float(lb.mean()),
    }
