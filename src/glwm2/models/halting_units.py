"""Jointly trained halting units: ACT (Graves 2016), PonderNet (Banino et al. 2021), v1 GLWM gate.

All functions take per-step halting logits g [T, B] and per-step, per-instance task losses
ce [T, B] (or per-step log-likelihoods for ACT), and return a scalar loss. The matching inference
statistics (cumulative halting mass, gate probability) are computed by `halting_stats`.
"""

from __future__ import annotations

import math

import torch


def act_loss(g: torch.Tensor, logp_y: torch.Tensor, tau: float, eps: float = 0.01):
    """ACT. logp_y [T, B, P]: log p_t(y_i) per position (invalid positions = 0).

    Halting probabilities lam_t accumulate until they reach 1 - eps at step N; the remainder
    R = 1 - sum_{t<N} lam_t is assigned to step N, and the output is the mixture of the per-step
    distributions with these weights. Loss = mixture NLL + tau * (N + R).
    """
    T = g.shape[0]
    lam = torch.sigmoid(g.float())
    lam = torch.cat([lam[:-1], torch.ones_like(lam[-1:])])  # force halting by the last step
    cum = lam.cumsum(0)
    cum_prev = cum - lam
    before = cum_prev >= 1 - eps
    at_n = (cum >= 1 - eps) & ~before
    w = torch.where(before, torch.zeros_like(lam), torch.where(at_n, 1 - cum_prev, lam))
    n_steps = (at_n.float() * torch.arange(1, T + 1, device=g.device)[:, None]).sum(0)
    remainder = (w * at_n.float()).sum(0)
    log_w = torch.log(w.clamp_min(1e-12))
    mix = torch.logsumexp(log_w[:, :, None] + logp_y.float(), dim=0)  # [B, P]
    return mix, n_steps + remainder


def geometric_prior(T: int, lambda_p: float, device) -> torch.Tensor:
    t = torch.arange(T, device=device, dtype=torch.float32)
    p = lambda_p * (1 - lambda_p) ** t
    return p / p.sum()


def halting_distribution(g: torch.Tensor, force_last: bool = True) -> torch.Tensor:
    lam = torch.sigmoid(g.float())
    if force_last:
        lam = torch.cat([lam[:-1], torch.ones_like(lam[-1:])])
    surv = torch.cumprod(torch.cat([torch.ones_like(lam[:1]), 1 - lam[:-1]]), 0)
    return lam * surv  # [T, B]


def ponder_loss(g: torch.Tensor, ce: torch.Tensor, beta: float, lambda_p: float):
    """PonderNet: L = sum_t p_t ce_t + beta * KL(p || truncated Geometric(lambda_p))."""
    p = halting_distribution(g, force_last=True)
    rec = (p * ce.float()).sum(0).mean()
    prior = geometric_prior(g.shape[0], lambda_p, g.device)[:, None]
    kl = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(prior))).sum(0).mean()
    return rec + beta * kl, {"rec": rec.detach(), "kl": kl.detach()}


def v1_gate_loss(g: torch.Tensor, ce: torch.Tensor, lam_reg: float):
    """v1 GLWM: halting distribution renormalised over T, task loss weighted by it,
    plus lam_reg * E[steps]."""
    p = halting_distribution(g, force_last=False)
    p = p / p.sum(0, keepdim=True).clamp_min(1e-8)
    steps = torch.arange(1, g.shape[0] + 1, device=g.device, dtype=torch.float32)[:, None]
    return (p * ce.float()).sum(0).mean() + lam_reg * (p * steps).sum(0).mean()


def halting_stats(g: torch.Tensor) -> dict[str, torch.Tensor]:
    """Inference statistics from halting logits over T_max steps (no forcing at the end)."""
    lam = torch.sigmoid(g.float())
    return {
        "lambda": lam,
        "act_cum": lam.cumsum(0),
        "ponder_cum": 1 - torch.cumprod(1 - lam, 0),
    }


def _check_ponder_formula() -> float:
    """Tiny self-check against the published formula (used by tests)."""
    g = torch.tensor([[0.0], [0.0], [0.0]])
    p = halting_distribution(g)
    expected = torch.tensor([[0.5], [0.25], [0.25]])
    return float((p - expected).abs().max())
