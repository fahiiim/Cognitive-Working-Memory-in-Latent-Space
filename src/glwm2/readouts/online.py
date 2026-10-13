"""Online early exit: actually stop computing for halted instances (used for latency and FLOP
measurements). Halted instances are removed from the active batch, so compute shrinks as
instances finish."""

from __future__ import annotations

from typing import Callable

import torch

from ..models.heads import PropCompletionHead


@torch.no_grad()
def run_online(model, x: torch.Tensor, t_max: int, halt_fn: Callable[[dict], torch.Tensor] | None,
               head=None, head_thr: float = 0.9) -> dict:
    """halt_fn(step_out) -> bool [B_active]; or pass a completion `head` and threshold for our rule.
    Returns answers, halting steps, and the number of position-steps actually computed."""
    B = x.shape[0]
    device = x.device
    mask_all = model.valid_mask(x)
    e_all = model.backbone.encode(x)
    h = model.backbone.init_state(e_all)
    active = torch.arange(B, device=device)
    answers = torch.zeros(x.shape[0], *x.shape[1:] if model.dim == 1 else x.shape[2:], dtype=torch.long, device=device)
    steps = torch.full((B,), t_max, dtype=torch.long, device=device)
    prev = None
    pos_steps = 0
    for t in range(1, t_max + 1):
        e = e_all[active]
        h = model.backbone.step(h, e)
        logits = model.backbone.decode(h)
        pos_steps += h.shape[0] * h[0, 0].numel()
        out = {"h": h, "logits": logits, "t": t, "mask": None if mask_all is None else mask_all[active]}
        if head is not None:
            mask = None if mask_all is None else mask_all[active]
            if isinstance(head, PropCompletionHead):
                d = torch.sigmoid(head(h, prev))
                prev = d
            else:
                d = torch.sigmoid(head(h))
            if mask is not None:
                d = torch.where(mask, d, torch.ones_like(d))
            done = d.flatten(1).min(1).values > head_thr
        elif halt_fn is not None:
            done = halt_fn(out)
        else:
            done = torch.zeros(h.shape[0], dtype=torch.bool, device=device)
        if t == t_max:
            done = torch.ones_like(done)
        if done.any():
            idx = active[done]
            answers[idx] = logits[done].argmax(-1)
            steps[idx] = t
            keep = ~done
            active, h = active[keep], h[keep]
            if prev is not None:
                prev = prev[keep]
        if active.numel() == 0:
            break
    return {"answers": answers, "steps": steps, "position_steps": pos_steps}
