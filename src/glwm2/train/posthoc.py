"""Heads fitted after training on a frozen backbone (post-hoc readouts).

- trm_post: pooled correctness head, target = exact match at step t (TRM-style, fitted post hoc).
- done_post: pointwise completion head, target = correct now and at every later step.
- done_prop_post: the dependency-propagating completion head, same target.

Training data: fresh ID training batches, rolled out to `horizon` steps without gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.addition import IGNORE
from ..data.splits import TrainSampler
from ..models.heads import CompletionHead, PooledHead, PropCompletionHead
from ..utils.config import RunConfig
from .objectives import done_targets, exact_match, masked_instance_mean, position_correct

POSTHOC_HEADS = ("trm_post", "done_post", "done_prop_post")


def make_posthoc_heads(width: int, dim: int, device) -> dict[str, torch.nn.Module]:
    return {
        "trm_post": PooledHead(width).to(device),
        "done_post": CompletionHead(width, dim=dim).to(device),
        "done_prop_post": PropCompletionHead(width, dim=dim).to(device),
    }


def fit_posthoc_heads(model, cfg: RunConfig, iters: int = 1500, batch_size: int = 256, horizon: int | None = None,
                      lr: float = 2e-3, seed: int = 10_000, device=None) -> tuple[dict, dict]:
    device = torch.device(device or next(model.parameters()).device)
    horizon = horizon or 2 * cfg.t_train
    heads = make_posthoc_heads(cfg.width, model.dim, device)
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    g = torch.Generator(device=device).manual_seed(cfg.seed + seed)
    sampler = TrainSampler(cfg.task, device, g, cfg.train_len, cfg.train_depth, cfg.maze_train_size)
    model.eval()
    logs = []
    for it in range(iters):
        b = sampler.sample(batch_size)
        x, y = b["x"], b["y"]
        valid = y != IGNORE
        mask = model.valid_mask(x)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hs, preds = [], []
            for o in model.rollout(x, horizon, mask):
                hs.append(o["h"].float())
                preds.append(o["logits"].argmax(-1))
            hs = torch.stack(hs)
            pc = position_correct(torch.stack(preds), y)
            em = exact_match(pc).float()
            tgt = done_targets(pc).float()
        T = hs.shape[0]
        flat_h = hs.flatten(0, 1)
        flat_mask = None if mask is None else mask.repeat(T, *([1] * (mask.dim() - 1)))
        l_trm = F.binary_cross_entropy_with_logits(heads["trm_post"](flat_h, flat_mask).view(T, -1), em)
        d = heads["done_post"](flat_h).view(T, *tgt.shape[1:])
        l_done = masked_instance_mean(F.binary_cross_entropy_with_logits(d, tgt, reduction="none"), valid).mean()
        prev, dp = None, []
        for t in range(T):
            lt = heads["done_prop_post"](hs[t], prev)
            prev = torch.sigmoid(lt)
            dp.append(lt)
        l_prop = masked_instance_mean(F.binary_cross_entropy_with_logits(torch.stack(dp), tgt, reduction="none"), valid).mean()
        loss = l_trm + l_done + l_prop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 250 == 0:
            logs.append({"it": it + 1, "trm": l_trm.item(), "done": l_done.item(), "prop": l_prop.item()})
    for h in heads.values():
        h.eval()
    return heads, {"log": logs, "horizon": horizon, "iters": iters}
