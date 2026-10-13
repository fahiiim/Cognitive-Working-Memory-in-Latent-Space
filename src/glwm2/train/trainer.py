"""Training loop for every method. Resumable: checkpoints/<run_id>/last.pt is written every
`ckpt_every` iterations and picked up automatically if the run is restarted."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import CKPT_DIR, RESULTS_DIR
from ..data.addition import IGNORE
from ..data.splits import TrainSampler
from ..models.halting_units import act_loss, ponder_loss, v1_gate_loss
from ..models.reasoner import Reasoner
from ..utils.config import RunConfig
from ..utils.io import append_jsonl, env_info, write_json
from ..utils.seed import seed_everything
from .objectives import done_targets, exact_match, fixed_prior_loss, masked_instance_mean, position_ce, position_correct


def run_dirs(cfg: RunConfig) -> tuple[Path, Path]:
    rid = cfg.run_id()
    return CKPT_DIR / rid, RESULTS_DIR / "runs" / rid


def lr_at(it: int, cfg: RunConfig) -> float:
    if it < cfg.warmup:
        return cfg.lr * (it + 1) / cfg.warmup
    prog = (it - cfg.warmup) / max(1, cfg.iters - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def compute_loss(model: Reasoner, batch: dict, cfg: RunConfig, g: torch.Generator):
    x, y = batch["x"], batch["y"]
    valid = y != IGNORE
    mask = model.valid_mask(x)
    T = cfg.t_train
    outs = list(model.rollout(x, T, mask))
    logits = torch.stack([o["logits"] for o in outs])  # [T, B, *S, C]
    ce_pos = position_ce(logits, y)
    ce = masked_instance_mean(ce_pos, valid)  # [T, B]
    logs = {}

    if cfg.method == "act":
        logp = -ce_pos
        g_logits = torch.stack([o["gate"] for o in outs])
        mix, ponder = act_loss(g_logits, logp, cfg.act_tau)
        nll = -(mix * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
        loss = nll.mean() + cfg.act_tau * ponder.mean()
        logs["ponder"] = ponder.mean().item()
    elif cfg.method == "ponder":
        g_logits = torch.stack([o["gate"] for o in outs])
        loss, parts = ponder_loss(g_logits, ce, cfg.ponder_beta, cfg.ponder_lambda_p)
        logs.update({k: v.item() for k, v in parts.items()})
    elif cfg.method == "v1gate":
        g_logits = torch.stack([o["gate"] for o in outs])
        loss = v1_gate_loss(g_logits, ce, cfg.v1_lambda)
    else:
        if cfg.objective == "fixed_prior":
            task = fixed_prior_loss(ce)
        elif cfg.objective == "progressive":
            # Bansal et al. 2022: (1 - alpha) * loss after T steps + alpha * loss after n (no grad) + k steps
            n = int(torch.randint(0, T, (1,), generator=g, device=g.device).item())
            k = int(torch.randint(1, T - n + 1, (1,), generator=g, device=g.device).item())
            e = model.backbone.encode(x)
            with torch.no_grad():
                h = model.backbone.init_state(e)
                for _ in range(n):
                    h = model.backbone.step(h, e)
            h = h.detach()
            for _ in range(k):
                h = model.backbone.step(h, e)
            ce_prog = masked_instance_mean(position_ce(model.backbone.decode(h)[None], y), valid)[0]
            task = (1 - cfg.progressive_alpha) * ce[-1].mean() + cfg.progressive_alpha * ce_prog.mean()
        else:
            raise ValueError(cfg.objective)
        loss = task
        if cfg.method in ("trm_joint", "done_joint", "done_prop_joint"):
            with torch.no_grad():
                pc = position_correct(logits, y)
            if cfg.method == "trm_joint":
                g_logits = torch.stack([o["gate"] for o in outs]).float()
                aux = F.binary_cross_entropy_with_logits(g_logits, exact_match(pc).float())
            else:
                d_logits = torch.stack([o["done"] for o in outs]).float()
                tgt = done_targets(pc).float()
                bce = F.binary_cross_entropy_with_logits(d_logits, tgt, reduction="none")
                aux = masked_instance_mean(bce, valid).mean()
            loss = loss + cfg.aux_weight * aux
            logs["aux"] = aux.item()

    with torch.no_grad():
        em = exact_match(position_correct(logits[-1:], y))[0]
        logs["acc_T"] = em.float().mean().item()
    return loss, logs


def build_model(cfg: RunConfig, device) -> Reasoner:
    return Reasoner(cfg).to(device)


def train(cfg: RunConfig, save: bool = True, deterministic: bool = False, device=None, verbose: bool = False,
          log_every: int = 200, ckpt_every: int = 1000) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(cfg.seed, deterministic=deterministic)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ckpt_dir, res_dir = run_dirs(cfg)
    start_it = 0
    if save:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        final = ckpt_dir / "model.pt"
        if final.exists():
            model.load_state_dict(torch.load(final, map_location=device))
            return {"model": model, "resumed": "final", "run_id": cfg.run_id()}
        last = ckpt_dir / "last.pt"
        if last.exists():
            st = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            g.set_state(st["gen"].to("cpu") if device.type == "cpu" else st["gen"])
            start_it = st["it"]
        write_json(res_dir / "config.json", cfg.to_dict())
    sampler = TrainSampler(cfg.task, device, g, cfg.train_len, cfg.train_depth, cfg.maze_train_size)
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda")
    t0 = time.time()
    model.train()
    for it in range(start_it, cfg.iters):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it, cfg)
        batch = sampler.sample(cfg.batch_size)
        with amp:
            loss, logs = compute_loss(model, batch, cfg, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if (it + 1) % log_every == 0 or it == cfg.iters - 1:
            rec = {"it": it + 1, "loss": loss.item(), "grad_norm": float(gn), "lr": lr_at(it, cfg),
                   "sec": round(time.time() - t0, 1), **logs}
            if save:
                append_jsonl(res_dir / "train_log.jsonl", rec)
            if verbose:
                print(rec)
        if save and (it + 1) % ckpt_every == 0 and it + 1 < cfg.iters:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "it": it + 1},
                       ckpt_dir / "last.pt")
    model.eval()
    out = {"model": model, "run_id": cfg.run_id(), "train_seconds": time.time() - t0, "n_filtered": sampler.n_filtered}
    if save:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        (ckpt_dir / "last.pt").unlink(missing_ok=True)
        write_json(res_dir / "train_summary.json", {"train_seconds": out["train_seconds"], "n_filtered": sampler.n_filtered,
                                                    "env": env_info()})
    return out


def load_model(cfg: RunConfig, device=None) -> Reasoner:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(run_dirs(cfg)[0] / "model.pt", map_location=device))
    return model.eval()
