"""Roll a model out to T_max once per test set and record every per-step statistic that any stopping
rule needs. All rules and thresholds are then evaluated from this record in a single pass
(src/glwm2/readouts/rules.py)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..data.addition import IGNORE
from ..data.depth import addition_rho, parity_rho
from ..models.halting_units import halting_stats
from .metrics import ece_bins

N_POS_SAVE = 64  # instances whose per-position done probabilities are kept for figures


def per_position_rho(task: str, data: dict) -> torch.Tensor | None:
    if task == "addition":
        return addition_rho(data["x"], data["length"])
    if task == "parity":
        return parity_rho(data["x"], data["length"])
    return None


def _batch_size(n_pos: int) -> int:
    return max(32, min(1024, int(16384 / max(1, n_pos) * 4)))


@torch.no_grad()
def collect_trajectory(model, data: dict, t_max: int, posthoc: dict | None = None, task: str | None = None,
                       batch_size: int | None = None, keep_preds: bool = True) -> dict:
    """data: frozen split dict (x, y, depth, length or size). Returns CPU tensors with a leading step
    axis T = t_max (steps 1..t_max)."""
    device = next(model.parameters()).device
    posthoc = posthoc or {}
    x_all, y_all = data["x"], data["y"]
    N = x_all.shape[0]
    n_pos = int(torch.tensor(y_all.shape[1:]).prod())
    bs = batch_size or _batch_size(n_pos)
    stats_keys = ["em", "conf_min", "ent_max", "kl_max", "dh", "logconf_sum", "run_len"]
    out = {k: [] for k in stats_keys}
    head_keys = []
    if model.gate is not None:
        head_keys += ["lambda", "act_cum", "ponder_cum"]
    if model.done is not None:
        head_keys += ["done_joint"]
    head_keys += [k for k in ("trm_post", "done_post", "done_prop_post") if k in posthoc]
    for k in head_keys:
        out[k] = []
    preds_all, pos_done = [], {}
    calib = {k: torch.zeros(2, 15, dtype=torch.float64) for k in ("done_joint", "done_post", "done_prop_post") if k in head_keys}
    calib_counts = {k: torch.zeros(15, dtype=torch.float64) for k in calib}

    for s in range(0, N, bs):
        x = x_all[s : s + bs].to(device)
        y = y_all[s : s + bs].to(device)
        x_in = x.float() if model.dim == 2 else x
        valid = y != IGNORE
        mask = model.valid_mask(x)
        B = x.shape[0]
        e, h_prev, logits0 = model.initial(x_in)
        logp_prev = F.log_softmax(logits0.float(), -1)
        pred_prev = logits0.argmax(-1)
        run = torch.ones(B, device=device)
        prev_prop = None
        rec = {k: [] for k in out}
        preds_b, done_b = [], {k: [] for k in calib}
        for o in model.rollout(x_in, t_max, mask, h0=h_prev, e=e):
            h, logits = o["h"], o["logits"].float()
            logp = F.log_softmax(logits, -1)
            p = logp.exp()
            pred = logits.argmax(-1)
            vflat = valid.flatten(1)
            maxp = p.max(-1).values.flatten(1)
            ent = -(p * logp).sum(-1).flatten(1)
            kl = (p * (logp - logp_prev)).sum(-1).flatten(1)
            rec["conf_min"].append(torch.where(vflat, maxp, torch.ones_like(maxp)).min(1).values)
            rec["ent_max"].append(torch.where(vflat, ent, torch.zeros_like(ent)).max(1).values)
            rec["kl_max"].append(torch.where(vflat, kl, torch.zeros_like(kl)).max(1).values)
            rec["logconf_sum"].append(torch.where(vflat, maxp.clamp_min(1e-30).log(), torch.zeros_like(maxp)).sum(1))
            vm = valid.unsqueeze(1).float() if model.dim == 1 else valid.unsqueeze(1).float()
            dh = ((h - h_prev).float() * vm).flatten(1).norm(dim=1) / (h.float() * vm).flatten(1).norm(dim=1).clamp_min(1e-12)
            rec["dh"].append(dh)
            same = ((pred == pred_prev) | ~valid).flatten(1).all(1)
            run = torch.where(same, run + 1, torch.ones_like(run))
            rec["run_len"].append(run.clone())
            correct = ((pred == y) | ~valid).flatten(1).all(1)
            rec["em"].append(correct)
            if model.gate is not None:
                rec["lambda"].append(o["gate"].float())
            if model.done is not None:
                dprob = torch.sigmoid(o["done"].float())
                rec["done_joint"].append(torch.where(valid, dprob, torch.ones_like(dprob)).flatten(1).min(1).values)
                if "done_joint" in calib:
                    done_b["done_joint"].append(dprob)
            if "trm_post" in posthoc:
                rec["trm_post"].append(torch.sigmoid(posthoc["trm_post"](h.float(), mask)))
            if "done_post" in posthoc:
                dprob = torch.sigmoid(posthoc["done_post"](h.float()))
                rec["done_post"].append(torch.where(valid, dprob, torch.ones_like(dprob)).flatten(1).min(1).values)
                done_b["done_post"].append(dprob)
            if "done_prop_post" in posthoc:
                dprob = torch.sigmoid(posthoc["done_prop_post"](h.float(), prev_prop))
                prev_prop = dprob
                rec["done_prop_post"].append(torch.where(valid, dprob, torch.ones_like(dprob)).flatten(1).min(1).values)
                done_b["done_prop_post"].append(dprob)
            preds_b.append(pred.to(torch.uint8))
            h_prev, logp_prev, pred_prev = h, logp, pred

        if model.gate is not None:
            hs = halting_stats(torch.stack(rec["lambda"]))  # rec["lambda"] holds gate logits here
            rec["lambda"] = list(hs["lambda"])
            rec["act_cum"] = list(hs["act_cum"])
            rec["ponder_cum"] = list(hs["ponder_cum"])
        for k in out:
            out[k].append(torch.stack(rec[k]).cpu())
        preds_t = torch.stack(preds_b)  # [T, B, *S]
        if keep_preds:
            preds_all.append(preds_t.cpu())
        if calib:
            pc = (preds_t.long() == y.unsqueeze(0)) | ~valid.unsqueeze(0)
            tgt = pc.flip(0).int().cummin(0).values.flip(0)
            vv = valid.unsqueeze(0).expand_as(tgt)
            for k, lst in done_b.items():
                probs = torch.stack(lst)
                c, n = ece_bins(probs[vv], tgt[vv].float())
                calib[k] += c.cpu()
                calib_counts[k] += n.cpu()
                if s == 0:
                    pos_done[k] = probs[:, :N_POS_SAVE].cpu().half()

    traj = {k: torch.cat(v, dim=1) for k, v in out.items()}
    if keep_preds:
        traj["preds"] = torch.cat(preds_all, dim=1)
    traj["t_max"] = t_max
    traj["calib"] = {k: {"sum": calib[k], "count": calib_counts[k]} for k in calib}
    traj["pos_done"] = pos_done
    return traj


def t_max_for(data: dict, r: int, extra: int = 16) -> int:
    return int(math.ceil(int(data["depth"].max()) / r)) + extra
