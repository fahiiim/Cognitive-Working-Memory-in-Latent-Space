"""Linear probes on the latent state: is "my answer is still pending" (t < rho_i) linearly
decodable from h_i(t), and does the probe transfer to deeper / longer inputs?"""

from __future__ import annotations

import numpy as np
import torch

from ..data.addition import IGNORE
from .trajectory import per_position_rho


@torch.no_grad()
def collect_probe_data(model, task: str, data: dict, t_max: int, max_rows: int = 200_000, seed: int = 0, batch: int = 256):
    """Returns h [N, C], pending [N] (t < rho_i), resolved-correct [N], and (t, rho) per row."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    rho_all = per_position_rho(task, data)
    feats, pend, corr, ts, rhos = [], [], [], [], []
    for s in range(0, data["x"].shape[0], batch):
        x = data["x"][s : s + batch].to(device)
        y = data["y"][s : s + batch].to(device)
        rho = rho_all[s : s + batch].to(device)
        valid = y != IGNORE
        for o in model.rollout(x, t_max, model.valid_mask(x)):
            t = o["t"]
            h = o["h"].transpose(1, 2)[valid]  # [n, C]
            r = rho[valid]
            feats.append(h.float().cpu())
            pend.append((t < r).cpu())
            corr.append((o["logits"].argmax(-1) == y)[valid].cpu())
            ts.append(torch.full((h.shape[0],), t))
            rhos.append(r.cpu())
    H, P, C, T, R = (torch.cat(v) for v in (feats, pend, corr, ts, rhos))
    if H.shape[0] > max_rows:
        idx = torch.randperm(H.shape[0], generator=g)[:max_rows]
        H, P, C, T, R = H[idx], P[idx], C[idx], T[idx], R[idx]
    return {"h": H, "pending": P, "correct": C, "t": T, "rho": R}


def fit_linear_probe(h: torch.Tensor, y: torch.Tensor, iters: int = 500, lr: float = 0.05, wd: float = 1e-4, device="cuda"):
    h, y = h.to(device), y.float().to(device)
    mu, sd = h.mean(0), h.std(0).clamp_min(1e-6)
    w = torch.zeros(h.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    hn = (h - mu) / sd
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(hn @ w + b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sd": sd}


@torch.no_grad()
def probe_scores(probe: dict, h: torch.Tensor, y: torch.Tensor) -> dict:
    dev = probe["w"].device
    logit = ((h.to(dev) - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]
    pred = logit > 0
    y = y.to(dev).bool()
    acc = (pred == y).float().mean().item()
    # AUC via the rank statistic
    s = logit.float().cpu().numpy()
    yy = y.cpu().numpy()
    order = s.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = yy.sum(), (~yy).sum()
    auc = float((ranks[yy].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))
    return {"acc": acc, "auc": auc, "base_rate": float(yy.mean())}
