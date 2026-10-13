"""Faithful re-implementation of the v1 GLWM model (main.tex) for the corrected re-run.

v1: operands with d ~ U{1..4} digits, most-significant first, zero-padded to 4 digits; sum padded
to 5 digits. Encoder: Embedding(11, 64) for a_i and b_i, concatenated (128) -> 2-layer GRU(256,
dropout 0.1). Latent loop z <- LayerNorm(z + MLP(z)) with MLP 256-512-256 (GELU). Output head
Linear(256, 55) = 5 digits x 11 classes. Gate Linear(256, 64) -> ReLU -> Linear(64, 1).
Loss: PonderNet-style halting distribution renormalised over T = 8, task loss weighted by it,
plus lambda * E[steps] with lambda = 0.01. Inference halts at the first step with gate > 0.7.

`gate_init="v1"` sets both gate biases to -2 as in the original notebook (the first one silences
most ReLU units); `gate_init="fixed"` keeps PyTorch's default first layer and sets only the
output bias to -2, which is what main.tex describes.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval.flops import count_flops

T_MAX = 8
THRESH = 0.7


class V1GLWM(nn.Module):
    def __init__(self, gate_init: str = "fixed", hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(11, 64)
        self.gru = nn.GRU(128, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.ln = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 55)
        self.gate = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        if gate_init == "v1":
            nn.init.constant_(self.gate[0].bias, -2.0)
            nn.init.constant_(self.gate[2].bias, -2.0)
        elif gate_init == "fixed":
            nn.init.constant_(self.gate[2].bias, -2.0)
        else:
            raise ValueError(gate_init)

    def encode(self, a, b):
        x = torch.cat([self.embed(a), self.embed(b)], dim=-1)  # [B, 4, 128]
        _, hn = self.gru(x)
        return hn[-1]

    def encode_manual(self, a, b):
        """Same computation as encode() with explicit matmuls, so FlopCounterMode can see it."""
        x = torch.cat([self.embed(a), self.embed(b)], dim=-1)
        H = self.gru.hidden_size
        for layer in range(2):
            w_ih, w_hh = getattr(self.gru, f"weight_ih_l{layer}"), getattr(self.gru, f"weight_hh_l{layer}")
            b_ih, b_hh = getattr(self.gru, f"bias_ih_l{layer}"), getattr(self.gru, f"bias_hh_l{layer}")
            h = x.new_zeros(x.shape[0], H)
            outs = []
            for t in range(x.shape[1]):
                gi, gh = F.linear(x[:, t], w_ih, b_ih), F.linear(h, w_hh, b_hh)
                i_r, i_z, i_n = gi.chunk(3, 1)
                h_r, h_z, h_n = gh.chunk(3, 1)
                r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h = (1 - z) * n + z * h
                outs.append(h)
            x = torch.stack(outs, 1)
        return h

    def step(self, z):
        return self.ln(z + self.mlp(z))

    def forward(self, a, b, T: int = T_MAX):
        z = self.encode(a, b)
        logits, lam = [], []
        for _ in range(T):
            z = self.step(z)
            logits.append(self.out(z).view(-1, 5, 11))
            lam.append(torch.sigmoid(self.gate(z)).squeeze(-1))
        return torch.stack(logits), torch.stack(lam)  # [T, B, 5, 11], [T, B]


def make_v1_data(n: int, rng: np.random.Generator):
    d = rng.integers(1, 5, size=n)
    lo = np.where(d == 1, 0, 10 ** (d - 1))
    a = rng.integers(lo, 10**d)
    b = rng.integers(lo, 10**d)
    return a, b, d


def digits(v: np.ndarray, width: int) -> torch.Tensor:
    return torch.tensor(np.stack([(v // 10 ** (width - 1 - k)) % 10 for k in range(width)], 1), dtype=torch.long)


def to_tensors(a, b, device):
    return digits(a, 4).to(device), digits(b, 4).to(device), digits(a + b, 5).to(device)


def v1_loss(logits, lam, y, lambda_reg: float = 0.01):
    T = logits.shape[0]
    surv = torch.cumprod(torch.cat([torch.ones_like(lam[:1]), 1 - lam[:-1]]), 0)
    p = lam * surv
    p = p / p.sum(0, keepdim=True).clamp_min(1e-8)
    ce = torch.stack([F.cross_entropy(logits[t].reshape(-1, 11), y.reshape(-1), reduction="none").view(y.shape).sum(1) for t in range(T)])
    steps = torch.arange(1, T + 1, device=lam.device, dtype=lam.dtype)[:, None]
    return (p * ce).sum(0).mean() + lambda_reg * (p * steps).sum(0).mean()


@torch.no_grad()
def v1_evaluate(model, a, b, y, batch: int = 4096):
    model.eval()
    halts, correct, correct_by_t = [], [], []
    for s in range(0, a.shape[0], batch):
        logits, lam = model(a[s : s + batch], b[s : s + batch])
        over = lam > THRESH
        halt = torch.where(over.any(0), over.float().argmax(0), torch.full_like(over[0], T_MAX - 1, dtype=torch.long))
        pred = logits.argmax(-1)  # [T, B, 5]
        ok = (pred == y[s : s + batch][None]).all(-1)  # [T, B]
        halts.append(halt + 1)
        correct.append(ok.gather(0, halt[None])[0])
        correct_by_t.append(ok)
    return torch.cat(halts), torch.cat(correct), torch.cat(correct_by_t, 1)


def train_v1(gate_init: str, seed: int, device="cuda", n_train=100_000, n_val=10_000, n_test=10_000, epochs=30, batch=512, lr=3e-4):
    """Corrected protocol: separate validation set for model selection, test de-duplicated against
    train+val, per-seed results. Returns a dict of metrics and the trained model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a_tr, b_tr, _ = make_v1_data(n_train, rng)
    a_va, b_va, d_va = make_v1_data(n_val, rng)
    a_te, b_te, d_te = make_v1_data(n_test * 3, rng)
    key = lambda a, b: a * 10**4 + b
    seen = set(key(a_tr, b_tr).tolist()) | set(key(a_va, b_va).tolist())
    te_key = key(a_te, b_te)
    dup = np.array([k in seen for k in te_key.tolist()])
    dup_share_by_d = {int(d): float(dup[d_te == d].mean()) for d in range(1, 5)}
    a_te_all, b_te_all, d_te_all = a_te[: n_test], b_te[: n_test], d_te[: n_test]
    keep = ~dup
    a_te_dd, b_te_dd, d_te_dd = a_te[keep][:n_test], b_te[keep][:n_test], d_te[keep][:n_test]

    model = V1GLWM(gate_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(n_train / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    A, B, Y = to_tensors(a_tr, b_tr, device)
    Av, Bv, Yv = to_tensors(a_va, b_va, device)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch):
            idx = perm[s : s + batch]
            logits, lam = model(A[idx], B[idx])
            loss = v1_loss(logits, lam, Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        _, ok, _ = v1_evaluate(model, Av, Bv, Yv)
        acc = ok.float().mean().item()
        if acc >= best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    out = {"gate_init": gate_init, "seed": seed, "val_acc": best, "train_seconds": time.time() - t0,
           "test_dup_share_by_digits": dup_share_by_d}
    for name, (a_, b_, d_) in {"test_v1_protocol": (a_te_all, b_te_all, d_te_all), "test_dedup": (a_te_dd, b_te_dd, d_te_dd)}.items():
        At, Bt, Yt = to_tensors(a_, b_, device)
        halt, ok, ok_t = v1_evaluate(model, At, Bt, Yt)
        out[name] = {
            "acc": ok.float().mean().item(),
            "mean_steps": halt.float().mean().item(),
            "steps_hist": torch.bincount(halt, minlength=T_MAX + 1)[1:].tolist(),
            "acc_by_digits": {int(d): ok[torch.tensor(d_ == d, device=device)].float().mean().item() for d in range(1, 5)},
            "steps_by_digits": {int(d): halt[torch.tensor(d_ == d, device=device)].float().mean().item() for d in range(1, 5)},
            "acc_at_step": ok_t.float().mean(1).tolist(),
        }
    # measured FLOPs per sample: encoder + one latent step (+ output head and gate)
    a1, b1, _ = to_tensors(a_te_all[:1], b_te_all[:1], "cpu")
    m_cpu = V1GLWM(gate_init).eval()
    m_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    z = torch.randn(1, 256)
    with torch.no_grad():
        assert torch.allclose(m_cpu.encode(a1, b1), m_cpu.encode_manual(a1, b1), atol=1e-5)
    enc = count_flops(lambda: m_cpu.encode_manual(a1, b1))
    step = count_flops(lambda: (m_cpu.step(z), m_cpu.out(z), m_cpu.gate(z)))
    out["flops"] = {"encoder": enc, "per_step": step,
                    "per_sample_at_mean_steps": enc + step * out["test_dedup"]["mean_steps"]}
    return out, model
