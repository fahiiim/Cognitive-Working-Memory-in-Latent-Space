"""GLWM-2 backbone: a local recurrent latent reasoner.

One latent vector per position (1D sequences or 2D grids). Every step applies the same block:

    h <- Norm(h + W_out GELU(Conv_k3(...GELU(Conv_k3([h; e]))...)))

with `convs_per_step` kernel-3 convolutions, so the receptive field grows by exactly
r = convs_per_step positions per step. The encoder, norm and decoder are pointwise, which keeps
the light cone exact (tests/test_theory.py checks this).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at every position (pointwise), for [B, C, *spatial]."""

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mu = h.mean(dim=1, keepdim=True)
        var = h.var(dim=1, keepdim=True, unbiased=False)
        h = (h - mu) * torch.rsqrt(var + self.eps)
        shape = (1, -1) + (1,) * (h.dim() - 2)
        return h * self.weight.view(shape) + self.bias.view(shape)


@dataclass
class StepOut:
    t: int
    h: torch.Tensor  # [B, C, *spatial]
    logits: torch.Tensor  # [B, *spatial, n_out]


class RecurrentBackbone(nn.Module):
    def __init__(
        self,
        n_out: int,
        width: int = 128,
        vocab: int | None = None,
        in_channels: int | None = None,
        dim: int = 1,
        convs_per_step: int = 1,
        recall: bool = True,
        norm: str = "layer",
    ):
        super().__init__()
        assert (vocab is None) != (in_channels is None), "token or image input"
        self.dim, self.width, self.recall = dim, width, recall
        self.r = convs_per_step
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        if vocab is not None:
            self.embed = nn.Embedding(vocab, width)
            self.in_proj = None
        else:
            self.embed = None
            self.in_proj = Conv(in_channels, width, 1)
        convs = []
        c_in = 2 * width if recall else width
        for _ in range(convs_per_step):
            convs.append(Conv(c_in, width, 3, padding=1))
            c_in = width
        self.convs = nn.ModuleList(convs)
        self.out_proj = Conv(width, width, 1)
        self.norm = ChannelLayerNorm(width) if norm == "layer" else nn.Identity()
        self.dec1 = Conv(width, width, 1)
        self.dec2 = Conv(width, n_out, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            e = self.embed(x)  # [B, P, C]
            return e.transpose(1, 2).contiguous()  # [B, C, P]
        return self.in_proj(x)

    def init_state(self, e: torch.Tensor) -> torch.Tensor:
        return self.norm(e)

    def step(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h, e], dim=1) if self.recall else h
        for conv in self.convs:
            z = F.gelu(conv(z))
        return self.norm(h + self.out_proj(z))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.dec2(F.gelu(self.dec1(h)))  # [B, n_out, *spatial]
        return logits.movedim(1, -1)

    def rollout(self, x: torch.Tensor, T: int, h0: torch.Tensor | None = None, start: int = 0):
        """Yield StepOut for t = start+1 .. start+T. Pass h0 to continue an earlier rollout."""
        e = self.encode(x)
        h = self.init_state(e) if h0 is None else h0
        for t in range(start + 1, start + T + 1):
            h = self.step(h, e)
            yield StepOut(t, h, self.decode(h))

    def forward(self, x: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-step logits [T, B, *spatial, n_out] and the final state."""
        outs, h = [], None
        for o in self.rollout(x, T):
            outs.append(o.logits)
            h = o.h
        return torch.stack(outs), h
