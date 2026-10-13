"""Read-out heads on the latent state h [B, C, *spatial].

- PooledHead: masked mean over positions -> MLP -> one logit per instance (ACT, PonderNet, v1 gate,
  TRM-style correctness head). This is the "global" signal family.
- CompletionHead: pointwise MLP -> one "my part is done" logit per position (ours).
- PropCompletionHead: like CompletionHead, but also reads its own and its neighbours' done
  probabilities from the previous step, mirroring the automaton of Proposition 3 (ours).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """h [B, C, *S], mask [B, *S] bool or None -> [B, C]."""
    if mask is None:
        return h.flatten(2).mean(-1)
    m = mask.flatten(1).unsqueeze(1).to(h.dtype)
    return (h.flatten(2) * m).sum(-1) / m.sum(-1).clamp_min(1.0)


class PooledHead(nn.Module):
    def __init__(self, width: int, hidden: int = 64, act: str = "gelu", out_bias: float | None = None):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        if out_bias is not None:
            nn.init.constant_(self.fc2.bias, out_bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(masked_mean(h, mask)))).squeeze(-1)  # [B]


class CompletionHead(nn.Module):
    def __init__(self, width: int, hidden: int = 64, dim: int = 1):
        super().__init__()
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.fc1 = Conv(width, hidden, 1)
        self.fc2 = Conv(hidden, 1, 1)

    def forward(self, h: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(h)))[:, 0]  # [B, *S]


class PropCompletionHead(nn.Module):
    """done_i(t) = f(h_i(t), done_{i-1..i+1}(t-1)). The constant-one channel is zero-padded at the
    border, so the head can see the input boundary exactly as the automaton does."""

    recurrent = True

    def __init__(self, width: int, hidden: int = 64, dim: int = 1):
        super().__init__()
        Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.fc_h = Conv(width, hidden, 1)
        self.fc_nb = Conv(2, hidden, 3, padding=1)
        self.fc2 = Conv(hidden, 1, 1)

    def forward(self, h: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        if prev is None:
            prev = torch.zeros_like(h[:, 0])
        nb = torch.stack([prev.to(h.dtype), torch.ones_like(prev, dtype=h.dtype)], dim=1)
        return self.fc2(F.gelu(self.fc_h(h) + self.fc_nb(nb)))[:, 0]
