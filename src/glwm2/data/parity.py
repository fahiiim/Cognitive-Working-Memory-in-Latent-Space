"""T2: segmented prefix parity.

Tokens: 0, 1, DELIM (2), PAD (3). y_i = parity of the bits since the last delimiter, inclusive;
y = 0 at delimiters. d* is the longest run of non-delimiter tokens. With no delimiters this is
the prefix-sum task of Schwarzschild et al. (2021).
"""

from __future__ import annotations

import torch

from .addition import IGNORE, _cap_runs, _sample_lengths_depths
from .depth import PAR_DELIM, PAR_PAD, depth_from_rho, parity_rho

VOCAB = 4
N_OUT = 2


def sample_parity(
    n: int,
    len_range: tuple[int, int],
    depth_range: tuple[int, int],
    g: torch.Generator,
    device: torch.device | str = "cpu",
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    L, d = _sample_lengths_depths(n, len_range, depth_range, g, device, extra=0)
    P = int(L.max().item()) if pad_to is None else pad_to
    pos = torch.arange(P, device=device).expand(n, P)
    real = pos < L[:, None]

    q = 0.03 + 0.35 * torch.rand(n, 1, generator=g, device=device)
    seg = torch.rand(n, P, generator=g, device=device) >= q  # True = bit, False = delimiter
    seg = _cap_runs(seg, d, real)

    u = torch.floor(torch.rand(n, generator=g, device=device) * (L - d + 1).float()).long()
    in_run = (pos >= u[:, None]) & (pos < (u + d)[:, None])
    flank = (pos == (u - 1)[:, None]) | (pos == (u + d)[:, None])
    seg = (seg | in_run) & ~flank & real

    bits = torch.randint(0, 2, (n, P), generator=g, device=device)
    x = torch.where(seg, bits, torch.full_like(bits, PAR_DELIM))
    x = torch.where(real, x, torch.full_like(x, PAR_PAD))

    y = torch.full((n, P), IGNORE, dtype=torch.long, device=device)
    par = torch.zeros(n, dtype=torch.long, device=device)
    for i in range(P):
        is_bit = seg[:, i]
        par = torch.where(is_bit, par ^ bits[:, i], torch.zeros_like(par))
        y[:, i] = torch.where(real[:, i], par, y[:, i])

    rho = parity_rho(x, L)
    depth = depth_from_rho(rho)
    return {"x": x, "y": y, "length": L, "depth": depth, "target_depth": d}
