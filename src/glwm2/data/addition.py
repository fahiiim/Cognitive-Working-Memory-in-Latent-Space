"""T1: decimal addition, least-significant digit first, with controlled carry-chain depth.

Each sample has L digit pairs at positions 0..L-1, a (0, 0) pair at position L for the final
carry, and PAD after that. Token = 10 * a + b. Target y_i is the i-th digit of a + b
(position L holds the carry-out). d* is the largest gap between consecutive "stops"
(a_j + b_j != 9), the virtual left boundary -1, and the final position L.
"""

from __future__ import annotations

import torch

from .depth import ADD_PAD, addition_rho, depth_from_rho

IGNORE = -100
VOCAB = 101
N_OUT = 10


def _sample_lengths_depths(n, len_range, depth_range, g, device, extra: int):
    """Sample L uniformly, then d uniformly in [d_lo, min(d_hi, L + extra)]."""
    l_lo, l_hi = len_range
    d_lo, d_hi = depth_range
    l_lo = max(l_lo, d_lo - extra)
    if l_lo > l_hi:
        raise ValueError(f"no length in {len_range} admits depth >= {d_lo}")
    L = torch.randint(l_lo, l_hi + 1, (n,), generator=g, device=device)
    d_max = torch.clamp(L + extra, max=d_hi)
    u = torch.rand(n, generator=g, device=device)
    d = d_lo + torch.floor(u * (d_max - d_lo + 1).float()).long()
    return L, torch.minimum(d, d_max)


def _cap_runs(is_run: torch.Tensor, cap: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Break every run of True longer than cap[b] by turning its (cap+1)-th element False."""
    is_run = is_run & real
    run = torch.zeros(is_run.shape[0], dtype=torch.long, device=is_run.device)
    for i in range(is_run.shape[1]):
        col = is_run[:, i]
        run = (run + 1) * col
        over = run > cap
        if over.any():
            is_run[over, i] = False
            run = torch.where(over, torch.zeros_like(run), run)
    return is_run


def sample_addition(
    n: int,
    len_range: tuple[int, int],
    depth_range: tuple[int, int],
    g: torch.Generator,
    device: torch.device | str = "cpu",
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    L, d = _sample_lengths_depths(n, len_range, depth_range, g, device, extra=1)
    P = int(L.max().item()) + 1 if pad_to is None else pad_to
    pos = torch.arange(P, device=device).expand(n, P)
    real = pos < L[:, None]

    # 1. random nine-runs with a per-sample density, capped so no gap exceeds d
    q = 0.05 + 0.45 * torch.rand(n, 1, generator=g, device=device)
    nine = torch.rand(n, P, generator=g, device=device) < q
    nine = _cap_runs(nine, d - 1, real)

    # 2. insert one run of exactly d-1 nines flanked by stops: gap from u-1 to u+d-1 equals d
    u = torch.floor(torch.rand(n, generator=g, device=device) * (L - d + 2).float()).long()
    in_run = (pos >= u[:, None]) & (pos < (u + d - 1)[:, None])
    flank = (pos == (u - 1)[:, None]) | (pos == (u + d - 1)[:, None])
    nine = (nine | in_run) & ~flank & real

    # 3. digits: nine positions get a + b = 9, stops get a + b != 9 (uniform over the 90 pairs)
    a = torch.randint(0, 10, (n, P), generator=g, device=device)
    off = torch.randint(1, 10, (n, P), generator=g, device=device)
    b_stop = (9 - a + off) % 10
    b = torch.where(nine, 9 - a, b_stop)
    a = torch.where(real, a, torch.zeros_like(a))
    b = torch.where(real, b, torch.zeros_like(b))

    x = 10 * a + b
    x = torch.where(pos <= L[:, None], x, torch.full_like(x, ADD_PAD))

    y = torch.full((n, P), IGNORE, dtype=torch.long, device=device)
    carry = torch.zeros(n, dtype=torch.long, device=device)
    for i in range(P):
        s = a[:, i] + b[:, i] + carry
        valid = i <= L
        y[:, i] = torch.where(valid, s % 10, y[:, i])
        carry = torch.where(i < L, s // 10, torch.zeros_like(carry))

    rho = addition_rho(x, L)
    depth = depth_from_rho(rho)
    return {"x": x, "y": y, "length": L, "depth": depth, "target_depth": d}


def decode_numbers(x: torch.Tensor, length: torch.Tensor) -> list[tuple[int, int]]:
    """Recover the integer operands (for tests against Python big integers)."""
    out = []
    for row, L in zip(x.tolist(), length.tolist()):
        a = sum((t // 10) * 10**i for i, t in enumerate(row[:L]))
        b = sum((t % 10) * 10**i for i, t in enumerate(row[:L]))
        out.append((a, b))
    return out


def decode_target(y: torch.Tensor, length: torch.Tensor) -> list[int]:
    return [sum(d * 10**i for i, d in enumerate(row[: L + 1])) for row, L in zip(y.tolist(), length.tolist())]
