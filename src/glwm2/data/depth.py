"""Per-instance depth oracles.

rho[b, i] is the light-cone radius of output position i: the smallest window radius around i
(in positions, counting the left boundary as a virtual position -1) that determines y_i.
d*(x) = max_i rho[b, i]. A local model that grows its receptive field by r positions per step
needs at least ceil(rho_i / r) steps before position i can be right on every input
(paper/sections/theory.tex, Theorem 1).
"""

from __future__ import annotations

import math

import torch

ADD_PAD = 100  # addition token = 10 * a + b, PAD = 100
PAR_DELIM, PAR_PAD = 2, 3  # parity tokens: 0, 1, DELIM, PAD


def carry_depth(a: list[int], b: list[int]) -> int:
    """Reference implementation. a, b: equal-length digit lists, least-significant first.

    Output digit i (i = 0..L, position L is the final carry) depends on inputs back to the
    nearest j < i with a[j] + b[j] != 9, or to the left edge (virtual position -1).
    """
    last_stop, d_star = -1, 0
    for i in range(len(a) + 1):
        if i > 0:
            d_star = max(d_star, i - last_stop)
        if i < len(a) and a[i] + b[i] != 9:
            last_stop = i
    return d_star


def segment_depth(bits: list[int]) -> int:
    """Reference implementation for segmented parity: longest run of non-delimiter tokens."""
    best = run = 0
    for t in bits:
        run = 0 if t == PAR_DELIM else run + 1
        best = max(best, run)
    return best


def _last_marker_scan(is_marker: torch.Tensor) -> torch.Tensor:
    """For each position i, the index of the nearest marker strictly before i (or -1)."""
    B, P = is_marker.shape
    idx = torch.arange(P, device=is_marker.device).expand(B, P)
    marked = torch.where(is_marker, idx, torch.full_like(idx, -1))
    incl = torch.cummax(marked, dim=1).values  # nearest marker at or before i
    excl = torch.cat([torch.full_like(incl[:, :1], -1), incl[:, :-1]], dim=1)
    return excl


def addition_rho(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x: [B, P] tokens (10a+b, PAD); lengths: [B] digit count L. Returns rho [B, P] (0 on pads).

    Valid output positions are 0..L (L+1 of them).
    """
    B, P = x.shape
    pos = torch.arange(P, device=x.device).expand(B, P)
    a, b = x // 10, x % 10
    real = pos < lengths[:, None]
    stop = (~real) | ((a + b) != 9)  # final position L and pads act as stops
    last = _last_marker_scan(stop)
    rho = pos - last
    valid = pos <= lengths[:, None]
    return torch.where(valid, rho, torch.zeros_like(rho))


def parity_rho(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x: [B, P] tokens (0, 1, DELIM, PAD). rho = distance back to the previous delimiter."""
    B, P = x.shape
    pos = torch.arange(P, device=x.device).expand(B, P)
    is_delim = x == PAR_DELIM
    last = _last_marker_scan(is_delim)
    rho = pos - last
    valid = (pos < lengths[:, None]) & (~is_delim)
    return torch.where(valid, rho, torch.zeros_like(rho))


def depth_from_rho(rho: torch.Tensor) -> torch.Tensor:
    return rho.amax(dim=1)


def lower_bound_steps(depth: torch.Tensor | int, r: int) -> torch.Tensor | int:
    if isinstance(depth, int):
        return math.ceil(depth / r)
    return torch.div(depth + r - 1, r, rounding_mode="floor")
