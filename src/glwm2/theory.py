"""Hand-built local completion automata (Proposition 3 in paper/sections/theory.tex).

Each position keeps a pointwise state and reads only its left and right neighbours per step
(r = 1). A position is "done" once its answer is certified; the input halts when every position
is done. The automata are size-invariant: the same update works for every length.
"""

from __future__ import annotations

import torch

from .data.depth import ADD_PAD, PAR_DELIM, PAR_PAD

UNKNOWN = -1


def addition_automaton(x: torch.Tensor, lengths: torch.Tensor, t_max: int):
    """x: [B, P] addition tokens. Returns (done [T+1, B, P], preds [T+1, B, P], halt [B]).

    State per position: carry-in c in {0, 1, UNKNOWN}. Carry-out of position j is known when
    a_j + b_j != 9 (it is (a_j + b_j) >= 10) or when a_j + b_j == 9 and c_j is known (it is c_j).
    The left boundary behaves as a position whose carry-out is 0.
    """
    B, P = x.shape
    a, b = x // 10, x % 10
    s = a + b
    pos = torch.arange(P, device=x.device).expand(B, P)
    valid = pos <= lengths[:, None]
    c = torch.full((B, P), UNKNOWN, dtype=torch.long, device=x.device)
    done_hist, pred_hist = [], []

    def record(c):
        done = (c != UNKNOWN) | ~valid
        pred = torch.where(c == UNKNOWN, s % 10, (s + c) % 10)
        done_hist.append(done)
        pred_hist.append(torch.where(valid, pred, torch.zeros_like(pred)))

    record(c)
    for _ in range(t_max):
        carry_out = torch.where(s != 9, (s >= 10).long(), c)  # UNKNOWN if s == 9 and c unknown
        carry_out = torch.where(x == ADD_PAD, torch.zeros_like(carry_out), carry_out)
        left = torch.cat([torch.zeros_like(carry_out[:, :1]), carry_out[:, :-1]], dim=1)
        c = torch.where(c == UNKNOWN, left, c)
        record(c)
    done = torch.stack(done_hist)
    preds = torch.stack(pred_hist)
    all_done = done.all(dim=2)  # [T+1, B]
    halt = torch.where(all_done.any(0), all_done.float().argmax(0), torch.full((B,), t_max, device=x.device))
    return done, preds, halt


def parity_automaton(x: torch.Tensor, lengths: torch.Tensor, t_max: int):
    """Same interface for segmented parity. State: prefix parity of the segment up to i, or UNKNOWN.

    Delimiters are known (value 0) from step 0; the left boundary acts as a known 0.
    """
    B, P = x.shape
    pos = torch.arange(P, device=x.device).expand(B, P)
    valid = pos < lengths[:, None]
    is_delim = x == PAR_DELIM
    bit = torch.where((x == 0) | (x == 1), x, torch.zeros_like(x))
    p = torch.where(is_delim | (x == PAR_PAD), torch.zeros_like(x), torch.full_like(x, UNKNOWN))
    done_hist, pred_hist = [], []

    def record(p):
        done = (p != UNKNOWN) | ~valid
        pred = torch.where(p == UNKNOWN, bit, p)
        done_hist.append(done)
        pred_hist.append(torch.where(valid, pred, torch.zeros_like(pred)))

    record(p)
    for _ in range(t_max):
        left = torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], dim=1)
        new = torch.where(left == UNKNOWN, left, left ^ bit)
        p = torch.where(p == UNKNOWN, new, p)
        record(p)
    done = torch.stack(done_hist)
    preds = torch.stack(pred_hist)
    all_done = done.all(dim=2)
    halt = torch.where(all_done.any(0), all_done.float().argmax(0), torch.full((B,), t_max, device=x.device))
    return done, preds, halt
