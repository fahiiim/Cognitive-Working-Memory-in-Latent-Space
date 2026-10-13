"""Flip tests for the depth oracles (Proposition 2).

For every output position i with determining radius rho_i:
- changing inputs strictly left of the window (positions < i - rho_i) never changes y_i;
- the counterexample from the proof (replace everything left of the window by one pair of the
  opposite carry class) does change y_i, so a smaller radius cannot suffice.
"""

import random

import torch

from glwm2.data.addition import sample_addition
from glwm2.data.depth import PAR_DELIM, addition_rho, carry_depth, parity_rho, segment_depth
from glwm2.data.parity import sample_parity


def add_targets(a, b):
    y, c = [], 0
    for ai, bi in zip(a + [0], b + [0]):
        s = ai + bi + c
        y.append(s % 10)
        c = s // 10
    return y


def test_carry_depth_examples_from_plan():
    # least-significant digit first
    assert carry_depth([9, 9, 9, 9], [1, 0, 0, 0]) == 4  # 9999 + 0001
    assert carry_depth([4, 3, 2, 1], [1, 1, 1, 1]) == 1  # 1234 + 1111
    assert carry_depth([5, 4], [4, 5]) == 3  # 45 + 54: no stops, boundary at -1


def test_vectorised_oracle_matches_reference():
    g = torch.Generator().manual_seed(0)
    b = sample_addition(500, (4, 40), (1, 30), g)
    for row, L, d in zip(b["x"].tolist(), b["length"].tolist(), b["depth"].tolist()):
        a = [t // 10 for t in row[:L]]
        bb = [t % 10 for t in row[:L]]
        assert carry_depth(a, bb) == d
    p = sample_parity(500, (4, 40), (1, 30), g)
    for row, L, d in zip(p["x"].tolist(), p["length"].tolist(), p["depth"].tolist()):
        assert segment_depth(row[:L]) == d


def test_addition_flip_outside_window_never_changes_output():
    rng = random.Random(1)
    g = torch.Generator().manual_seed(1)
    b = sample_addition(200, (6, 24), (1, 12), g)
    rho = addition_rho(b["x"], b["length"])
    for k in range(200):
        L = int(b["length"][k])
        row = b["x"][k, :L].tolist()
        a = [t // 10 for t in row]
        bb = [t % 10 for t in row]
        y = add_targets(a, bb)
        for i in range(L + 1):
            left = i - int(rho[k, i])  # nearest stop, or -1
            for _ in range(5):
                a2, b2 = a[:], bb[:]
                for j in range(0, max(left, 0)):
                    a2[j], b2[j] = rng.randrange(10), rng.randrange(10)
                assert add_targets(a2, b2)[i] == y[i]


def test_addition_counterexample_at_smaller_radius():
    g = torch.Generator().manual_seed(2)
    b = sample_addition(200, (6, 24), (2, 12), g)
    rho = addition_rho(b["x"], b["length"])
    for k in range(200):
        L = int(b["length"][k])
        row = b["x"][k, :L].tolist()
        a = [t // 10 for t in row]
        bb = [t % 10 for t in row]
        y = add_targets(a, bb)
        for i in range(1, L + 1):
            s = i - int(rho[k, i])
            carry_in = (y[i] - (a[i] + bb[i] if i < L else 0)) % 10
            new = (5, 5) if carry_in == 0 else (1, 1)
            a2, b2 = [new[0]] + a[s + 1 :], [new[1]] + bb[s + 1 :]
            i2 = i - s
            assert add_targets(a2, b2)[i2] != y[i]


def test_parity_flip_tests():
    rng = random.Random(3)
    g = torch.Generator().manual_seed(3)
    p = sample_parity(200, (6, 24), (2, 12), g)
    rho = parity_rho(p["x"], p["length"])

    def targets(row):
        out, par = [], 0
        for t in row:
            par = 0 if t == PAR_DELIM else par ^ t
            out.append(par)
        return out

    for k in range(200):
        L = int(p["length"][k])
        row = p["x"][k, :L].tolist()
        y = targets(row)
        for i in range(L):
            if row[i] == PAR_DELIM:
                continue
            s = i - int(rho[k, i])
            for _ in range(3):
                row2 = row[:]
                for j in range(0, max(s, 0)):
                    row2[j] = rng.choice([0, 1, PAR_DELIM])
                assert targets(row2)[i] == y[i]
            # counterexample: one bit left of the window, without the delimiter
            row3 = [1] + row[s + 1 :]
            assert targets(row3)[i - s] != y[i]
