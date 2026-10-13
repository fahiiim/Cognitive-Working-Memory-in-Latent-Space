"""Executable checks for paper/sections/theory.tex."""

import pytest
import torch

from glwm2.data.addition import IGNORE, sample_addition
from glwm2.data.depth import ADD_PAD, addition_rho, parity_rho
from glwm2.data.parity import sample_parity
from glwm2.models.backbone import RecurrentBackbone
from glwm2.theory import addition_automaton, parity_automaton


def _states(model, x, T):
    e = model.encode(x)
    hs = [model.init_state(e)]
    for o in model.rollout(x, T):
        hs.append(o.h)
    return hs


@pytest.mark.parametrize("r", [1, 2])
def test_light_cone_flip_1d(r):
    """Lemma 1: inputs outside the radius-rt window never change h_i(t)."""
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = RecurrentBackbone(n_out=10, width=16, vocab=101, convs_per_step=r).double().eval()
    P, T = 40, 6
    x = torch.randint(0, 100, (1, P), generator=g)
    with torch.no_grad():
        base = _states(model, x, T)
        for t in range(T + 1):
            for i in (0, 7, 20, P - 1):
                far = (torch.arange(P) - i).abs() > r * t
                if not far.any():
                    continue
                x2 = x.clone()
                x2[0, far] = torch.randint(0, 100, (int(far.sum()),), generator=g)
                h2 = _states(model, x2, t)[t]
                assert torch.equal(h2[0, :, i], base[t][0, :, i]), (t, i)
        # sanity: an input inside the cone does change the state
        x3 = x.clone()
        x3[0, 20 - r * 3] = (x3[0, 20 - r * 3] + 1) % 100
        h3 = _states(model, x3, 3)[3]
        assert not torch.allclose(h3[0, :, 20], base[3][0, :, 20])


def test_light_cone_flip_2d():
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    model = RecurrentBackbone(n_out=2, width=8, in_channels=3, dim=2, convs_per_step=2).double().eval()
    x = torch.rand(1, 3, 21, 21, generator=g, dtype=torch.float64)
    T = 3
    with torch.no_grad():
        base = _states(model, x, T)
        ii, jj = torch.meshgrid(torch.arange(21), torch.arange(21), indexing="ij")
        for t in range(T + 1):
            far = torch.maximum((ii - 10).abs(), (jj - 10).abs()) > 2 * t
            x2 = x.clone()
            x2[0, :, far] = torch.rand(3, int(far.sum()), generator=g, dtype=torch.float64)
            h2 = _states(model, x2, t)[t]
            assert torch.equal(h2[0, :, 10, 10], base[t][0, :, 10, 10])


def _addition_targets(x, L):
    a, b = x // 10, x % 10
    y, c = [], 0
    for i in range(L + 1):
        s = int(a[i]) + int(b[i]) + c
        y.append(s % 10)
        c = s // 10 if i < L else 0
    return y


def test_pair_construction_lower_bound():
    """Theorem 1: the proof's pair (x', i') has the same prediction at step t < rho_i but a
    different target."""
    g = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    model = RecurrentBackbone(n_out=10, width=16, vocab=101).double().eval()
    batch = sample_addition(64, (6, 16), (3, 8), g)
    rho = addition_rho(batch["x"], batch["length"])
    checked = 0
    for k in range(64):
        L = int(batch["length"][k])
        x = batch["x"][k, : L + 1]
        i = int(rho[k].argmax())
        s = i - int(rho[k, i])  # nearest stop before i, or -1
        y = _addition_targets(x, L)
        carry_in = (y[i] - (int(x[i]) // 10 + int(x[i]) % 10)) % 10 if i < L else y[i]
        new_pair = torch.tensor([55 if carry_in == 0 else 11])  # 5+5 >= 10 or 1+1 < 9
        x_prime = torch.cat([new_pair, x[s + 1 :]])
        i_prime = i - s
        y_prime = _addition_targets(x_prime, len(x_prime) - 1)
        assert y_prime[i_prime] != y[i]
        with torch.no_grad():
            for t in range(1, int(rho[k, i])):
                yh = list(model.rollout(x[None], t))[-1].logits[0, i]
                yh2 = list(model.rollout(x_prime[None], t))[-1].logits[0, i_prime]
                assert torch.allclose(yh, yh2, atol=1e-10, rtol=0)
        checked += 1
    assert checked == 64


@pytest.mark.parametrize("L_range,d_range", [((4, 20), (1, 10)), ((64, 256), (32, 128)), ((400, 512), (1, 400))])
def test_addition_automaton_halts_at_bound(L_range, d_range):
    """Proposition 3 for addition: halts exactly at d*, and committed answers are correct."""
    g = torch.Generator().manual_seed(3)
    batch = sample_addition(200, L_range, d_range, g)
    x, y, L, d = batch["x"], batch["y"], batch["length"], batch["depth"]
    t_max = int(d.max()) + 3
    done, preds, halt = addition_automaton(x, L, t_max)
    assert torch.equal(halt, d)
    valid = y != IGNORE
    for k in range(x.shape[0]):
        assert torch.equal(preds[halt[k], k][valid[k]], y[k][valid[k]])
    # commit-on-done: once a position is done its prediction is correct from then on
    committed_wrong = done & (preds != y[None]) & valid[None]
    assert not committed_wrong.any()
    # each position is done exactly at rho_i
    rho = addition_rho(x, L)
    first_done = done.float().argmax(0)
    assert torch.equal(first_done[valid], rho[valid])


@pytest.mark.parametrize("L_range,d_range", [((4, 20), (1, 10)), ((64, 256), (32, 128)), ((400, 512), (1, 400))])
def test_parity_automaton_halts_at_bound(L_range, d_range):
    g = torch.Generator().manual_seed(4)
    batch = sample_parity(200, L_range, d_range, g)
    x, y, L, d = batch["x"], batch["y"], batch["length"], batch["depth"]
    done, preds, halt = parity_automaton(x, L, int(d.max()) + 3)
    assert torch.equal(halt, d)
    valid = y != IGNORE
    for k in range(x.shape[0]):
        assert torch.equal(preds[halt[k], k][valid[k]], y[k][valid[k]])
    rho = parity_rho(x, L)
    first_done = done.float().argmax(0)
    assert torch.equal(first_done[valid], rho[valid])


def test_length_only_rule_corollary():
    """Corollary 2: at each length the all-nines input has d* = L + 1."""
    for L in (1, 5, 37, 256):
        x = torch.tensor([[45] * L + [0]])
        rho = addition_rho(x, torch.tensor([L]))
        assert int(rho.max()) == L + 1
    assert ADD_PAD == 100
