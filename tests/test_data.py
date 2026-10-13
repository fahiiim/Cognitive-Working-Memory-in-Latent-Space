import numpy as np
import pytest
import torch

from glwm2.data import mazes
from glwm2.data.addition import IGNORE, decode_numbers, decode_target, sample_addition
from glwm2.data.depth import PAR_DELIM, segment_depth
from glwm2.data.parity import sample_parity
from glwm2.data.splits import SEQ_SPLITS, row_hash


@pytest.mark.parametrize("L_range,d_range", [((4, 20), (1, 10)), ((10, 20), (11, 20)), ((64, 256), (32, 128))])
def test_addition_matches_python_bigints(L_range, d_range):
    g = torch.Generator().manual_seed(0)
    b = sample_addition(300, L_range, d_range, g)
    for (a_int, b_int), y_int in zip(decode_numbers(b["x"], b["length"]), decode_target(b["y"], b["length"])):
        assert a_int + b_int == y_int


@pytest.mark.parametrize("L_range,d_range", [((4, 20), (1, 10)), ((10, 20), (11, 20)), ((64, 256), (32, 128))])
def test_addition_depth_control(L_range, d_range):
    g = torch.Generator().manual_seed(1)
    b = sample_addition(500, L_range, d_range, g)
    assert torch.equal(b["depth"], b["target_depth"])
    assert int(b["depth"].min()) >= d_range[0] and int(b["depth"].max()) <= d_range[1]
    assert int(b["length"].min()) >= L_range[0] and int(b["length"].max()) <= L_range[1]
    assert (b["depth"] <= b["length"] + 1).all()


def test_addition_depth_covers_range():
    g = torch.Generator().manual_seed(2)
    b = sample_addition(4000, (4, 20), (1, 10), g)
    assert set(b["depth"].tolist()) == set(range(1, 11))


@pytest.mark.parametrize("L_range,d_range", [((4, 20), (1, 10)), ((11, 20), (11, 20)), ((64, 256), (32, 128))])
def test_parity_targets_and_depth(L_range, d_range):
    g = torch.Generator().manual_seed(3)
    b = sample_parity(300, L_range, d_range, g)
    assert torch.equal(b["depth"], b["target_depth"])
    for row, yrow, L in zip(b["x"].tolist(), b["y"].tolist(), b["length"].tolist()):
        par = 0
        for t, yt in zip(row[:L], yrow[:L]):
            par = 0 if t == PAR_DELIM else par ^ t
            assert yt == par
        assert all(v == IGNORE for v in yrow[L:])
        assert segment_depth(row[:L]) == max(1, segment_depth(row[:L]))


def test_ood_splits_are_disjoint_by_construction():
    for task, splits in SEQ_SPLITS.items():
        tr_len, tr_depth = (4, 20), (1, 10)
        for name, spec in splits.items():
            if name.startswith("ood"):
                assert spec.len_range[0] > tr_len[1] or spec.depth_range[0] > tr_depth[1], (task, name)


def test_row_hash_ignores_padding():
    g = torch.Generator().manual_seed(4)
    b = sample_addition(50, (4, 8), (1, 5), g)
    extra = 31
    x2 = torch.cat([b["x"], torch.full((50, extra), 100)], dim=1)
    y2 = torch.cat([b["y"], torch.full((50, extra), IGNORE)], dim=1)
    assert torch.equal(row_hash(b["x"], b["y"]), row_hash(x2, y2))
    b3 = sample_addition(50, (4, 8), (1, 5), torch.Generator().manual_seed(4), pad_to=40)
    assert b3["x"].shape[1] == 40 and torch.equal(b3["depth"], b3["target_depth"])


def test_maze_generator_and_depths():
    data = mazes.make_mazes(20, 5, seed=0)
    x, y = data["x"], data["y"]
    assert x.shape == (20, 3, 11, 11) and y.shape == (20, 11, 11)
    # the path is a connected chain of open pixels from start to goal
    assert ((y == 1) <= (x[:, 0] == 1)).all()
    assert (y[x[:, 1] == 1] == 1).all() and (y[x[:, 2] == 1] == 1).all()
    assert torch.equal(data["path_len"], y.flatten(1).sum(1) - 1)
    # dead-end filling leaves exactly the path in a perfect maze
    _, remaining = mazes.dead_end_fill_depth(x[:, 0].bool(), x[:, 1].bool(), x[:, 2].bool())
    assert torch.equal(remaining.long(), y)
    assert (data["depth"] >= 0).all()
