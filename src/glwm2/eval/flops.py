"""Measured FLOPs with torch.utils.flop_counter.FlopCounterMode (no hand-derived formulas)."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_flops(fn: Callable[[], object]) -> int:
    """Total FLOPs (multiply-add = 2) of the ops run inside fn()."""
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            fn()
    return int(fc.get_total_flops())


def backbone_flops(model, x: torch.Tensor) -> dict[str, int]:
    """FLOPs for encoding + init, one recurrent step, and one decode, for a single input batch."""
    e = model.encode(x)
    h = model.init_state(e)
    return {
        "encode": count_flops(lambda: model.init_state(model.encode(x))),
        "step": count_flops(lambda: model.step(h, e)),
        "decode": count_flops(lambda: model.decode(h)),
    }


def module_flops(module, *args) -> int:
    return count_flops(lambda: module(*args))
