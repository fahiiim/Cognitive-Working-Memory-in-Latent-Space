"""Wall-clock latency with CUDA events."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def cuda_time_ms(fn: Callable[[], object], repeats: int = 20, warmup: int = 3) -> dict[str, float]:
    """Median and IQR of fn() latency in milliseconds, measured with CUDA events."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    t = np.asarray(times)
    return {"median_ms": float(np.median(t)), "q25_ms": float(np.quantile(t, 0.25)), "q75_ms": float(np.quantile(t, 0.75))}
