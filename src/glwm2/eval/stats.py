"""Statistics: bootstrap CIs, paired tests, Holm correction, (partial) Spearman correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    """Percentile bootstrap CI for `stat` of a 1-D sample. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1) if stat in (np.mean, np.median) else np.array([stat(v[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(a - b) over paired per-instance values (same instances, two rules)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


def clustered_bootstrap_diff(a_by_seed, b_by_seed, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Two-level bootstrap over seeds then instances, for per-instance values from several seeds.

    a_by_seed, b_by_seed: lists (one entry per seed) of paired per-instance arrays.
    """
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(a, float) - np.asarray(b, float) for a, b in zip(a_by_seed, b_by_seed)]
    point = float(np.mean([d.mean() for d in diffs]))
    boots = np.empty(n_boot)
    S = len(diffs)
    for k in range(n_boot):
        seeds = rng.integers(0, S, size=S)
        boots[k] = np.mean([diffs[s][rng.integers(0, diffs[s].size, size=diffs[s].size)].mean() for s in seeds])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def wilcoxon_seeds(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value on per-seed paired values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    try:
        return float(sps.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction. Returns {name: {p, p_adj, reject}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = {"p": p, "p_adj": running, "reject": running < alpha}
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(sps.spearmanr(x, y).statistic)


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z (via rank correlations)."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / den)


def bootstrap_stat_ci(fn, arrays, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for an arbitrary statistic of several aligned arrays (resampling rows)."""
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    point = fn(*arrays)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = fn(*[a[idx] for a in arrays])
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
