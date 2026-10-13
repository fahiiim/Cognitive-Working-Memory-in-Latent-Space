"""Shared analysis for notebooks 04, 05, 07, 08 and 09: the stopping-rule entries of the paper,
hypothesis and gate checks, and figure helpers."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from ..utils.config import RunConfig
from .results import finished, instance_frame, metric_rows
from .stats import bootstrap_stat_ci, clustered_bootstrap_diff, partial_spearman

# (entry name) -> (training method, rule evaluated on that model, family, label)
ENTRIES = OrderedDict(
    [
        ("fixed_T", ("base", "fixed_T", "reference", "Fixed $T$")),
        ("oracle_lb", ("base", "oracle_lb", "reference", "Oracle lower bound")),
        ("oracle_stable", ("base", "oracle_stable", "reference", "Oracle (stable)")),
        ("oracle_iter", ("base", "oracle_iter", "reference", "Oracle (any step)")),
        ("max_softmax", ("base", "max_softmax", "posthoc", "Max-softmax")),
        ("entropy", ("base", "entropy", "posthoc", "Entropy")),
        ("stability", ("base", "stability", "posthoc", "$k$-step stability")),
        ("kl", ("base", "kl", "posthoc", "KL between steps")),
        ("hidden_change", ("base", "hidden_change", "posthoc", "Hidden-state change")),
        ("max_conf_fan", ("base", "max_conf_fan", "posthoc", "Max confidence (Fan et al.)")),
        ("trm_post", ("base", "trm_post", "posthoc", "Correctness head (post hoc)")),
        ("act", ("act", "act", "joint", "ACT")),
        ("ponder", ("ponder", "ponder", "joint", "PonderNet")),
        ("trm_joint", ("trm_joint", "gate", "joint", "Correctness head (joint)")),
        ("v1gate", ("v1gate", "gate", "joint", "v1 GLWM gate")),
        ("done_post", ("base", "done_post", "ours", "Completion head (post hoc)")),
        ("done_prop_post", ("base", "done_prop_post", "ours", "Propagating head (post hoc)")),
        ("done_joint", ("done_joint", "done_joint", "ours", "Completion head (joint)")),
        ("done_prop_joint", ("done_prop_joint", "done_joint", "ours", "Propagating head (joint)")),
    ]
)
BASELINES = [k for k, v in ENTRIES.items() if v[2] in ("posthoc", "joint")]
OURS = [k for k, v in ENTRIES.items() if v[2] == "ours"]
POOLED_JOINT = ["act", "ponder", "trm_joint", "v1gate"]
CONF_RULES = ["max_softmax", "entropy", "stability", "kl", "hidden_change"]


def entry_rows(cfgs: list[RunConfig]) -> pd.DataFrame:
    """Metric rows for every entry: one row per (entry, seed, split)."""
    df = metric_rows(cfgs)
    if df.empty:
        return df
    out = []
    for name, (method, rule, fam, label) in ENTRIES.items():
        sub = df[(df.method == method) & (df.rule == rule)].copy()
        sub["entry"], sub["family"], sub["label"] = name, fam, label
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def entry_instances(cfgs: list[RunConfig], split: str, entries=None) -> pd.DataFrame:
    """Per-instance outcomes for each entry (all finished seeds)."""
    entries = entries or list(ENTRIES)
    frames = []
    for c in finished(cfgs):
        wanted = {n: r for n, (m, r, _, _) in ENTRIES.items() if m == c.method and n in entries}
        if not wanted:
            continue
        inst = instance_frame(c, split, list(set(wanted.values())))
        for name, rule in wanted.items():
            sub = inst[inst.rule == rule].copy()
            sub["entry"] = name
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary_table(rows: pd.DataFrame, splits, metric: str) -> pd.DataFrame:
    t = rows[rows.split.isin(splits)].pivot_table(index="entry", columns="split", values=metric, aggfunc=["mean", "std"])
    return t.reindex([e for e in ENTRIES if e in t.index])


def premature_ci(inst: pd.DataFrame, entry: str, seed=0) -> tuple[float, float, float]:
    sub = inst[inst.entry == entry]
    return bootstrap_stat_ci(lambda s, lb: float(np.mean(s < lb)), [sub.steps.values, sub.lb.values], seed=seed)


def premature_error_ci(inst: pd.DataFrame, entry: str, seed=0) -> tuple[float, float, float]:
    """Share of instances halted before the lower bound AND answered wrongly. Halting before the
    bound means the answer is not certified (Theorem 1); this measures how often that costs accuracy."""
    sub = inst[inst.entry == entry]
    return bootstrap_stat_ci(lambda s, lb, c: float(np.mean((s < lb) & (c < 0.5))),
                             [sub.steps.values, sub.lb.values, sub.correct.values], seed=seed)


def length_leakage_ci(inst: pd.DataFrame, entry: str, seed=0) -> tuple[float, float, float]:
    sub = inst[inst.entry == entry]
    return bootstrap_stat_ci(lambda s, L, d: partial_spearman(s, L, d), [sub.steps.values, sub.length.values, sub.depth.values],
                             n_boot=1000, seed=seed)


def g2_check(inst_deep: pd.DataFrame, inst_len: pd.DataFrame, min_premature=0.05, min_leak=0.1) -> pd.DataFrame:
    """inst_deep: OOD-depth (+ OOD-both) instances; inst_len: ID-test + OOD-length instances."""
    rows = []
    for e in BASELINES:
        if e not in set(inst_deep.entry):
            continue
        p, plo, phi = premature_ci(inst_deep, e)
        q, qlo, qhi = premature_error_ci(inst_deep, e)
        r = {"entry": e, "family": ENTRIES[e][2], "premature": p, "premature_lo": plo, "premature_hi": phi,
             "premature_err": q, "premature_err_lo": qlo, "premature_err_hi": qhi,
             "acc": float(inst_deep[inst_deep.entry == e].correct.mean())}
        if e in set(inst_len.entry):
            l, llo, lhi = length_leakage_ci(inst_len, e)
            r.update(leak=l, leak_lo=llo, leak_hi=lhi)
        r["shows_premature"] = p >= min_premature and plo > 0
        r["premature_costs_acc"] = q >= min_premature and qlo > 0
        r["shows_leak"] = bool(r.get("leak", 0) >= min_leak and r.get("leak_lo", 0) > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def g3_check(inst: pd.DataFrame, min_acc_gain=0.10, min_prem_reduction=0.5) -> dict:
    """inst: OOD-depth per-instance outcomes for every entry and seed. Best ours vs best baseline,
    paired per instance within seed, clustered bootstrap over seeds and instances."""
    acc = inst.groupby("entry").correct.mean()
    ours = [e for e in OURS if e in acc.index]
    base = [e for e in BASELINES if e in acc.index and e != "max_conf_fan"]
    if not ours or not base:
        return {}
    best_ours = max(ours, key=lambda e: acc[e])
    best_base = max(base, key=lambda e: acc[e])
    a_by, b_by, pa_by, pb_by = [], [], [], []
    seeds = sorted(set(inst[inst.entry == best_ours].seed) & set(inst[inst.entry == best_base].seed))
    for s in seeds:
        A = inst[(inst.entry == best_ours) & (inst.seed == s)].sort_values("idx")
        B = inst[(inst.entry == best_base) & (inst.seed == s)].sort_values("idx")
        a_by.append(A.correct.values); b_by.append(B.correct.values)
        pa_by.append((A.steps < A.lb).values.astype(float)); pb_by.append((B.steps < B.lb).values.astype(float))
    d_acc = clustered_bootstrap_diff(a_by, b_by, n_boot=2000)
    d_prem = clustered_bootstrap_diff(pa_by, pb_by, n_boot=2000)
    prem_b = float(np.mean(np.concatenate(pb_by)))
    prem_a = float(np.mean(np.concatenate(pa_by)))
    passes_acc = d_acc[0] >= min_acc_gain and d_acc[1] > 0
    passes_prem = prem_b > 0 and (prem_b - prem_a) / prem_b >= min_prem_reduction and d_prem[2] < 0
    return {"best_ours": best_ours, "best_baseline": best_base, "seeds": seeds,
            "acc_ours": float(acc[best_ours]), "acc_base": float(acc[best_base]), "acc_diff": d_acc,
            "prem_ours": prem_a, "prem_base": prem_b, "prem_diff": d_prem,
            "passes_acc": bool(passes_acc), "passes_premature": bool(passes_prem), "G3": bool(passes_acc or passes_prem)}


def binned(inst: pd.DataFrame, col: str, bins, value: str = "steps") -> pd.DataFrame:
    d = inst.copy()
    d["bin"] = pd.cut(d[col], bins, include_lowest=True)
    g = d.groupby(["entry", "bin"], observed=True)[value].mean().reset_index()
    g["x"] = g["bin"].apply(lambda b: b.right)
    return g
