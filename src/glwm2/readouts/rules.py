"""Stopping rules evaluated in one pass over a recorded trajectory.

Every threshold rule halts at the first step t (1-indexed) at which its statistic crosses the
threshold, or at T_max if it never does. All thresholds of all rules are evaluated at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..eval.metrics import first_stable_correct, halting_metrics

PROB_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
LOG_GRID = [float(v) for v in np.logspace(-6, 0, 19)]


@dataclass
class Rule:
    name: str
    family: str  # reference | posthoc | joint | ours
    stat: str | None
    op: str  # ">" or "<" or ">=" (halt when stat op thr)
    grid: list = field(default_factory=list)
    requires: str | None = None  # stat key that must be present in the trajectory


RULES = [
    Rule("max_softmax", "posthoc", "conf_min", ">", PROB_GRID),
    Rule("entropy", "posthoc", "ent_max", "<", LOG_GRID),
    Rule("stability", "posthoc", "run_len", ">=", [2, 3, 5]),
    Rule("kl", "posthoc", "kl_max", "<", LOG_GRID),
    Rule("hidden_change", "posthoc", "dh", "<", LOG_GRID),
    Rule("trm_post", "posthoc", "trm_post", ">", PROB_GRID, requires="trm_post"),
    Rule("act", "joint", "act_cum", ">=", [0.5, 0.7, 0.9, 0.95, 0.99], requires="act_cum"),
    Rule("ponder", "joint", "ponder_cum", ">=", [0.3, 0.5, 0.7, 0.9, 0.95, 0.99], requires="ponder_cum"),
    Rule("gate", "joint", "lambda", ">", PROB_GRID, requires="lambda"),
    Rule("done_post", "ours", "done_post", ">", PROB_GRID, requires="done_post"),
    Rule("done_prop_post", "ours", "done_prop_post", ">", PROB_GRID, requires="done_prop_post"),
    Rule("done_joint", "ours", "done_joint", ">", PROB_GRID, requires="done_joint"),
]


def first_cross(cond: torch.Tensor) -> torch.Tensor:
    """cond [K, T, B] bool -> halting step [K, B] in 1..T (T if never)."""
    T = cond.shape[1]
    any_ = cond.any(1)
    first = cond.float().argmax(1) + 1
    return torch.where(any_, first, torch.full_like(first, T))


def threshold_steps(stat: torch.Tensor, op: str, grid) -> torch.Tensor:
    thr = torch.tensor(grid, dtype=torch.float32)[:, None, None]
    s = stat.float()[None]
    cond = {">": s > thr, "<": s < thr, ">=": s >= thr}[op]
    return first_cross(cond)


def em_at(em: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """em [T, B], steps [K, B] (1-indexed) -> correctness [K, B] at those steps."""
    idx = (steps.long() - 1).clamp(0, em.shape[0] - 1)
    return torch.gather(em, 0, idx)


def all_rule_outcomes(traj: dict, lb: torch.Tensor, include_fixed: bool = True) -> dict[str, dict]:
    """Returns {rule: {"family", "grid", "steps" [K, B], "correct" [K, B]}} for every applicable rule."""
    em = traj["em"]
    T = em.shape[0]
    out = {}
    if include_fixed:
        grid = list(range(1, T + 1))
        steps = torch.tensor(grid)[:, None].expand(T, em.shape[1])
        out["fixed_T"] = {"family": "reference", "grid": grid, "steps": steps, "correct": em_at(em, steps)}
    lb_c = lb.clamp(1, T)
    out["oracle_lb"] = {"family": "reference", "grid": [None], "steps": lb_c[None], "correct": em_at(em, lb_c[None])}
    first_ok = torch.where(em.any(0), em.float().argmax(0) + 1, torch.full((em.shape[1],), T))
    out["oracle_iter"] = {"family": "reference", "grid": [None], "steps": first_ok[None], "correct": em.any(0)[None]}
    fsc = first_stable_correct(em)
    out["oracle_stable"] = {"family": "reference", "grid": [None], "steps": torch.where(fsc > 0, fsc, torch.full_like(fsc, T))[None],
                            "correct": (fsc > 0)[None]}
    # Fan et al.: run to T_max, answer at the most confident step
    best = traj["logconf_sum"].argmax(0) + 1
    out["max_conf_fan"] = {"family": "posthoc", "grid": [None], "steps": torch.full_like(best, T)[None],
                           "correct": em_at(em, best[None]), "answer_step": best[None]}
    for rule in RULES:
        if rule.stat not in traj:
            continue
        steps = threshold_steps(traj[rule.stat], rule.op, rule.grid)
        out[rule.name] = {"family": rule.family, "grid": list(rule.grid), "steps": steps, "correct": em_at(em, steps)}
    return out


def select_threshold(steps: torch.Tensor, correct: torch.Tensor, tol: float = 0.005) -> int:
    """Pre-registered rule: among thresholds within `tol` of the best accuracy, the fewest mean
    steps; ties go to the later (more conservative) grid entry."""
    acc = correct.float().mean(1)
    ms = steps.float().mean(1)
    ok = acc >= acc.max() - tol - 1e-9
    cand = torch.where(ok, ms, torch.full_like(ms, float("inf")))
    best = cand.min()
    idx = torch.nonzero(cand <= best + 1e-9).flatten()
    return int(idx[-1])


def evaluate_rules(trajs: dict[str, dict], metas: dict[str, dict], val_split: str = "id_val") -> tuple[list[dict], dict]:
    """trajs/metas keyed by split. metas[split] = {lb, depth, length} tensors. Thresholds are chosen
    on `val_split` and applied to every split. Returns (metric rows, per-instance outcomes)."""
    outcomes = {s: all_rule_outcomes(trajs[s], metas[s]["lb"]) for s in trajs}
    chosen = {}
    for rule, o in outcomes[val_split].items():
        chosen[rule] = select_threshold(o["steps"], o["correct"]) if len(o["grid"]) > 1 else 0
    rows, per_instance = [], {}
    for s, outs in outcomes.items():
        meta = metas[s]
        em = trajs[s]["em"]
        fsc = first_stable_correct(em).numpy()
        o_steps = outs["oracle_stable"]["steps"][0].numpy()
        o_corr = outs["oracle_stable"]["correct"][0].numpy().astype(float)
        per_instance[s] = {}
        for rule, o in outs.items():
            k = min(chosen[rule], o["steps"].shape[0] - 1)
            steps = o["steps"][k].numpy()
            corr = o["correct"][k].numpy().astype(float)
            m = halting_metrics(steps, corr, meta["lb"].numpy(), meta["depth"].numpy(), meta["length"].numpy(), fsc,
                                o_steps, o_corr)
            rows.append({"split": s, "rule": rule, "family": o["family"], "threshold": o["grid"][chosen[rule]], **m})
            per_instance[s][rule] = {"steps": steps.astype(np.int16), "correct": corr.astype(np.int8)}
    return rows, per_instance


def tradeoff_curves(trajs: dict[str, dict], metas: dict[str, dict]) -> list[dict]:
    """Accuracy and mean steps for every rule at every threshold, per split (for Fig. 5)."""
    rows = []
    for s, traj in trajs.items():
        outs = all_rule_outcomes(traj, metas[s]["lb"])
        lb = metas[s]["lb"]
        for rule, o in outs.items():
            for k, thr in enumerate(o["grid"]):
                st = o["steps"][k]
                rows.append({"split": s, "rule": rule, "family": o["family"], "threshold": thr,
                             "acc": float(o["correct"][k].float().mean()), "mean_steps": float(st.float().mean()),
                             "premature": float((st < lb).float().mean())})
    return rows
