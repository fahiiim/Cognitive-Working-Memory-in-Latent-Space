"""One run = train (or resume) -> fit post-hoc heads -> trajectories on every frozen split ->
all stopping rules with thresholds chosen on ID-val -> metrics files under results/runs/<run_id>/."""

from __future__ import annotations

import time

import torch

from ..data.splits import load_split, split_names
from ..eval.metrics import ece_from_bins
from ..eval.trajectory import collect_trajectory, t_max_for
from ..readouts.rules import evaluate_rules, tradeoff_curves
from ..utils.config import RunConfig
from ..utils.io import read_json, write_json
from .posthoc import fit_posthoc_heads, make_posthoc_heads
from .trainer import load_model, run_dirs, train


def split_meta(task: str, data: dict, r: int) -> dict:
    depth = data["depth"].cpu()
    length = (data["size"] if task == "mazes" else data["length"]).cpu()
    lb = torch.div(depth + r - 1, r, rounding_mode="floor").clamp_min(1)
    return {"depth": depth, "length": length, "lb": lb}


def get_posthoc_heads(model, cfg: RunConfig, device, iters: int = 1500):
    ckpt_dir, res_dir = run_dirs(cfg)
    path = ckpt_dir / "posthoc.pt"
    heads = make_posthoc_heads(cfg.width, model.dim, device)
    if path.exists():
        st = torch.load(path, map_location=device)
        for k, h in heads.items():
            h.load_state_dict(st[k])
            h.eval()
        return heads
    heads, info = fit_posthoc_heads(model, cfg, iters=iters, device=device,
                                    batch_size=cfg.batch_size if cfg.task == "mazes" else 256)
    torch.save({k: h.state_dict() for k, h in heads.items()}, path)
    write_json(res_dir / "posthoc_log.json", info)
    return heads


def evaluate_run(cfg: RunConfig, splits: list[str] | None = None, force: bool = False, device=None,
                 posthoc_iters: int = 1500, keep_traj: bool = False) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir, res_dir = run_dirs(cfg)
    done_flag = res_dir / "metrics.json"
    if done_flag.exists() and not force:
        return read_json(done_flag)
    t0 = time.time()
    model = load_model(cfg, device)
    heads = get_posthoc_heads(model, cfg, device, posthoc_iters)
    splits = splits or split_names(cfg.task)
    trajs, metas, step_acc, calib, pos_done = {}, {}, {}, {}, {}
    for s in splits:
        data = load_split(cfg.task, s)
        t_max = t_max_for(data, cfg.convs_per_step, cfg.t_max_extra)
        traj = collect_trajectory(model, data, t_max, posthoc=heads, task=cfg.task, keep_preds=keep_traj)
        trajs[s] = traj
        metas[s] = split_meta(cfg.task, data, cfg.convs_per_step)
        step_acc[s] = traj["em"].float().mean(1).tolist()
        calib[s] = {k: ece_from_bins(v["sum"], v["count"]) for k, v in traj["calib"].items()}
        pos_done[s] = traj["pos_done"]
    rows, per_inst = evaluate_rules(trajs, metas)
    curves = tradeoff_curves(trajs, metas)
    base = {k: v for k, v in cfg.to_dict().items() if k != "tags"}
    rows = [{**base, "run_id": cfg.run_id(), **r} for r in rows]
    write_json(res_dir / "curves.json", curves)
    write_json(res_dir / "step_acc.json", step_acc)
    write_json(res_dir / "calibration.json", calib)
    torch.save({"per_instance": per_inst, "metas": metas, "pos_done": pos_done,
                "stats": {s: {k: v for k, v in t.items() if isinstance(v, torch.Tensor) and k != "preds"} for s, t in trajs.items()}},
               res_dir / "per_instance.pt")
    if keep_traj:
        torch.save(trajs, res_dir / "trajectories.pt")
    write_json(done_flag, rows)
    write_json(res_dir / "eval_summary.json", {"eval_seconds": time.time() - t0})
    return rows


def run(cfg: RunConfig, verbose: bool = False, **eval_kw) -> list[dict]:
    train(cfg, save=True, verbose=verbose)
    return evaluate_run(cfg, **eval_kw)
