"""2x2 split design, frozen test sets and the training sampler.

Frozen sets are generated with fixed seeds, saved to data/frozen/<task>_<split>.pt, and their
SHA-256 hashes are written to data/frozen/HASHES.json (committed). The training sampler drops any
generated instance whose hash matches a frozen ID set, so ID-test is disjoint from training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .. import DATA_DIR
from ..utils.io import read_json, tensor_sha256, write_json
from . import addition, mazes, parity
from .addition import IGNORE

FROZEN_DIR = DATA_DIR / "frozen"


@dataclass(frozen=True)
class SplitSpec:
    len_range: tuple[int, int]
    depth_range: tuple[int, int]
    n: int
    seed: int


SEQ_SPLITS: dict[str, dict[str, SplitSpec]] = {
    "addition": {
        "id_val": SplitSpec((4, 20), (1, 10), 2000, 1001),
        "id_test": SplitSpec((4, 20), (1, 10), 2000, 1002),
        "ood_val": SplitSpec((24, 32), (12, 16), 1000, 1003),
        "ood_depth": SplitSpec((10, 20), (11, 20), 2000, 1004),
        "ood_length": SplitSpec((64, 256), (1, 10), 1000, 1005),
        "ood_both": SplitSpec((64, 256), (32, 128), 1000, 1006),
    },
    "parity": {
        "id_val": SplitSpec((4, 20), (1, 10), 2000, 2001),
        "id_test": SplitSpec((4, 20), (1, 10), 2000, 2002),
        "ood_val": SplitSpec((24, 32), (12, 16), 1000, 2003),
        "ood_depth": SplitSpec((11, 20), (11, 20), 2000, 2004),
        "ood_length": SplitSpec((64, 256), (1, 10), 1000, 2005),
        "ood_both": SplitSpec((64, 256), (32, 128), 1000, 2006),
    },
}
TRAIN_SPEC = {"len_range": (4, 20), "depth_range": (1, 10)}

MAZE_SPLITS = {
    "id_val": (9, 1000, 3001),
    "id_test": (9, 1000, 3002),
    "ood_13": (13, 500, 3013),
    "ood_17": (17, 500, 3017),
    "ood_21": (21, 500, 3021),
    "ood_25": (25, 500, 3025),
    "ood_33": (33, 500, 3033),
}
MAZE_TRAIN_POOL = (9, 50000, 3999)

ID_SPLITS = ("id_val", "id_test")
SEQ_SPLIT_ORDER = ["id_val", "id_test", "ood_val", "ood_depth", "ood_length", "ood_both"]

SAMPLERS = {"addition": addition.sample_addition, "parity": parity.sample_parity}
TASK_META = {
    "addition": {"vocab": addition.VOCAB, "n_out": addition.N_OUT, "pad": 100},
    "parity": {"vocab": parity.VOCAB, "n_out": parity.N_OUT, "pad": 3},
    "mazes": {"in_channels": mazes.N_IN, "n_out": mazes.N_OUT},
}

_HASH_W = None


def _hash_weights(device) -> torch.Tensor:
    global _HASH_W
    if _HASH_W is None:
        g = torch.Generator().manual_seed(12345)
        _HASH_W = torch.randint(1, 2**62, (1 << 15,), generator=g, dtype=torch.long)
    return _HASH_W.to(device)


def row_hash(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hash each instance from its valid positions (y != IGNORE). Works for any padding width."""
    flat_x = x.flatten(1).long()
    flat_y = y.flatten(1)
    if flat_x.shape[1] == flat_y.shape[1]:
        valid = flat_y != IGNORE
    else:  # images: every input pixel counts
        valid = torch.ones_like(flat_x, dtype=torch.bool)
    w_all = _hash_weights(x.device)
    w = w_all[: flat_x.shape[1]]
    vals = torch.where(valid, (flat_x + 1) * w, torch.zeros_like(flat_x))
    return vals.sum(1) + valid.sum(1) * w_all[-1]


def _frozen_path(task: str, split: str) -> Path:
    return FROZEN_DIR / f"{task}_{split}.pt"


def build_frozen_splits(task: str, force: bool = False) -> dict[str, str]:
    """Generate (or reuse) every frozen split of a task. Returns {split: sha256}."""
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    hashes_path = FROZEN_DIR / "HASHES.json"
    hashes = read_json(hashes_path) if hashes_path.exists() else {}
    out = {}
    if task == "mazes":
        items = {k: v for k, v in MAZE_SPLITS.items()}
    else:
        items = SEQ_SPLITS[task]
    seen = torch.empty(0, dtype=torch.long)
    for split, spec in items.items():
        path = _frozen_path(task, split)
        if path.exists() and not force:
            data = torch.load(path)
        else:
            if task == "mazes":
                n_cells, count, seed = spec
                data = mazes.make_mazes(count, n_cells, seed)
            else:
                g = torch.Generator().manual_seed(spec.seed)
                data = SAMPLERS[task](spec.n, spec.len_range, spec.depth_range, g)
                if split in ID_SPLITS:  # de-duplicate ID-test against ID-val
                    h = row_hash(data["x"], data["y"])
                    keep = ~torch.isin(h, seen)
                    data = {k: v[keep] for k, v in data.items()}
                    seen = torch.cat([seen, h[keep]])
            torch.save(data, path)
        sha = tensor_sha256(data["x"], data["y"])
        out[split] = sha
        hashes[f"{task}/{split}"] = {"sha256": sha, "n": int(data["x"].shape[0])}
    write_json(hashes_path, hashes)
    return out


def load_split(task: str, split: str, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    path = _frozen_path(task, split)
    if not path.exists():
        build_frozen_splits(task)
    data = torch.load(path)
    expected = read_json(FROZEN_DIR / "HASHES.json").get(f"{task}/{split}")
    if expected is not None and tensor_sha256(data["x"], data["y"]) != expected["sha256"]:
        raise RuntimeError(f"frozen split {task}/{split} does not match HASHES.json")
    return {k: v.to(device) for k, v in data.items()}


def split_names(task: str) -> list[str]:
    return list(MAZE_SPLITS) if task == "mazes" else SEQ_SPLIT_ORDER


class TrainSampler:
    """Online training batches on the GPU, with frozen ID instances filtered out."""

    def __init__(self, task: str, device, g: torch.Generator, len_range=None, depth_range=None, maze_size: int = 9):
        self.task, self.device, self.g = task, torch.device(device), g
        if task == "mazes":
            pool_path = DATA_DIR / "mazes" / f"train_pool_{MAZE_TRAIN_POOL[0]}_{MAZE_TRAIN_POOL[1]}.pt"
            if not pool_path.exists():
                pool_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(mazes.make_mazes(MAZE_TRAIN_POOL[1], maze_size, MAZE_TRAIN_POOL[2]), pool_path)
            pool = torch.load(pool_path)
            banned = torch.cat([row_hash(load_split("mazes", s)["x"], load_split("mazes", s)["y"]) for s in ID_SPLITS])
            keep = ~torch.isin(row_hash(pool["x"], pool["y"]), banned)
            self.pool = {k: v[keep].to(self.device) for k, v in pool.items()}
            self.banned = None
        else:
            self.len_range = tuple(len_range or TRAIN_SPEC["len_range"])
            self.depth_range = tuple(depth_range or TRAIN_SPEC["depth_range"])
            banned = []
            for s in ID_SPLITS:
                d = load_split(task, s)
                banned.append(row_hash(d["x"], d["y"]))
            self.banned = torch.cat(banned).to(self.device)
        self.n_filtered = 0
        self._buf: dict[str, torch.Tensor] | None = None
        self._pos = 0

    def _refill(self, n: int) -> None:
        batch = SAMPLERS[self.task](n, self.len_range, self.depth_range, self.g, self.device)
        keep = ~torch.isin(row_hash(batch["x"], batch["y"]), self.banned)
        if not keep.all():
            self.n_filtered += int((~keep).sum())
            batch = {k: v[keep] for k, v in batch.items()}
        self._buf, self._pos = batch, 0

    def sample(self, n: int, chunk: int = 64) -> dict[str, torch.Tensor]:
        if self.task == "mazes":
            idx = torch.randint(0, self.pool["x"].shape[0], (n,), generator=self.g, device=self.device)
            x, y = mazes.augment_dihedral(self.pool["x"][idx].float(), self.pool["y"][idx], self.g)
            return {"x": x, "y": y, "depth": self.pool["depth"][idx], "size": self.pool["size"][idx]}
        if self._buf is None or self._pos + n > self._buf["x"].shape[0]:
            self._refill(n * chunk)
        sl = slice(self._pos, self._pos + n)
        self._pos += n
        batch = {k: v[sl] for k, v in self._buf.items()}
        P = int(batch["length"].max()) + (1 if self.task == "addition" else 0)
        return {k: (v[:, :P] if v.dim() == 2 else v) for k, v in batch.items()}
