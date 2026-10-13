import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy and torch. Returns a torch.Generator on the default device.

    deterministic=True turns on cuDNN determinism and deterministic algorithms; it is
    slower and is only used by the determinism test and final reproductions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return make_generator(seed)


def make_generator(seed: int, device: str | torch.device | None = None) -> torch.Generator:
    device = torch.device(device or default_device())
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
