import torch

from glwm2.train.trainer import train
from glwm2.utils.config import RunConfig


def _tiny(device):
    cfg = RunConfig(task="addition", iters=8, batch_size=16, width=16, t_train=4, warmup=2, bf16=False)
    return train(cfg, save=False, deterministic=True, device=device)["model"]


def test_cpu_training_is_deterministic():
    m1, m2 = _tiny("cpu"), _tiny("cpu")
    for p, q in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(p, q)


def test_config_run_id_is_stable():
    a = RunConfig(task="parity", seed=3)
    b = RunConfig(task="parity", seed=3, tags={"note": "tags do not change the id"})
    assert a.run_id() == b.run_id()
    assert a.run_id() != RunConfig(task="parity", seed=4).run_id()
