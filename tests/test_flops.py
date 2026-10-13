import torch
import torch.nn as nn

from glwm2.eval.flops import backbone_flops, count_flops
from glwm2.models.backbone import RecurrentBackbone


def test_linear_flops():
    lin = nn.Linear(64, 32)
    x = torch.randn(10, 64)
    assert count_flops(lambda: lin(x)) == 2 * 10 * 64 * 32


def test_backbone_step_flops_match_conv_arithmetic():
    C, P, B = 32, 50, 2
    m = RecurrentBackbone(n_out=10, width=C, vocab=101, convs_per_step=1).eval()
    x = torch.randint(0, 100, (B, P))
    f = backbone_flops(m, x)
    conv3 = 2 * B * P * (2 * C) * C * 3  # recall concatenates h and e
    conv1 = 2 * B * P * C * C
    assert f["step"] == conv3 + conv1
    assert f["decode"] == 2 * B * P * C * C + 2 * B * P * C * 10


def test_v1_manual_gru_matches_cudnn_gru():
    from glwm2.models.v1_glwm import V1GLWM

    m = V1GLWM("fixed").eval()
    a = torch.randint(0, 10, (3, 4))
    b = torch.randint(0, 10, (3, 4))
    with torch.no_grad():
        assert torch.allclose(m.encode(a, b), m.encode_manual(a, b), atol=1e-5)
