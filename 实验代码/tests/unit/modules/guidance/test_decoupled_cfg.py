from __future__ import annotations

import torch

from hs_mad.modules.guidance.decoupled_cfg import DecoupledCFG
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D, UNetConfig


def build_unet():
    cfg = UNetConfig(
        in_channels=4,
        base_channels=8,
        mid_channels=8,
        bottleneck_channels=12,
        time_dim=16,
        style_dim=6,
        cond_dims={"event": 10, "local": 12, "global": 14},
        attention_heads={"event": 2, "local": 2, "global": 2},
        r1=2,
        r2=2,
        scm_layers=["high", "mid"],
    )
    return MRCIUNet1D(cfg)


def test_decoupled_cfg_combination():
    model = build_unet()
    cfg = DecoupledCFG()
    z = torch.randn(1, 4, 16)
    t = torch.randn(1, 16)
    structure = {
        "event": torch.randn(1, 16, 10),
        "local": torch.randn(1, 8, 12),
        "global": torch.randn(1, 4, 14),
    }
    style = torch.randn(1, 6)
    output = cfg(model, z, t, structure, style, w_structure=2.0, w_style=1.0)
    assert output.shape == z.shape
