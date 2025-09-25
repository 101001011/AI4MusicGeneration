from __future__ import annotations

import torch

from hs_mad.modules.unet.mrci_unet import MRCIUNet1D, UNetConfig


def test_mrci_unet_forward_shapes():
    cfg = UNetConfig(
        in_channels=8,
        base_channels=16,
        mid_channels=32,
        bottleneck_channels=48,
        time_dim=32,
        style_dim=16,
        cond_dims={"event": 20, "local": 24, "global": 28},
        attention_heads={"event": 4, "local": 4, "global": 4},
        r1=4,
        r2=4,
        scm_layers=["high", "mid"],
    )
    model = MRCIUNet1D(cfg)
    z = torch.randn(2, 8, 64)
    t = torch.randn(2, 32)
    style = torch.randn(2, 16)
    cond = {
        "event": torch.randn(2, 64, 20),
        "local": torch.randn(2, 16, 24),
        "global": torch.randn(2, 4, 28),
    }
    out = model(z, t, cond, style)
    assert out.shape == z.shape


def test_mrci_unet_from_config():
    cfg = {
        "latent_channels": 8,
        "hse": {"d_event": 20, "d_local": 24, "d_global": 28, "r1": 4, "r2": 4},
        "unet": {
            "base_channels": 16,
            "mid_channels": 32,
            "bottleneck_channels": 48,
            "diffusion_embedding_dim": 32,
            "adaln_dim": 16,
            "attention_heads": {"event": 4, "local": 4, "global": 4},
            "scm_layers": ["high", "mid"],
        },
        "style_encoder": {"embedding_dim": 16},
    }
    model = MRCIUNet1D.from_config(cfg)
    assert isinstance(model, MRCIUNet1D)
