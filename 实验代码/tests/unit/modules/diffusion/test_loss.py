from __future__ import annotations

import torch

from hs_mad.modules.diffusion.loss import DiffusionLoss
from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D, UNetConfig


def build_dummy_unet() -> MRCIUNet1D:
    cfg = UNetConfig(
        in_channels=4,
        base_channels=8,
        mid_channels=12,
        bottleneck_channels=16,
        time_dim=16,
        style_dim=8,
        cond_dims={"event": 10, "local": 12, "global": 14},
        attention_heads={"event": 2, "local": 2, "global": 2},
        r1=2,
        r2=2,
        scm_layers=["high", "mid"],
    )
    return MRCIUNet1D(cfg)


def test_diffusion_loss_forward():
    scheduler = DiffusionScheduler(SchedulerConfig(num_train_timesteps=20))
    model = build_dummy_unet()
    loss_module = DiffusionLoss(scheduler, time_dim=model.cfg.time_dim)

    x0 = torch.randn(2, 4, 16)
    cond = {
        "event": torch.randn(2, 16, 10),
        "local": torch.randn(2, 8, 12),
        "global": torch.randn(2, 4, 14),
    }
    style = torch.randn(2, 8)
    output = loss_module(model, x0, cond, style)
    assert output.loss.requires_grad
    output.loss.backward()
    assert model.out_proj.weight.grad is not None
