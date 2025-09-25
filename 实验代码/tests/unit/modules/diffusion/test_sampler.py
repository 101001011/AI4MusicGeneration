from __future__ import annotations

import torch

from hs_mad.modules.diffusion.sampler import DiffusionSampler, SamplerConfig
from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig
from hs_mad.modules.guidance.decoupled_cfg import DecoupledCFG
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D, UNetConfig


def build_components():
    scheduler = DiffusionScheduler(SchedulerConfig(num_train_timesteps=20))
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
    model = MRCIUNet1D(cfg)
    sampler = DiffusionSampler(scheduler, SamplerConfig(num_inference_steps=5))
    return scheduler, model, sampler


def test_diffusion_sampler_runs():
    scheduler, model, sampler = build_components()
    cfg_module = DecoupledCFG()
    z_T = torch.randn(1, 4, 16)
    structure = {
        "event": torch.randn(1, 16, 10),
        "local": torch.randn(1, 8, 12),
        "global": torch.randn(1, 4, 14),
    }
    style = torch.randn(1, 6)
    out = sampler.sample(model, cfg_module, codec=None, z_T=z_T, structure_cond=structure, style_cond=style)
    assert out.shape == z_T.shape
