"""Diffusion sampler with decoupled CFG support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from hs_mad.modules.guidance.decoupled_cfg import DecoupledCFG

from .scheduler import DiffusionScheduler, get_timestep_embedding

__all__ = ["SamplerConfig", "DiffusionSampler"]


@dataclass
class SamplerConfig:
    num_inference_steps: int = 50
    w_structure: float = 3.0
    w_style: float = 1.5


class DiffusionSampler:
    def __init__(self, scheduler: DiffusionScheduler, cfg: SamplerConfig) -> None:
        self.scheduler = scheduler
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: Dict[str, any], scheduler: DiffusionScheduler) -> "DiffusionSampler":
        sampler_cfg = SamplerConfig(
            num_inference_steps=cfg.get("diffusion_steps", 50),
            w_structure=cfg.get("cfg_weights", {}).get("structure", 3.0),
            w_style=cfg.get("cfg_weights", {}).get("style", 1.5),
        )
        return cls(scheduler, sampler_cfg)

    def sample(
        self,
        model: torch.nn.Module,
        cfg_module: DecoupledCFG,
        codec: Optional[torch.nn.Module],
        z_T: Optional[torch.Tensor],
        structure_cond: Dict[str, torch.Tensor],
        style_cond: Optional[torch.Tensor],
        steps: Optional[int] = None,
        cfg_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        batch_size = next(iter(structure_cond.values())).size(0)
        device = next(iter(structure_cond.values())).device
        if z_T is None:
            latent_channels = model.cfg.in_channels if hasattr(model, "cfg") else structure_cond["event"].size(-1)
            z = torch.randn(batch_size, latent_channels, structure_cond["event"].size(1), device=device)
        else:
            z = z_T
        steps = steps or self.cfg.num_inference_steps
        weights = cfg_weights or {"structure": self.cfg.w_structure, "style": self.cfg.w_style}
        timesteps = self.scheduler.timesteps(steps)
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            time_dim = model.cfg.time_dim if hasattr(model, "cfg") else structure_cond["event"].size(-1)
            t_embed = get_timestep_embedding(t_tensor, time_dim).to(device)
            eps = cfg_module(
                model,
                z,
                t_embed,
                structure_cond,
                style_cond,
                weights["structure"],
                weights["style"],
            )
            z = self.scheduler.step(eps, int(t.item()), z)
        if codec is not None:
            return codec.decode(z)
        return z
