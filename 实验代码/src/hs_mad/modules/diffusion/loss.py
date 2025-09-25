"""Diffusion training loss for HS-MAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheduler import DiffusionScheduler, get_timestep_embedding

__all__ = ["DiffusionLoss", "LossOutput"]


@dataclass
class LossOutput:
    loss: torch.Tensor
    pred_noise: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


class DiffusionLoss(nn.Module):
    def __init__(self, scheduler: DiffusionScheduler, time_dim: int) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.time_dim = time_dim

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        structure_cond: Dict[str, torch.Tensor],
        style_cond: Optional[torch.Tensor],
    ) -> LossOutput:
        noise = torch.randn_like(x0)
        timesteps = self.scheduler.sample_timesteps(x0.size(0), x0.device)
        noisy = self.scheduler.add_noise(x0, noise, timesteps)
        t_embed = get_timestep_embedding(timesteps, self.time_dim).to(x0.device)
        pred_noise = model(noisy, t_embed, structure_cond, style_cond)
        loss = F.mse_loss(pred_noise, noise)
        return LossOutput(loss=loss, pred_noise=pred_noise, noise=noise, timesteps=timesteps)
