"""Diffusion scheduler supporting linear and cosine beta schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["DiffusionScheduler", "get_timestep_embedding"]


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    schedule: str = "cosine"  # or "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "epsilon"


class DiffusionScheduler(nn.Module):
    def __init__(self, cfg: SchedulerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        betas = self._build_betas(cfg)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def _build_betas(self, cfg: SchedulerConfig) -> torch.Tensor:
        if cfg.schedule == "linear":
            return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps, dtype=torch.float32)
        if cfg.schedule == "cosine":
            steps = cfg.num_train_timesteps
            t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((t / steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.float().clamp(min=1e-8)
        raise ValueError(f"Unknown schedule {cfg.schedule}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.cfg.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        if self.cfg.prediction_type != "epsilon":
            raise NotImplementedError("Only epsilon prediction type is implemented")
        beta_t = self.betas[timestep]
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        pred_x0 = (sample - sqrt_one_minus * model_output) / sqrt_alpha_cumprod
        if timestep == 0:
            return pred_x0
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        sqrt_alpha_t = torch.sqrt(self.alphas[timestep])
        coef_x0 = beta_t * sqrt_alpha_cumprod_prev / (1 - alpha_cumprod_t)
        coef_xt = (1 - beta_t) * sqrt_alpha_t / (1 - alpha_cumprod_t)
        posterior_mean = coef_x0 * pred_x0 + coef_xt * sample
        variance = self.posterior_variance[timestep]
        noise = torch.randn_like(sample)
        return posterior_mean + torch.sqrt(variance) * noise

    def timesteps(self, num_inference_steps: int) -> torch.Tensor:
        stride = self.cfg.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.cfg.num_train_timesteps, stride, dtype=torch.long)
        return torch.flip(timesteps[:num_inference_steps], dims=[0])
