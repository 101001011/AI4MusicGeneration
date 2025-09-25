"""Descript Audio Codec wrapper with differentiable fallback implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DACWrapper", "DACConfig"]


@dataclass
class DACConfig:
    sample_rate: int = 44100
    latent_rate: float = 137.8
    latent_channels: int = 8
    stride: int = 320


class DACWrapper(nn.Module):
    """Lightweight wrapper that mimics DAC encode/decode interfaces."""

    def __init__(self, cfg: DACConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_mean", torch.tensor(0.0), persistent=False)

    @property
    def latent_channels(self) -> int:
        return self.cfg.latent_channels

    def latent_frames(self, duration_seconds: float) -> int:
        return int(duration_seconds * self.cfg.latent_rate)

    @classmethod
    def from_config(cls, cfg: DACConfig) -> "DACWrapper":
        return cls(cfg)

    @classmethod
    def from_pretrained(cls) -> "DACWrapper":  # pragma: no cover - runtime convenience
        return cls(DACConfig())

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Project waveform into a latent grid using average pooling."""

        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        latent = F.avg_pool1d(audio, kernel_size=self.cfg.stride, stride=self.cfg.stride)
        repeat_factor = self.cfg.latent_channels
        latent = latent.repeat(1, repeat_factor, 1)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        scale = self.cfg.stride
        audio = F.interpolate(latent, scale_factor=scale, mode="linear", align_corners=False)
        # Average across latent channels to recover mono waveform
        audio = audio.mean(dim=1, keepdim=True)
        return audio
