"""Multi-resolution conditioned 1D U-Net with SCM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


import torch.nn as nn

from hs_mad.modules.unet.attention import StandardCrossAttention

from .blocks import Downsample1D, ResBlock1D, Upsample1D
from .scm import SCMDecoderBlock

__all__ = ["MRCIUNet1D", "UNetConfig"]


@dataclass
class UNetConfig:
    in_channels: int
    base_channels: int
    mid_channels: int
    bottleneck_channels: int
    time_dim: int
    style_dim: int
    cond_dims: Dict[str, int]
    attention_heads: Dict[str, int]
    r1: int
    r2: int
    scm_layers: Optional[list[str]] = None


class MRCIUNet1D(nn.Module):
    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Conv1d(cfg.in_channels, cfg.base_channels, kernel_size=1)
        self.encoder_high = ResBlock1D(cfg.base_channels, cfg.base_channels, cfg.time_dim)
        self.down1 = Downsample1D(cfg.base_channels, cfg.mid_channels, cfg.r1)
        self.encoder_mid = ResBlock1D(cfg.mid_channels, cfg.mid_channels, cfg.time_dim)
        self.down2 = Downsample1D(cfg.mid_channels, cfg.bottleneck_channels, cfg.r2)
        self.bottleneck = ResBlock1D(cfg.bottleneck_channels, cfg.bottleneck_channels, cfg.time_dim)
        self.global_proj = nn.Linear(cfg.cond_dims["global"], cfg.bottleneck_channels)
        self.global_attn = StandardCrossAttention(
            query_dim=cfg.bottleneck_channels,
            kv_dim=cfg.bottleneck_channels,
            n_heads=cfg.attention_heads.get("global", 8),
        )
        self.upsample_mid = Upsample1D(cfg.bottleneck_channels, cfg.mid_channels, cfg.r2)
        self.decoder_mid = SCMDecoderBlock(
            c_in=cfg.mid_channels,
            c_skip=cfg.mid_channels,
            c_out=cfg.mid_channels,
            cond_dim=cfg.cond_dims["local"],
            time_dim=cfg.time_dim,
            style_dim=cfg.style_dim,
            n_heads=cfg.attention_heads.get("local", 8),
            enable_scm=(cfg.scm_layers is None or "mid" in cfg.scm_layers),
        )
        self.upsample_high = Upsample1D(cfg.mid_channels, cfg.base_channels, cfg.r1)
        self.decoder_high = SCMDecoderBlock(
            c_in=cfg.base_channels,
            c_skip=cfg.base_channels,
            c_out=cfg.base_channels,
            cond_dim=cfg.cond_dims["event"],
            time_dim=cfg.time_dim,
            style_dim=cfg.style_dim,
            n_heads=cfg.attention_heads.get("event", 8),
            enable_scm=(cfg.scm_layers is None or "high" in cfg.scm_layers),
        )
        self.out_proj = nn.Conv1d(cfg.base_channels, cfg.in_channels, kernel_size=1)

    def forward(
        self,
        z_t: torch.Tensor,
        t_embed: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        style: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.in_proj(z_t)
        skip_high = self.encoder_high(x, t_embed)
        x_mid = self.down1(skip_high)
        skip_mid = self.encoder_mid(x_mid, t_embed)
        x_low = self.down2(skip_mid)
        bott = self.bottleneck(x_low, t_embed)

        global_cond = self.global_proj(cond["global"])
        bott = bott + self.global_attn(bott.transpose(1, 2), global_cond).transpose(1, 2)

        x_mid = self.upsample_mid(bott, skip_mid.size(-1))
        x_mid = self.decoder_mid(x_mid, skip_mid, cond["local"], t_embed, style)
        x_high = self.upsample_high(x_mid, skip_high.size(-1))
        x_high = self.decoder_high(x_high, skip_high, cond["event"], t_embed, style)
        return self.out_proj(x_high)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MRCIUNet1D":
        if 'DictConfig' in globals() and DictConfig is not None and isinstance(cfg, DictConfig):
            container = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        else:
            container = cfg
        hse_cfg = container["hse"] if "hse" in container else container.get("hse_cfg")
        unet_cfg = container["unet"] if "unet" in container else container
        style_dim = container.get("style_encoder", {}).get("embedding_dim", unet_cfg.get("adaln_dim", 512))
        cond_dims = {
            "event": hse_cfg["d_event"],
            "local": hse_cfg["d_local"],
            "global": hse_cfg["d_global"],
        }
        scm_layers = unet_cfg.get("scm_layers")
        if scm_layers is not None:
            scm_layers = list(scm_layers)
        config = UNetConfig(
            in_channels=container.get("latent_channels", 8),
            base_channels=unet_cfg["base_channels"],
            mid_channels=unet_cfg["mid_channels"],
            bottleneck_channels=unet_cfg["bottleneck_channels"],
            time_dim=unet_cfg["diffusion_embedding_dim"],
            style_dim=style_dim,
            cond_dims=cond_dims,
            attention_heads=dict(unet_cfg["attention_heads"]),
            r1=hse_cfg["r1"],
            r2=hse_cfg["r2"],
            scm_layers=scm_layers,
        )
        return cls(config)
