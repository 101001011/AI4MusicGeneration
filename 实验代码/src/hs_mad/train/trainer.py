"""Training orchestration for HS-MAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from hs_mad.modules.diffusion.loss import DiffusionLoss
from hs_mad.modules.codecs.dac_wrapper import DACWrapper
from hs_mad.modules.codecs.clap_encoder import CLAPEncoder, CLAPConfig
from hs_mad.modules.hse.hse import HierarchicalSymbolicEncoder
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D
from hs_mad.data.datamodules import HSMadDataModule
from hs_mad.utils.logging import setup_logging

from .optim import build_optimizer, build_scheduler

__all__ = ["TrainerConfig", "Trainer"]


@dataclass
class TrainerConfig:
    max_steps: int = 1000
    grad_accum_steps: int = 1
    amp: bool = False
    ema_decay: float = 0.9999
    lambda_aux: float = 0.1
    log_every: int = 50


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:  # pragma: no cover - utility
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        unet: MRCIUNet1D,
        hse: HierarchicalSymbolicEncoder,
        codec: DACWrapper,
        diffusion_loss: DiffusionLoss,
        datamodule: HSMadDataModule,
        optim_cfg: Dict,
        sched_cfg: Dict,
        style_encoder: Optional[CLAPEncoder] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.logger = setup_logging("trainer")

        self.unet = unet.to(self.device)
        self.hse = hse.to(self.device)
        self.codec = codec.to(self.device)
        self.diffusion_loss = diffusion_loss
        self.datamodule = datamodule
        style_dim = getattr(self.unet.cfg, 'style_dim', diffusion_loss.time_dim)
        self.style_encoder = style_encoder or CLAPEncoder(CLAPConfig(embedding_dim=style_dim), self.device)

        self.optimizer = build_optimizer(self.unet, optim_cfg)
        self.scheduler = build_scheduler(self.optimizer, sched_cfg, cfg.max_steps)
        self.ema = EMA(self.unet, cfg.ema_decay) if cfg.ema_decay else None
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        self.global_step = 0

    def _prepare_style(self, style_batch: Dict[str, Optional[torch.Tensor | str]]) -> Optional[torch.Tensor]:
        if self.style_encoder is None:
            return None
        return self.style_encoder(style_batch).to(self.device)

    def _structure_to_device(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in features.items()}

    def train_step(self, batch: Dict[str, any]) -> Dict[str, float]:
        wave = batch["wave"].to(self.device)
        structure_cond, aux_losses = self.hse(batch["midi"], batch["dur_sec"].tolist())
        structure_cond = self._structure_to_device(structure_cond)
        style_vec = self._prepare_style(batch["style"])
        with torch.no_grad():
            latents = self.codec.encode(wave)
        loss_output = self.diffusion_loss(self.unet, latents, structure_cond, style_vec)
        aux_total = sum(loss for loss in aux_losses.values())
        total_loss = loss_output.loss + self.cfg.lambda_aux * aux_total
        return {
            "total": total_loss,
            "diffusion": loss_output.loss,
            "aux": aux_total,
        }

    def fit(self) -> None:
        self.unet.train()
        self.hse.train()
        self.datamodule.setup("fit")
        data_loader = self.datamodule.train_dataloader()
        data_iter = iter(data_loader)

        self.optimizer.zero_grad()
        while self.global_step < self.cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp):
                losses = self.train_step(batch)
                loss = losses["total"] / self.cfg.grad_accum_steps
            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % self.cfg.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema:
                    self.ema.update(self.unet)

            if self.global_step % self.cfg.log_every == 0:
                self.logger.info(
                    "step=%d loss=%.4f diffusion=%.4f aux=%.4f",
                    self.global_step,
                    losses["total"].item(),
                    losses["diffusion"].item(),
                    losses["aux"].item(),
                )

            self.global_step += 1
