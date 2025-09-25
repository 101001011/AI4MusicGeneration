"""CLI entry-point for HS-MAD training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from hs_mad.data.datamodules import create_datamodule
from hs_mad.modules.codecs.dac_wrapper import DACConfig, DACWrapper
from hs_mad.modules.diffusion.loss import DiffusionLoss
from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig
from hs_mad.modules.hse.hse import HierarchicalSymbolicEncoder
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D
from hs_mad.train.trainer import Trainer, TrainerConfig
from hs_mad.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HS-MAD")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = OmegaConf.load(args.data_config)
    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)

    seed_everything(train_cfg.get("seed", 42))

    datamodule = create_datamodule(data_cfg)

    hse_cfg = model_cfg.hse
    hse = HierarchicalSymbolicEncoder(
        r1=hse_cfg.r1,
        r2=hse_cfg.r2,
        d_in=hse_cfg.d_in,
        d_event=hse_cfg.d_event,
        d_local=hse_cfg.d_local,
        d_global=hse_cfg.d_global,
        n_blocks=tuple(hse_cfg.n_blocks),
        conformer_cfg=dict(hse_cfg.conformer),
        sr_latent=data_cfg.latent_rate,
    )

    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)  # type: ignore[arg-type]
    model_cfg_dict["latent_channels"] = data_cfg.latent_channels
    unet = MRCIUNet1D.from_config(model_cfg_dict)

    scheduler = DiffusionScheduler(
        SchedulerConfig(num_train_timesteps=train_cfg.scheduler.get("num_train_timesteps", 1000))
    )
    diffusion_loss = DiffusionLoss(scheduler, time_dim=unet.cfg.time_dim)

    codec = DACWrapper.from_config(
        DACConfig(
            sample_rate=data_cfg.sample_rate,
            latent_rate=data_cfg.latent_rate,
            latent_channels=data_cfg.latent_channels,
            stride=data_cfg.latent_stride,
        )
    )

    trainer_cfg = TrainerConfig(
        max_steps=train_cfg.max_steps,
        grad_accum_steps=train_cfg.batching.grad_accum_steps,
        amp=train_cfg.get("precision", "fp32") == "amp",
        ema_decay=train_cfg.ema.decay if train_cfg.ema.enable else 0.0,
        lambda_aux=train_cfg.get("lambda_aux", 0.1),
        log_every=train_cfg.log_every_n_steps,
    )

    optim_cfg = OmegaConf.to_container(train_cfg.optimizer, resolve=True)  # type: ignore[arg-type]
    sched_cfg = OmegaConf.to_container(train_cfg.scheduler, resolve=True)  # type: ignore[arg-type]
    trainer = Trainer(
        trainer_cfg,
        unet,
        hse,
        codec,
        diffusion_loss,
        datamodule,
        optim_cfg=optim_cfg,
        sched_cfg=sched_cfg,
        device=torch.device(args.device),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
