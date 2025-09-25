from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from hs_mad.data.datamodules import create_datamodule
from omegaconf import OmegaConf
from hs_mad.modules.codecs.dac_wrapper import DACConfig, DACWrapper
from hs_mad.modules.diffusion.loss import DiffusionLoss
from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig
from hs_mad.modules.hse.hse import HierarchicalSymbolicEncoder
from hs_mad.modules.unet.mrci_unet import UNetConfig, MRCIUNet1D
from hs_mad.train.trainer import Trainer, TrainerConfig
from hs_mad.utils.io import write_json
from hs_mad.utils.seed import seed_everything


def _create_audio(path: Path, sr: int = 44100) -> None:
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    import soundfile as sf

    sf.write(path, wave, sr)


def _create_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.5, pitch=60, velocity=90))
    midi.instruments.append(inst)
    midi.write(str(path))


def _build_manifests(tmp_path: Path) -> Path:
    audio = tmp_path / "sample.wav"
    midi = tmp_path / "sample.mid"
    _create_audio(audio)
    _create_midi(midi)
    manifest = [{"uid": "sample", "audio": str(audio), "midi": str(midi), "style_text": "sample"}]
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def test_trainer_single_step(tmp_path: Path):
    seed_everything(123)
    manifest_path = _build_manifests(tmp_path)
    data_cfg = OmegaConf.create({
        "sample_rate": 44100,
        "segment_seconds": 1.0,
        "latent_rate": 137.8,
        "latent_channels": 4,
        "latent_stride": 320,
        "train_datasets": {"maestro": {"manifest": str(manifest_path)}},
        "val_datasets": {"maestro": {"manifest": str(manifest_path)}},
        "collate": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
    })
    datamodule = create_datamodule(data_cfg)

    hse = HierarchicalSymbolicEncoder(r1=2, r2=2, d_in=384, d_event=16, d_local=24, d_global=32, n_blocks=(1, 1, 1), sr_latent=137.8)
    unet_cfg = UNetConfig(
        in_channels=4,
        base_channels=8,
        mid_channels=8,
        bottleneck_channels=12,
        time_dim=16,
        style_dim=6,
        cond_dims={"event": 16, "local": 24, "global": 32},
        attention_heads={"event": 2, "local": 2, "global": 2},
        r1=2,
        r2=2,
        scm_layers=["high", "mid"],
    )
    unet = MRCIUNet1D(unet_cfg)
    scheduler = DiffusionScheduler(SchedulerConfig(num_train_timesteps=20))
    diffusion_loss = DiffusionLoss(scheduler, time_dim=unet.cfg.time_dim)
    codec = DACWrapper.from_config(DACConfig(sample_rate=44100, latent_rate=137.8, latent_channels=4, stride=320))

    trainer = Trainer(
        TrainerConfig(max_steps=1, grad_accum_steps=1, amp=False, ema_decay=0.0, log_every=1),
        unet,
        hse,
        codec,
        diffusion_loss,
        datamodule,
        optim_cfg={"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        sched_cfg={"warmup_steps": 0, "min_lr": 1e-6},
        device=torch.device("cpu"),
    )

    trainer.fit()
