"""Benchmark HS-MAD inference latency/throughput."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from hs_mad.modules.diffusion.sampler import DiffusionSampler
from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig
from hs_mad.modules.diffusion.sampler import SamplerConfig
from hs_mad.modules.guidance.decoupled_cfg import DecoupledCFG
from hs_mad.modules.unet.mrci_unet import MRCIUNet1D
from hs_mad.modules.codecs.dac_wrapper import DACWrapper
from hs_mad.utils.logging import setup_logging
from hs_mad.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HS-MAD inference speed")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--infer-config", type=Path, default=Path("configs/infer.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(1234)
    logger = setup_logging("bench_infer")

    model_cfg = OmegaConf.load(args.model_config)
    infer_cfg = OmegaConf.load(args.infer_config)

    device = torch.device(args.device)

    unet = MRCIUNet1D.from_config(model_cfg).to(device)
    unet.eval()

    codec = DACWrapper.from_pretrained().to(device)
    scheduler = DiffusionScheduler(SchedulerConfig(num_train_timesteps=infer_cfg.get("diffusion_steps", 50) * 2))
    sampler = DiffusionSampler(scheduler, SamplerConfig(num_inference_steps=infer_cfg.get("diffusion_steps", 50), w_structure=infer_cfg.cfg_weights.structure, w_style=infer_cfg.cfg_weights.style))
    cfg = DecoupledCFG()

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(args.samples):
            start = time.perf_counter()
            frames = codec.latent_frames(infer_cfg.chunk_seconds)
            z_T = torch.randn(1, codec.latent_channels, frames, device=device)
            dummy_cond = {
                "event": torch.zeros(1, frames, unet.cfg.cond_dims["event"], device=device),
                "local": torch.zeros(1, frames // unet.cfg.r1, unet.cfg.cond_dims["local"], device=device),
                "global": torch.zeros(1, max(1, frames // (unet.cfg.r1 * unet.cfg.r2)), unet.cfg.cond_dims["global"], device=device),
            }
            audio = sampler.sample(
                model=unet,
                cfg=cfg,
                codec=codec,
                z_T=z_T,
                structure_cond=dummy_cond,
                style_cond=torch.zeros(1, unet.cfg.style_dim, device=device),
                steps=args.steps,
                cfg_weights=infer_cfg.cfg_weights,
            )
            _ = audio.cpu()
            timings.append(time.perf_counter() - start)

    logger.info("Median latency: %.3f s", statistics.median(timings))
    logger.info("P90 latency: %.3f s", statistics.quantiles(timings, n=10)[8])


if __name__ == "__main__":
    main()
