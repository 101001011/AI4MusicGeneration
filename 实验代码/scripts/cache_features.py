"""预渲染 SRM 与 HSE 特征的脚本模板。"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

import torch

from hs_mad.data.datamodules import create_datamodule
from hs_mad.modules.hse.hse import HierarchicalSymbolicEncoder
from hs_mad.utils.logging import setup_logging
from hs_mad.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SRM/HSE features for HS-MAD")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("cache/hse"))
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logging("cache_features")

    data_cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load(args.model_config)

    datamodule = create_datamodule(data_cfg)
    datamodule.setup(stage="fit")

    hse = HierarchicalSymbolicEncoder(
        d_in=model_cfg.hse.d_in,
        d_event=model_cfg.hse.d_event,
        d_local=model_cfg.hse.d_local,
        d_global=model_cfg.hse.d_global,
        r1=model_cfg.hse.r1,
        r2=model_cfg.hse.r2,
        n_blocks=tuple(model_cfg.hse.n_blocks),
        conformer_cfg=model_cfg.hse.conformer,
    )
    hse.eval()

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    with datamodule.prefetch_iterator() as iterator:
        for batch in iterator:
            feats, _ = hse(batch["midi"], batch["dur_sec"])
            uid = batch["uid"]
            dest = output_root / f"{uid}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Saving features to %s", dest)
            torch.save(feats, dest)
            n_processed += 1
            if 0 < args.max_items <= n_processed:
                break

    logger.info("Finished caching %d items", n_processed)


if __name__ == "__main__":
    main()
