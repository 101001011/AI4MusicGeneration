"""DataModule abstraction for HS-MAD datasets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from hs_mad.utils.audio import time_to_latent_frames

from .datasets.maestro import MaestroDataset
from .datasets.slakh import SlakhDataset

__all__ = ["HSMadDataModule", "create_datamodule"]


@dataclass
class DataConfig:
    sample_rate: int
    segment_seconds: float
    latent_rate: float
    collate: Dict[str, Any]
    train_datasets: Dict[str, Dict[str, Any]]
    val_datasets: Dict[str, Dict[str, Any]]


def _collate_fn(batch: List[Dict[str, Any]], sample_rate: int, segment_seconds: float, latent_rate: float) -> Dict[str, Any]:
    waves = torch.stack([item["wave"] for item in batch], dim=0)
    durs = torch.tensor([item["dur_sec"] for item in batch], dtype=torch.float32)
    latent_frames = torch.tensor([time_to_latent_frames(d.item(), latent_rate) for d in durs], dtype=torch.long)
    style_texts = [item["style"]["text"] for item in batch]
    style_refs = [item["style"]["ref"] for item in batch]

    return {
        "uid": [item["uid"] for item in batch],
        "wave": waves,
        "midi": [item["midi"] for item in batch],
        "dur_sec": durs,
        "latent_frames": latent_frames,
        "style": {"text": style_texts, "ref": style_refs},
        "sample_rate": sample_rate,
        "segment_seconds": segment_seconds,
    }


class HSMadDataModule:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._train_dataset: Optional[ConcatDataset] = None
        self._val_dataset: Optional[ConcatDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_parts = []
            if "maestro" in self.cfg.train_datasets:
                manifest_path = self.cfg.train_datasets["maestro"]["manifest"]
                train_parts.append(
                    MaestroDataset(
                        manifest_path,
                        sample_rate=self.cfg.sample_rate,
                        segment_seconds=self.cfg.segment_seconds,
                    )
                )
            if "slakh" in self.cfg.train_datasets:
                manifest_path = self.cfg.train_datasets["slakh"]["manifest"]
                train_parts.append(
                    SlakhDataset(
                        manifest_path,
                        sample_rate=self.cfg.sample_rate,
                        segment_seconds=self.cfg.segment_seconds,
                    )
                )
            self._train_dataset = ConcatDataset(train_parts)
        if stage in ("validate", "fit", None):
            val_parts = []
            if "maestro" in self.cfg.val_datasets:
                manifest_path = self.cfg.val_datasets["maestro"]["manifest"]
                val_parts.append(
                    MaestroDataset(
                        manifest_path,
                        sample_rate=self.cfg.sample_rate,
                        segment_seconds=self.cfg.segment_seconds,
                    )
                )
            self._val_dataset = ConcatDataset(val_parts)

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.collate.get("num_workers", 0)
        loader_kwargs = {
            "batch_size": self.cfg.collate["batch_size"],
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": self.cfg.collate.get("pin_memory", False),
            "persistent_workers": self.cfg.collate.get("persistent_workers", False) and num_workers > 0,
            "collate_fn": lambda batch: _collate_fn(
                batch,
                self.cfg.sample_rate,
                self.cfg.segment_seconds,
                self.cfg.latent_rate,
            ),
        }
        prefetch_factor = self.cfg.collate.get("prefetch_factor")
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(self._train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        num_workers = self.cfg.collate.get("num_workers", 0)
        loader_kwargs = {
            "batch_size": self.cfg.collate["batch_size"],
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self.cfg.collate.get("pin_memory", False),
            "collate_fn": lambda batch: _collate_fn(
                batch,
                self.cfg.sample_rate,
                self.cfg.segment_seconds,
                self.cfg.latent_rate,
            ),
        }
        prefetch_factor = self.cfg.collate.get("prefetch_factor")
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(self._val_dataset, **loader_kwargs)

    @contextmanager
    def prefetch_iterator(self) -> Iterator[Dict[str, Any]]:
        loader = self.train_dataloader()
        iterator = iter(loader)
        try:
            yield iterator
        finally:
            del iterator


def create_datamodule(cfg: Any) -> HSMadDataModule:
    data_cfg = DataConfig(
        sample_rate=int(cfg.sample_rate),
        segment_seconds=float(cfg.segment_seconds),
        latent_rate=float(cfg.latent_rate),
        collate=dict(cfg.collate),
        train_datasets=dict(cfg.train_datasets),
        val_datasets=dict(cfg.val_datasets),
    )
    module = HSMadDataModule(data_cfg)
    return module
