"""Disk-backed caching for SRM features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

__all__ = ["FeatureCache"]


class FeatureCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, uid: str) -> Path:
        return self.root / f"{uid}.pt"

    def has(self, uid: str) -> bool:
        return self.path_for(uid).exists()

    def load(self, uid: str, map_location: Optional[str | torch.device] = None) -> torch.Tensor:
        return torch.load(self.path_for(uid), map_location=map_location)

    def save(self, uid: str, tensor: torch.Tensor) -> None:
        path = self.path_for(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.cpu(), path)
