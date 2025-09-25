"""Dataset wrapper for Slakh2100."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from hs_mad.utils.io import read_json

from .base import BaseMusicDataset

__all__ = ["SlakhDataset"]


class SlakhDataset(BaseMusicDataset):
    def __init__(
        self,
        manifest_path: Path | str,
        sample_rate: int = 44100,
        segment_seconds: float = 30.0,
    ) -> None:
        manifest: List[Dict[str, Any]] = read_json(manifest_path)
        super().__init__(manifest=manifest, sample_rate=sample_rate, segment_seconds=segment_seconds)
