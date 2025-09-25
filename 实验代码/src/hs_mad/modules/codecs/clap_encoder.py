"""CLAP encoder wrapper with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - heavy dependency
    from laion_clap import CLAP_Module  # type: ignore
except ImportError:  # pragma: no cover
    CLAP_Module = None

__all__ = ["CLAPEncoder", "CLAPConfig"]


@dataclass
class CLAPConfig:
    embedding_dim: int = 512
    enable_text: bool = True
    enable_audio: bool = True


class CLAPEncoder(nn.Module):
    def __init__(self, cfg: CLAPConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.available = CLAP_Module is not None
        if self.available:
            self.model = CLAP_Module(enable_fusion=True, device=device)
        else:
            self.register_buffer("_zero", torch.zeros(cfg.embedding_dim), persistent=False)

    def forward(self, style: Dict[str, Optional[torch.Tensor | str]]) -> torch.Tensor:
        return self.encode(style)

    def encode(self, style: Dict[str, Optional[torch.Tensor | str]]) -> torch.Tensor:
        if self.available:
            texts: List[str] = style.get("text", []) or []
            refs: Optional[torch.Tensor] = style.get("ref")
            embeddings: List[torch.Tensor] = []
            if texts and self.cfg.enable_text:
                embeddings.append(self.model.get_text_embedding(texts))
            if refs is not None and self.cfg.enable_audio:
                embeddings.append(self.model.get_audio_embedding(refs.to(self.device)))
            if not embeddings:
                return torch.zeros((1, self.cfg.embedding_dim), device=self.device)
            return torch.mean(torch.stack(embeddings, dim=0), dim=0)
        texts = style.get("text") or []
        if isinstance(texts, str):
            batch = 1
        elif isinstance(texts, list) and len(texts) > 0:
            batch = len(texts)
        else:
            ref = style.get("ref")
            batch = ref.size(0) if isinstance(ref, torch.Tensor) else 1
        return torch.zeros((batch, self.cfg.embedding_dim), device=self.device)
