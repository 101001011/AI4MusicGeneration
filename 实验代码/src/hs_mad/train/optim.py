"""Optimizer and LR scheduler utilities."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["build_optimizer", "build_scheduler"]


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name}")


def build_scheduler(optimizer: Optimizer, cfg: Dict[str, Any], max_steps: int) -> LambdaLR:
    warmup_steps = cfg.get("warmup_steps", 0)
    min_lr = cfg.get("min_lr", 1e-6)
    base_lr = cfg.get("lr", optimizer.param_groups[0]["lr"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda)
