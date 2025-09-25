"""Utilities wrapping torch.distributed with graceful fallbacks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

import torch

__all__ = [
    "is_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "barrier",
    "broadcast_object",
    "distributed_zero_first",
]


def is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    if not is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_initialized():
        torch.distributed.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_initialized():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


@contextmanager
def distributed_zero_first(local_rank: int) -> Generator[None, None, None]:
    """Context manager ensuring rank zero runs first in DDP."""

    need_barrier = get_world_size() > 1
    if need_barrier and local_rank != 0:
        torch.distributed.barrier()
    yield
    if need_barrier and local_rank == 0:
        torch.distributed.barrier()


def infer_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 0


__all__.append("infer_local_rank")
