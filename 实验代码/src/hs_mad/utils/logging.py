"""Structured logging utilities tailored for HS-MAD."""

from __future__ import annotations

import logging
from typing import Optional

from .distributed import get_rank

__all__ = ["setup_logging"]


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a rank-aware logger.

    Multiple calls with the same name will reuse handlers to avoid duplicate logs.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    rank = get_rank()
    formatter = logging.Formatter(
        fmt=f"%(asctime)s | rank={rank} | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler: logging.Handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def silence_external_loggers(level: int = logging.WARNING) -> None:
    """Reduce verbosity of common third-party libraries."""

    for lib_name in [
        "numba",
        "madmom",
        "librosa",
        "matplotlib",
        "torch",
        "transformers",
    ]:
        logging.getLogger(lib_name).setLevel(level)


__all__.append("silence_external_loggers")
