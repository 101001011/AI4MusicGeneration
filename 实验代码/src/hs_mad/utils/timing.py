"""Timing utilities for profiling HS-MAD training and inference."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import DefaultDict, Dict, Generator

__all__ = ["Timer", "time_block"]


@dataclass
class Timer:
    """Accumulates durations for named events."""

    counts: DefaultDict[str, int]
    totals: DefaultDict[str, float]

    def __init__(self) -> None:
        self.counts = defaultdict(int)
        self.totals = defaultdict(float)

    def record(self, name: str, duration: float) -> None:
        self.counts[name] += 1
        self.totals[name] += duration

    def summary(self) -> Dict[str, float]:
        return {key: self.totals[key] / self.counts[key] for key in self.totals}


@contextmanager
def time_block(timer: Timer, name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    timer.record(name, end - start)
