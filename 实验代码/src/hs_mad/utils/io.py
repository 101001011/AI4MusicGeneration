"""I/O helpers for manifests, caching and serialization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
    "atomic_write",
]


def read_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike[str] | str, data: Any, *, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_yaml(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: os.PathLike[str] | str, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target.parent), suffix=".yaml") as tmp_file:
        yaml.safe_dump(data, tmp_file)
        tmp_name = tmp_file.name
    os.replace(tmp_name, target)


def atomic_write(path: os.PathLike[str] | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target.parent)) as tmp_file:
        tmp_file.write(data)
        tmp_name = tmp_file.name
    os.replace(tmp_name, target)
