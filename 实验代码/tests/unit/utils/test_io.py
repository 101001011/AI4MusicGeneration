from __future__ import annotations

import json
from pathlib import Path

from hs_mad.utils.io import atomic_write, read_json, read_yaml, write_json, write_yaml


def test_json_roundtrip(tmp_path: Path):
    sample = {"hello": "世界", "values": [1, 2, 3]}
    path = tmp_path / "sample.json"
    write_json(path, sample)
    assert path.exists()
    assert read_json(path) == sample


def test_yaml_roundtrip(tmp_path: Path):
    sample = {"list": [1, 2], "flag": True}
    path = tmp_path / "config.yaml"
    write_yaml(path, sample)
    assert read_yaml(path) == sample


def test_atomic_write(tmp_path: Path):
    path = tmp_path / "binary.bin"
    atomic_write(path, b"abc")
    with path.open("rb") as f:
        assert f.read() == b"abc"
