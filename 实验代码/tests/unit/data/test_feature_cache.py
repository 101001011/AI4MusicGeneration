from __future__ import annotations

from pathlib import Path

import torch

from hs_mad.data.srm.features import FeatureCache


def test_feature_cache_roundtrip(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    uid = "sample"
    tensor = torch.randn(3, 4)
    cache.save(uid, tensor)
    assert cache.has(uid)
    loaded = cache.load(uid)
    assert torch.allclose(loaded, tensor)
