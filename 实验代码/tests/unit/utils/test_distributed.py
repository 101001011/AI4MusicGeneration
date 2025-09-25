from __future__ import annotations

from unittest import mock

import torch

from hs_mad.utils import distributed as dist_utils


def test_distributed_fallback_when_uninitialized():
    assert not dist_utils.is_initialized()
    assert dist_utils.get_world_size() == 1
    assert dist_utils.get_rank() == 0
    assert dist_utils.is_main_process()
    # barrier should not raise
    dist_utils.barrier()
    obj = {"value": 1}
    assert dist_utils.broadcast_object(obj) == obj


def test_distributed_zero_first_with_mocked_world_size():
    with mock.patch.object(dist_utils, "get_world_size", return_value=2):
        with mock.patch.object(torch.distributed, "barrier") as barrier_mock:
            with dist_utils.distributed_zero_first(local_rank=1):
                pass
            assert barrier_mock.call_count == 1
            barrier_mock.reset_mock()
            with dist_utils.distributed_zero_first(local_rank=0):
                pass
            assert barrier_mock.call_count == 1


def test_infer_local_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert dist_utils.infer_local_rank() == 3
    monkeypatch.delenv("LOCAL_RANK")
    monkeypatch.setenv("SLURM_LOCALID", "5")
    assert dist_utils.infer_local_rank() == 5
