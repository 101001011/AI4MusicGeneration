from __future__ import annotations

import time

from hs_mad.utils.timing import Timer, time_block


def test_timer_records_average():
    timer = Timer()
    with time_block(timer, "sleep"):
        time.sleep(0.01)
    with time_block(timer, "sleep"):
        time.sleep(0.02)
    summary = timer.summary()
    assert "sleep" in summary
    assert summary["sleep"] >= 0.01
