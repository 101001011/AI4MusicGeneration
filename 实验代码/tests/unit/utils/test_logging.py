from __future__ import annotations

import logging

from hs_mad.utils.logging import setup_logging, silence_external_loggers


def test_setup_logging_reuses_handlers(capsys):
    logger = setup_logging("hs_mad.test", level=logging.INFO)
    logger.info("hello")
    first_handler_count = len(logger.handlers)

    logger_again = setup_logging("hs_mad.test", level=logging.DEBUG)
    logger_again.debug("world")

    assert logger is logger_again
    assert len(logger_again.handlers) == first_handler_count

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "hello" in combined


def test_silence_external_loggers():
    silence_external_loggers(level=logging.ERROR)
    for name in ["numba", "madmom", "librosa", "matplotlib", "torch", "transformers"]:
        assert logging.getLogger(name).level == logging.ERROR
