"""
tests/unit/test_logging_utils.py
=================================
Unit tests for ``network_anomaly_autoencoder.utils.logging_utils``.

Validates Requirements 10.4, 10.5.

Each test uses a unique logger name (via ``uuid.uuid4().hex``) to prevent
handler accumulation from cross-test pollution inside Python's global
logging registry.
"""

from __future__ import annotations

import logging
import re
import uuid
from io import StringIO

import pytest

from network_anomaly_autoencoder.utils.logging_utils import get_logger


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unique_name() -> str:
    """Return a logger name that cannot collide with any other test."""
    return f"test_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Test 1 – Log format contains timestamp, level, and module name
# ---------------------------------------------------------------------------

def test_log_format_contains_timestamp_level_and_module():
    """Emitted log records must contain a date stamp, a level field, and a
    module name field.

    Validates: Requirements 10.4
    """
    logger = get_logger(_unique_name())

    # Capture output via a StringIO stream handler so we can inspect it.
    buf = StringIO()
    capture = logging.StreamHandler(buf)
    capture.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(capture)

    try:
        logger.info("probe message")
    finally:
        logger.removeHandler(capture)

    output = buf.getvalue()

    # Timestamp: YYYY-MM-DD
    assert re.search(r"\d{4}-\d{2}-\d{2}", output), (
        f"Expected a date stamp in log output, got: {output!r}"
    )
    # Log level
    assert "INFO" in output, (
        f"Expected 'INFO' level in log output, got: {output!r}"
    )
    # Module name – the formatter uses %(module)s which is the basename of the
    # calling source file without the .py extension.
    assert re.search(r"[A-Za-z_]\w*", output), (
        f"Expected a module name field in log output, got: {output!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 – No FileHandler when log_file is not provided
# ---------------------------------------------------------------------------

def test_no_file_handler_without_log_file():
    """When *log_file* is omitted, the returned logger must NOT have any
    ``FileHandler`` attached.

    Validates: Requirements 10.5
    """
    logger = get_logger(_unique_name())

    file_handlers = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers == [], (
        f"Expected no FileHandler, but found: {file_handlers}"
    )


# ---------------------------------------------------------------------------
# Test 3 – FileHandler IS present when log_file is provided
# ---------------------------------------------------------------------------

def test_file_handler_added_when_log_file_provided(tmp_path):
    """When *log_file* is provided, a ``FileHandler`` pointing to that path
    must be attached to the returned logger.

    Validates: Requirements 10.5
    """
    log_path = tmp_path / "app.log"
    logger = get_logger(_unique_name(), log_file=str(log_path))

    file_handlers = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1, (
        f"Expected exactly one FileHandler, got: {file_handlers}"
    )

    # Confirm it actually writes to the requested file.
    logger.info("hello file")
    for h in file_handlers:
        h.flush()

    assert log_path.exists(), "Log file was not created on disk."
    content = log_path.read_text(encoding="utf-8")
    assert "hello file" in content, (
        f"Expected log message in file, got: {content!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 – Duplicate handlers are NOT added on repeated calls
# ---------------------------------------------------------------------------

def test_no_duplicate_handlers_on_repeated_calls(tmp_path):
    """Calling ``get_logger`` twice with the same name must NOT add duplicate
    handlers (neither StreamHandler nor FileHandler).

    Validates: Requirements 10.4
    """
    log_path = tmp_path / "dedup.log"
    name = _unique_name()

    logger_first = get_logger(name, log_file=str(log_path))
    handler_count_after_first = len(logger_first.handlers)

    logger_second = get_logger(name, log_file=str(log_path))
    handler_count_after_second = len(logger_second.handlers)

    assert logger_first is logger_second, (
        "Both calls must return the same logger object."
    )
    assert handler_count_after_second == handler_count_after_first, (
        f"Handler count grew from {handler_count_after_first} to "
        f"{handler_count_after_second} after a second get_logger call — "
        "duplicate handlers were added."
    )


# ---------------------------------------------------------------------------
# Test 5 – Logger level is INFO
# ---------------------------------------------------------------------------

def test_logger_level_is_info():
    """The returned logger must be configured at ``logging.INFO`` level.

    Validates: Requirements 10.4
    """
    logger = get_logger(_unique_name())

    assert logger.level == logging.INFO, (
        f"Expected logger level INFO ({logging.INFO}), got {logger.level}."
    )
