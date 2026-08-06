from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).parents[2] / "docker" / "openevolve-executor" / "worker.py"
SPEC = importlib.util.spec_from_file_location("hardened_executor_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


@pytest.mark.parametrize(
    ("payload", "limit", "expected", "truncated"),
    (
        (b"", 8, b"", False),
        (b"short", 8, b"short", False),
        (b"12345678", 8, b"12345678", False),
        (b"123456789", 8, b"12345678", True),
    ),
)
def test_capped_capture_boundaries(payload, limit, expected, truncated):
    stream = io.BytesIO(payload)
    capture = worker.CappedCapture()

    worker.read_capped(stream, limit, capture)

    assert capture.data == expected
    assert capture.truncated is truncated
    assert stream.read() == b""


def test_capped_capture_drains_large_binary_stream_and_safe_log_replaces_invalid_utf8():
    payload = b"\xff\xfe" + b"x" * 32_000
    stream = io.BytesIO(payload)
    capture = worker.CappedCapture()

    worker.read_capped(stream, 32, capture)
    safe, truncated = worker.safe_log(capture.data, b"", 64)

    assert capture.data == payload[:32]
    assert capture.truncated is True
    assert stream.tell() == len(payload)
    assert "\ufffd" in safe
    assert truncated is False


def test_stdout_stderr_and_control_captures_are_independent_typed_channels():
    captures = [worker.CappedCapture() for _ in range(3)]
    payloads = (b"stdout", b"stderr", b'{"result":true}')

    for capture, payload in zip(captures, payloads, strict=True):
        worker.read_capped(io.BytesIO(payload), 1_000, capture)

    assert [capture.data for capture in captures] == list(payloads)
    assert all(isinstance(capture.data, bytes) for capture in captures)
    assert all(type(capture.truncated) is bool for capture in captures)
    assert len({id(capture) for capture in captures}) == 3


@pytest.mark.parametrize("payload", (b"", b"not-json", b"[]"))
def test_control_parser_rejects_empty_malformed_or_wrong_schema_bytes(payload):
    with pytest.raises((ValueError, worker.json.JSONDecodeError)):
        worker.parse_control(payload, 1_000)


def test_control_parser_preserves_valid_bytes_and_enforces_output_limit():
    payload = b'{"model_family":"linear"}'

    configuration, canonical = worker.parse_control(payload, len(payload))

    assert configuration == {"model_family": "linear"}
    assert canonical == payload
    with pytest.raises(OverflowError):
        worker.parse_control(payload, len(payload) - 1)
