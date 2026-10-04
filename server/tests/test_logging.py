import json
import logging

from svd_server.logging_setup import JsonFormatter


def test_json_formatter_outputs_event_and_fields() -> None:
    record = logging.LogRecord(
        "svd_server.access", logging.INFO, __file__, 1, "request", None, None
    )
    record.svd = {"route": "/health", "status": 200}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "request"
    assert payload["level"] == "INFO"
    assert payload["route"] == "/health"
    assert payload["status"] == 200
    assert "ts" in payload


def test_json_formatter_without_fields() -> None:
    record = logging.LogRecord("svd_server", logging.WARNING, __file__, 1, "hello", None, None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "hello"
