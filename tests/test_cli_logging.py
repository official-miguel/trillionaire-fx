from __future__ import annotations

import json
import logging

import pytest

import tfx.__main__ as cli
from tests.conftest import FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.service import MarketDataService
from tfx.logging_setup import JsonFormatter, configure_logging, fields, get_logger


def test_cli_fetch_and_status(cfg: AppConfig, cache: CandleCache,
                              monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    frame = synthetic_candles("2003-01-01", "2026-11-20")
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", frame),
                                         "mt5": FakeProvider("mt5", frame)},
                            clock=MutableClock(ny(2026, 11, 20, 21)))
    monkeypatch.setattr(cli, "_service", lambda: svc)
    assert cli.main(["fetch"]) == 0
    out = capsys.readouterr().out
    assert "EURUSD" in out and "FRESH" in out and "12/12 symbols eligible" in out
    assert cli.main(["status"]) == 0
    assert "FRESH" in capsys.readouterr().out


def test_json_formatter_includes_fields_and_exceptions() -> None:
    log = get_logger("unit")
    assert log.name == "tfx.unit" and get_logger("tfx.x").name == "tfx.x"
    rec = log.makeRecord("tfx.unit", logging.INFO, __file__, 1, "hello %s", ("w",), None,
                         extra=fields(symbol="EURUSD"))
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "hello w" and payload["symbol"] == "EURUSD"
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        rec2 = log.makeRecord("tfx.unit", logging.ERROR, __file__, 1, "x", (), sys.exc_info())
    assert "ValueError" in json.loads(JsonFormatter().format(rec2))["exc"]


def test_configure_logging_idempotent() -> None:
    configure_logging()
    configure_logging()
    ours = [h for h in logging.getLogger("tfx").handlers if isinstance(h.formatter, JsonFormatter)]
    assert len(ours) == 1
