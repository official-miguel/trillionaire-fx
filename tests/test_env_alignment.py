from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import tfx.__main__ as cli
from tests.conftest import FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.providers import Alignment, OandaProvider, parse_oanda_candles
from tfx.data.service import MarketDataService
from tfx.env import load_env


# ------------------------------------------------------------------ .env
def test_load_env_parses_and_never_overrides(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "﻿# comment\nTFX_T_A=one\nexport TFX_T_B = \"two words\"\nTFX_T_C='x'\n"
        "TFX_T_EMPTY=\nnot a pair\nTFX_T_KEEP=file\n",
        encoding="utf-8",
    )
    for k in ("TFX_T_A", "TFX_T_B", "TFX_T_C", "TFX_T_EMPTY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TFX_T_KEEP", "shell")
    loaded = load_env(env)
    assert loaded == ["TFX_T_A", "TFX_T_B", "TFX_T_C"]
    assert os.environ["TFX_T_B"] == "two words" and os.environ["TFX_T_C"] == "x"
    assert os.environ["TFX_T_KEEP"] == "shell"  # real environment wins
    assert "TFX_T_EMPTY" not in os.environ
    assert load_env(tmp_path / "missing.env") == []


def test_env_file_is_gitignored() -> None:
    root = Path(__file__).resolve().parent.parent
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").split()


# ------------------------------------------------------------- alignment
def _c(t: str) -> dict[str, object]:
    return {"time": t, "complete": True, "volume": 1,
            "mid": {"o": "1", "h": "1.1", "l": "0.9", "c": "1.05"}}


def test_london_alignment_labels_by_open_date() -> None:
    rows = parse_oanda_candles(
        [_c("2026-11-19T00:00:00Z"),        # winter: London midnight == 00:00 UTC
         _c("2026-07-15T23:00:00Z")],       # summer: London midnight == 23:00 UTC prev day
        Alignment.LONDON_MIDNIGHT,
    )
    assert [r["date"] for r in rows] == [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-07-16")]


def test_oanda_request_uses_requested_alignment() -> None:
    from tests.data.test_providers import FakeHttp, FakeResp

    http = FakeHttp([FakeResp(payload={"candles": [_c("2026-11-19T00:00:00Z")]})])
    prov = OandaProvider(token="t", http_get=http)
    prov.fetch_daily("EUR_USD", date(2026, 11, 1), alignment=Alignment.LONDON_MIDNIGHT)
    assert http.calls[0]["params"]["dailyAlignment"] == 0
    assert http.calls[0]["params"]["alignmentTimezone"] == "Europe/London"


def test_reference_requested_on_primary_alignment(cfg: AppConfig, cache: CandleCache) -> None:
    full = synthetic_candles("2003-01-01", "2026-11-20")
    oanda = FakeProvider("oanda", full)
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": FakeProvider("mt5", fail=True),
                                         "oanda": oanda},
                            clock=MutableClock(ny(2026, 11, 20, 21)))
    md = svc.get("EURUSD")
    assert md.signal_allowed and md.check is not None and md.check.reference_source == "oanda"
    assert oanda.alignments == [Alignment.UTC_MIDNIGHT]  # matches the rebuilt Yahoo bars


# ------------------------------------------------------- sources report
def test_sources_report(cfg: AppConfig, cache: CandleCache, monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture[str]) -> None:
    full = synthetic_candles("2003-01-01", "2026-11-20")
    class Off(FakeProvider):
        last_error = "mt5: terminal not ready"

        def available(self) -> bool:
            return False

    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": Off("mt5", full)},
                            clock=MutableClock(ny(2026, 11, 20, 21)))
    monkeypatch.setattr(cli, "_service", lambda: svc)
    monkeypatch.setattr(cli, "load_env", lambda: [])
    assert cli.main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "MT5: NOT CONNECTED" in out and "No reference data" in out
    assert "File > Login to Trade Account" in out
    svc.providers = {"yfinance": FakeProvider("yfinance", full),
                     "oanda": FakeProvider("oanda", full)}
    assert cli.main(["sources"]) == 0
    out = capsys.readouterr().out
    assert "EURUSD" in out and "oanda" in out and "Average share" in out and "0.0%" in out
