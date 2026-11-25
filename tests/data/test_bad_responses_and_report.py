"""Bad provider responses are rejected (never parsed) and the quality report is accurate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import pytest

import tfx.__main__ as cli
from tests.conftest import FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import AppConfig
from tfx.data.cache import CandleCache
from tfx.data.providers import (
    OandaProvider,
    ProviderError,
    StooqProvider,
    YFinanceProvider,
    looks_like_html,
    native_alignment,
)
from tfx.data.quality import DataQualityError, clean_candles
from tfx.data.report import build_quality_report, format_quality_report
from tfx.data.service import DataStatus, MarketDataService
from tfx.data.session import SessionCalendar

HTML = "<!DOCTYPE html><html><head><script>verify()</script></head><body>Hi</body></html>"
FULL = synthetic_candles("2003-01-01", "2026-11-20")
CLOCK = ny(2026, 11, 20, 21)


@dataclass
class Resp:
    status_code: int = 200
    text: str = ""
    payload: Any = None
    bad_json: bool = False

    def json(self) -> Any:
        if self.bad_json:
            raise ValueError("Expecting value")
        return self.payload


def _http(resp: Resp) -> Any:
    return lambda *_a, **_k: resp


@pytest.mark.parametrize("text", [HTML, "  <html>x</html>", "<?xml version='1.0'?><e/>",
                                  "<head></head>"])
def test_html_detection(text: str) -> None:
    assert looks_like_html(text)
    assert not looks_like_html("Date,Open,High,Low,Close\n")


def test_stooq_html_rejected_without_parsing() -> None:
    with pytest.raises(ProviderError, match="HTML page instead of CSV"):
        StooqProvider(http_get=_http(Resp(text=HTML))).fetch_daily("eurusd", date(2026, 1, 1))


@pytest.mark.parametrize("resp", [Resp(text=HTML, bad_json=True), Resp(bad_json=True),
                                  Resp(payload=["not", "a", "dict"]),
                                  Resp(payload={"candles": "oops"})])
def test_oanda_html_and_malformed_rejected(resp: Resp) -> None:
    with pytest.raises(ProviderError):
        OandaProvider(token="t", http_get=_http(resp)).fetch_daily("EUR_USD", date(2026, 1, 1))


@pytest.mark.parametrize("bad", [None, HTML, pd.Series([1.0, 2.0])])
def test_yfinance_non_frame_rejected(bad: Any) -> None:
    with pytest.raises(ProviderError):
        YFinanceProvider(history_fn=lambda _t, _s: bad).fetch_daily("EURUSD=X", date(2026, 1, 1))


def test_frame_of_html_strings_is_unusable() -> None:
    junk = pd.DataFrame({c: [HTML, HTML] for c in ["open", "high", "low", "close"]},
                        index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
    with pytest.raises(DataQualityError):
        clean_candles(junk, SessionCalendar(150), date(2026, 11, 20))


def test_html_from_reference_source_means_no_signal(cfg: AppConfig, cache: CandleCache) -> None:
    md = MarketDataService(cfg, cache, {
        "yfinance": FakeProvider("yfinance", FULL),
        "mt5": FakeProvider("mt5", fail="HTML page instead of data"),
    }, clock=MutableClock(CLOCK)).get("EURUSD")
    assert md.status is DataStatus.UNVERIFIED and not md.signal_allowed  # never falls back


def test_native_alignment() -> None:
    from tfx.data.providers import Alignment

    assert native_alignment("yfinance", "EURUSD=X") is Alignment.UTC_MIDNIGHT
    assert native_alignment("yfinance", "GC=F") is Alignment.NY_CLOSE
    assert native_alignment("mt5", "EURUSD") is Alignment.NY_CLOSE
    assert native_alignment("stooq", "eurusd") is None


# ------------------------------------------------------------ quality report
def _bump(df: pd.DataFrame, when: str, pct: float) -> pd.DataFrame:
    out = df.copy()
    out.loc[pd.Timestamp(when), "close"] *= 1 + pct / 100
    return out


def test_quality_report_numbers(cfg: AppConfig, cache: CandleCache) -> None:
    primary = FULL.drop([pd.Timestamp("2026-06-10")])        # one gap
    primary = _bump(primary, "2015-01-15", 12)                # one outlier (+ reversal)
    reference = _bump(FULL, "2026-11-18", 1.5)                # one recent disagreement
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", primary),
                                         "mt5": FakeProvider("mt5", reference)},
                            clock=MutableClock(CLOCK))
    rows = {r.symbol: r for r in build_quality_report(svc)}
    eu = rows["EURUSD"]
    assert eu.primary == "yfinance" and eu.reference == "mt5"
    assert eu.first_bar == date(2003, 1, 2) and eu.last_bar == date(2026, 11, 20)
    assert eu.gaps >= 1 and eu.outliers >= 1
    assert eu.compared_bars > 50 and eu.agreement_pct is not None and eu.agreement_pct < 100
    assert eu.latest_confirmed == date(2026, 11, 20) and eu.signal_allowed
    gold = rows["XAUUSD"]
    assert gold.is_proxy
    text = format_quality_report(list(rows.values()), "MT5: connected", True)
    assert "(proxy)" in text and "EURUSD" in text and "Login to Trade Account" not in text


def test_quality_report_without_mt5_says_what_to_do(cfg: AppConfig, cache: CandleCache,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    class Off(FakeProvider):
        last_error = "mt5: terminal not ready"

        def available(self) -> bool:
            return False

    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", FULL),
                                         "mt5": Off("mt5", FULL)}, clock=MutableClock(CLOCK))
    monkeypatch.setattr(cli, "_service", lambda: svc)
    monkeypatch.setattr(cli, "load_env", lambda: [])
    assert cli.main(["quality"]) == 0
    out = capsys.readouterr().out
    assert "MT5: NOT CONNECTED" in out and "NO SIGNALS" in out
    assert "File > Login to Trade Account" in out
    assert "UNVERIFIED" in out and " yes" not in out
    rows = build_quality_report(svc, refresh=False)
    assert all(not r.signal_allowed and r.latest_confirmed is None for r in rows)
