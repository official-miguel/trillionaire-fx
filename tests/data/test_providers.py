from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tfx.data.providers import (
    OandaProvider,
    ProviderError,
    StooqProvider,
    YFinanceProvider,
    build_providers,
    parse_oanda_candles,
    parse_stooq_csv,
)

STOOQ_CSV = """Date,Open,High,Low,Close
2026-11-18,1.0550,1.0600,1.0500,1.0580
2026-11-19,1.0580,1.0620,1.0540,1.0560
2026-11-20,1.0560,1.0590,1.0510,1.0530
"""


@dataclass
class FakeResp:
    status_code: int = 200
    text: str = ""
    payload: Any = None

    def json(self) -> Any:
        return self.payload


@dataclass
class FakeHttp:
    responses: list[FakeResp]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, url: str, params: Mapping[str, Any], headers: Mapping[str, str],
                 timeout: float) -> FakeResp:
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        return self.responses.pop(0)


# ------------------------------------------------------------------ stooq
def test_stooq_parses_csv() -> None:
    http = FakeHttp([FakeResp(text=STOOQ_CSV)])
    df = StooqProvider(http_get=http).fetch_daily("EURUSD", date(2026, 11, 1))
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert df.index[-1] == pd.Timestamp("2026-11-20")
    assert http.calls[0]["params"]["s"] == "eurusd"
    assert http.calls[0]["params"]["d1"] == "20261101"
    assert "TrillionaireFX" in http.calls[0]["headers"]["User-Agent"]


@pytest.mark.parametrize("text", ["No data", "<html>limit exceeded</html>", "", "Date\n"])
def test_stooq_rejects_non_csv(text: str) -> None:
    with pytest.raises(ProviderError):
        StooqProvider(http_get=FakeHttp([FakeResp(text=text)])).fetch_daily("x", date(2026, 1, 1))


def test_stooq_http_error_and_network_error() -> None:
    with pytest.raises(ProviderError, match="HTTP 503"):
        StooqProvider(http_get=FakeHttp([FakeResp(503)])).fetch_daily("x", date(2026, 1, 1))

    def boom(*_: Any) -> FakeResp:
        raise ConnectionError("offline")

    with pytest.raises(ProviderError, match="offline"):
        StooqProvider(http_get=boom).fetch_daily("x", date(2026, 1, 1))


def test_stooq_unparseable_dates() -> None:
    with pytest.raises(ProviderError):
        parse_stooq_csv("Date,Open,High,Low,Close\nnot-a-date,1,1,1,1\n")


# ---------------------------------------------------------------- yfinance
def test_yfinance_normalises_frame() -> None:
    idx = pd.DatetimeIndex(["2026-11-19", "2026-11-20"]).tz_localize("Europe/London")
    raw = pd.DataFrame({"Open": [1.0, 1.1], "High": [1.2, 1.2], "Low": [0.9, 1.0],
                        "Close": [1.1, 1.15], "Volume": [0, 0], "Dividends": [0, 0]}, index=idx)
    seen: list[tuple[str, date]] = []

    def hist(ticker: str, start: date) -> pd.DataFrame:
        seen.append((ticker, start))
        return raw

    df = YFinanceProvider(history_fn=hist).fetch_daily("EURUSD=X", date(2026, 11, 1))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is None and seen == [("EURUSD=X", date(2026, 11, 1))]


def test_yfinance_errors_wrapped() -> None:
    def fail(_t: str, _s: date) -> pd.DataFrame:
        raise ValueError("rate limited")

    with pytest.raises(ProviderError, match="rate limited"):
        YFinanceProvider(history_fn=fail).fetch_daily("X", date(2026, 1, 1))
    with pytest.raises(ProviderError, match="empty"):
        empty = YFinanceProvider(history_fn=lambda _t, _s: pd.DataFrame())
        empty.fetch_daily("X", date(2026, 1, 1))
    assert YFinanceProvider().available()


def test_ascii_ca_bundle_is_openable_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import certifi

    from tfx.data import providers as prov

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    star_dir = tmp_path / "W\u2b50rld"
    star_dir.mkdir()
    fake = star_dir / "cacert.pem"
    fake.write_text("CERTS", encoding="utf-8")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    out = prov.ascii_ca_bundle()
    assert out.isascii() and Path(out).read_text(encoding="utf-8") == "CERTS"
    monkeypatch.setattr(certifi, "where", lambda: str(tmp_path / "plain.pem"))
    assert prov.ascii_ca_bundle() == str(tmp_path / "plain.pem")


# ------------------------------------------------------------------- oanda
def _candle(t: str, c: float, complete: bool = True) -> dict[str, Any]:
    return {"time": t, "complete": complete, "volume": 100,
            "mid": {"o": str(c), "h": str(c + 0.01), "l": str(c - 0.01), "c": str(c)}}


def test_oanda_labels_by_ny_close_and_skips_incomplete() -> None:
    rows = parse_oanda_candles([
        _candle("2026-11-15T22:00:00.000000000Z", 1.05),  # Sun 17:00 NY open -> Mon 16
        _candle("2026-11-19T22:00:00.000000000Z", 1.06),  # Thu 17:00 NY open -> Fri 20
        _candle("2026-11-22T22:00:00.000000000Z", 1.07, complete=False),
    ])
    assert [r["date"] for r in rows] == [pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-20")]


def test_oanda_requires_token_and_paginates() -> None:
    assert not OandaProvider(token="").available()
    with pytest.raises(ProviderError, match="OANDA_API_TOKEN"):
        OandaProvider(token="").fetch_daily("EUR_USD", date(2026, 1, 1))

    page1 = [_candle(f"2026-11-{d:02d}T22:00:00Z", 1.0) for d in (15, 16)]
    page2 = [_candle("2026-11-17T22:00:00Z", 1.0)]
    http = FakeHttp([FakeResp(payload={"candles": page1}), FakeResp(payload={"candles": page2})])
    prov = OandaProvider(token="t", http_get=http)
    prov.MAX_COUNT = 2  # type: ignore[misc]
    df = prov.fetch_daily("EUR_USD", date(2026, 11, 15))
    assert len(df) == 3 and len(http.calls) == 2
    assert http.calls[0]["headers"]["Authorization"] == "Bearer t"
    assert http.calls[0]["params"]["alignmentTimezone"] == "America/New_York"


def test_oanda_errors() -> None:
    with pytest.raises(ProviderError, match="HTTP 401"):
        OandaProvider(token="t", http_get=FakeHttp([FakeResp(401)])).fetch_daily("X", date.today())
    with pytest.raises(ProviderError, match="no complete"):
        http = FakeHttp([FakeResp(payload={"candles": []})])
        OandaProvider(token="t", http_get=http).fetch_daily("X", date.today())

    def boom(*_: Any) -> FakeResp:
        raise TimeoutError("slow")

    with pytest.raises(ProviderError, match="slow"):
        OandaProvider(token="t", http_get=boom).fetch_daily("X", date.today())


def test_build_providers_registry() -> None:
    assert set(build_providers()) == {"yfinance", "mt5", "stooq", "oanda"}


# ---------------------------------------------------------------- network
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("TFX_NETWORK_TESTS") != "1", reason="set TFX_NETWORK_TESTS=1")
def test_live_yfinance_smoke() -> None:  # pragma: no cover - optional live check
    df = YFinanceProvider().fetch_daily("EURUSD=X", date(2026, 10, 1))
    assert not df.empty
