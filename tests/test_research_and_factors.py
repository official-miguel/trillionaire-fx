"""Research-only sources never confirm live bars; the news factor stays informational."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.conftest import ROOT, FakeProvider, MutableClock, ny, synthetic_candles
from tfx.config import INFORMATIONAL_MAX_WEIGHT, AppConfig, SoftFactorConfig, load_config
from tfx.data.cache import CandleCache
from tfx.data.intraday import INTRADAY_SOURCES
from tfx.data.service import DataStatus, MarketDataService


def _write(tmp_path: Path, mutate: object) -> Path:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mutate(raw)  # type: ignore[operator]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


# ------------------------------------------------------ research-only sources
def test_dukascopy_is_research_only(cfg: AppConfig) -> None:
    assert "dukascopy" in cfg.data.research_only_sources
    assert not cfg.is_live_source("dukascopy") and cfg.is_live_source("mt5")
    for symbol in cfg.symbols:
        assert "dukascopy" not in cfg.providers_for(symbol)
    assert not set(INTRADAY_SOURCES) & set(cfg.data.research_only_sources)


def test_config_rejects_research_source_as_live_provider(tmp_path: Path) -> None:
    def put_in_order(raw: dict) -> None:  # type: ignore[type-arg]
        raw["data"]["research_only_sources"] = ["oanda"]

    with pytest.raises(ValidationError, match="research-only"):
        load_config(_write(tmp_path, put_in_order))


def test_research_source_never_confirms_a_live_bar(cfg: AppConfig,
                                                   cache: CandleCache) -> None:
    """Even if a research provider is plugged in and agrees perfectly, no confirmation."""
    full = synthetic_candles("2003-01-01", "2026-11-20")
    cfg.data.research_only_sources = ["mt5"]  # pretend MT5 were research-only
    svc = MarketDataService(cfg, cache, {"yfinance": FakeProvider("yfinance", full),
                                         "mt5": FakeProvider("mt5", full)},
                            clock=MutableClock(ny(2026, 11, 20, 21)))
    md = svc.get("EURUSD")
    assert md.status is DataStatus.UNVERIFIED and not md.signal_allowed
    _, ref_source = cache.load_reference("EURUSD")
    assert ref_source is None


# ------------------------------------------------------------- soft factors
def test_news_factor_is_informational_with_small_fixed_weight(cfg: AppConfig) -> None:
    factors = cfg.scoring_factors()
    news = factors["news_proximity"]
    assert news.mode == "informational" and not news.is_validated_candidate
    assert 0 < news.weight <= INFORMATIONAL_MAX_WEIGHT
    assert news.display_label == "informational - not validated"
    assert "cannot be validated" in news.note
    assert abs(sum(f.weight for f in factors.values()) - 1.0) < 1e-9
    assert factors["trend_clarity"].display_label.startswith("validated on training data")


def test_informational_weight_cap_enforced() -> None:
    with pytest.raises(ValidationError, match="capped"):
        SoftFactorConfig(mode="informational", weight=0.2)
    assert SoftFactorConfig(mode="candidate", weight=0.2).weight == 0.2


def test_no_ui_text_claims_news_factor_is_validated() -> None:
    text = (ROOT / "config.yaml").read_text(encoding="utf-8").lower()
    news_block = text[text.index("news_proximity:"):text.index("risk:\n")]
    assert "mode: informational" in news_block and "cannot be validated" in news_block
