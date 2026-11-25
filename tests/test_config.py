from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.conftest import ROOT
from tfx.config import AccountConfig, AppConfig, ServerConfig, load_config

EXPECTED = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
            "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"}


def test_repo_config_loads(cfg: AppConfig) -> None:
    assert set(cfg.symbols) == EXPECTED
    assert cfg.account.currency == "USD" and cfg.account.starting_balance == 1000
    assert cfg.account.risk_per_trade_pct == 0.5
    assert cfg.symbol("eurusd").typical_spread_pips == 0.8
    assert cfg.providers_for("XAUUSD")[:2] == ["yfinance", "mt5"]
    assert cfg.providers_for("EURUSD") == ["yfinance", "mt5", "oanda"]  # stooq removed
    assert cfg.section("strategies")["trend"]["donchian_lookbacks"] == [20, 55, 100]
    assert cfg.section("does_not_exist") == {}
    assert cfg.server.host == "127.0.0.1"


def test_risk_hard_cap_cannot_be_raised() -> None:
    with pytest.raises(ValidationError, match="hard cap"):
        AccountConfig(risk_per_trade_pct=1.5)
    assert AccountConfig(risk_per_trade_pct=1.0).risk_per_trade_pct == 1.0


def test_server_must_bind_localhost() -> None:
    with pytest.raises(ValidationError, match="localhost"):
        ServerConfig(host="0.0.0.0")


def test_currency_normalised_and_validated() -> None:
    assert AccountConfig(currency="kes").currency == "KES"
    with pytest.raises(ValidationError):
        AccountConfig(currency="dollars")


def test_bad_symbol_keys_and_unknown_symbol(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["symbols"]["eurusd"] = raw["symbols"]["EURUSD"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="upper-case"):
        load_config(p)
    with pytest.raises(KeyError):
        load_config(ROOT / "config.yaml").symbol("BTCUSD")


def test_non_mapping_config_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_non_session_days_format() -> None:
    from tfx.config import DataConfig

    with pytest.raises(ValidationError):
        DataConfig(non_session_days=["Christmas"])
