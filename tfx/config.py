"""Typed loading of ``config.yaml``.

Only the sections needed by the current build steps are strictly modelled; the
remaining sections are kept as plain dictionaries until their modules exist.
Safety-critical limits (risk hard cap, localhost binding) are validated here so
that an edited config file cannot silently loosen them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tfx.core_types import Timeframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

RISK_HARD_CAP_PCT = 1.0
"""Absolute ceiling on risk per trade. Not configurable upwards."""

ProviderName = Literal["yfinance", "stooq", "oanda", "mt5"]


class AccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    starting_balance: float = Field(default=1000.0, gt=0)
    risk_per_trade_pct: float = Field(default=0.5, gt=0)
    broker_profile: str = "typical_retail_ecn"

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"currency must be a 3-letter ISO code, got {v!r}")
        return v

    @field_validator("risk_per_trade_pct")
    @classmethod
    def _risk_cap(cls, v: float) -> float:
        if v > RISK_HARD_CAP_PCT:
            raise ValueError(
                f"risk_per_trade_pct {v} exceeds the hard cap of {RISK_HARD_CAP_PCT}%"
            )
        return v


class CrossCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    require_second_source: bool = True
    max_disagreement_atr: float = Field(default=0.2, gt=0)
    window_bars: int = Field(default=20, ge=1)
    reference_history_days: int = Field(default=120, ge=40)


class Mt5Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_time: str = "ny_close"   # "ny_close" | "utc" | fixed hours such as "+2"
    terminal_path: str | None = None

    @field_validator("server_time")
    @classmethod
    def _server_time(cls, v: str) -> str:
        if v in {"ny_close", "utc"}:
            return v
        try:
            float(v)
        except ValueError as exc:
            raise ValueError("mt5.server_time must be 'ny_close', 'utc' or an hour offset") from exc
        return v


class IntradayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    history_days: int = Field(default=3650, ge=30)  # requested; Yahoo caps 1h at ~730 days


class ResearchConfig(BaseModel):
    """Pre-declared quality thresholds for RESEARCH-ONLY MT5 H1 history.

    Fixed before any data is seen; they are not tuned to make a market pass.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    start: date = date(2005, 1, 1)
    min_years: float = Field(default=8.0, ge=4)        # training span + 3-year holdout
    min_coverage_pct: float = Field(default=95.0, gt=0, le=100)
    gap_hours: int = Field(default=4, ge=2)            # missing runs this long count as gaps
    outlier_atr: float = Field(default=15.0, gt=0)     # one-hour move > 15 x ATR(20) H1
    max_outlier_pct: float = Field(default=0.05, gt=0)
    max_disagree_pct: float = Field(default=5.0, gt=0)  # share of bars beyond the ATR limit
    min_hourly_compared: int = Field(default=2000, ge=100)
    min_daily_compared: int = Field(default=1000, ge=100)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_path: str = "var/tfx.sqlite"
    provider_order: list[ProviderName] = ["yfinance", "mt5", "oanda"]
    history_start: date = date(2003, 1, 1)
    day_boundary: Literal["utc_midnight", "ny_close"] = "utc_midnight"
    bar_complete_delay_minutes: int = Field(default=30, ge=0, le=6 * 60)
    max_signal_lag_sessions: int = Field(default=0, ge=0, le=2)
    incremental_overlap_days: int = Field(default=14, ge=1)
    overlap_tolerance_pct: float = Field(default=0.5, gt=0)
    spike_flag_pct: float = Field(default=8.0, gt=0)
    request_timeout_s: float = Field(default=20.0, gt=0)
    non_session_days: list[str] = ["12-25", "01-01"]
    cross_check: CrossCheckConfig = Field(default_factory=lambda: CrossCheckConfig())
    mt5: Mt5Config = Field(default_factory=lambda: Mt5Config())
    intraday: IntradayConfig = Field(default_factory=lambda: IntradayConfig())
    research: ResearchConfig = Field(default_factory=lambda: ResearchConfig())
    # Sources allowed ONLY for backtests/validation. They can never confirm a live bar.
    research_only_sources: list[str] = ["dukascopy"]

    @field_validator("non_session_days")
    @classmethod
    def _mmdd(cls, v: list[str]) -> list[str]:
        for item in v:
            month, _, day = item.partition("-")
            if not (month.isdigit() and day.isdigit() and 1 <= int(month) <= 12):
                raise ValueError(f"non_session_days entries must be MM-DD, got {item!r}")
        return v

    def resolved_db_path(self, root: Path = PROJECT_ROOT) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else root / p


class SymbolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: str
    quote: str
    pip: float = Field(gt=0)
    yahoo: str | None = None
    yahoo_is_proxy: bool = False
    stooq: str | None = None
    oanda: str | None = None
    mt5: str | None = None          # broker symbol; resolved automatically when omitted
    typical_spread_pips: float = Field(gt=0)
    providers: list[ProviderName] | None = None

    def ticker_for(self, provider: ProviderName) -> str | None:
        return {"yfinance": self.yahoo, "stooq": self.stooq, "oanda": self.oanda,
                "mt5": self.mt5}[provider]


class StrategyTimeframes(BaseModel):
    """Timeframes a strategy may run on. Each candidate is validated SEPARATELY.

    * ``candidates`` – signal/entry timeframes to test (a small pre-declared grid)
    * ``default``    – the one shown first; it still needs its own PASS to signal
    * ``bias``       – optional higher timeframe feeding the soft score (never a trigger)
    * ``execution``  – optional lower timeframe used only for entry timing and costs
    * ``rebalance``  – optional rebalance period (carry / cross-sectional momentum)
    """

    model_config = ConfigDict(extra="forbid")

    candidates: list[Timeframe] = Field(min_length=1, max_length=3)
    default: Timeframe
    bias: Timeframe | None = None
    execution: Timeframe | None = None
    rebalance: Timeframe | None = None

    @model_validator(mode="after")
    def _consistent(self) -> StrategyTimeframes:
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("duplicate timeframe candidates")
        if self.default not in self.candidates:
            raise ValueError("default timeframe must be one of the candidates")
        if self.bias is not None and any(c.is_higher_than(self.bias) for c in self.candidates):
            raise ValueError("bias timeframe must not be lower than any candidate")
        if self.execution is not None and any(self.execution.is_higher_than(c)
                                              for c in self.candidates):
            raise ValueError("execution timeframe must not be higher than any candidate")
        if self.rebalance is not None and any(c.is_higher_than(self.rebalance)
                                              for c in self.candidates):
            raise ValueError("rebalance period must not be shorter than the signal timeframe")
        return self


INFORMATIONAL_MAX_WEIGHT = 0.05


class SoftFactorConfig(BaseModel):
    """A soft factor in the setup score (never blocks a trade by itself).

    * ``mode: candidate``     – its weight is estimated on training data, frozen before
      the holdout, and kept only if it improves out-of-sample results.
    * ``mode: informational`` – shown to the user with a small FIXED weight because no
      history exists to validate it. It is never described as validated.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["candidate", "informational"] = "candidate"
    weight: float = Field(ge=0, le=1)
    enabled: bool = True
    note: str = ""

    @model_validator(mode="after")
    def _informational_is_small(self) -> SoftFactorConfig:
        if self.mode == "informational" and self.weight > INFORMATIONAL_MAX_WEIGHT:
            raise ValueError(f"informational factors are capped at weight "
                             f"{INFORMATIONAL_MAX_WEIGHT}")
        return self

    @property
    def is_validated_candidate(self) -> bool:
        return self.mode == "candidate"

    @property
    def display_label(self) -> str:
        return "informational - not validated" if self.mode == "informational" else \
            "validated on training data (see Validation tab)"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    # Extra names the app answers to. ONLY private Tailscale names (*.ts.net) are
    # allowed: they are reachable solely from your own signed-in devices through
    # `tailscale serve`, while the server itself stays bound to localhost.
    allowed_hosts: list[str] = Field(default_factory=list)

    @field_validator("allowed_hosts")
    @classmethod
    def _private_tailnet_only(cls, v: list[str]) -> list[str]:
        out = []
        for h in v:
            name = h.strip().lower()
            labels = name.split(".")
            if (not name.endswith(".ts.net") or len(labels) < 4 or "*" in name
                    or any(not lab or not all(c.isalnum() or c == "-" for c in lab)
                           for lab in labels)):
                raise ValueError(
                    f"server.allowed_hosts: {h!r} is not a private Tailscale name "
                    "(expected e.g. my-pc.tail1234.ts.net); public hosts are not allowed")
            out.append(name)
        return out

    @field_validator("host")
    @classmethod
    def _localhost_only(cls, v: str) -> str:
        if v not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "server.host must be localhost (127.0.0.1 / ::1); remote binding is disabled"
            )
        return v


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    random_seed: int = 42
    account: AccountConfig = AccountConfig()
    data: DataConfig = DataConfig()
    symbols: dict[str, SymbolConfig]
    server: ServerConfig = ServerConfig()

    @model_validator(mode="after")
    def _research_sources_not_live(self) -> AppConfig:
        research = set(self.data.research_only_sources)
        lists = [self.data.provider_order] + [s.providers or [] for s in self.symbols.values()]
        clash = sorted({p for lst in lists for p in lst if p in research})
        if clash:
            raise ValueError(f"research-only sources cannot be live providers: {clash}")
        self.scoring_factors()  # validates the soft-factor declarations
        return self

    def scoring_factors(self) -> dict[str, SoftFactorConfig]:
        raw = self.section("scoring").get("soft_factors", {})
        return {k: SoftFactorConfig.model_validate(v) for k, v in raw.items()}

    @model_validator(mode="after")
    def _strategy_timeframes_valid(self) -> AppConfig:
        self.strategy_timeframes()  # raises on any inconsistent declaration
        return self

    @model_validator(mode="after")
    def _symbols_upper(self) -> AppConfig:
        bad = [s for s in self.symbols if s != s.upper() or len(s) != 6]
        if bad:
            raise ValueError(f"symbol keys must be 6 upper-case letters: {bad}")
        for key, sym in self.symbols.items():
            if sym.mt5 is None:
                sym.mt5 = key  # Mt5Provider.resolve() maps it to the broker's name
        return self

    def providers_for(self, symbol: str) -> list[ProviderName]:
        """LIVE data sources for ``symbol`` (research-only sources are never included)."""
        sym = self.symbol(symbol)
        research = set(self.data.research_only_sources)
        return [p for p in (sym.providers or self.data.provider_order) if p not in research]

    def is_live_source(self, source: str) -> bool:
        return source not in set(self.data.research_only_sources)

    def symbol(self, symbol: str) -> SymbolConfig:
        try:
            return self.symbols[symbol.upper()]
        except KeyError as exc:
            known = sorted(self.symbols)
            raise KeyError(f"unknown symbol {symbol!r}; configured: {known}") from exc

    def strategy_timeframes(self) -> dict[str, StrategyTimeframes]:
        """Every strategy that declares ``timeframes`` in ``config.yaml``."""
        out: dict[str, StrategyTimeframes] = {}
        for name, body in self.section("strategies").items():
            if isinstance(body, dict) and "timeframes" in body:
                out[name] = StrategyTimeframes.model_validate(body["timeframes"])
        return out

    def section(self, name: str) -> dict[str, Any]:
        """Raw access to sections not yet strictly modelled (strategies, risk, ...)."""
        extra = self.model_extra or {}
        value = extra.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"config section {name!r} must be a mapping")
        return value


def load_config(path: Path | str | None = None) -> AppConfig:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return AppConfig.model_validate(raw)
