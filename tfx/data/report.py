"""Per-symbol data-quality report.

For each market: history span, gaps, outliers, how often the two sources agree,
the latest bar both sources confirm, and whether signals are allowed. When the
MT5 terminal is not connected, :data:`MT5_SETUP_STEPS` says exactly what to do.
Nothing here relaxes the rules: an unconfirmed bar never becomes a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tfx.data.crosscheck import disagreement_series
from tfx.data.quality import profile_series
from tfx.data.service import MarketDataService

MT5_SETUP_STEPS = """\
MetaTrader 5 is not connected, so no bar can be confirmed and NO SIGNALS will be shown.
To fix it:
  1. Open a free MT5 *demo* account with a broker licensed by Kenya's CMA (cma.or.ke).
  2. Start MetaTrader 5 and use File > Login to Trade Account (login, password, server).
  3. Check the bottom-right corner shows traffic (e.g. "1234 / 5 kb"), not "No connection".
  4. Leave MetaTrader 5 open and run:  .\\make.ps1 quality
TrillionaireFX only reads prices from MT5. It never places orders."""


@dataclass(frozen=True)
class SymbolQuality:
    symbol: str
    status: str
    primary: str | None
    is_proxy: bool
    first_bar: date | None
    last_bar: date | None
    bars: int
    gaps: int
    outliers: int
    reference: str | None
    compared_bars: int
    agreement_pct: float | None  # share of compared bars within the ATR limit
    latest_confirmed: date | None
    signal_allowed: bool
    message: str


def build_quality_report(svc: MarketDataService, refresh: bool = True) -> list[SymbolQuality]:
    limit = svc.cfg.data.cross_check.max_disagreement_atr
    out: list[SymbolQuality] = []
    for symbol in svc.cfg.symbols:
        md = svc.get(symbol, refresh=refresh)
        candles = md.candles
        gaps, spikes = profile_series(candles, svc.calendar, svc.cfg.data.spike_flag_pct)
        ref, ref_source = svc.cache.load_reference(symbol)
        if ref_source is not None and ref_source != md.source and not candles.empty:
            dis = disagreement_series(candles, ref)
        else:
            dis, ref_source = candles["close"].iloc[0:0], None
        ok = dis[dis <= limit]
        out.append(SymbolQuality(
            symbol=symbol, status=md.status.value, primary=md.source, is_proxy=md.is_proxy,
            first_bar=candles.index[0].date() if not candles.empty else None,
            last_bar=md.last_bar, bars=len(candles), gaps=len(gaps), outliers=len(spikes),
            reference=ref_source, compared_bars=len(dis),
            agreement_pct=float(len(ok) / len(dis) * 100) if len(dis) else None,
            latest_confirmed=ok.index[-1].date() if len(ok) else None,
            signal_allowed=md.signal_allowed, message=md.message,
        ))
    return out


def format_quality_report(rows: list[SymbolQuality], mt5_line: str, mt5_ready: bool) -> str:
    lines = [mt5_line, ""]
    head = (f"{'SYMBOL':7} {'STATUS':11} {'PRIMARY':16} {'HISTORY':23} {'GAPS':>4} "
            f"{'OUTL':>4} {'REF':5} {'CMP':>4} {'AGREE':>6} {'CONFIRMED':10} SIGNAL")
    lines.append(head)
    for r in rows:
        span = f"{r.first_bar} to {r.last_bar}" if r.first_bar else "-"
        prim = (r.primary or "-") + (" (proxy)" if r.is_proxy else "")
        agree = f"{r.agreement_pct:5.1f}%" if r.agreement_pct is not None else "     -"
        lines.append(f"{r.symbol:7} {r.status:11} {prim:16} {span:23} {r.gaps:>4} "
                     f"{r.outliers:>4} {r.reference or '-':5} {r.compared_bars:>4} {agree:>6} "
                     f"{r.latest_confirmed or '-'!s:10} {'yes' if r.signal_allowed else 'NO'}")
    lines.append("")
    lines.append("GAPS = bars preceded by missing sessions; OUTL = daily moves over the spike "
                 "limit (review, may be bad ticks);")
    lines.append("AGREE = share of compared bars within the cross-check ATR limit; CONFIRMED = "
                 "latest bar both sources agree on.")
    if any(r.is_proxy for r in rows):
        lines.append("(proxy) = Yahoo gold futures used as a stand-in for spot gold; each bar must "
                     "still be confirmed by MT5 spot gold.")
    if not mt5_ready:
        lines += ["", MT5_SETUP_STEPS]
    return "\n".join(lines)
