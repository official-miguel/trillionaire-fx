"""Local REST API + static web app. Bound to localhost only; never places orders."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tfx import __version__, coach
from tfx.alerts.notify import Notifier
from tfx.alerts.scheduler import run_daily
from tfx.core_types import Timeframe
from tfx.data.chart_bars import ALL_TIMEFRAMES, INTRADAY, ChartData
from tfx.data.intraday import IntradayService
from tfx.data.report import build_quality_report
from tfx.data.timeframes import resample_daily
from tfx.execution import cost_in_r, execution_quality, slippage_report
from tfx.mindset import CHECKLIST, PRINCIPLES, DisciplineError, Mindset
from tfx.missed import hypothetical_r, missed_report
from tfx.paper import paper_status
from tfx.pipeline import Pipeline, account
from tfx.store import DrawingRow, FillRow, JournalRow, TradeRow
from tfx.strategies.features import build_features
from tfx.strategies.rules import StrategyParams
from tfx.validation.proof import costs_for, load_report

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
RISK_NOTICE = ("Educational decision support. Trading carries risk of loss. Past performance "
               "does not guarantee future results.")


class SettingsIn(BaseModel):
    balance: float | None = Field(default=None, gt=0)
    risk_pct: float | None = Field(default=None, gt=0, le=1.0)
    currency: str | None = None
    mode: str | None = None
    theme: str | None = None
    user_spread_pips: float | None = Field(default=None, ge=0)
    user_commission_pips: float | None = Field(default=None, ge=0)


class EmotionIn(BaseModel):
    confidence: int = Field(ge=1, le=5)
    stress: int = Field(ge=1, le=5)
    urgency: int = Field(ge=1, le=5)


class TradeIn(BaseModel):
    signal_id: int | None = None
    symbol: str
    direction: int = Field(description="+1 long, -1 short")
    entry: float = Field(gt=0)
    stop: float = Field(gt=0)
    units: float | None = Field(default=None, gt=0)
    planned_entry: float | None = None
    checklist: dict[str, bool] = {}
    override_reason: str = ""
    notes: str = ""
    screenshot: str = ""


class StopIn(BaseModel):
    stop: float = Field(gt=0)


class CloseIn(BaseModel):
    exit_price: float = Field(gt=0)
    followed_rules: bool
    lesson: str = ""
    planned_exit: float | None = None


class JournalIn(BaseModel):
    kind: str = "note"
    text: str


class DrawingIn(BaseModel):
    kind: str
    data: dict[str, Any]


def _ts(idx: pd.DatetimeIndex) -> list[int]:
    i = idx if idx.tz is not None else idx.tz_localize("UTC")
    return [int(x) for x in (i.as_unit("s").asi8)]


def _series(times: list[int], values: pd.Series) -> list[dict[str, Any]]:
    v = values.to_numpy(float)
    return [{"time": t, "value": round(float(x), 6)} for t, x in zip(times, v, strict=True)
            if np.isfinite(x)]


def create_app(pipeline: Pipeline, notifier: Notifier, intraday: IntradayService | None = None,
               clock: Any = None, chart: ChartData | None = None) -> FastAPI:
    app = FastAPI(title="TrillionaireFX", version=__version__, docs_url="/api/docs")
    svc, store, cfg = pipeline.svc, pipeline.store, pipeline.cfg
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]",
                                      *cfg.server.allowed_hosts])
    chart = chart or ChartData(svc.providers.get("mt5"))
    now = clock or (lambda: datetime.now(UTC))
    mind = Mindset(store, cfg.section("mindset"), now)

    @app.middleware("http")
    async def invalidate(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            pipeline.clear_cache()
        path = request.url.path
        if path == "/" or (path.startswith("/static/") and "/vendor/" not in path):
            response.headers["Cache-Control"] = "no-cache"  # always pick up app updates
        return response

    def ensure_symbol(symbol: str) -> str:
        s = symbol.upper()
        if s not in cfg.symbols:
            raise HTTPException(404, f"unknown symbol {symbol}")
        return s

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "time": now().isoformat()}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        menus = {}
        for name, tfs in cfg.strategy_timeframes().items():
            items = pipeline.registry.timeframe_menu(name, tfs.candidates, now())
            menus[name] = [{"timeframe": m.timeframe.value, "validated": m.validated,
                            "label": m.label} for m in items]
        return {"version": __version__, "symbols": list(cfg.symbols), "principles": PRINCIPLES,
                "checklist": CHECKLIST, "risk_notice": RISK_NOTICE, "timeframes": menus,
                "view_timeframes": list(ALL_TIMEFRAMES),
                "attribution": "Charts by TradingView Lightweight Charts (Apache-2.0)."}

    # -------------------------------------------------------- market / signals
    @app.get("/api/watchlist")
    def watchlist() -> list[dict[str, Any]]:
        return pipeline.watchlist(refresh=False)

    @app.get("/api/signal/{symbol}")
    def signal(symbol: str) -> dict[str, Any]:
        return pipeline.evaluate(ensure_symbol(symbol), refresh=False, now=now())

    @app.get("/api/analysis/{symbol}")
    def analysis(symbol: str) -> dict[str, Any]:
        """Top-down market read (deterministic, analysis only - never a signal)."""
        from tfx.analysis import market_read

        sym = ensure_symbol(symbol)
        md = pipeline.svc.get(sym, refresh=False)
        digits = 3 if pipeline.cfg.symbol(sym).pip >= 0.01 else 5
        return {"symbol": sym, "data_status": md.status.value,
                **market_read(md.candles, digits, pipeline.sd_params())}

    @app.get("/api/candles/{symbol}")
    def candles(symbol: str, tf: str = Query("D1")) -> dict[str, Any]:
        sym = ensure_symbol(symbol)
        if tf not in ALL_TIMEFRAMES:
            raise HTTPException(422, f"timeframe must be one of {', '.join(ALL_TIMEFRAMES)}")
        md = svc.get(sym, refresh=False)
        validated_view = tf == "D1"
        source, src_note = md.source or "", ""
        if tf in INTRADAY:
            sc = cfg.symbol(sym)
            bars = chart.intraday(sym, sc.mt5, sc.yahoo, tf)
            df, source, src_note = bars.frame, bars.source, bars.note
        elif tf in ("W1", "MN1"):
            df = resample_daily(md.candles, Timeframe(tf), svc.calendar,
                                md.last_bar or now().date())
        else:
            df = md.candles
        if df.empty:
            msg = src_note or md.message
            return {"symbol": sym, "tf": tf, "candles": [], "overlays": {}, "markers": [],
                    "data": {"status": md.status.value, "message": msg}, "source": source}
        # D1: the validated feature frame; other timeframes: display-only indicators
        f = pipeline.features(md) if tf == "D1" else (
            build_features(df) if len(df) >= 60 else None)
        times = _ts(pd.DatetimeIndex(df.index))
        out: dict[str, Any] = {
            "symbol": sym, "tf": tf, "validated_view": validated_view,
            "note": "" if validated_view else (
                f"{tf} view only - not validated, no signals" +
                (f" · {src_note}" if src_note else "")),
            "source": source, "live": tf in INTRADAY and source == "mt5",
            "candles": [{"time": t, "open": float(o), "high": float(h), "low": float(lo),
                         "close": float(c)} for t, o, h, lo, c in zip(
                times, df["open"], df["high"], df["low"], df["close"], strict=True)],
            "data": {"status": md.status.value, "message": md.message, "proxy": md.is_proxy},
            "overlays": {}, "markers": [], "regimes": []}
        if f is not None:
            ft = _ts(pd.DatetimeIndex(f.index))
            ov = out["overlays"]
            for n in (20, 55, 100):
                ov[f"donchian_high_{n}"] = _series(ft, f[f"dch_{n}"])
                ov[f"donchian_low_{n}"] = _series(ft, f[f"dcl_{n}"])
            for n in (20, 100, 200):
                ov[f"ema_{n}"] = _series(ft, f[f"ema_{n}"])
            ov["atr_stop_long"] = _series(ft, f["close"] - 2 * f["atr_20"])
            ov["adx"] = _series(ft, f["adx_14"])
            ov["rsi2"] = _series(ft, f["rsi_2"])
            ov["atr"] = _series(ft, f["atr_20"])
            reg = f["regime"]
            seg_start = 0
            for i in range(1, len(reg) + 1):
                if i == len(reg) or reg.iloc[i] != reg.iloc[seg_start]:
                    out["regimes"].append({"from": ft[seg_start], "to": ft[i - 1],
                                           "regime": str(reg.iloc[seg_start])})
                    seg_start = i
            for t in (store.trades() if tf == "D1" else []):
                if t.symbol != sym:
                    continue
                d0 = int(pd.Timestamp(t.opened_at.date(), tz="UTC").timestamp())
                out["markers"].append({"time": d0, "position": "belowBar" if t.direction == 1
                                       else "aboveBar", "shape": "arrowUp" if t.direction == 1
                                       else "arrowDown", "text": f"#{t.id} entry {t.entry:.5g}"})
                if t.closed_at is not None:
                    d1 = int(pd.Timestamp(t.closed_at.date(), tz="UTC").timestamp())
                    out["markers"].append({"time": d1, "position": "aboveBar", "shape": "circle",
                                           "text": f"#{t.id} exit {t.r_result:+.2f}R"})
        try:  # ICT / SMC annotations for the viewed timeframe (display only)
            from tfx.strategies.ict import ICTParams, chart_annotations
            out["ict"] = chart_annotations(df, ICTParams(swing_n=3 if tf in ("D1", "W1", "MN1")
                                                         else 5))
        except Exception:  # never break the chart because of an annotation
            out["ict"] = None
        try:
            from tfx.strategies.supply_demand import SDParams, chart_zones
            if out.get("ict") is not None:
                out["ict"]["sd_zones"] = chart_zones(df, pipeline.sd_params() if tf == "D1"
                                                     else SDParams())
        except Exception:
            pass
        if tf == "D1":
            ev = pipeline.evaluate(sym, refresh=False, now=now(), persist=False)
            if ev["card"].get("watch_level"):
                out["watch_level"] = ev["card"]["watch_level"]
            if ev["card"].get("stop"):
                out["signal_levels"] = {"entry": ev["card"]["entry"], "stop": ev["card"]["stop"]}
            if ev["card"].get("setup"):
                out["setup"] = ev["card"]["setup"]
        return out

    @app.get("/api/live/{symbol}")
    def live(symbol: str, tf: str = Query("M15")) -> dict[str, Any]:
        """Latest bars + MT5 bid/ask for live chart updates (display only)."""
        sym = ensure_symbol(symbol)
        sc = cfg.symbol(sym)
        price = chart.live_price(sc.mt5)
        if tf not in INTRADAY:
            return {"symbol": sym, "tf": tf, "price": price, "bars": []}
        bars = chart.latest(sym, sc.mt5, sc.yahoo, tf, 3)
        tail = bars.frame
        return {"symbol": sym, "tf": tf, "price": price, "source": bars.source,
                "bars": [{"time": t, "open": float(o), "high": float(h), "low": float(lo),
                          "close": float(c)} for t, o, h, lo, c in zip(
                    _ts(pd.DatetimeIndex(tail.index)), tail["open"], tail["high"],
                    tail["low"], tail["close"], strict=True)]}

    @app.get("/api/replay")
    def replay_endpoint(start: str, end: str, balance: float = Query(100.0, gt=0),
                        risk_pct: float = Query(0.5, gt=0, le=1.0),
                        strategy: str = Query("all")) -> dict[str, Any]:
        from datetime import date as _date

        from tfx.replay import as_dict, replay
        from tfx.strategies.rules import STRATEGIES
        if strategy != "all" and strategy not in STRATEGIES:
            raise HTTPException(422, f"strategy must be all or one of {', '.join(STRATEGIES)}")
        chosen = STRATEGIES if strategy == "all" else (strategy,)
        try:
            s, e = _date.fromisoformat(start), _date.fromisoformat(end)
        except ValueError as exc:
            raise HTTPException(422, "dates must be YYYY-MM-DD") from exc
        if e < s or (e - s).days > 370:
            raise HTTPException(422, "end must be after start and within a year")
        currency = str(store.get_setting("currency", cfg.account.currency))
        return as_dict(replay(cfg, svc.cache, s, e, balance, risk_pct, currency, chosen))

    @app.get("/api/replay/compare")
    def replay_compare(start: str, end: str, balance: float = Query(100.0, gt=0),
                       risk_pct: float = Query(0.5, gt=0, le=1.0)) -> dict[str, Any]:
        from datetime import date as _date

        from tfx.replay import compare
        try:
            s, e = _date.fromisoformat(start), _date.fromisoformat(end)
        except ValueError as exc:
            raise HTTPException(422, "dates must be YYYY-MM-DD") from exc
        if e < s or (e - s).days > 370:
            raise HTTPException(422, "end must be after start and within a year")
        currency = str(store.get_setting("currency", cfg.account.currency))
        return compare(cfg, svc.cache, s, e, balance, risk_pct, currency)

    @app.get("/api/quality")
    def quality() -> list[dict[str, Any]]:
        return [r.__dict__ | {"first_bar": str(r.first_bar or ""),
                              "last_bar": str(r.last_bar or ""),
                              "latest_confirmed": str(r.latest_confirmed or "")}
                for r in build_quality_report(svc, refresh=False)]

    @app.get("/api/validation")
    def validation() -> dict[str, Any]:
        rep = load_report()
        gates = {}
        for name, tfs in cfg.strategy_timeframes().items():
            for tf in tfs.candidates:
                g = pipeline.registry.gate(name, tf, now())
                gates[f"{name}:{tf.value}"] = {"allowed": g.allowed, "reason": g.reason}
        decay = []
        from sqlalchemy import select

        from tfx.validation.registry import ValidationResult
        with pipeline.registry.engine.connect() as c:
            for row in c.execute(select(ValidationResult).order_by(
                    ValidationResult.validated_at)):
                m = json.loads(row.metrics or "{}")
                if "edge_confidence" in m:
                    decay.append({"strategy": row.strategy, "timeframe": row.timeframe,
                                  "at": row.validated_at.isoformat(),
                                  "edge_confidence": m["edge_confidence"],
                                  "verdict": row.verdict})
        return {"report": rep, "gates": gates, "edge_decay": decay,
                "trials_counted": pipeline.registry.trial_count()}

    # --------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        a = account(cfg, store)
        return {"balance": a.balance, "risk_pct": a.risk_pct, "currency": a.currency,
                "mode": store.get_setting("mode", "paper"),
                "theme": store.get_setting("theme", "dark"),
                "risk_lock": store.get_setting("risk_lock"),
                "risk_hard_cap": 1.0,
                "user_spread_pips": store.get_setting("user_spread_pips"),
                "user_commission_pips": store.get_setting("user_commission_pips"),
                "telegram_configured": notifier.telegram_configured()}

    @app.put("/api/settings")
    def put_settings(body: SettingsIn) -> dict[str, Any]:
        if body.risk_pct is not None:
            try:
                mind.set_risk_pct(body.risk_pct)
            except DisciplineError as exc:
                raise HTTPException(409, str(exc)) from exc
        if body.balance is not None:
            store.set_setting("balance", body.balance)
            peak = float(store.get_setting("equity_peak", body.balance))
            store.set_setting("equity_peak", max(peak, body.balance))
        if body.currency is not None:
            ccy = body.currency.upper()
            if len(ccy) != 3 or not ccy.isalpha():
                raise HTTPException(422, "currency must be a 3-letter code")
            store.set_setting("currency", ccy)
        for k in ("mode", "theme", "user_spread_pips", "user_commission_pips"):
            v = getattr(body, k)
            if v is not None:
                if k == "mode" and v not in ("paper", "live"):
                    raise HTTPException(422, "mode is paper or live")
                store.set_setting(k, v)
                if k == "mode" and v == "paper" and not store.get_setting("paper_started"):
                    store.set_setting("paper_started", now().isoformat())
        return get_settings()

    # ---------------------------------------------------------------- mindset
    @app.post("/api/emotions")
    def emotions(body: EmotionIn) -> dict[str, Any]:
        g = mind.check_in(body.confidence, body.stress, body.urgency)
        return {"allowed": g.allowed, "blockers": g.blockers,
                "wait_until": g.wait_until.isoformat() if g.wait_until else None}

    @app.get("/api/mindset")
    def mindset() -> dict[str, Any]:
        gate = mind.entry_gate({k: True for k in CHECKLIST})
        return {"principles": PRINCIPLES, "checklist": CHECKLIST,
                "entry_allowed": gate.allowed, "blockers": gate.blockers,
                "wait_until": gate.wait_until.isoformat() if gate.wait_until else None,
                "trades_this_week": mind.trades_this_week(),
                "reflection_required": mind.reflection_required(),
                "weekly_review": mind.weekly_review(),
                "expectations": _expectations(),
                "breaches": [{"at": b.at.isoformat(), "kind": b.kind, "detail": b.detail,
                              "trade_id": b.trade_id} for b in store.breaches()[-50:]]}

    def _expectations() -> dict[str, Any]:
        rep = load_report() or {}
        out = {}
        for p in rep.get("pairs", []):
            if p.get("monte_carlo"):
                out[p["strategy"]] = {**p["monte_carlo"], "verdict": p["verdict"],
                                      "note": "Historical ranges from validation, not a "
                                              "forecast."}
        return out

    # ----------------------------------------------------------------- trades
    def _trade_dict(t: TradeRow) -> dict[str, Any]:
        d = {k: getattr(t, k) for k in ("id", "signal_id", "mode", "symbol", "strategy",
                                        "direction", "entry", "units", "initial_stop", "stop",
                                        "risk_pct", "amount_at_risk", "exit_price", "r_result",
                                        "off_signal", "notes", "followed_rules", "lesson",
                                        "partial_done", "adds")}
        d["opened_at"] = t.opened_at.isoformat()
        d["closed_at"] = t.closed_at.isoformat() if t.closed_at else None
        return d

    @app.get("/api/trades")
    def trades(status: str | None = None) -> list[dict[str, Any]]:
        return [_trade_dict(t) for t in store.trades(status)]

    @app.post("/api/trades")
    def log_trade(body: TradeIn) -> dict[str, Any]:
        sym = ensure_symbol(body.symbol)
        if body.direction not in (1, -1):
            raise HTTPException(422, "direction must be +1 or -1")
        a = account(cfg, store)
        risk_cash = a.balance * a.risk_pct / 100
        units = body.units
        if units is None:
            from tfx.pipeline import fx_converter
            from tfx.risk.engine import position_size
            try:
                ps = position_size(a.balance, a.risk_pct, body.entry, body.stop,
                                   cfg.symbol(sym), a.currency, fx_converter(svc, store))
            except (KeyError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
            units = ps.units
        sig = store.signal(body.signal_id) if body.signal_id else None
        t = TradeRow(signal_id=body.signal_id, mode=store.get_setting("mode", "paper"),
                     symbol=sym, strategy=sig.strategy if sig else "manual",
                     direction=body.direction, planned_entry=body.planned_entry or (
                         sig.order_price if sig else None),
                     planned_stop=sig.stop if sig else None, planned_units=units,
                     entry=body.entry, units=units, initial_stop=body.stop, stop=body.stop,
                     risk_pct=a.risk_pct, amount_at_risk=round(risk_cash, 2), notes=body.notes,
                     screenshot=body.screenshot)
        try:
            saved = mind.open_trade(t, body.checklist, body.override_reason)
        except DisciplineError as exc:
            raise HTTPException(409, str(exc)) from exc
        if t.planned_entry:
            pip = cfg.symbol(sym).pip
            slip = body.direction * (body.entry - t.planned_entry) / pip
            store.add(FillRow(trade_id=saved.id, symbol=sym, side="entry",
                              planned=t.planned_entry, actual=body.entry, slippage_pips=slip,
                              utc_hour=now().hour))
        if sig is not None:
            sig.outcome = "TAKEN"
            store.save(sig)
        return _trade_dict(saved)

    @app.patch("/api/trades/{trade_id}/stop")
    def move_stop(trade_id: int, body: StopIn) -> dict[str, Any]:
        try:
            return _trade_dict(mind.update_stop(trade_id, body.stop))
        except KeyError as exc:
            raise HTTPException(404, "trade not found") from exc
        except DisciplineError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/trades/{trade_id}/close")
    def close_trade(trade_id: int, body: CloseIn) -> dict[str, Any]:
        try:
            t = mind.close_trade(trade_id, body.exit_price, body.followed_rules, body.lesson)
        except KeyError as exc:
            raise HTTPException(404, "trade not found") from exc
        if body.planned_exit:
            pip = cfg.symbol(t.symbol).pip
            store.add(FillRow(trade_id=t.id, symbol=t.symbol, side="exit",
                              planned=body.planned_exit, actual=body.exit_price,
                              slippage_pips=-t.direction * (body.exit_price - body.planned_exit)
                              / pip, utc_hour=now().hour))
        bal = float(store.get_setting("balance", cfg.account.starting_balance))
        new_bal = bal + (t.r_result or 0) * t.amount_at_risk
        store.set_setting("balance", round(new_bal, 2))
        store.set_setting("equity_peak", max(float(store.get_setting("equity_peak", bal)),
                                             new_bal))
        return {"trade": _trade_dict(t), "review": coach.post_trade_review(
            t, len(store.stop_changes(t.id)), t.followed_rules)}

    @app.get("/api/trades/{trade_id}/review")
    def review(trade_id: int) -> dict[str, Any]:
        t = store.trade(trade_id)
        if t is None or t.closed_at is None:
            raise HTTPException(404, "closed trade not found")
        return coach.post_trade_review(t, len(store.stop_changes(t.id)), t.followed_rules)

    # ---------------------------------------------------------------- journal
    @app.get("/api/journal")
    def journal() -> list[dict[str, Any]]:
        return [{"at": j.at.isoformat(), "kind": j.kind, "text": j.text,
                 "trade_id": j.trade_id} for j in store.journal()]

    @app.post("/api/journal")
    def add_journal(body: JournalIn) -> dict[str, Any]:
        try:
            if body.kind == "reflection":
                mind.submit_reflection(body.text)
                store.set_setting("breaker_reviewed_at", now().isoformat())
            elif body.kind == "cooldown_review":
                mind.submit_cooldown_review(body.text)
            else:
                store.add(JournalRow(kind="note", text=body.text))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------- signals / missed / paper
    @app.get("/api/signals")
    def signals_history() -> list[dict[str, Any]]:
        return [{"id": s.id, "session_date": s.session_date, "symbol": s.symbol,
                 "strategy": s.strategy, "status": s.status, "direction": s.direction,
                 "grade": s.grade, "score": s.score, "entry": s.entry_ref, "stop": s.stop,
                 "order": f"{s.order_type} {s.order_price} ({s.expiry})",
                 "reasons": json.loads(s.reasons), "outcome": s.outcome}
                for s in store.signals()]

    @app.get("/api/missed")
    def missed() -> dict[str, Any]:
        sigs = store.signals(limit=300)
        taken = {t.signal_id for t in store.trades() if t.signal_id}
        p = StrategyParams.from_config(cfg.section("strategies"))
        from tfx.backtest.engine import Management
        m = Management.from_config(cfg.section("strategies").get("management", {}))
        hypo: dict[int, float | None] = {}
        rows = []
        for s in sigs:
            reason = ", ".join(json.loads(s.reasons)) if s.reasons != "[]" else (
                "low score" if s.grade in ("WAIT", "PROBE") else "")
            rows.append({"id": s.id, "status": s.status, "symbol": s.symbol,
                         "date": s.session_date, "size_mult": s.size_mult,
                         "strategy": s.strategy, "reason": reason})
            if s.direction and s.stop and s.id not in taken:
                md = svc.get(s.symbol, refresh=False)
                f = pipeline.features(md)
                hypo[s.id] = hypothetical_r(s.symbol, s.strategy, s.direction, s.stop,
                                            pd.Timestamp(s.session_date), md.candles, f, p, m,
                                            costs_for(cfg, s.symbol))
        rep = missed_report(rows, taken, hypo)
        return {"rows": rep.rows, "by_reason": rep.by_reason, "capture_rate": rep.capture_rate,
                "valid_setups": rep.valid_setups, "taken": rep.taken,
                "net_effect_r": rep.net_effect_r}

    @app.get("/api/paper")
    def paper() -> dict[str, Any]:
        started = store.get_setting("paper_started")
        rs = [t.r_result for t in store.trades("closed", "paper") if t.r_result is not None]
        rep = load_report() or {}
        cis = [tuple(p["expectancy_ci"]) for p in rep.get("pairs", [])
               if p.get("verdict") == "PASS" and p.get("expectancy_ci")]
        ci = (min(c[0] for c in cis), max(c[1] for c in cis)) if cis else None
        return paper_status(rs, datetime.fromisoformat(started).replace(tzinfo=None)
                            if started else None, now().replace(tzinfo=None), ci,
                            int(cfg.section("paper").get("min_weeks", 8)))

    @app.get("/api/execution/{symbol}")
    def execution(symbol: str) -> dict[str, Any]:
        sym = ensure_symbol(symbol)
        sbh = intraday.spread_by_hour(sym) if intraday else pd.Series(dtype=float)
        q = execution_quality(sbh, None, now())
        sym_cfg = cfg.symbol(sym)
        prof = cfg.section("costs").get("profiles", {}).get(cfg.account.broker_profile, {})
        user_spread = store.get_setting("user_spread_pips")
        spread = float(user_spread) if user_spread is not None else sym_cfg.typical_spread_pips
        comm = float(store.get_setting("user_commission_pips") or prof.get("commission_pips",
                                                                             0.7))
        md = svc.get(sym, refresh=False)
        f = pipeline.features(md)
        stop_pips = float(2 * f["atr_20"].iloc[-1] / sym_cfg.pip) if len(f) else 0.0
        c_r = cost_in_r(spread, comm, float(prof.get("slippage_pips", 0.2)), stop_pips)
        assumed_r = cost_in_r(sym_cfg.typical_spread_pips, float(prof.get("commission_pips", 0.7)),
                              float(prof.get("slippage_pips", 0.2)), stop_pips)
        rep = load_report() or {}
        exps = [p["full"]["expectancy_r"] for p in rep.get("pairs", []) if p.get("full")]
        best_exp = max(exps) if exps else None
        slip = slippage_report([(x.symbol, x.slippage_pips) for x in store.fills()],
                               float(prof.get("slippage_pips", 0.2)))
        warn = None
        if best_exp is not None and best_exp - (c_r - assumed_r) <= 0:
            warn = "At your costs the historical edge would be negative. Do not trade this."
        if slip.revalidate:
            warn = ((warn + " ") if warn else "") + ("Real slippage exceeds the assumption by "
                                                     "more than 25%: re-run validation.")
            store.set_setting("revalidate_required", True)
        return {"symbol": sym, "quality": q, "spread_by_hour": {int(k): float(v) for k, v in
                                                                 sbh.items()},
                "cost_per_trade_r": round(c_r, 3), "assumed_cost_per_trade_r": round(
                    assumed_r, 3), "your_spread_pips": spread, "assumed_spread_pips":
                    sym_cfg.typical_spread_pips, "slippage": slip.__dict__, "warning": warn}

    # --------------------------------------------------------------- drawings
    @app.get("/api/drawings/{symbol}")
    def drawings(symbol: str) -> list[dict[str, Any]]:
        sym = ensure_symbol(symbol)
        return [{"id": d.id, "kind": d.kind, "data": json.loads(d.data)}
                for d in store.drawings(sym)]

    @app.post("/api/drawings/{symbol}")
    def add_drawing(symbol: str, body: DrawingIn) -> dict[str, Any]:
        sym = ensure_symbol(symbol)
        if body.kind not in ("hline", "trendline", "rect", "measure", "text", "long", "short"):
            raise HTTPException(422, "unknown drawing kind")
        row = store.add(DrawingRow(symbol=sym, kind=body.kind, data=json.dumps(body.data)))
        return {"id": row.id, "kind": row.kind, "data": body.data}

    @app.delete("/api/drawings/item/{drawing_id}")
    def del_drawing(drawing_id: int) -> dict[str, bool]:
        if not store.delete_drawing(drawing_id):
            raise HTTPException(404, "drawing not found")
        return {"ok": True}

    # ----------------------------------------------------------------- alerts
    @app.get("/api/alerts")
    def alerts() -> list[dict[str, Any]]:
        return [{"at": a.at.isoformat(), "channel": a.channel, "kind": a.kind, "text": a.text,
                 "ok": a.ok} for a in store.alerts()]

    @app.post("/api/run-daily")
    def run_daily_now() -> dict[str, Any]:
        return run_daily(pipeline, notifier, refresh=True, intraday=intraday)

    # ---------------------------------------------------------------- static
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    @app.exception_handler(Exception)
    async def errors(_: Any, exc: Exception) -> JSONResponse:  # pragma: no cover
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
