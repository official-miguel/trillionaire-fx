"""Command line: ``python -m tfx {run,daily,fetch,status,quality,sources,research,validate}``.

* ``run``      start the local web app at http://127.0.0.1:8765 plus the daily scheduler
* ``daily``    one daily evaluation + alerts (for Windows Task Scheduler / cron)
* ``validate`` run the strategy proof phase (validation gate)
"""

from __future__ import annotations

import argparse
import sys

from tfx.config import load_config
from tfx.data import CandleCache, MarketDataService, build_providers
from tfx.data.crosscheck import disagreement_series
from tfx.data.report import MT5_SETUP_STEPS, build_quality_report, format_quality_report
from tfx.env import load_env
from tfx.logging_setup import configure_logging


def _service() -> MarketDataService:
    cfg = load_config()
    cache = CandleCache(cfg.data.resolved_db_path())
    providers = build_providers(cfg.data.request_timeout_s, cfg.data.mt5.server_time,
                                cfg.data.mt5.terminal_path)
    return MarketDataService(cfg, cache, providers)


def _app_parts(svc: MarketDataService):  # type: ignore[no-untyped-def]
    from tfx.alerts.notify import Notifier
    from tfx.data.intraday import IntradayService
    from tfx.pipeline import Pipeline
    from tfx.store import Store
    from tfx.validation.registry import ValidationRegistry

    db = svc.cfg.data.resolved_db_path()
    store = Store(db)
    pipeline = Pipeline(svc, store, ValidationRegistry(db))
    return pipeline, Notifier(store), IntradayService(svc.cfg, svc.cache, svc.providers,
                                                      svc.clock)


def _run_server(svc: MarketDataService) -> int:  # pragma: no cover - starts a server
    import threading

    import uvicorn

    from tfx.alerts.scheduler import start_scheduler
    from tfx.api.app import create_app

    pipeline, notifier, intraday = _app_parts(svc)
    start_scheduler(pipeline, notifier, intraday)
    # warm the evaluation cache so the first page load is fast
    threading.Thread(target=pipeline.watchlist, daemon=True).start()
    host, port = svc.cfg.server.host, svc.cfg.server.port
    print(f"TrillionaireFX running at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(pipeline, notifier, intraday), host=host, port=port,
                log_level="warning")
    return 0


def _daily(svc: MarketDataService) -> int:
    from tfx.alerts.scheduler import run_daily

    pipeline, notifier, intraday = _app_parts(svc)
    out = run_daily(pipeline, notifier, refresh=True, intraday=intraday)
    print(f"Evaluated {out['evaluated']} markets; alerts sent for: {out['alerts'] or 'none'}")
    return 0


def _print_status(svc: MarketDataService, refresh: bool) -> int:
    print(f"{'SYMBOL':8} {'STATUS':12} {'LAST BAR':11} {'SOURCE':9} MESSAGE")
    unavailable = 0
    for symbol in svc.cfg.symbols:
        md = svc.get(symbol, refresh=refresh)
        last = md.last_bar.isoformat() if md.last_bar else "-"
        print(f"{symbol:8} {md.status.value:12} {last:11} {md.source or '-':9} {md.message}")
        unavailable += not md.signal_allowed
    print(f"\n{len(svc.cfg.symbols) - unavailable}/{len(svc.cfg.symbols)} symbols eligible "
          "for signals.")
    return 0


def _mt5_line(svc: MarketDataService) -> tuple[str, bool]:
    mt5 = svc.providers.get("mt5")
    if mt5 is None:
        return "MT5: provider not configured", False
    if mt5.available():
        summary = getattr(mt5, "account_summary", None)
        if summary is None:
            return "MT5: connected", True
        info = summary()
        kind = "DEMO" if info["is_demo"] else "LIVE (read-only use)"
        return (f"MT5: connected to {info['company']} / {info['server']} ({kind}), "
                f"server_time={svc.cfg.data.mt5.server_time}"), True
    return f"MT5: NOT CONNECTED - {getattr(mt5, 'last_error', None) or 'unavailable'}", False


def _print_quality(svc: MarketDataService, refresh: bool) -> int:
    line, ready = _mt5_line(svc)
    print(format_quality_report(build_quality_report(svc, refresh), line, ready))
    return 0


def _print_research(svc: MarketDataService) -> int:
    """Download and assess RESEARCH-ONLY MT5 H1 history (never used for live bars)."""
    from tfx.data.intraday import IntradayService
    from tfx.data.research import ResearchHistory

    line, ready = _mt5_line(svc)
    print(line)
    intraday = IntradayService(svc.cfg, svc.cache, svc.providers, svc.clock)
    research = ResearchHistory(svc.cfg, svc.cache, svc.providers.get("mt5"), intraday,
                               svc.clock)
    print(f"\n{'SYMBOL':7} {'VERDICT':8} {'HISTORY':25} {'YRS':>5} {'COVER':>6} {'GAPS':>5} "
          f"{'OUTL':>5} {'H1 CMP':>7} {'H1 DIS':>7} {'D1 CMP':>7} {'D1 DIS':>7}")
    for symbol in svc.cfg.symbols:
        if ready:
            intraday.refresh(symbol)
            research.download(symbol)
        q = research.assess(symbol)
        span = (f"{q.first_bar:%Y-%m-%d} to {q.last_bar:%Y-%m-%d}" if q.first_bar is not None
                else "-")
        def pct(v: float | None) -> str:
            return f"{v:6.1f}%" if v is not None else "      -"
        print(f"{symbol:7} {q.verdict:8} {span:25} {q.years:5.1f} {pct(q.coverage_pct)} "
              f"{q.gaps:>5} {q.outliers:>5} {q.hourly_compared:>7} "
              f"{pct(q.hourly_disagree_pct)} {q.daily_compared:>7} {pct(q.daily_disagree_pct)}")
        for reason in q.reasons:
            print(f"        - {reason}")
    print("\nRESEARCH ONLY: this history is used for H1/H4 backtests, never to confirm live "
          "bars.\nA FAIL keeps that market's H1/H4 strategies 'not validated'.")
    if not ready:
        print("\n" + MT5_SETUP_STEPS)
    return 0


def _replay(svc: MarketDataService, start: str | None, end: str | None, balance: float,
            risk: float, strategy: str = "all", do_compare: bool = False) -> int:
    from datetime import date, timedelta

    from tfx.replay import replay

    today = date.today()
    last_mon = today - timedelta(days=today.weekday() + 7)
    s = date.fromisoformat(start) if start else last_mon
    e = date.fromisoformat(end) if end else s + timedelta(days=4)
    if do_compare:
        from tfx.replay import compare

        c = compare(svc.cfg, svc.cache, s, e, balance, risk, svc.cfg.account.currency)
        print(f"Strategy comparison {c['start']} to {c['end']}, start {balance:g}, "
              f"{risk}% risk per trade\n")
        for x in c["rows"]:
            wr = x.get("win_rate_closed")
            wr_txt = "-" if wr is None else f"{wr:.0%}"
            extra = (f"trades {x['trades']:>4} win {wr_txt:>4} max DD "
                     f"{x['max_drawdown_pct']:6.1f}%  below 0.01 lot {x['untradable']}"
                     if x["kind"] == "per-trade" else f"holdings {x['trades']:>4} (sleeve)")
            print(f"  {x['strategy']:15} {x['starting_balance']:g} -> {x['ending_balance']:<9} "
                  f"{x['pnl']:+9.2f} ({x['return_pct']:+8.2f}%)  {extra}")
        print("\n" + c["disclaimer"])
        return 0
    from tfx.strategies.rules import STRATEGIES

    chosen = STRATEGIES if strategy == "all" else (strategy,)
    r = replay(svc.cfg, svc.cache, s, e, balance, risk, svc.cfg.account.currency, chosen)
    print(f"Replay {r.start} to {r.end}: {r.starting_balance:g} -> {r.ending_balance:g} "
          f"({r.pnl:+.2f}, {r.return_pct:+.2f}%) at {r.risk_pct}% risk per trade\n")
    for t in r.trades:
        flag = "" if t.tradable else f"  [below 0.01 lot: would risk {t.min_lot_risk} " \
                                     f"({t.min_lot_risk_pct}%)]"
        print(f"{t.symbol} {t.strategy:15} {t.side:5} signal {t.signal_date} entry {t.entry} "
              f"stop {t.stop} {t.status:24} {t.r:+.2f}R  P&L {t.pnl:+.2f}{flag}")
    for n in r.notes:
        print("\nNote: " + n)
    print("\n" + r.disclaimer)
    return 0


def _study(svc: MarketDataService) -> int:
    """Missed-trade study on EURUSD/GBPUSD training data (holdout untouched)."""
    from tfx.study_missed import CONDITION_TEXT, run_study, save
    from tfx.validation.proof import costs_for

    syms = list(svc.cfg.section("missed_study").get("symbols", ["EURUSD", "GBPUSD"]))
    data = {s: svc.cache.load(s) for s in syms}
    res = run_study(svc.cfg, data, {s: costs_for(svc.cfg, s) for s in syms})
    save(res)
    print(f"Missed-trade study {', '.join(res.symbols)}  {res.period[0]} -> {res.period[1]} "
          f"(holdout from {res.holdout_start} NOT used)\n")
    print(f"{'GROUP':28} {'CAND':>5} {'FILLED':>6} {'WIN':>4} {'LOSS':>4} {'MEAN R':>7} "
          f"{'TOTAL R':>8} {'BOOT P5':>8}")
    for g in res.groups.values():
        mr = "-" if g.mean_r is None else f"{g.mean_r:+.3f}"
        p5 = "-" if g.boot_p5 is None else f"{g.boot_p5:+.3f}"
        print(f"{g.name:28} {g.candidates:>5} {g.filled:>6} {g.wins:>4} {g.losses:>4} "
              f"{mr:>7} {g.total_r:>+8.2f} {p5:>8}")
        if g.why and g.name.startswith("missed: only"):
            cond = g.name.removeprefix("missed: only ")
            print(f"    ({CONDITION_TEXT[cond]}) {g.why}")
    bm = res.big_moves
    print(f"\nBig moves (>= 3 ATR within 10 days before 1 ATR against): {bm['moves']} "
          f"(base rate {bm['base_rate'] * 100:.1f}% of days)")
    for k, v in bm["had_a_setup"].items():
        print(f"    {k:16} had a setup for {v} ({100 * v / max(bm['moves'], 1):.0f}%)")
    print("\n" + res.summary)
    print("Full study: var/missed_study.json")
    return 0


def _moves(svc: MarketDataService) -> int:
    """Move study: what came before the big moves, and would it have paid?"""
    from datetime import UTC, datetime

    from tfx.core_types import Timeframe
    from tfx.strategies.precursors import RULE_TEXT
    from tfx.study_moves import run_move_study, save
    from tfx.validation.proof import costs_for
    from tfx.validation.registry import ValidationRegistry

    syms = list(svc.cfg.section("move_study").get("symbols", ["EURUSD", "GBPUSD"]))
    data = {s: svc.cache.load(s) for s in syms}
    res = run_move_study(svc.cfg, data, {s: costs_for(svc.cfg, s) for s in syms})
    save(res)
    reg = ValidationRegistry(svc.cfg.data.resolved_db_path())   # every rule is a trial
    now = datetime.now(UTC)
    for name in res.rules:
        reg.record_trial("move-study", f"move_study:{name}", Timeframe.D1, {"rule": name}, now)
    print(f"Move study {', '.join(res.symbols)}  {res.period[0]} -> {res.period[1]} "
          f"(holdout NOT used; halves split {res.split})")
    print(f"{res.moves} big moves ({res.up_moves} up, {res.down_moves} down) in {res.bars} "
          "trading days\n")
    print(f"{'RULE':26} {'FIRED':>6} {'CAUGHT':>6} {'HIT%':>5} {'BASE%':>5} {'TRADES':>6} "
          f"{'WIN%':>5} {'MEAN R':>7} {'1ST':>6} {'2ND':>6} {'LOW':>6}")
    for r in res.rules.values():
        def f(x: float | None) -> str:
            return "-" if x is None else f"{x:+.3f}"
        print(f"{r.rule:26} {r.signals:>6} {r.moves_captured:>6} {100 * r.hit_rate:5.1f} "
              f"{100 * r.base_rate:5.1f} {r.trades:>6} {100 * r.win_rate:5.1f} "
              f"{f(r.mean_r):>7} {f(r.mean_r_first_half):>6} {f(r.mean_r_second_half):>6} "
              f"{f(r.boot_low):>6}  {'ADOPTED' if r.adopted else ''}")
    print("\nRules: " + "; ".join(f"{k} = {v}" for k, v in RULE_TEXT.items()))
    print(f"Adoption: >= {svc.cfg.section('move_study').get('min_trades', 100)} trades, mean "
          f"R > 0, positive in both halves, bootstrap {res.boot_percentile}th pct > 0.")
    print("\n" + res.summary)
    print("Full study: var/move_study.json")
    return 0


def _validate(svc: MarketDataService, only: list[str] | None = None) -> int:
    """Run the strategy proof phase (validation gate) on cached, confirmed data."""
    from tfx.validation.proof import ProofRunner
    from tfx.validation.registry import ValidationRegistry

    reg = ValidationRegistry(svc.cfg.data.resolved_db_path())
    report = ProofRunner(svc.cfg, svc.cache, reg).run_all(only)
    print(f"Proof phase {report['run_id']}\n")
    print(f"{'STRATEGY':15} {'TF':3} {'VERDICT':7} {'TRADES':>6} {'WIN%':>5} {'EXP R':>6} "
          f"{'SHARPE':>6} {'MAXDD':>7} {'HOLDOUT SH':>10} {'PBO':>5} {'DSR':>5} EDGE")
    for p in report["pairs"]:
        full, ho = p.get("full") or {}, p.get("holdout") or {}
        if full:
            print(f"{p['strategy']:15} {p['timeframe']:3} {p['verdict']:7} {full['trades']:>6} "
                  f"{full['win_rate'] * 100:5.1f} {full['expectancy_r']:+6.2f} "
                  f"{full['sharpe']:6.2f} {full['max_drawdown'] * 100:6.1f}% "
                  f"{ho.get('sharpe', 0):10.2f} {p['pbo']:5.2f} {p['dsr']:5.2f} "
                  f"{p['edge_confidence']}")
        else:
            print(f"{p['strategy']:15} {p['timeframe']:3} {p['verdict']:7}")
        for r in p["reasons"]:
            print(f"    - {r}")
    print("\n" + report["summary"])
    print("Full report: var/validation_report.json")
    return 0


def _print_sources(svc: MarketDataService, refresh: bool) -> int:
    """How often do the primary and reference feeds disagree? (measure before trusting)"""
    limit = svc.cfg.data.cross_check.max_disagreement_atr
    line, ready = _mt5_line(svc)
    print(line)
    print(f"Disagreement = |change(primary) - change(reference)| / ATR20; limit {limit:.2f}\n")
    print(f"{'SYMBOL':8} {'PRIMARY':9} {'REFERENCE':10} {'BARS':>5} {'MEDIAN':>7} "
          f"{'P95':>6} {'MAX':>6} {'>LIMIT':>7}  LATEST")
    rates: list[float] = []
    for symbol in svc.cfg.symbols:
        md = svc.get(symbol, refresh=refresh)
        ref, ref_source = svc.cache.load_reference(symbol)
        dis = disagreement_series(md.candles, ref) if ref_source != md.source else ref["close"][:0]
        latest = md.check.status.value if md.check else md.status.value
        if dis.empty:
            print(f"{symbol:8} {md.source or '-':9} {ref_source or '-':10} {0:>5} "
                  f"{'-':>7} {'-':>6} {'-':>6} {'-':>7}  {latest}")
            continue
        over = float((dis > limit).mean() * 100)
        rates.append(over)
        print(f"{symbol:8} {md.source or '-':9} {ref_source:10} {len(dis):>5} "
              f"{dis.median():>7.3f} {dis.quantile(0.95):>6.3f} {dis.max():>6.2f} "
              f"{over:>6.1f}%  {latest}")
    if rates:
        avg = sum(rates) / len(rates)
        print(f"\nAverage share of bars over the limit: {avg:.1f}%. If this is high for clean "
              "data, the feeds are not comparable bar-for-bar and the limit needs review.")
    else:
        print("\nNo reference data to compare.")
    if not ready:
        print("\n" + MT5_SETUP_STEPS)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tfx", description="TrillionaireFX (decision support)")
    parser.add_argument("command",
                        choices=["run", "daily", "fetch", "status", "quality", "sources",
                                 "research", "validate", "replay", "study", "moves", "publish"],
                        nargs="?", default="run")
    parser.add_argument("--start", help="replay start date YYYY-MM-DD (default last Monday)")
    parser.add_argument("--end", help="replay end date YYYY-MM-DD (default start + 4 days)")
    parser.add_argument("--balance", type=float, default=100.0, help="replay starting balance")
    parser.add_argument("--risk", type=float, default=0.5, help="replay risk %% per trade")
    parser.add_argument("--strategy", default="all",
                        choices=["all", "trend", "pullback", "mean_reversion", "ict",
                                 "ict_flex", "supply_demand", "ict_then_sd"],
                        help="replay one strategy (default: all)")
    parser.add_argument("--only", help="validate only these strategies (comma separated); "
                        "earlier results are kept and not re-counted")
    parser.add_argument("--push", action="store_true",
                        help="publish: commit site/ and push it (only if it changed)")
    parser.add_argument("--compare", action="store_true",
                        help="replay every strategy separately and compare")
    args = parser.parse_args(argv)
    configure_logging()
    load_env()  # .env values are never printed
    svc = _service()
    if args.command == "run":  # pragma: no cover
        return _run_server(svc)
    if args.command == "daily":
        return _daily(svc)
    if args.command == "replay":
        return _replay(svc, args.start, args.end, args.balance, args.risk, args.strategy,
                       args.compare)
    if args.command == "validate":
        return _validate(svc, args.only.split(",") if args.only else None)
    if args.command == "study":
        return _study(svc)
    if args.command == "moves":
        return _moves(svc)
    if args.command == "publish":
        from tfx.publish import publish

        print("Public site: " + publish(svc, args.push))
        return 0
    if args.command == "research":
        return _print_research(svc)
    if args.command == "quality":
        return _print_quality(svc, refresh=True)
    if args.command == "sources":
        return _print_sources(svc, refresh=True)
    if args.command == "status":
        return _print_status(svc, refresh=False)
    return _print_status(svc, refresh=True)


if __name__ == "__main__":
    sys.exit(main())
