"""Publish the public, view-only TrillionaireFX site (static files for Vercel).

``python -m tfx publish [--push]`` writes ``site/``:

* the same web UI (web/), switched to view-only mode;
* ``data/*.json``: market data, signals, the market read, ICT / supply & demand zones,
  validation and data quality, generated through the app's own API.

Privacy by construction: the data is produced by a pipeline with a BLANK in-memory
store, so your trades, journal, mindset log, settings, balance, risk, drawings and
alerts can never appear. A whitelist filter then drops position sizes and amounts at
risk as a second line of defence. Live prices on the site come from a small Vercel
function (api/live/[symbol].js), not from this snapshot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tfx.config import PROJECT_ROOT
from tfx.logging_setup import get_logger

log = get_logger(__name__)

SITE_DIR = PROJECT_ROOT / "site"
WEB_DIR = PROJECT_ROOT / "web"
SNAPSHOT_TFS = ("D1", "W1", "MN1")              # full history from the checked data
DAILY_BARS = 1000                               # ~4 years of D1 on the public site
INTRADAY_TFS = ("M1", "M5", "M15", "M30", "H1", "H4")   # zones only; candles are live
PRIVATE_KEYS = {"size", "amount_at_risk", "risk_amount", "lots", "units", "signal_id",
                "actions", "trade_r_now", "markers", "balance", "risk_pct", "equity"}
STATIC_FLAG = "<script>window.TFX_STATIC = true;</script>"


def scrub(obj: Any) -> Any:
    """Remove every private key, at any depth."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in PRIVATE_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def slim_candles(d: dict[str, Any], tf: str) -> dict[str, Any]:
    """Keep the published history small: no precomputed indicator lines (the browser
    recomputes them from the live candles) and only the recent daily bars."""
    d = dict(d)
    d.pop("overlays", None)
    keep = DAILY_BARS if tf == "D1" else None
    if keep and len(d.get("candles", [])) > keep:
        d["candles"] = d["candles"][-keep:]
        t0 = d["candles"][0]["time"]
        d["regimes"] = [r for r in d.get("regimes", []) if r["to"] >= t0]
    return d


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(scrub(data)), separators=(",", ":"), default=str),
                    encoding="utf-8")


def public_client(svc: Any) -> Any:
    """The app's API on a blank in-memory account (no personal data exists in it)."""
    from fastapi.testclient import TestClient

    from tfx.alerts.notify import Notifier
    from tfx.api.app import create_app
    from tfx.data.chart_bars import ChartData
    from tfx.pipeline import Pipeline
    from tfx.store import Store
    from tfx.validation.registry import ValidationRegistry

    blank = Store(":memory:")
    pipe = Pipeline(svc, blank, ValidationRegistry(svc.cfg.data.resolved_db_path()))
    # intraday zones from Yahoo only: the same feed the public site draws live, and no
    # waiting on MT5
    return TestClient(create_app(pipe, Notifier(blank, desktop=lambda t, b: True), None,
                                 chart=ChartData(None)))


def copy_ui(site: Path = SITE_DIR) -> None:
    for name in ("styles.css", "app.js", "icons.svg"):
        if (WEB_DIR / name).exists():
            shutil.copy2(WEB_DIR / name, site / name)
    shutil.copytree(WEB_DIR / "vendor", site / "vendor", dirs_exist_ok=True)
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/', 'href="/').replace('src="/static/', 'src="/')
    html = html.replace('<script src="/vendor/', STATIC_FLAG + '\n<script src="/vendor/', 1)
    (site / "index.html").write_text(html, encoding="utf-8")


def build(svc: Any, site: Path = SITE_DIR, intraday_zones: bool = False) -> dict[str, Any]:
    site.mkdir(parents=True, exist_ok=True)
    data = site / "data"
    if data.exists():
        shutil.rmtree(data)
    copy_ui(site)
    c = public_client(svc)

    def get(path: str) -> Any:
        r = c.get(path)
        r.raise_for_status()
        return r.json()

    meta = get("/api/meta")
    meta.update({"static": True, "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
                 "snapshot_timeframes": list(SNAPSHOT_TFS),
                 "live_note": "Live prices from Yahoo Finance (may lag a few seconds; not "
                              "confirmed by MT5). Signals and analysis are from the latest "
                              "published run."})
    _write(data / "meta.json", meta)
    _write(data / "watchlist.json", get("/api/watchlist"))
    _write(data / "validation.json", get("/api/validation"))
    _write(data / "quality.json", get("/api/quality"))
    symbols = list(meta["symbols"])
    for sym in symbols:
        _write(data / "signal" / f"{sym}.json", get(f"/api/signal/{sym}"))
        _write(data / "analysis" / f"{sym}.json", get(f"/api/analysis/{sym}"))
        for tf in SNAPSHOT_TFS:
            _write(data / "candles" / f"{sym}_{tf}.json",
                   slim_candles(get(f"/api/candles/{sym}?tf={tf}"), tf))
        if intraday_zones:
            for tf in INTRADAY_TFS:
                try:
                    d = get(f"/api/candles/{sym}?tf={tf}")
                except Exception as exc:   # zones are optional; candles are live anyway
                    log.warning("intraday zones skipped: %s %s %s", sym, tf, exc)
                    continue
                _write(data / "zones" / f"{sym}_{tf}.json", {"ict": d.get("ict")})
    digest = fingerprint(data)
    (data / "version.json").write_text(json.dumps({"published_at": meta["published_at"],
                                                   "digest": digest}), encoding="utf-8")
    return {"symbols": symbols, "digest": digest, "published_at": meta["published_at"]}


def fingerprint(data: Path) -> str:
    """Hash of the published content, ignoring the publish time (so re-publishing
    unchanged data does not create a new commit)."""
    h = hashlib.sha256()
    for p in sorted(data.rglob("*.json")):
        if p.name in ("meta.json", "version.json"):
            continue
        h.update(p.relative_to(data).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=False)


def push(previous_digest: str | None, result: dict[str, Any]) -> str:
    """Commit ONLY site/ and push, if the published content changed."""
    if previous_digest == result["digest"]:
        return "unchanged: nothing to push"
    _git("add", "--", "site")
    msg = f"site: publish snapshot {result['published_at']}"
    done = _git("commit", "-m", msg, "--", "site")
    if done.returncode != 0:
        return "nothing to commit" if "nothing" in done.stdout + done.stderr else \
            f"commit failed: {done.stderr.strip()[:200]}"
    up = _git("push")
    return "pushed" if up.returncode == 0 else f"push failed: {up.stderr.strip()[:200]}"


def publish(svc: Any, do_push: bool) -> str:
    prev = previous_digest()
    res = build(svc)
    status = push(prev, res) if do_push else "built (not pushed)"
    log.info("site published: %s", status)
    return f"{len(res['symbols'])} symbols, digest {res['digest'][:12]}: {status}"


def publish_job(svc: Any) -> None:  # pragma: no cover - scheduler entry point
    cfg = svc.cfg.section("publish")
    if not cfg.get("enabled", False):
        return
    try:
        publish(svc, bool(cfg.get("push", True)))
    except Exception:
        log.exception("site publish failed")


def previous_digest(site: Path = SITE_DIR) -> str | None:
    try:
        d: str = json.loads((site / "data" / "version.json").read_text(encoding="utf-8"))[
            "digest"]
        return d
    except (OSError, ValueError, KeyError):
        return None
