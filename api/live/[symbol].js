// Live prices for the public TrillionaireFX site (Vercel serverless function).
// GET /api/live/EURUSD?tf=M15[&full=1]
//
// Source: Yahoo Finance chart data (free; can lag a few seconds; NOT confirmed by MT5).
// Display only: it never feeds a signal. Only the 12 configured pairs and the fixed
// timeframes are accepted, and answers are cached at the CDN for a few seconds so all
// viewers share one upstream request.
"use strict";

const SYMBOLS = {
  EURUSD: "EURUSD=X", GBPUSD: "GBPUSD=X", USDJPY: "USDJPY=X", AUDUSD: "AUDUSD=X",
  USDCAD: "USDCAD=X", NZDUSD: "NZDUSD=X", USDCHF: "USDCHF=X", EURJPY: "EURJPY=X",
  GBPJPY: "GBPJPY=X", EURGBP: "EURGBP=X", AUDJPY: "AUDJPY=X", XAUUSD: "GC=F",
};
// timeframe -> [Yahoo interval, range, bar size in seconds]. Every bar is snapped to its
// boundary: Yahoo stamps the still-forming bar with the current second.
const TF = {
  M1: ["1m", "2d", 60], M5: ["5m", "5d", 300], M15: ["15m", "1mo", 900],
  M30: ["30m", "1mo", 1800], H1: ["60m", "3mo", 3600], H4: ["60m", "6mo", 4 * 3600],
  // daily and above: only TODAY's bar (UTC day, built from hourly bars); the history
  // comes from the published, MT5-checked snapshot
  D1: ["60m", "5d", 86400], W1: ["60m", "5d", 86400], MN1: ["60m", "5d", 86400],
};
const DAILY = new Set(["D1", "W1", "MN1"]);
const CACHE = { intraday: "public, s-maxage=5, stale-while-revalidate=20",
                daily: "public, s-maxage=30, stale-while-revalidate=60" };

function bucket(bars, size) {
  const out = [];
  for (const b of bars) {
    const t = b.time - (b.time % size);
    const last = out[out.length - 1];
    if (last && last.time === t) {
      last.high = Math.max(last.high, b.high);
      last.low = Math.min(last.low, b.low);
      last.close = b.close;
    } else out.push({ time: t, open: b.open, high: b.high, low: b.low, close: b.close });
  }
  return out;
}

async function yahoo(ticker, interval, range) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
              `?interval=${interval}&range=${range}&includePrePost=false`;
  const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 (compatible; tfx-viewer)",
                                          Accept: "application/json" } });
  if (!r.ok) throw new Error(`upstream HTTP ${r.status}`);
  const j = await r.json();
  const res = j?.chart?.result?.[0];
  if (!res || !res.timestamp) throw new Error("no data from upstream");
  const q = res.indicators.quote[0];
  const bars = [];
  for (let i = 0; i < res.timestamp.length; i++) {
    const o = q.open[i], h = q.high[i], l = q.low[i], c = q.close[i];
    if ([o, h, l, c].some((x) => x === null || x === undefined || !isFinite(x))) continue;
    bars.push({ time: res.timestamp[i], open: o, high: h, low: l, close: c });
  }
  return { bars, price: res.meta?.regularMarketPrice ?? (bars.length ? bars[bars.length - 1].close : null) };
}

module.exports = async (req, res) => {
  const symbol = String(req.query.symbol || "").toUpperCase();
  const tf = String(req.query.tf || "M15").toUpperCase();
  const full = req.query.full === "1";
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("X-Content-Type-Options", "nosniff");
  if (!SYMBOLS[symbol] || !TF[tf]) {
    res.statusCode = 400;
    return res.end(JSON.stringify({ detail: "unknown symbol or timeframe" }));
  }
  const [interval, range, size] = TF[tf];
  try {
    const y = await yahoo(SYMBOLS[symbol], interval, range);
    let bars = bucket(y.bars, size);
    if (DAILY.has(tf)) bars = bars.slice(-1);                 // today's UTC bar only
    else if (!full) bars = bars.slice(-3);
    res.setHeader("Cache-Control", DAILY.has(tf) ? CACHE.daily : CACHE.intraday);
    return res.end(JSON.stringify({
      symbol, tf, source: "yahoo", price: y.price, bars, server_time: Math.floor(Date.now() / 1000),
      note: "Live prices from Yahoo Finance: may lag a few seconds, not confirmed by MT5.",
    }));
  } catch (e) {
    res.statusCode = 502;
    res.setHeader("Cache-Control", "public, s-maxage=5");
    return res.end(JSON.stringify({ detail: `live feed unavailable: ${e.message}` }));
  }
};
