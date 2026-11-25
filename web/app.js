/* TrillionaireFX web app (vanilla JS, no build step). Decision support only. */
"use strict";
(() => {
const LW = window.LightweightCharts;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;",
  ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 5) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  Number(v).toFixed(d).replace(/0+$/, "").replace(/\.$/, "");
const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : `${(v * 100).toFixed(d)}%`;
const ICON = { BUY: "▲", SELL: "▼", EXIT: "■", HOLD: "●", WATCH: "◎", WAIT: "–" };
const ico = (name, cls = "") => `<svg class="ico ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

const TF_LABEL = { M1: "1m", M5: "5m", M15: "15m", M30: "30m", H1: "1H", H4: "4H",
  D1: "D", W1: "W", MN1: "M" };
const PHONE = window.matchMedia("(max-width: 767px)");
const STATIC = !!window.TFX_STATIC;            // public view-only site
if (STATIC) document.documentElement.classList.add("static");
const state = {
  symbol: localStorage.getItem("tfx.symbol") || "EURUSD",
  tf: localStorage.getItem("tfx.tf") || "D1", ctype: "candles", live: null, liveTimer: null,
  tool: "cursor", meta: null, settings: null, data: null, drawings: [], pending: null,
  ind: JSON.parse(localStorage.getItem("tfx.ind") || "null") || {
    dc20: true, dc55: false, dc100: false, ema20: true, ema100: false, ema200: true,
    atrstop: false, regime: true, levels: true, markers: true, adx: true, rsi: false, atr: false,
    setup: true, ict: !PHONE.matches },
  tab: STATIC ? "validation" : "trades", hover: null,
};
if (PHONE.matches && !localStorage.getItem("tfx.ind")) {
  Object.assign(state.ind, { dc20: false, adx: false, markers: false, regime: false });
}

/* ------------------------------------------------------------ phone views */
function setView(v) {
  document.body.dataset.view = v;
  $$(".mnav button").forEach((b) => {
    const on = b.dataset.view === v;
    b.classList.toggle("active", on);
    b.setAttribute("aria-current", on ? "page" : "false");
  });
  document.body.classList.remove("acct-open");
  $("#moreBtn")?.setAttribute("aria-expanded", "false");
  if (v === "chart") requestAnimationFrame(() => { sizeOverlay(); requestDraw(); });
  if (PHONE.matches) window.scrollTo(0, 0);
}

async function api(path, opts = {}) {
  if (STATIC) return staticApi(path, opts);
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

/* ------------------------------------------- public view-only site (Vercel) */
/* Published analysis comes from /data/*.json (written by `python -m tfx publish` on
   the owner's PC from a BLANK account: no personal data). Candles are LIVE from the
   /api/live function (Yahoo, display only); indicators are recomputed here with the
   same formulas as tfx/strategies/indicators.py so they extend with the live bars. */
const SNAP_TF = new Set(["D1", "W1", "MN1"]);
async function getJson(url) {
  const r = await fetch(url, { cache: "no-cache" });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}
async function staticApi(path, opts) {
  if (opts.method && opts.method !== "GET")
    throw new Error("View-only site: this is available only in the owner's own app.");
  const u = new URL(path, location.origin);
  const p = u.pathname.split("/").filter(Boolean);       // ["api", name, ...]
  const tf = u.searchParams.get("tf") || "D1";
  switch (p[1]) {
    case "meta": return getJson("/data/meta.json");
    case "watchlist": return getJson("/data/watchlist.json");
    case "signal": return getJson(`/data/signal/${p[2]}.json`);
    case "analysis": return getJson(`/data/analysis/${p[2]}.json`);
    case "validation": return getJson("/data/validation.json");
    case "quality": return getJson("/data/quality.json");
    case "settings": return { theme: safeGet("tfx.theme") || "dark", balance: "", risk_pct: "",
                              currency: "", static: true };
    case "drawings": case "alerts": return [];
    case "live": return getJson(`/api/live/${p[2]}?tf=${tf}`);
    case "candles": return staticCandles(p[2], tf);
    default: throw new Error("Not available on the public view-only site.");
  }
}
function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
async function staticCandles(sym, tf) {
  let d, live = null;
  if (SNAP_TF.has(tf)) {
    d = await getJson(`/data/candles/${sym}_${tf}.json`);
    live = await getJson(`/api/live/${sym}?tf=${tf}`).catch(() => null);
    if (live?.bars?.length) mergeToday(d.candles, live.bars[0], tf);
  } else {
    live = await getJson(`/api/live/${sym}?tf=${tf}&full=1`);
    const z = await getJson(`/data/zones/${sym}_${tf}.json`).catch(() => ({}));
    d = { symbol: sym, tf, validated_view: false, candles: live.bars, ict: z.ict || null,
          markers: [], regimes: [], data: { status: "LIVE", message: live.note } };
  }
  d.overlays = computeOverlays(d.candles);
  d.live = !!live;
  d.source = "yahoo (live)";
  d.note = `${tf === "D1" ? "" : `${tf} view only - not validated, no signals · `}` +
           (live ? "live prices (Yahoo, may lag a few seconds)" : "live feed unavailable: last published data");
  return d;
}
/* today's live bar into a D1 / W1 / MN1 series */
function mergeToday(c, b, tf) {
  if (!c.length || !b) return;
  const last = c[c.length - 1];
  const sameBar = tf === "D1" ? b.time === last.time :
    tf === "W1" ? b.time - last.time < 7 * 86400 && b.time >= last.time :
    new Date(b.time * 1000).getUTCMonth() === new Date(last.time * 1000).getUTCMonth() &&
    b.time >= last.time;
  if (b.time < last.time) return;
  if (sameBar) {
    if (tf === "D1") Object.assign(last, b);
    else { last.high = Math.max(last.high, b.high); last.low = Math.min(last.low, b.low);
           last.close = b.close; }
  } else c.push(tf === "D1" ? { ...b } : { ...b, time: b.time });
}
function alignLive(bars) {
  if (!STATIC || !SNAP_TF.has(state.tf) || !state.data?.candles?.length) return bars;
  const c = state.data.candles; const before = c.length;
  for (const b of bars) mergeToday(c, b, state.tf);
  return c.slice(c.length === before ? -1 : -2);
}
/* indicators (same formulas as tfx/strategies/indicators.py) */
function ewm(x, alpha, minp) {
  const out = new Array(x.length).fill(NaN); let y = NaN, n = 0;
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) { out[i] = n >= minp ? y : NaN; continue; }
    y = n === 0 ? x[i] : (1 - alpha) * y + alpha * x[i]; n++;
    if (n >= minp) out[i] = y;
  }
  return out;
}
function computeOverlays(c) {
  const n = c.length, H = c.map((b) => b.high), L = c.map((b) => b.low),
        C = c.map((b) => b.close), T = c.map((b) => b.time);
  const ser = (v) => v.map((x, i) => ({ time: T[i], value: x }))
    .filter((p) => Number.isFinite(p.value));
  const ema = (k) => ewm(C, 2 / (k + 1), k);
  const tr = c.map((b, i) => i === 0 ? b.high - b.low : Math.max(b.high - b.low,
    Math.abs(b.high - C[i - 1]), Math.abs(b.low - C[i - 1])));
  const atr = (k) => ewm(tr, 1 / k, k);
  const dch = (k) => H.map((_, i) => i < k ? NaN : Math.max(...H.slice(i - k, i)));
  const dcl = (k) => L.map((_, i) => i < k ? NaN : Math.min(...L.slice(i - k, i)));
  const d = C.map((x, i) => i === 0 ? NaN : x - C[i - 1]);
  const up = ewm(d.map((x) => Number.isNaN(x) ? NaN : Math.max(x, 0)), 1 / 2, 2);
  const dn = ewm(d.map((x) => Number.isNaN(x) ? NaN : Math.max(-x, 0)), 1 / 2, 2);
  const rsi2 = up.map((u, i) => Number.isNaN(u) ? NaN : dn[i] === 0 ? 100 : 100 - 100 / (1 + u / dn[i]));
  const pdm = H.map((h, i) => { if (!i) return 0; const u = h - H[i - 1], v = L[i - 1] - L[i];
    return u > v && u > 0 ? u : 0; });
  const mdm = L.map((l, i) => { if (!i) return 0; const u = H[i] - H[i - 1], v = L[i - 1] - l;
    return v > u && v > 0 ? v : 0; });
  const tr14 = atr(14), p14 = ewm(pdm, 1 / 14, 14), m14 = ewm(mdm, 1 / 14, 14);
  const dx = tr14.map((t, i) => { const pd = 100 * p14[i] / t, md = 100 * m14[i] / t;
    return pd + md === 0 ? NaN : 100 * Math.abs(pd - md) / (pd + md); });
  const a20 = atr(20);
  const ov = {};
  for (const k of [20, 55, 100]) { ov[`donchian_high_${k}`] = ser(dch(k)); ov[`donchian_low_${k}`] = ser(dcl(k)); }
  for (const k of [20, 100, 200]) ov[`ema_${k}`] = ser(ema(k));
  ov.atr_stop_long = ser(C.map((x, i) => x - 2 * a20[i]));
  ov.adx = ser(ewm(dx, 1 / 14, 14)); ov.rsi2 = ser(rsi2); ov.atr = ser(a20);
  return n ? ov : {};
}
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ------------------------------------------------------------------ chart */
let chart, main, overlaySeries = {}, subSeries = {}, markersApi = null, priceLines = [];
function chartColors() {
  return { bg: css("--panel"), text: css("--text"), grid: css("--border"), up: css("--up"),
           down: css("--down"), muted: css("--muted"), accent: css("--accent"),
           warn: css("--warn"), bad: css("--bad") };
}
function buildChart() {
  const c = chartColors();
  const el = $("#chart");
  if (chart) chart.remove();
  chart = LW.createChart(el, {
    autoSize: true,
    layout: { background: { color: c.bg }, textColor: c.text, fontFamily: css("--font"),
              panes: { separatorColor: c.grid } },
    grid: { vertLines: { color: c.grid + "55" }, horzLines: { color: c.grid + "55" } },
    crosshair: { mode: LW.CrosshairMode.Normal },
    rightPriceScale: { borderColor: c.grid }, timeScale: { borderColor: c.grid, rightOffset: 6 },
    localization: { priceFormatter: (p) => fmt(p, state.symbol.endsWith("JPY") ||
      state.symbol === "XAUUSD" ? 3 : 5) },
  });
  overlaySeries = {}; subSeries = {}; markersApi = null; priceLines = [];
  chart.subscribeCrosshairMove(onCrosshair);
  chart.subscribeClick(onChartClick);
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestDraw());
  new ResizeObserver(() => { sizeOverlay(); requestDraw(); }).observe($("#chartWrap"));
}
function heikinAshi(c) {
  const out = []; let po, pc;
  for (const b of c) {
    const cl = (b.open + b.high + b.low + b.close) / 4;
    const op = po === undefined ? (b.open + b.close) / 2 : (po + pc) / 2;
    out.push({ time: b.time, open: op, close: cl, high: Math.max(b.high, op, cl),
               low: Math.min(b.low, op, cl) });
    po = op; pc = cl;
  }
  return out;
}
function renderChart() {
  const d = state.data; const c = chartColors();
  buildChart();
  if (!d || !d.candles.length) { showState(d?.data?.message || "No data available."); return; }
  hideState();
  const candles = state.ctype === "ha" ? heikinAshi(d.candles) : d.candles;
  if (state.ctype === "line") {
    main = chart.addSeries(LW.LineSeries, { color: c.accent, lineWidth: 2 });
    main.setData(candles.map((b) => ({ time: b.time, value: b.close })));
  } else {
    const def = state.ctype === "bars" ? LW.BarSeries : LW.CandlestickSeries;
    main = chart.addSeries(def, { upColor: c.up, downColor: c.down, borderUpColor: c.up,
      borderDownColor: c.down, wickUpColor: c.up, wickDownColor: c.down });
    main.setData(candles);
  }
  const ov = d.overlays || {};
  const line = (key, color, width = 1, style = 0) => {
    if (!ov[key]) return;
    const s = chart.addSeries(LW.LineSeries, { color, lineWidth: width, lineStyle: style,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(ov[key]); overlaySeries[key] = s;
  };
  const I = state.ind;
  if (I.dc20) { line("donchian_high_20", c.up, 1, 2); line("donchian_low_20", c.down, 1, 2); }
  if (I.dc55) { line("donchian_high_55", c.up, 1, 1); line("donchian_low_55", c.down, 1, 1); }
  if (I.dc100) { line("donchian_high_100", c.up, 1, 3); line("donchian_low_100", c.down, 1, 3); }
  if (I.ema20) line("ema_20", "#9b8cff", 1);
  if (I.ema100) line("ema_100", "#2ec4b6", 1);
  if (I.ema200) line("ema_200", c.muted, 2);
  if (I.atrstop) line("atr_stop_long", c.bad, 1, 2);
  let pane = 1;
  const sub = (key, color, label, show) => {
    if (!show || !ov[key]) return;
    const s = chart.addSeries(LW.LineSeries, { color, lineWidth: 1, priceLineVisible: false,
      title: label }, pane);
    s.setData(ov[key]); subSeries[key] = s;
    chart.panes()[pane]?.setHeight(90); pane += 1;
  };
  sub("adx", c.accent, "ADX 14", I.adx);
  sub("rsi2", "#9b8cff", "RSI 2", I.rsi);
  sub("atr", c.warn, "ATR 20", I.atr);
  if (I.markers && d.markers?.length) {
    markersApi = LW.createSeriesMarkers(main, d.markers.map((m) => ({ ...m,
      color: m.shape === "arrowUp" ? c.up : (m.shape === "arrowDown" ? c.down : c.muted) })));
  }
  if (I.levels) {
    if (d.watch_level) priceLines.push(main.createPriceLine({ price: d.watch_level,
      color: c.warn, lineStyle: 2, title: "watch trigger" }));
    if (d.signal_levels) {
      priceLines.push(main.createPriceLine({ price: d.signal_levels.entry, color: c.accent,
        title: "entry" }));
      priceLines.push(main.createPriceLine({ price: d.signal_levels.stop, color: c.bad,
        lineStyle: 2, title: "stop" }));
    }
  }
  const note = $("#chartNote");
  note.hidden = !d.note; note.textContent = d.note || "";
  const intraday = !["D1", "W1", "MN1"].includes(state.tf);
  chart.timeScale().applyOptions({ timeVisible: intraday, secondsVisible: false });
  chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, candles.length - 180),
                                             to: candles.length + 5 });
  startLive();
  setTool(state.tool);
  sizeOverlay(); requestDraw();
}
/* ------------------------------------------------- live MT5 updates (display only) */
function stopLive() { clearInterval(state.liveTimer); state.liveTimer = null; }
function startLive() {
  stopLive();
  if (!state.data?.live) return;
  let inflight = false;
  const tick = async () => {
    if (document.hidden || !main || inflight) return;
    inflight = true;
    try {
      const lv = await api(`/api/live/${state.symbol}?tf=${state.tf}`);
      if (lv.symbol !== state.symbol || lv.tf !== state.tf) return;
      state.live = lv.price;
      for (const b of alignLive(lv.bars)) {
        if (state.ctype === "line") main.update({ time: b.time, value: b.close });
        else if (state.ctype !== "ha") main.update(b);
      }
      onCrosshair(null);
    } catch { /* keep the last good chart; next tick retries */ }
    finally { inflight = false; }
  };
  // once a second locally (MT5); every 3 s on the public site (CDN-cached Yahoo feed)
  state.liveTimer = setInterval(tick, STATIC ? 3000 : 1000);
}
function showState(msg) { const s = $("#chartState"); s.hidden = false; s.textContent = msg; }
function hideState() { $("#chartState").hidden = true; }

function onCrosshair(p) {
  const lg = $("#legend");
  if (!p || !p.time || !main) {
    const lv = state.live;
    lg.innerHTML = `${esc(state.symbol)} · ${esc(TF_LABEL[state.tf] || state.tf)}` +
      (state.data?.live ? ` · <span class="livedot"></span>MT5 live` : "") +
      (lv ? ` · bid ${fmt(lv.bid)} ask ${fmt(lv.ask)}` : "") +
      (state.data?.source ? ` · ${esc(state.data.source)}` : "");
  }
  else {
    const b = p.seriesData.get(main);
    if (b) {
      const o = b.open ?? b.value, cl = b.close ?? b.value;
      const dir = cl >= o ? `<span class="up">▲</span>` : `<span class="down">▼</span>`;
      lg.innerHTML = `${esc(state.symbol)} · ${esc(state.tf)} ${dir} O ${fmt(o)} H ${fmt(b.high ?? cl)}` +
        ` L ${fmt(b.low ?? cl)} C ${fmt(cl)}`;
    }
  }
  if (state.pending && p?.point) { state.hover = pointToData(p.point.x, p.point.y, p.time); requestDraw(); }
}

/* --------------------------------------------------------------- drawings */
const canvas = () => $("#overlay");
function sizeOverlay() {
  const cv = canvas(), w = $("#chartWrap"); const r = devicePixelRatio || 1;
  cv.width = w.clientWidth * r; cv.height = w.clientHeight * r;
  cv.style.width = w.clientWidth + "px"; cv.style.height = w.clientHeight + "px";
}
let drawQueued = false;
function requestDraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(drawAll); } }
function xOf(t) { const x = chart?.timeScale().timeToCoordinate(t); return x ?? null; }
function yOf(p) { const y = main?.priceToCoordinate(p); return y ?? null; }
function pointToData(x, y, time) {
  const t = time ?? chart.timeScale().coordinateToTime(x);
  const p = main.coordinateToPrice(y);
  return t && p ? { time: t, price: p } : null;
}
function pipOf() { return state.symbol.endsWith("JPY") || state.symbol === "XAUUSD" ? 0.01 : 0.0001; }
function drawAll() {
  drawQueued = false;
  const cv = canvas(); if (!cv || !chart || !main) return;
  const ctx = cv.getContext("2d"); const r = devicePixelRatio || 1;
  ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
  const c = chartColors();
  if (state.ind.regime && state.data?.regimes) {
    for (const seg of state.data.regimes) {
      const x1 = xOf(seg.from), x2 = xOf(seg.to);
      if (x1 === null && x2 === null) continue;
      const col = { TRENDING: c.up, CHOPPY: c.muted, SHOCK: c.bad }[seg.regime];
      if (!col) continue;
      ctx.fillStyle = col + "55";
      ctx.fillRect(x1 ?? 0, 0, Math.max(2, (x2 ?? cv.clientWidth) - (x1 ?? 0)), 4);
    }
  }
  const all = [...state.drawings];
  if (state.pending && state.hover) all.push({ kind: state.pending.kind, data: {
    a: state.pending.a, b: state.hover }, preview: true });
  for (const d of all) drawOne(ctx, d, c);
  if (state.ind.ict && state.data?.ict) drawIct(ctx, state.data.ict, c);
  if (state.ind.setup && state.data?.setup) drawSetup(ctx, state.data.setup, c);
}
function drawOne(ctx, d, c) {
  const { a, b, text } = d.data; ctx.lineWidth = 1.5; ctx.setLineDash(d.preview ? [4, 4] : []);
  ctx.font = `12px ${css("--font")}`;
  const ax = a ? xOf(a.time) : null, ay = a ? yOf(a.price) : null;
  const bx = b ? xOf(b.time) : null, by = b ? yOf(b.price) : null;
  if (d.kind === "hline" && ay !== null) {
    ctx.strokeStyle = c.accent; ctx.beginPath(); ctx.moveTo(0, ay); ctx.lineTo(9999, ay); ctx.stroke();
    ctx.fillStyle = c.accent; ctx.fillText(fmt(a.price), 6, ay - 4);
  } else if (d.kind === "trendline" && ax !== null && bx !== null) {
    ctx.strokeStyle = c.accent; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  } else if (d.kind === "rect" && ax !== null && bx !== null) {
    ctx.strokeStyle = c.accent; ctx.fillStyle = c.accent + "22";
    ctx.fillRect(ax, ay, bx - ax, by - ay); ctx.strokeRect(ax, ay, bx - ax, by - ay);
  } else if (d.kind === "measure" && ax !== null && bx !== null) {
    ctx.strokeStyle = c.warn; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    const dp = b.price - a.price; const bars = barsBetween(a.time, b.time);
    ctx.fillStyle = c.warn;
    ctx.fillText(`${fmt(dp / pipOf(), 1)} pips · ${(dp / a.price * 100).toFixed(2)}% · ${bars} bars`, bx + 6, by);
  } else if (d.kind === "text" && ax !== null) {
    ctx.fillStyle = c.text; ctx.fillText(text || "", ax, ay);
  } else if ((d.kind === "long" || d.kind === "short") && ax !== null && bx !== null) {
    const dir = d.kind === "long" ? 1 : -1;
    const entry = a.price, stop = b.price, risk = Math.abs(entry - stop);
    const target = entry + dir * 2 * risk;
    const ty = yOf(target), w = Math.max(120, Math.abs((bx ?? ax) - ax) || 160);
    ctx.fillStyle = c.bad + "33"; ctx.fillRect(ax, Math.min(ay, by), w, Math.abs(by - ay));
    if (ty !== null) { ctx.fillStyle = c.up + "33"; ctx.fillRect(ax, Math.min(ay, ty), w, Math.abs(ty - ay)); }
    const st = state.settings || {}; const money = (st.balance || 0) * (st.risk_pct || 0) / 100;
    ctx.fillStyle = c.text;
    ctx.fillText(`${d.kind.toUpperCase()} R:R 1:2 · risk ${fmt(risk / pipOf(), 1)} pips · ` +
      `${money.toFixed(2)} ${st.currency || ""} at risk`, ax + 4, Math.min(ay, by) - 6);
  }
}
const SETUP_COLORS = (c) => ({ entry: c.accent, stop: c.bad, target: c.up, trail: c.warn,
  breakeven: c.muted, channel_hi: c.up, channel_lo: c.down, ema_100: "#2ec4b6",
  ema_200: c.muted, broken: c.warn, lq: "#9b8cff", mss: c.warn });
function drawIct(ctx, a, c) {
  /* Display-only ICT objects; every object has an icon and a text label, not just colour. */
  const W = canvas().clientWidth; ctx.save(); ctx.font = `10px ${css("--font")}`;
  const band = (lo, hi, t0, col, label) => {
    const y1 = yOf(hi), y2 = yOf(lo), x = xOf(t0) ?? 0; if (y1 === null || y2 === null) return;
    ctx.fillStyle = col + "26"; ctx.fillRect(x, y1, W - x - 60, Math.max(2, y2 - y1));
    ctx.strokeStyle = col + "88"; ctx.strokeRect(x, y1, W - x - 60, Math.max(2, y2 - y1));
    ctx.fillStyle = col; ctx.fillText(label, x + 3, y1 - 2);
  };
  for (const z of a.fvgs || []) band(z.low, z.high, z.time, z.dir === "bullish" ? c.up : c.down,
                                       `${z.icon} ${z.label}`);
  for (const z of a.zones || []) band(z.low, z.high, z.time, z.dir === "demand" ? c.up : c.down,
                                        `${z.icon} ${z.label}`);
  for (const z of a.sd_zones || []) band(z.low, z.high, z.time,
                                          z.dir === "demand" ? "#2ec4b6" : "#ff8a65",
                                          `${z.icon} ${z.label}`);
  ctx.setLineDash([2, 3]);
  for (const l of a.liquidity || []) {
    const y = yOf(l.price), x = xOf(l.time) ?? 0; if (y === null) continue;
    ctx.strokeStyle = "#9b8cff"; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(W - 60, y); ctx.stroke();
    ctx.fillStyle = "#9b8cff"; ctx.fillText(`${l.icon} ${l.label} ${fmt(l.price)}`, W - 230, y - 2);
  }
  ctx.setLineDash([]);
  for (const e of a.events || []) {
    const x = xOf(e.time), y = yOf(e.price); if (x === null || y === null) continue;
    ctx.fillStyle = c.warn; ctx.fillText(`${e.icon} ${e.label}`, x + 4, y - 4);
  }
  ctx.restore();
}
function drawSetup(ctx, s, c) {
  /* Closed-bar setup objects from the server: the same levels as the signal card. */
  const col = SETUP_COLORS(c); const W = canvas().clientWidth; const x0 = xOf(s.box.from_time);
  ctx.save(); ctx.font = `11px ${css("--font")}`; ctx.lineWidth = 1;
  for (const z of s.zones) {
    const y1 = yOf(z.top), y2 = yOf(z.bottom); if (y1 === null || y2 === null) continue;
    ctx.fillStyle = c.accent + "22"; ctx.fillRect(x0 ?? 0, y1, W - (x0 ?? 0), y2 - y1);
    ctx.fillStyle = c.text; ctx.fillText(`${z.icon} ${z.label}`, (x0 ?? 0) + 4, y1 - 3);
  }
  for (const l of s.lines) {
    const y = yOf(l.price); if (y === null) continue;
    ctx.strokeStyle = col[l.id] || c.muted; ctx.setLineDash(l.id === "stop" || l.id === "trail" ? [5, 4] : []);
    ctx.beginPath(); ctx.moveTo(x0 ?? 0, y); ctx.lineTo(W - 60, y); ctx.stroke();
    ctx.fillStyle = col[l.id] || c.muted;
    ctx.fillText(`${l.icon} ${l.label} ${fmt(l.price)}`, (x0 ?? 0) + 4, y - 3);
  }
  ctx.setLineDash([]);
  const b = s.box, ye = yOf(b.entry), ys = yOf(b.stop), yt = yOf(b.target);
  if (x0 !== null && ye !== null && ys !== null && yt !== null) {
    const w = Math.min(160, W - x0 - 70);
    ctx.fillStyle = c.bad + "30"; ctx.fillRect(x0, Math.min(ye, ys), w, Math.abs(ys - ye));
    ctx.fillStyle = c.up + "30"; ctx.fillRect(x0, Math.min(ye, yt), w, Math.abs(yt - ye));
    ctx.fillStyle = c.text;
    ctx.fillText(`${b.side === "long" ? "▲ LONG" : "▼ SHORT"} R:R ${b.rr}` +
      (b.risk_amount !== null ? ` · ${b.risk_amount} ${b.currency} at risk` : ""), x0 + 4,
      Math.min(ye, ys, yt) - 6);
  }
  for (const m of s.markers) {
    const x = xOf(m.time), y = yOf(m.price); if (x === null || y === null) continue;
    ctx.fillStyle = c.text; ctx.fillText(`${m.icon} ${m.label}`, x + 6, y + 14);
  }
  ctx.fillStyle = c.panel ?? css("--panel"); ctx.globalAlpha = 0.9;
  const text = s.summary; const tw = Math.min(ctx.measureText(text).width + 12, W - 20);
  ctx.fillRect(8, 8, tw, 20); ctx.globalAlpha = 1; ctx.fillStyle = s.validated ? c.text : c.warn;
  ctx.fillText(text, 14, 22, tw - 12); ctx.restore();
}
function barsBetween(t1, t2) {
  const cs = state.data?.candles || []; const i1 = cs.findIndex((b) => b.time >= t1);
  const i2 = cs.findIndex((b) => b.time >= t2); return i1 < 0 || i2 < 0 ? 0 : Math.abs(i2 - i1);
}
const TWO_POINT = new Set(["trendline", "rect", "measure", "long", "short"]);
async function onChartClick(p) {
  if (!p?.point || !main) return;
  const pt = pointToData(p.point.x, p.point.y, p.time); if (!pt) return;
  const tool = state.tool;
  if (tool === "eraser") return eraseNear(p.point.x, p.point.y);
  if (tool === "hline") return saveDrawing("hline", { a: pt });
  if (tool === "text") {
    const text = prompt("Note text:"); if (text) await saveDrawing("text", { a: pt, text });
    return;
  }
  if (TWO_POINT.has(tool)) {
    if (!state.pending) { state.pending = { kind: tool, a: pt }; toast("Click the second point (Esc cancels)."); }
    else { const a = state.pending.a; state.pending = null; state.hover = null;
           await saveDrawing(tool, { a, b: pt }); }
  }
}
async function saveDrawing(kind, data) {
  try { const d = await api(`/api/drawings/${state.symbol}`, { method: "POST",
          body: JSON.stringify({ kind, data }) });
        state.drawings.push(d); requestDraw(); }
  catch (e) { toast(`Could not save drawing: ${e.message}`); }
}
async function eraseNear(x, y) {
  let best = null, bestD = 14;
  for (const d of state.drawings) {
    const ax = xOf(d.data.a?.time), ay = yOf(d.data.a?.price);
    const dist = d.kind === "hline" ? Math.abs(y - ay) : Math.hypot(x - (ax ?? -1e9), y - (ay ?? -1e9));
    if (dist < bestD) { bestD = dist; best = d; }
  }
  if (!best) return toast("Nothing to erase here (click on a drawing's first point).");
  await api(`/api/drawings/item/${best.id}`, { method: "DELETE" });
  state.drawings = state.drawings.filter((d) => d.id !== best.id); requestDraw();
}
function setTool(tool) {
  state.tool = tool; state.pending = null; state.hover = null;
  $$(".toolbar button").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
  chart?.applyOptions({ crosshair: { mode: tool === "crosshair" ? LW.CrosshairMode.Magnet
                                                                 : LW.CrosshairMode.Normal } });
  requestDraw();
}

/* -------------------------------------------------------------- watchlist */
async function loadWatchlist() {
  try {
    const rows = await api("/api/watchlist");
    $("#watchTable tbody").innerHTML = rows.map((r) => `
      <tr data-sym="${esc(r.symbol)}" class="${r.symbol === state.symbol ? "sel" : ""}"
          tabindex="0" aria-label="${esc(r.symbol)} ${esc(r.status)}">
        <td><strong>${esc(r.symbol)}</strong>${r.data !== "FRESH" ? ` <span class="muted" title="${esc(r.data)}">${ico("alert", "warn")}</span>` : ""}</td>
        <td><span class="chip ${esc(r.status)}">${ICON[r.status] || ""} ${esc(r.status)}</span>${r.locked ? ' <span class="muted" title="strategy not validated">${ico("lock")}</span>' : ""}</td>
        <td>${esc(r.grade || "")}</td><td>${r.score ?? "–"}</td>
        <td class="${(r.change_pct ?? 0) >= 0 ? "up" : "down"}">${r.change_pct === null ? "–" :
          `${r.change_pct >= 0 ? "▲" : "▼"} ${Math.abs(r.change_pct).toFixed(2)}%`}</td></tr>`).join("");
    $$("#watchTable tbody tr").forEach((tr) => {
      const go = () => { selectSymbol(tr.dataset.sym); if (PHONE.matches) setView("chart"); };
      tr.addEventListener("click", go);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    });
  } catch (e) {
    $("#watchTable tbody").innerHTML = `<tr><td colspan="5" class="muted">Could not load: ${esc(e.message)}</td></tr>`;
  }
}

/* ------------------------------------------------------------ signal panel */
function factorIcon(s) { return s === "good" ? "✓" : (s === "mixed" ? "~" : "✗"); }
async function loadSignal() {
  const panel = $("#signalPanel"); panel.innerHTML = `<div class="muted">Loading signal…</div>`;
  try {
    const ev = await api(`/api/signal/${state.symbol}`); state.signal = ev;
    const c = ev.card; const reg = c.regime || {};
    $("#regimeBanner").innerHTML = `<span class="tag ${esc(reg.regime)}">${esc(reg.regime)}</span>${esc(reg.guidance || "")}`;
    let h = `<div class="headline ${esc(ev.status)}">${ICON[ev.status] || ""} ${esc(c.headline)}</div>`;
    const setupObj = c.setup && typeof c.setup === "object" ? c.setup : null;
    const setupName = typeof c.setup === "string" ? c.setup
      : (c.setup_name || ({ trend: "Trend breakout", pullback: "Trend pullback",
          mean_reversion: "Mean reversion" }[setupObj?.strategy] || ""));
    if (setupName) h += `<div class="muted">${esc(setupName)}</div>`;
    h += `<p>${esc(c.why || "")}</p>`;
    if (c.blocks?.length) h += `<div class="box blocks"><h3>Why no trade</h3><ul>${c.blocks.map((b) => `<li>${esc(b)}</li>`).join("")}</ul></div>`;
    if (c.entry !== undefined && c.size) {
      h += `<div class="kv"><div>Entry</div><div class="mono">${fmt(c.entry)}</div>
        <div>Stop</div><div class="mono">${fmt(c.stop)}</div>
        <div>Size</div><div class="mono">${c.size.lots} lots (${c.size.units} units)</div>
        <div>Amount at risk</div><div class="mono"><strong>${c.size.amount_at_risk} ${esc(c.size.currency)}</strong> (${c.size.risk_pct}%)</div>
        <div>Order</div><div>${esc(c.order?.text || "")}</div></div>`;
      h += `<div class="box"><h3>Playbook</h3><ol>${c.playbook.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>
        <p class="muted">${esc(c.invalidation)}</p></div>`;
      h += `<div class="box"><h3>Skip if</h3><ul>${c.skip_if.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>`;
    }
    if (c.score) {
      h += `<div class="box"><h3>Setup score ${c.score.score} · ${esc(c.score.grade)} · size ×${c.score.size_multiplier}</h3>
        <div class="meter" role="meter" aria-valuenow="${c.score.score}" aria-valuemin="0" aria-valuemax="100"><div style="width:${c.score.score}%"></div></div>
        ${Object.entries(c.score.factors).map(([k, f]) => `<div class="factor" title="${esc(f.label)}">
          <span class="${f.status}">${factorIcon(f.status)}</span><span>${esc(k.replaceAll("_", " "))}${f.mode === "informational" ? " (info)" : ""}</span>
          <span class="mono">${(f.value * 100).toFixed(0)}%</span></div>`).join("")}</div>`;
    }
    if (c.actions) h += `<div class="box"><h3>Next action</h3><ul>${c.actions.map((a) => `<li>${esc(a.text)}</li>`).join("")}</ul>
      <div>Trade now: <strong>${c.trade_r_now >= 0 ? "+" : ""}${c.trade_r_now}R</strong></div></div>`;
    if (c.trace) {
      const tr = c.trace;
      h += `<details class="box"><summary><strong>Why this signal</strong> · rule ${esc(tr.strategy)} v${esc(tr.rule_version)} · ${esc(tr.validator_state)}</summary>
        <div class="muted">${esc(tr.reason)} · ${esc(tr.consensus)}</div>
        <div class="kv"><div>Bar</div><div>${esc(tr.bar_date)} (${esc(tr.data_source)}, data ${esc(String(tr.data_version).slice(0, 12))})</div>
        <div>Rule hash</div><div class="mono">${esc(tr.rule_hash.slice(0, 16))}…</div>
        <div>Validated hash</div><div class="mono">${tr.validated_hash ? esc(tr.validated_hash.slice(0, 16)) + "…" : "none"}</div></div>
        <h3>Bars (closed)</h3><table class="mono"><tbody>${tr.bars.map((b) => `<tr><td>${esc(b.date)}</td><td>O ${fmt(b.open)}</td><td>H ${fmt(b.high)}</td><td>L ${fmt(b.low)}</td><td>C ${fmt(b.close)}</td></tr>`).join("")}</tbody></table>
        <h3>Indicator values</h3><div class="kv mono">${Object.entries(tr.indicators).map(([k, v]) => `<div>${esc(k)}</div><div>${fmt(v, 6)}</div>`).join("")}</div>
        <h3>Levels</h3><div class="kv mono">${Object.entries(tr.levels).map(([k, v]) => `<div>${esc(k)}</div><div>${fmt(v, 6)}</div>`).join("")}</div>
        <h3>Thresholds</h3><div class="kv mono">${Object.entries(tr.thresholds).map(([k, v]) => `<div>${esc(k)}</div><div>${esc(Array.isArray(v) ? v.join(", ") : v)}</div>`).join("")}</div></details>`;
    }
    if (setupObj) h += `<div class="box"><h3>${esc(setupName || "Setup")}</h3><div>${esc(setupObj.summary)}</div></div>`;
    if (c.historical) {
      const hs = c.historical;
      h += `<div class="box"><h3>Historical (validation)</h3><div class="kv">
        <div>Verdict</div><div class="${hs.verdict === "PASS" ? "pass" : "fail"}">${esc(hs.verdict)}</div>
        <div>Win rate</div><div>${pct(hs.win_rate)}</div><div>Avg R</div><div>${fmt(hs.avg_r, 2)}R</div>
        <div>Trades</div><div>${hs.trades}</div></div><p class="muted">${esc(hs.label)}</p></div>`;
    }
    const canLog = ev.status === "BUY" || ev.status === "SELL";
    h += `<button class="primary" id="startChecklist" ${canLog ? "" : "disabled"}
      title="${canLog ? "Run the pre-trade checklist" : "Only available for a validated BUY/SELL signal"}">Start pre-trade checklist</button>
      <button id="logOff" title="Log a trade without a system signal (tagged DISCIPLINE BREACH)">Log off-signal trade…</button>`;
    h += `<div id="marketRead"></div>`;
    panel.innerHTML = h;
    loadMarketRead();
    $("#startChecklist")?.addEventListener("click", () => openChecklist(ev));
    $("#logOff")?.addEventListener("click", () => openChecklist(null));
  } catch (e) { panel.innerHTML = `<div class="box blocks">Could not load the signal: ${esc(e.message)}</div>`; }
}

async function loadMarketRead() {
  const el = $("#marketRead"); if (!el) return;
  try {
    const a = await api(`/api/analysis/${state.symbol}`);
    if (!a.available) { el.innerHTML = `<div class="box muted">${esc(a.text)}</div>`; return; }
    el.innerHTML = `<details class="box" open><summary><strong>Market read (top-down)</strong> ·
        bias <span class="tag">${esc(a.bias)}</span> · ${esc(a.zone)}</summary>
      <ol class="read">${a.steps.map((s) => `<li><strong>${esc(s.step.replace(/^\d+\.\s*/, ""))}</strong>: ${esc(s.text)}</li>`).join("")}</ol>
      <p class="muted">${esc(a.disclaimer)} Bar ${esc(a.bar_date)} · data ${esc(a.data_status)}</p></details>`;
  } catch (e) { el.innerHTML = `<div class="box muted">Market read unavailable: ${esc(e.message)}</div>`; }
}

/* ------------------------------------------------------------- checklist */
function modal(html) { $("#modalBody").innerHTML = html; $("#modal").showModal(); }
function closeModal() { $("#modal").close(); }
function openChecklist(ev) {
  modal(`<h2>Emotion check-in</h2><p class="muted">Rate honestly (1 = low, 5 = high).</p>
    ${["confidence", "stress", "urgency"].map((k) => `<div class="slider"><label for="em_${k}">${k}</label>
      <input id="em_${k}" type="range" min="1" max="5" value="${k === "confidence" ? 3 : 2}">
      <output id="o_${k}">${k === "confidence" ? 3 : 2}</output></div>`).join("")}
    <div class="actions"><button id="mCancel">Cancel</button><button class="primary" id="mNext">Continue</button></div>`);
  ["confidence", "stress", "urgency"].forEach((k) => $(`#em_${k}`).addEventListener("input",
    (e) => ($(`#o_${k}`).textContent = e.target.value)));
  $("#mCancel").onclick = closeModal;
  $("#mNext").onclick = async () => {
    const body = { confidence: +$("#em_confidence").value, stress: +$("#em_stress").value,
                   urgency: +$("#em_urgency").value };
    const g = await api("/api/emotions", { method: "POST", body: JSON.stringify(body) });
    if (!g.allowed) {
      modal(`<div class="wait">Wait. Re-read the plan.</div>
        <p>Stress or urgency is high. The checklist unlocks at ${esc(new Date(g.wait_until).toLocaleTimeString())}.</p>
        <ul>${state.meta.principles.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
        <div class="actions"><button class="primary" id="mOk">OK</button></div>`);
      $("#mOk").onclick = closeModal; return;
    }
    checklistStep(ev);
  };
}
function checklistStep(ev) {
  const c = ev?.card || {};
  modal(`<h2>${ev ? "Pre-trade checklist" : "Off-signal trade"}</h2>
    ${ev ? "" : `<div class="box blocks">This trade has no system signal. It will be logged as a <strong>DISCIPLINE BREACH</strong>.</div>`}
    ${Object.entries(state.meta.checklist).map(([k, t]) => `<label class="check"><input type="checkbox" data-ck="${k}"> ${esc(t)}</label>`).join("")}
    <div class="kv">
      <div>Symbol</div><div>${esc(state.symbol)}</div>
      <div>Direction</div><div><select id="tDir"><option value="1" ${ev?.status === "SELL" ? "" : "selected"}>▲ Long</option><option value="-1" ${ev?.status === "SELL" ? "selected" : ""}>▼ Short</option></select></div>
      <div>Actual entry</div><div><input id="tEntry" type="number" step="any" value="${c.entry ?? ""}"></div>
      <div>Stop (required)</div><div><input id="tStop" type="number" step="any" value="${c.stop ?? ""}" ${ev ? "readonly" : ""}></div>
      <div>Notes</div><div><textarea id="tNotes" rows="2"></textarea></div></div>
    <label class="check"><input type="checkbox" id="tOverride"> Override a blocked rule (logged as a breach): <input id="tReason" placeholder="reason"></label>
    <div class="actions"><button id="mCancel">Cancel</button><button class="primary" id="mLog">Log trade</button></div>`);
  $("#mCancel").onclick = closeModal;
  $("#mLog").onclick = async () => {
    const checklist = Object.fromEntries($$("[data-ck]").map((x) => [x.dataset.ck, x.checked]));
    let shot = "";
    try { shot = chart.takeScreenshot().toDataURL("image/png"); } catch { /* optional */ }
    const body = { signal_id: ev?.signal_id ?? null, symbol: state.symbol, direction: +$("#tDir").value,
      entry: +$("#tEntry").value, stop: +$("#tStop").value, checklist, notes: $("#tNotes").value,
      planned_entry: c.order?.price ?? null, screenshot: shot,
      override_reason: $("#tOverride").checked ? ($("#tReason").value || "override") : "" };
    try { const t = await api("/api/trades", { method: "POST", body: JSON.stringify(body) });
          closeModal(); toast(`Trade #${t.id} logged (${t.mode}).`); refreshAll(); }
    catch (e) { toast(e.message, 6000); }
  };
}

/* ------------------------------------------------------------------ tabs */
const TABS = {
  async trades(el) {
    const ts = await api("/api/trades?status=open");
    el.innerHTML = ts.length ? `<table><thead><tr><th>#</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Stop</th><th>At risk</th><th>Mode</th><th></th></tr></thead><tbody>
      ${ts.map((t) => `<tr><td>${t.id}${t.off_signal ? ' <span class="fail" title="DISCIPLINE BREACH">⚑</span>' : ""}</td><td>${esc(t.symbol)}</td>
        <td class="${t.direction > 0 ? "up" : "down"}">${t.direction > 0 ? "▲ Long" : "▼ Short"}</td><td>${fmt(t.entry)}</td><td>${fmt(t.stop)}</td>
        <td>${t.amount_at_risk}</td><td>${esc(t.mode)}</td>
        <td><button data-stop="${t.id}">Tighten stop</button> <button data-close="${t.id}">Close…</button></td></tr>`).join("")}</tbody></table>`
      : `<p class="muted">No open trades. WAIT is a valid answer.</p>`;
    $$("[data-stop]", el).forEach((b) => (b.onclick = async () => {
      const v = prompt("New stop (it may only move in your favour):"); if (!v) return;
      try { await api(`/api/trades/${b.dataset.stop}/stop`, { method: "PATCH", body: JSON.stringify({ stop: +v }) });
            toast("Stop tightened."); renderTab(); }
      catch (e) { toast(e.message, 6000); renderTab(); }
    }));
    $$("[data-close]", el).forEach((b) => (b.onclick = () => closeTrade(+b.dataset.close)));
  },
  async signals(el) {
    const ss = await api("/api/signals");
    el.innerHTML = ss.length ? `<table><thead><tr><th>Date</th><th>Symbol</th><th>Strategy</th><th>Status</th><th>Grade</th><th>Score</th><th>Order</th><th>Outcome</th><th>Reasons</th></tr></thead><tbody>
      ${ss.map((s) => `<tr><td>${esc(s.session_date)}</td><td>${esc(s.symbol)}</td><td>${esc(s.strategy)}</td>
        <td><span class="chip ${esc(s.status === "LOCKED" ? "WAIT" : s.status)}">${esc(s.status)}</span></td><td>${esc(s.grade)}</td><td>${s.score}</td>
        <td>${esc(s.order)}</td><td>${esc(s.outcome)}</td><td class="muted">${esc(s.reasons.join("; "))}</td></tr>`).join("")}</tbody></table>`
      : `<p class="muted">No signals recorded yet.</p>`;
  },
  async journal(el) {
    const [j, m, closed] = await Promise.all([api("/api/journal"), api("/api/mindset"), api("/api/trades?status=closed")]);
    const w = m.weekly_review;
    el.innerHTML = `<div class="row"><div class="box"><h3>This week</h3>
        <div class="big">${w.plan_adherence_pct}% plan adherence</div>
        <div>Trades ${w.trades} · breaches ${w.breaches} · result ${w.result_r >= 0 ? "+" : ""}${w.result_r}R</div>
        <div class="muted">Best habit: ${esc(w.best_habit || "–")} · Worst: ${esc(w.worst_habit || "–")}</div></div>
      <div class="box" style="flex:1"><h3>Add entry</h3><select id="jKind"><option value="note">Note</option>
        <option value="cooldown_review">Cooldown review</option><option value="reflection">Written reflection</option></select>
        <textarea id="jText" rows="3" style="width:100%"></textarea><button class="primary" id="jAdd">Save</button></div></div>
      <h3>Closed trades</h3><table><thead><tr><th>#</th><th>Symbol</th><th>R</th><th>Followed rules</th><th>Lesson</th></tr></thead><tbody>
      ${closed.map((t) => `<tr><td>${t.id}</td><td>${esc(t.symbol)}</td><td>${fmt(t.r_result, 2)}</td><td>${t.followed_rules ? "✓ yes" : "✗ no"}</td><td>${esc(t.lesson)}</td></tr>`).join("")}</tbody></table>
      <h3>Journal</h3>${j.slice().reverse().map((x) => `<div class="box"><span class="muted">${esc(x.at)} · ${esc(x.kind)}</span><div>${esc(x.text)}</div></div>`).join("") || '<p class="muted">Empty.</p>'}`;
    $("#jAdd").onclick = async () => {
      try { await api("/api/journal", { method: "POST", body: JSON.stringify({ kind: $("#jKind").value, text: $("#jText").value }) });
            toast("Saved."); renderTab(); } catch (e) { toast(e.message, 6000); }
    };
  },
  async validation(el) {
    const v = await api("/api/validation"); const r = v.report;
    if (!r) { el.innerHTML = `<p class="muted">Validation has not been run yet. Run <code>.\\make.ps1 validate</code>.</p>`; return; }
    el.innerHTML = `<p><strong>${esc(r.summary)}</strong> Trials counted: ${v.trials_counted}. Universe: ${esc(r.universe.join(", "))}.
      ${Object.entries(r.excluded).map(([k, x]) => `${esc(k)} ${esc(x)}`).join(" ")}</p>
      <table><thead><tr><th>Strategy</th><th>TF</th><th>Verdict</th><th>Trades</th><th>Win</th><th>Payoff</th><th>Exp R</th><th>PF</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Holdout Sharpe</th><th>PBO</th><th>DSR</th><th>Edge confidence</th></tr></thead><tbody>
      ${r.pairs.map((p) => { const f = p.full || {}; const h = p.holdout || {};
        return `<tr><td>${esc(p.strategy)}</td><td>${esc(p.timeframe)}</td><td class="${p.verdict === "PASS" ? "pass" : "fail"}">${p.verdict === "PASS" ? "✓" : "✗"} ${esc(p.verdict)}</td>
        <td>${f.trades ?? "–"}</td><td>${f.trades ? pct(f.win_rate) : "–"}</td><td>${f.trades ? fmt(f.payoff, 2) : "–"}</td><td>${f.trades ? fmt(f.expectancy_r, 3) : "–"}</td>
        <td>${f.trades ? fmt(f.profit_factor, 2) : "–"}</td><td>${f.trades ? fmt(f.sharpe, 2) : "–"}</td><td>${f.trades ? pct(f.cagr, 2) : "–"}</td><td>${f.trades ? pct(f.max_drawdown) : "–"}</td>
        <td>${h.trades ? fmt(h.sharpe, 2) : "–"}</td><td>${p.pbo ?? "–"}</td><td>${p.dsr === null || p.dsr === undefined ? "–" : fmt(p.dsr, 2)}</td><td title="${esc(p.edge_derivation)}">${p.edge_confidence}</td></tr>
        <tr><td colspan="15" class="muted">${p.reasons.length ? "Reasons: " + esc(p.reasons.join("; ")) : "Passed all gates."}
        ${p.edge_derivation ? " · Edge = " + esc(p.edge_derivation) : ""}
        ${p.cost_stress && Object.keys(p.cost_stress).length ? " · Cost stress Sharpe: " + Object.entries(p.cost_stress).map(([k, x]) => `${k} ${fmt(x, 2)}`).join(", ") : ""}
        ${p.stress_periods && Object.keys(p.stress_periods).length ? " · Stress: " + Object.entries(p.stress_periods).map(([k, x]) => `${k} ${pct(x)}`).join(", ") : ""}
        ${p.walk_forward?.oos_sharpe !== undefined ? ` · Walk-forward OOS Sharpe ${fmt(p.walk_forward.oos_sharpe, 2)}` : ""}</td></tr>`; }).join("")}</tbody></table>
      <h3>Edge decay (edge confidence per validation run)</h3><div id="decay" style="height:160px"></div>`;
    const d = LW.createChart($("#decay"), { autoSize: true, layout: { background: { color: css("--panel") }, textColor: css("--text") },
      grid: { vertLines: { visible: false }, horzLines: { color: css("--border") + "55" } } });
    const by = {};
    for (const x of v.edge_decay) (by[x.strategy] ||= []).push({ time: Math.floor(new Date(x.at + "Z") / 1000), value: x.edge_confidence });
    const cols = [css("--up"), css("--down"), "#9b8cff", "#2ec4b6"]; let i = 0;
    for (const [k, pts] of Object.entries(by)) {
      const s = d.addSeries(LW.LineSeries, { color: cols[i++ % cols.length], title: k, pointMarkersVisible: true });
      const uniq = [...new Map(pts.map((p) => [p.time, p])).values()].sort((a, b) => a.time - b.time);
      s.setData(uniq);
    }
  },
  async mindset(el) {
    const m = await api("/api/mindset");
    el.innerHTML = `<div class="row"><div class="box"><h3>Principles</h3><ul>${m.principles.map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>
      <div class="box"><h3>Entry gate</h3><div class="${m.entry_allowed ? "pass" : "fail"}">${m.entry_allowed ? "✓ Entries allowed" : "✗ Entries blocked"}</div>
        <ul>${m.blockers.map((b) => `<li>${esc(b)}</li>`).join("")}</ul><div>New trades this week: ${m.trades_this_week} / 3</div>
        ${m.reflection_required ? '<div class="fail">Written reflection required (Journal tab).</div>' : ""}</div>
      <div class="box"><h3>What normal looks like (historical)</h3>${Object.entries(m.expectations).map(([k, x]) => `
        <div><strong>${esc(k)}</strong> (${esc(x.verdict)}): typical losing streak ${x.losing_streak_p50}, bad case ${x.losing_streak_p95};
        typical max drawdown ${pct(x.max_drawdown_p50)}, bad case ${pct(x.max_drawdown_p95)}</div>`).join("") || '<span class="muted">Run validation first.</span>'}
        <p class="muted">Ranges from validation, not a forecast. Ordinary losing streaks are expected.</p></div></div>
      <h3>Breaches</h3>${m.breaches.length ? `<table><tbody>${m.breaches.map((b) => `<tr><td>${esc(b.at)}</td><td class="fail">⚑ ${esc(b.kind)}</td><td>${esc(b.detail)}</td></tr>`).join("")}</tbody></table>` : '<p class="muted">None.</p>'}`;
  },
  async missed(el) {
    const m = await api("/api/missed");
    el.innerHTML = `<div class="row"><div class="box"><h3>Opportunity capture rate</h3><div class="big">${m.capture_rate === null ? "–" : pct(m.capture_rate, 0)}</div>
      <div>${m.taken} taken of ${m.valid_setups} valid setups · net effect of missed ${m.net_effect_r >= 0 ? "+" : ""}${fmt(m.net_effect_r, 2)}R (hypothetical)</div></div>
      <div class="box"><h3>By reason</h3>${Object.entries(m.by_reason).map(([k, x]) => `<div>${esc(k)}: ${x.count} · avg ${fmt(x.avg_r, 2)}R</div>`).join("") || '<span class="muted">Nothing missed yet.</span>'}</div></div>
      <table><thead><tr><th>Date</th><th>Symbol</th><th>Strategy</th><th>Status</th><th>Reason</th><th>Hypothetical R</th></tr></thead><tbody>
      ${m.rows.map((r) => `<tr><td>${esc(r.date)}</td><td>${esc(r.symbol)}</td><td>${esc(r.strategy)}</td><td>${esc(r.status)}</td><td>${esc(r.reason)}</td><td>${r.hypothetical_r === null ? "open" : fmt(r.hypothetical_r, 2)}</td></tr>`).join("")}</tbody></table>`;
  },
  async execution(el) {
    const x = await api(`/api/execution/${state.symbol}`);
    el.innerHTML = `<div class="row"><div class="box"><h3>Execution quality now (${esc(state.symbol)})</h3><div class="big">${x.quality.score}/100</div>
      <div>${esc(x.quality.note)}</div><div class="muted">Best UTC hours: ${x.quality.best_utc_hours.join(", ")} · ${esc(x.quality.prefer_overlap)}</div></div>
      <div class="box"><h3>Broker cost check</h3><div>Your spread ${x.your_spread_pips} pips vs assumed ${x.assumed_spread_pips}</div>
      <div>Cost per trade: <strong>${x.cost_per_trade_r}R</strong> (assumed ${x.assumed_cost_per_trade_r}R)</div>
      ${x.warning ? `<div class="fail">${esc(x.warning)}</div>` : ""}<div class="muted">Set your real spread/commission in Settings.</div></div>
      <div class="box"><h3>Slippage learning</h3>${Object.entries(x.slippage.by_symbol).map(([k, v]) => `<div>${esc(k)}: ${fmt(v, 2)} pips</div>`).join("") || '<span class="muted">No fills logged yet.</span>'}</div></div>
      <h3>Median broker spread by UTC hour</h3><div class="row mono">${Object.entries(x.spread_by_hour).map(([h, v]) => `<span>${h}h:${fmt(v, 0)}</span>`).join(" ") || '<span class="muted">Run the intraday refresh with MT5 connected.</span>'}</div>`;
  },
  async paper(el) {
    const p = await api("/api/paper");
    el.innerHTML = `<div class="box"><h3>Paper trading</h3><div class="big ${p.ready_for_small_live_use ? "pass" : "fail"}">${p.ready_for_small_live_use ? "✓ Inside expected range" : "✗ Not ready for live use"}</div>
      <p>${esc(p.message)}</p><div class="kv"><div>Weeks</div><div>${p.weeks}</div><div>Closed paper trades</div><div>${p.closed_trades}</div>
      <div>Paper expectancy</div><div>${p.paper_expectancy_r === null ? "–" : fmt(p.paper_expectancy_r, 2) + "R"}</div>
      <div>Expected range</div><div>${p.expected_range_r ? p.expected_range_r.map((v) => fmt(v, 2)).join(" to ") + "R" : "–"}</div></div></div>`;
  },
  async replay(el) {
    const today = new Date(); const day = (today.getUTCDay() + 6) % 7;
    const mon = new Date(today); mon.setUTCDate(today.getUTCDate() - day - 7);
    const fri = new Date(mon); fri.setUTCDate(mon.getUTCDate() + 4);
    const iso = (d) => d.toISOString().slice(0, 10);
    el.innerHTML = `<div class="row"><label>From <input id="rpS" type="date" value="${iso(mon)}"></label>
      <label>To <input id="rpE" type="date" value="${iso(fri)}"></label>
      <label>Starting balance <input id="rpB" type="number" min="1" value="100"></label>
      <label>Risk % <input id="rpR" type="number" step="0.05" min="0.05" max="1" value="0.5"></label>
      <label>Strategy <select id="rpStrat"><option value="all">All</option><option value="trend">Trend</option>
        <option value="pullback">Pullback</option><option value="mean_reversion">Mean reversion</option></select></label>
      <button class="primary" id="rpGo">Run replay</button> <button id="rpCmp">Compare all strategies</button></div><div id="rpOut" class="muted">Choose a window and run.</div>`;
    $("#rpGo").onclick = async () => {
      const out = $("#rpOut"); out.textContent = "Running…";
      try {
        const r = await api(`/api/replay?start=${$("#rpS").value}&end=${$("#rpE").value}&balance=${+$("#rpB").value}&risk_pct=${+$("#rpR").value}&strategy=${$("#rpStrat").value}`);
        out.innerHTML = `<div class="box blocks">${esc(r.disclaimer)}</div>
          <div class="big">${r.starting_balance} → ${r.ending_balance} (${r.pnl >= 0 ? "+" : ""}${r.pnl}, ${r.return_pct >= 0 ? "+" : ""}${r.return_pct}%)</div>
          ${r.notes.map((n) => `<p class="fail">${esc(n)}</p>`).join("")}
          <table><thead><tr><th>Symbol</th><th>Rule</th><th>Side</th><th>Signal</th><th>Entry</th><th>Stop</th><th>Status</th><th>R</th><th>Risked</th><th>P&L</th><th>Lots</th><th>0.01-lot risk</th></tr></thead><tbody>
          ${r.trades.map((t) => `<tr><td>${esc(t.symbol)}</td><td>${esc(t.strategy)}</td><td class="${t.side === "LONG" ? "up" : "down"}">${t.side === "LONG" ? "▲" : "▼"} ${esc(t.side)}</td>
            <td>${esc(t.signal_date)}</td><td>${fmt(t.entry)}</td><td>${fmt(t.stop)}</td><td>${esc(t.status)}</td><td>${t.r >= 0 ? "+" : ""}${t.r}</td>
            <td>${t.risk_amount}</td><td>${t.pnl >= 0 ? "+" : ""}${t.pnl}</td><td>${t.lots}</td>
            <td class="${t.tradable ? "" : "fail"}">${t.tradable ? "ok" : `✗ ${t.min_lot_risk} (${t.min_lot_risk_pct}%)`}</td></tr>`).join("")}</tbody></table>`;
      } catch (e) { out.innerHTML = `<p class="fail">${esc(e.message)}</p>`; }
    };
    $("#rpCmp").onclick = async () => {
      const out = $("#rpOut"); out.textContent = "Comparing strategies… (about a minute)";
      try {
        const c = await api(`/api/replay/compare?start=${$("#rpS").value}&end=${$("#rpE").value}&balance=${+$("#rpB").value}&risk_pct=${+$("#rpR").value}`);
        out.innerHTML = `<div class="box blocks">${esc(c.disclaimer)}</div>
          <table><thead><tr><th>Strategy</th><th>Start</th><th>End</th><th>P&L</th><th>Return</th><th>Trades</th><th>Closed win rate</th><th>Below 0.01 lot</th><th>Note</th></tr></thead><tbody>
          ${c.rows.map((x) => `<tr><td><strong>${esc(x.strategy)}</strong></td><td>${x.starting_balance}</td><td>${x.ending_balance}</td>
            <td class="${x.pnl >= 0 ? "up" : "down"}">${x.pnl >= 0 ? "▲ +" : "▼ "}${x.pnl}</td><td>${x.return_pct >= 0 ? "+" : ""}${x.return_pct}%</td>
            <td>${x.trades}</td><td>${x.win_rate_closed === null || x.win_rate_closed === undefined ? "–" : pct(x.win_rate_closed, 0)}</td>
            <td>${x.untradable === null || x.untradable === undefined ? "–" : x.untradable}</td><td class="muted">${esc(x.note || "")}</td></tr>`).join("")}</tbody></table>
          <p class="muted">Short windows swing a lot; compare with the 23-year results in "How reliable is this?".</p>`;
      } catch (e) { out.innerHTML = `<p class="fail">${esc(e.message)}</p>`; }
    };
  },
  async quality(el) {
    const q = await api("/api/quality");
    el.innerHTML = `<table><thead><tr><th>Symbol</th><th>Status</th><th>Primary</th><th>History</th><th>Gaps</th><th>Outliers</th><th>Ref</th><th>Agree</th><th>Confirmed</th><th>Signal</th></tr></thead><tbody>
      ${q.map((r) => `<tr><td>${esc(r.symbol)}</td><td>${esc(r.status)}</td><td>${esc(r.primary)}${r.is_proxy ? " (proxy)" : ""}</td><td>${esc(r.first_bar)} → ${esc(r.last_bar)}</td>
      <td>${r.gaps}</td><td>${r.outliers}</td><td>${esc(r.reference || "–")}</td><td>${r.agreement_pct === null ? "–" : r.agreement_pct.toFixed(1) + "%"}</td>
      <td>${esc(r.latest_confirmed || "–")}</td><td>${r.signal_allowed ? "✓ yes" : "✗ no"}</td></tr>`).join("")}</tbody></table>`;
  },
  async settings(el) {
    const s = await api("/api/settings");
    el.innerHTML = `<div class="row"><div class="box"><h3>Account</h3><div class="kv">
      <div>Balance</div><div><input id="sBal" type="number" value="${s.balance}"></div>
      <div>Currency</div><div><input id="sCcy" value="${esc(s.currency)}" maxlength="3"></div>
      <div>Risk % (cap 1%)</div><div><input id="sRisk" type="number" step="0.05" value="${s.risk_pct}"> <span class="muted">locked for ${esc(s.risk_lock?.month || "this month")} (can be lowered)</span></div>
      <div>Mode</div><div><select id="sMode"><option ${s.mode === "paper" ? "selected" : ""}>paper</option><option ${s.mode === "live" ? "selected" : ""}>live</option></select></div>
      <div>Your spread (pips)</div><div><input id="sSpread" type="number" step="0.1" value="${s.user_spread_pips ?? ""}"></div>
      <div>Your commission (pips)</div><div><input id="sComm" type="number" step="0.1" value="${s.user_commission_pips ?? ""}"></div></div>
      <button class="primary" id="sSave">Save</button></div>
      <div class="box"><h3>Alerts</h3><div>Telegram: ${s.telegram_configured ? "✓ configured" : "not configured (add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env)"}</div>
      <div>Desktop notifications: on</div><button id="sRun">Run daily evaluation now</button></div></div>`;
    $("#sSave").onclick = async () => {
      const body = { balance: +$("#sBal").value || null, currency: $("#sCcy").value || null,
        risk_pct: +$("#sRisk").value || null, mode: $("#sMode").value,
        user_spread_pips: $("#sSpread").value === "" ? null : +$("#sSpread").value,
        user_commission_pips: $("#sComm").value === "" ? null : +$("#sComm").value };
      try { await api("/api/settings", { method: "PUT", body: JSON.stringify(body) }); toast("Saved."); refreshAll(); }
      catch (e) { toast(e.message, 6000); }
    };
    $("#sRun").onclick = async () => { toast("Running… this can take a minute.");
      try { const r = await api("/api/run-daily", { method: "POST" }); toast(`Evaluated ${r.evaluated} markets.`); refreshAll(); }
      catch (e) { toast(e.message, 6000); } };
  },
};
async function renderTab() {
  const el = $("#tabBody"); el.innerHTML = `<p class="muted">Loading…</p>`;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
  const act = $(".tabs button.active");
  if (act) act.parentElement.scrollLeft = act.offsetLeft - 8;   // keep the chosen tab visible
  try { await TABS[state.tab](el); } catch (e) { el.innerHTML = `<p class="fail">Could not load: ${esc(e.message)}</p>`; }
}
async function closeTrade(id) {
  modal(`<h2>Close trade #${id}</h2><div class="kv"><div>Exit price</div><div><input id="cPx" type="number" step="any"></div>
    <div>Exit followed the rules?</div><div><select id="cRules"><option value="1">Yes</option><option value="0">No</option></select></div>
    <div>One lesson</div><div><input id="cLesson"></div></div>
    <div class="actions"><button id="mCancel">Cancel</button><button class="primary" id="mClose">Close trade</button></div>`);
  $("#mCancel").onclick = closeModal;
  $("#mClose").onclick = async () => {
    try { const r = await api(`/api/trades/${id}/close`, { method: "POST", body: JSON.stringify({
            exit_price: +$("#cPx").value, followed_rules: $("#cRules").value === "1", lesson: $("#cLesson").value }) });
          const v = r.review;
          modal(`<h2>Post-trade review</h2><div class="kv"><div>Result</div><div class="big">${v.r_result >= 0 ? "+" : ""}${v.r_result}R</div>
            <div>Plan adherence</div><div>${v.plan_adherence ? "✓ yes" : "✗ no"}</div><div>MFE / MAE</div><div>${v.mfe_r}R / ${v.mae_r}R</div>
            <div>Exit per rules</div><div>${v.exit_followed_rules ? "✓" : "✗"}</div><div>Lesson</div><div>${esc(v.lesson)}</div></div>
            <div class="actions"><button class="primary" id="mOk">Done</button></div>`);
          $("#mOk").onclick = () => { closeModal(); refreshAll(); };
    } catch (e) { toast(e.message, 6000); }
  };
}

/* ------------------------------------------------------------- top bar */
async function loadCandles() {
  stopLive(); state.live = null;
  showState("Loading chart…");
  try {
    state.data = await api(`/api/candles/${state.symbol}?tf=${state.tf}`);
    state.drawings = await api(`/api/drawings/${state.symbol}`);
    renderChart();
  } catch (e) { showState(`Could not load chart: ${e.message}`); }
}
async function selectSymbol(sym) {
  state.symbol = sym; localStorage.setItem("tfx.symbol", sym); $("#symbolSearch").value = sym;
  $$("#watchTable tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.sym === sym));
  await Promise.all([loadCandles(), loadSignal()]);
  if (state.tab === "execution") renderTab();
}
function buildTfSelect() {
  const validated = new Set();
  for (const menu of Object.values(state.meta.timeframes))
    for (const m of menu) if (m.validated) validated.add(m.timeframe);
  if (!state.meta.view_timeframes.includes(state.tf)) state.tf = "D1";
  $("#tfBar").innerHTML = state.meta.view_timeframes.map((tf) => {
    const ok = validated.has(tf);
    return `<button data-tf="${tf}" class="${tf === state.tf ? "active" : ""} ${ok ? "" : "unvalidated"}"
      title="${tf}: ${ok ? "validated strategy timeframe" : "view only - not validated, no signals"}"
      aria-pressed="${tf === state.tf}">${TF_LABEL[tf] || tf}</button>`; }).join("");
  $$("#tfBar button").forEach((b) => b.addEventListener("click", () => {
    state.tf = b.dataset.tf; localStorage.setItem("tfx.tf", state.tf);
    $$("#tfBar button").forEach((x) => { x.classList.toggle("active", x === b);
      x.setAttribute("aria-pressed", String(x === b)); });
    loadCandles();
  }));
}
function buildIndMenu() {
  const items = [["dc20", "Donchian 20"], ["dc55", "Donchian 55"], ["dc100", "Donchian 100"],
    ["ema20", "EMA 20"], ["ema100", "EMA 100"], ["ema200", "EMA 200"], ["atrstop", "ATR stop (2x)"],
    ["regime", "Regime shading"], ["levels", "Signal / watch levels"], ["markers", "Trade markers"],
    ["setup", "Strategy setup overlay"], ["ict", "ICT / SMC + supply & demand overlay"],
    ["adx", "ADX pane"], ["rsi", "RSI(2) pane"], ["atr", "ATR pane"]];
  $("#indMenu").innerHTML = items.map(([k, t]) => `<label><input type="checkbox" data-ind="${k}" ${state.ind[k] ? "checked" : ""}> ${t}</label>`).join("");
  $$("[data-ind]").forEach((cb) => cb.addEventListener("change", () => {
    state.ind[cb.dataset.ind] = cb.checked; localStorage.setItem("tfx.ind", JSON.stringify(state.ind)); renderChart();
  }));
}
async function loadSettings() {
  state.settings = await api("/api/settings");
  $("#balanceInput").value = state.settings.balance; $("#riskInput").value = state.settings.risk_pct;
  $("#ccyLabel").textContent = state.settings.currency;
  const theme = state.settings.theme || "dark";
  document.documentElement.dataset.theme = theme;
}
async function loadAlerts() {
  const a = await api("/api/alerts").catch(() => []);
  const n = a.filter((x) => Date.now() - new Date(x.at + "Z") < 86400000).length;
  $("#bellCount").hidden = !n; $("#bellCount").textContent = n;
  $("#bellMenu").innerHTML = a.length ? a.slice(0, 30).map((x) => `<div class="box"><span class="muted">${esc(x.at)} · ${esc(x.channel)}</span><div>${esc(x.text)}</div></div>`).join("") : '<p class="muted">No alerts yet.</p>';
}
function helpDialog() {
  modal(`<h2>Keyboard shortcuts</h2><div class="kv">
    <div>Alt+H</div><div>Horizontal line</div><div>Alt+T</div><div>Trend line</div><div>Alt+R</div><div>Rectangle</div>
    <div>Alt+M</div><div>Measure</div><div>Alt+N</div><div>Text note</div><div>Alt+L / Alt+S</div><div>Long / short position tool</div>
    <div>Alt+E</div><div>Eraser</div><div>Esc</div><div>Cancel / cursor</div><div>+ / −</div><div>Zoom in / out</div><div>?</div><div>This help</div></div>
    <div class="actions"><button class="primary" id="mOk">Close</button></div>`);
  $("#mOk").onclick = closeModal;
}
function zoom(f) {
  const ts = chart?.timeScale(); const r = ts?.getVisibleLogicalRange(); if (!r) return;
  const mid = (r.from + r.to) / 2, half = (r.to - r.from) / 2 * f;
  ts.setVisibleLogicalRange({ from: mid - half, to: mid + half });
}
function bindUI() {
  $$(".toolbar button").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
  $$("[data-ctype]").forEach((b) => b.addEventListener("click", () => {
    state.ctype = b.dataset.ctype; $$("[data-ctype]").forEach((x) => x.classList.toggle("active", x === b)); renderChart(); }));
  $("#symbolSearch").addEventListener("change", (e) => {
    const s = e.target.value.toUpperCase().trim(); if (state.meta.symbols.includes(s)) selectSymbol(s); else toast("Unknown symbol."); });
  $("#indBtn").addEventListener("click", () => { const m = $("#indMenu"); m.hidden = !m.hidden; $("#indBtn").setAttribute("aria-expanded", String(!m.hidden)); });
  $("#bellBtn").addEventListener("click", () => { const m = $("#bellMenu"); m.hidden = !m.hidden; });
  $("#themeBtn").addEventListener("click", async () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    if (STATIC) { try { localStorage.setItem("tfx.theme", t); } catch { /* private mode */ } }
    else await api("/api/settings", { method: "PUT", body: JSON.stringify({ theme: t }) });
    renderChart(); });
  $("#settingsBtn").addEventListener("click", () => {
    state.tab = "settings"; renderTab(); if (PHONE.matches) setView("more"); });
  $$(".mnav button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#moreBtn").addEventListener("click", () => {
    const open = document.body.classList.toggle("acct-open");
    $("#moreBtn").setAttribute("aria-expanded", String(open)); });
  PHONE.addEventListener("change", () => setView(document.body.dataset.view || "chart"));
  $("#helpBtn").addEventListener("click", helpDialog);
  const saveAcct = async () => {
    try { await api("/api/settings", { method: "PUT", body: JSON.stringify({ balance: +$("#balanceInput").value, risk_pct: +$("#riskInput").value }) });
          toast("Account updated."); await loadSettings(); loadSignal(); }
    catch (e) { toast(e.message, 6000); await loadSettings(); }
  };
  $("#balanceInput").addEventListener("change", saveAcct); $("#riskInput").addEventListener("change", saveAcct);
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => { state.tab = b.dataset.tab; renderTab(); }));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    const map = { h: "hline", t: "trendline", r: "rect", m: "measure", n: "text", l: "long", s: "short", e: "eraser" };
    if (e.altKey && map[e.key.toLowerCase()]) { e.preventDefault(); setTool(map[e.key.toLowerCase()]); }
    else if (e.key === "Escape") setTool("cursor");
    else if (e.key === "+" || e.key === "=") zoom(0.8);
    else if (e.key === "-") zoom(1.25);
    else if (e.key === "?") helpDialog();
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".menu")) { $("#indMenu").hidden = true; $("#bellMenu").hidden = true; } });
}
async function refreshAll() { await Promise.all([loadWatchlist(), loadSignal(), loadAlerts()]); renderTab(); }

async function init() {
  if (!LW) { showState("Chart library failed to load."); return; }
  try {
    state.meta = await api("/api/meta");
    if (STATIC) {
      const when = new Date(state.meta.published_at).toLocaleString();
      $(".footer .muted").textContent = `Public view-only · analysis published ${when} · ${state.meta.live_note}`;
    }
    $("#symbolList").innerHTML = state.meta.symbols.map((s) => `<option value="${s}">`).join("");
    if (!state.meta.symbols.includes(state.symbol)) state.symbol = state.meta.symbols[0];
    buildTfSelect(); buildIndMenu(); bindUI(); await loadSettings();
    await Promise.all([loadWatchlist(), selectSymbol(state.symbol), loadAlerts()]);
    renderTab();
    setInterval(() => { loadWatchlist(); loadAlerts(); }, 5 * 60 * 1000);
  } catch (e) { showState(`Could not start: ${e.message}`); }
}
window.TFX = { state, selectSymbol, setTool, renderTab, setView };  // for tests / debugging
init();
})();
