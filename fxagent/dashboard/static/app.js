/*
 * The panel's front end. Vanilla, deliberately.
 *
 * It does four things: stay current with the server, draw the price series in whichever form
 * is selected, paint the session bands, and render the feed. It computes nothing. Every price,
 * level, session boundary and marker is calculated on the server from the same code the
 * strategies read — this file would be the easiest place in the system to introduce a second,
 * subtly different definition of "the London session" or "the 20-period EMA", so it is not
 * allowed to have one.
 *
 * Two exceptions, both deliberate. The grant countdown ticks locally against `expires_at`,
 * because a countdown pushed over a socket would measure network latency as well as time
 * remaining. And the chart *type* is a client-side view of the same candles — switching from
 * candles to a line re-reads `close`, it does not ask the server for anything.
 *
 * All user-visible text goes through `textContent`. Agent narration is prose from a language
 * model and reasons are stored strings; neither is ever interpolated into markup.
 */

"use strict";

const LWC = window.LightweightCharts;
const SVG_NS = "http://www.w3.org/2000/svg";
const STORE_KEY = "fxagent.panel.v1";

/** Every form the price series can take. Same bars, different reading of them. */
const CHART_TYPES = [
  { key: "candles", label: "Candles", icon: "i-candles" },
  { key: "bars", label: "Bars", icon: "i-bars" },
  { key: "line", label: "Line", icon: "i-line" },
  { key: "area", label: "Area", icon: "i-area" },
  { key: "baseline", label: "Baseline", icon: "i-baseline" },
];

/* Mirrors --tokyo/--london/--newyork/--overlap in styles.css. Duplicated because the bands are
   painted onto a canvas, which cannot read a CSS custom property; kept adjacent in both files
   so a change to one is obvious in the other. */
const SESSION_FILL = {
  TOKYO: "rgba(120, 160, 220, 0.06)",
  LONDON: "rgba(230, 170, 80, 0.06)",
  NEW_YORK: "rgba(150, 120, 210, 0.06)",
  OVERLAP: "rgba(230, 120, 120, 0.085)",
};

const PALETTE = {
  up: "#31c9a4",
  down: "#f2555a",
  line: "#e9d8b4",
  text: "#e8ebf1",
  muted: "#6a7383",
  hairline: "rgba(255, 255, 255, 0.06)",
  surface: "#07090c",
};

const state = {
  socket: null,
  transport: "ws",
  pollSeconds: 15,
  pollTimer: null,
  revision: null,
  socketFailures: 0,
  snapshot: null,
  chart: null,
  priceSeries: null,
  chartType: "candles",
  overlays: new Map(), // key -> ISeriesApi
  tradeLines: [],
  times: [],
  bands: [],
  hidden: new Set(), // overlay group keys the user switched off
  layers: { sessions: true, markers: true, trades: true },
  drawers: { options: false, feed: true },
  seenEntries: new Set(),
  grantExpiry: null,
  reconnectDelay: 1000,
};

// --- storage -----------------------------------------------------------------

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    if (CHART_TYPES.some((t) => t.key === saved.chartType)) state.chartType = saved.chartType;
    if (Array.isArray(saved.hidden)) state.hidden = new Set(saved.hidden);
    if (saved.layers) Object.assign(state.layers, saved.layers);
    if (saved.drawers) Object.assign(state.drawers, saved.drawers);
  } catch {
    // A corrupt preference is not worth a broken panel; the defaults are all usable.
  }
}

function save() {
  try {
    localStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        chartType: state.chartType,
        hidden: [...state.hidden],
        layers: state.layers,
        drawers: state.drawers,
      }),
    );
  } catch {
    /* private mode, quota — neither is a reason to stop working */
  }
}

// --- DOM helpers ---------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#${name}`);
  svg.appendChild(use);
  return svg;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmt(value, digits) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

/** ISO instant -> "2026-08-15 08:00Z". The panel is a UTC instrument: local time would make
 *  two people reading the same screen in two places disagree about when a bar was. */
function stamp(iso) {
  if (!iso) return "—";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  return at.toISOString().replace("T", " ").slice(0, 16) + "Z";
}

// --- tab groups ----------------------------------------------------------------

/**
 * A glass tab strip with one sliding indicator.
 *
 * The indicator is a single element moved with a transform rather than a border on each tab,
 * so selection animates without touching layout — and so the movement itself says *which way*
 * you went, which a border appearing somewhere else cannot.
 */
function renderTabs(container, items, selected, onSelect) {
  const pill = container.querySelector(".tabs__pill");
  clear(container);
  container.appendChild(pill);

  items.forEach((item) => {
    const tab = el("button", "tab");
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(item.key === selected));
    tab.dataset.key = item.key;

    if (item.icon) tab.appendChild(icon(item.icon));
    tab.appendChild(el("span", "tab__label", item.label));
    if (item.icon) tab.title = item.label;

    tab.addEventListener("click", () => onSelect(item.key));
    container.appendChild(tab);
  });

  movePill(container);
}

function movePill(container) {
  const active = container.querySelector('.tab[aria-selected="true"]');
  const pill = container.querySelector(".tabs__pill");
  if (!active || !pill) return;

  container.style.setProperty("--pill-x", `${active.offsetLeft - container.clientLeft}px`);
  container.style.setProperty("--pill-w", `${active.offsetWidth}px`);
  container.classList.add("tabs--ready");
}

function selectTab(container, key) {
  container.querySelectorAll(".tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.key === key));
  });
  movePill(container);
}

// --- chart ---------------------------------------------------------------------

function buildChart() {
  const chart = LWC.createChart(document.getElementById("chart"), {
    // v4's own ResizeObserver. The drawers animate their width, so the chart is resized on
    // every frame of that transition; letting the library own it beats racing it.
    autoSize: true,
    layout: {
      background: { color: "transparent" },
      textColor: PALETTE.muted,
      fontFamily: getComputedStyle(document.body).fontFamily,
    },
    grid: {
      vertLines: { color: PALETTE.hairline },
      horzLines: { color: PALETTE.hairline },
    },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.09)" },
    timeScale: {
      borderColor: "rgba(255,255,255,0.09)",
      timeVisible: true,
      secondsVisible: false,
    },
    crosshair: {
      mode: LWC.CrosshairMode.Normal,
      vertLine: { color: "rgba(233,216,180,0.35)", labelBackgroundColor: "#1a1d24" },
      horzLine: { color: "rgba(233,216,180,0.35)", labelBackgroundColor: "#1a1d24" },
    },
    localization: {
      // Axis labels in UTC, matching everything else on the page.
      timeFormatter: (t) => new Date(t * 1000).toISOString().slice(11, 16) + "Z",
    },
  });

  state.chart = chart;
  chart.timeScale().subscribeVisibleLogicalRangeChange(drawSessions);

  const wrap = document.querySelector(".chart-wrap");
  new ResizeObserver(() => {
    resizeCanvas();
    drawSessions();
  }).observe(wrap);
}

/** Create the price series in the currently selected form. Same bars either way. */
function createPriceSeries(precision, candles) {
  const format = { type: "price", precision, minMove: Math.pow(10, -precision) };
  const common = { priceFormat: format, priceLineVisible: true, lastValueVisible: true };

  switch (state.chartType) {
    case "bars":
      return state.chart.addBarSeries({
        ...common,
        upColor: PALETTE.up,
        downColor: PALETTE.down,
        thinBars: false,
      });
    case "line":
      return state.chart.addLineSeries({ ...common, color: PALETTE.line, lineWidth: 2 });
    case "area":
      return state.chart.addAreaSeries({
        ...common,
        lineColor: PALETTE.line,
        lineWidth: 2,
        topColor: "rgba(233, 216, 180, 0.22)",
        bottomColor: "rgba(233, 216, 180, 0.01)",
      });
    case "baseline":
      // Anchored on the first close on screen, so the shading answers "up or down over this
      // window" — which is the only question a baseline is any good at.
      return state.chart.addBaselineSeries({
        ...common,
        baseValue: { type: "price", price: candles.length ? candles[0].close : 0 },
        topLineColor: PALETTE.up,
        topFillColor1: "rgba(49, 201, 164, 0.22)",
        topFillColor2: "rgba(49, 201, 164, 0.02)",
        bottomLineColor: PALETTE.down,
        bottomFillColor1: "rgba(242, 85, 90, 0.02)",
        bottomFillColor2: "rgba(242, 85, 90, 0.22)",
      });
    default:
      return state.chart.addCandlestickSeries({
        ...common,
        upColor: PALETTE.up,
        downColor: PALETTE.down,
        borderUpColor: PALETTE.up,
        borderDownColor: PALETTE.down,
        wickUpColor: "rgba(49, 201, 164, 0.8)",
        wickDownColor: "rgba(242, 85, 90, 0.8)",
      });
  }
}

/** OHLC forms take the bar; single-value forms take its close. */
function priceData(candles) {
  if (state.chartType === "candles" || state.chartType === "bars") return candles;
  return candles.map((candle) => ({ time: candle.time, value: candle.close }));
}

function setChartType(next) {
  if (next === state.chartType) return;
  state.chartType = next;
  save();
  selectTab(document.getElementById("type-tabs"), next);

  // Tear the whole price layer down and rebuild it from the cached snapshot. Lightweight
  // Charts draws series in creation order, so recreating only the price series would put the
  // candles on top of the overlays in one view and underneath in another.
  if (state.priceSeries) {
    state.chart.removeSeries(state.priceSeries);
    state.priceSeries = null;
  }
  for (const series of state.overlays.values()) state.chart.removeSeries(series);
  state.overlays.clear();
  for (const line of state.tradeLines) state.chart.removeSeries(line);
  state.tradeLines = [];

  if (state.snapshot) applyChart(state.snapshot.chart);
}

function applyChart(payload) {
  const precision = payload.price_precision;

  if (!state.priceSeries) {
    state.priceSeries = createPriceSeries(precision, payload.candles);
  }
  state.priceSeries.setData(priceData(payload.candles));

  state.times = payload.candles.map((candle) => candle.time);
  state.bands = payload.session_bands;

  applyOverlays(payload.overlays);
  applyTradeLines(payload.trades, precision);
  applyMarkers(payload.markers);

  renderLegend(payload);
  renderOverlayToggles(payload);
  renderNotes(document.getElementById("chart-notes"), payload.notes);
  resizeCanvas();
  drawSessions();
}

/** `bb_upper` and `bb_lower` are one thing to a reader, so they are one switch. */
function groupOf(key) {
  if (key.startsWith("bb_")) return "bollinger";
  if (key.startsWith("asian_")) return "asian";
  return key;
}

function groupLabel(overlay) {
  const group = groupOf(overlay.key);
  if (group === "bollinger") return overlay.label.replace(/ (upper|mid|lower)$/, "");
  if (group === "asian") return "Asian range";
  return overlay.label;
}

/** Overlays are keyed, so a redraw updates the series in place instead of dropping and
 *  recreating it — which would reset the price scale and make the chart jump every refresh. */
function applyOverlays(overlays) {
  const seen = new Set();

  for (const overlay of overlays) {
    seen.add(overlay.key);
    let series = state.overlays.get(overlay.key);
    if (!series) {
      series = state.chart.addLineSeries({
        color: overlay.colour,
        lineWidth: overlay.width || 1,
        lineType: overlay.style === "step" ? LWC.LineType.WithSteps : LWC.LineType.Simple,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      state.overlays.set(overlay.key, series);
    }
    series.applyOptions({ visible: !state.hidden.has(groupOf(overlay.key)) });
    // A point with a null value becomes whitespace, which breaks the line rather than drawing
    // through a gap. See the note on holes in `dashboard/models.py`.
    series.setData(
      overlay.points.map((point) =>
        point.value === null || point.value === undefined
          ? { time: point.time }
          : { time: point.time, value: point.value },
      ),
    );
  }

  for (const [key, series] of state.overlays) {
    if (!seen.has(key)) {
      state.chart.removeSeries(series);
      state.overlays.delete(key);
    }
  }
}

function applyMarkers(markers) {
  if (!state.priceSeries) return;
  state.priceSeries.setMarkers(
    state.layers.markers
      ? markers.map((marker) => ({
          time: marker.time,
          position: marker.position,
          shape: marker.shape,
          color: marker.colour,
          text: marker.text,
        }))
      : [],
  );
}

/** Entry, stop and target for each trade, as three two-point lines spanning the position.
 *  Recreated wholesale: trades are few, and a stop that moved would otherwise need diffing. */
function applyTradeLines(trades, precision) {
  for (const line of state.tradeLines) state.chart.removeSeries(line);
  state.tradeLines = [];
  if (!state.layers.trades) return;

  for (const trade of trades) {
    const levels = [
      { price: trade.entry_price, colour: PALETTE.text, dash: LWC.LineStyle.Solid },
      { price: trade.stop_price, colour: PALETTE.down, dash: LWC.LineStyle.Dashed },
      { price: trade.target_price, colour: PALETTE.up, dash: LWC.LineStyle.Dashed },
    ];
    for (const level of levels) {
      const series = state.chart.addLineSeries({
        color: level.colour,
        lineWidth: 1,
        lineStyle: level.dash,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        priceFormat: { type: "price", precision, minMove: Math.pow(10, -precision) },
      });
      series.setData([
        { time: trade.start, value: level.price },
        { time: Math.max(trade.end, trade.start + 1), value: level.price },
      ]);
      state.tradeLines.push(series);
    }
  }
}

// --- legend and view options ------------------------------------------------------

function toggleGroup(group) {
  if (state.hidden.has(group)) state.hidden.delete(group);
  else state.hidden.add(group);
  save();
  if (state.snapshot) applyChart(state.snapshot.chart);
}

function toggleLayer(name) {
  state.layers[name] = !state.layers[name];
  save();
  if (state.snapshot) applyChart(state.snapshot.chart);
}

/** The legend doubles as the fastest way to switch a series off — one click, where you are
 *  already looking, instead of opening a drawer to find the same switch. */
function renderLegend(payload) {
  const legend = document.getElementById("legend");
  clear(legend);

  const head = el("span", "legend__item legend__item--head");
  head.appendChild(el("span", null, `${payload.symbol} · ${payload.timeframe}`));
  legend.appendChild(head);

  const groups = new Map();
  for (const overlay of payload.overlays) {
    const group = groupOf(overlay.key);
    if (!groups.has(group)) groups.set(group, { label: groupLabel(overlay), colour: overlay.colour });
  }

  for (const [group, meta] of groups) {
    const button = el("button", "legend__item");
    button.type = "button";
    button.setAttribute("aria-pressed", String(!state.hidden.has(group)));
    button.title = `Show or hide ${meta.label}`;

    const swatch = el("span", "swatch");
    swatch.style.background = meta.colour;
    button.appendChild(swatch);
    button.appendChild(el("span", null, meta.label));
    button.addEventListener("click", () => toggleGroup(group));
    legend.appendChild(button);
  }

  const strategies = new Map();
  for (const marker of payload.markers) {
    if (marker.strategy && !strategies.has(marker.strategy)) {
      strategies.set(marker.strategy, marker.colour);
    }
  }
  for (const [strategy, colour] of strategies) {
    const item = el("span", "legend__item");
    const dot = el("span", "dot");
    dot.style.background = colour;
    item.appendChild(dot);
    item.appendChild(el("span", null, strategy));
    legend.appendChild(item);
  }
}

function switchRow(label, checked, colour, onToggle) {
  const row = el("button", "switch");
  row.type = "button";
  row.setAttribute("role", "switch");
  row.setAttribute("aria-checked", String(checked));

  const left = el("span", "switch__label");
  if (colour) {
    const swatch = el("span", "switch__swatch");
    swatch.style.background = colour;
    left.appendChild(swatch);
  }
  left.appendChild(el("span", null, label));

  row.appendChild(left);
  row.appendChild(el("span", "switch__track"));
  row.addEventListener("click", onToggle);
  return row;
}

function renderOverlayToggles(payload) {
  const host = document.getElementById("overlay-toggles");
  clear(host);

  const groups = new Map();
  for (const overlay of payload.overlays) {
    const group = groupOf(overlay.key);
    if (!groups.has(group)) groups.set(group, { label: groupLabel(overlay), colour: overlay.colour });
  }
  for (const [group, meta] of groups) {
    host.appendChild(
      switchRow(meta.label, !state.hidden.has(group), meta.colour, () => toggleGroup(group)),
    );
  }

  host.appendChild(el("div", "drawer__rule"));
  host.appendChild(
    switchRow("Session shading", state.layers.sessions, null, () => toggleLayer("sessions")),
  );
  host.appendChild(
    switchRow("Signal markers", state.layers.markers, null, () => toggleLayer("markers")),
  );
  host.appendChild(switchRow("Trade levels", state.layers.trades, null, () => toggleLayer("trades")));
}

// --- session shading ------------------------------------------------------------

function resizeCanvas() {
  const canvas = document.getElementById("sessions");
  const wrap = canvas.parentElement;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(wrap.clientWidth * ratio);
  canvas.height = Math.floor(wrap.clientHeight * ratio);
  canvas.style.width = wrap.clientWidth + "px";
  canvas.style.height = wrap.clientHeight + "px";
  canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
}

/**
 * Position a UTC instant on the chart's logical (bar-index) axis.
 *
 * `timeToCoordinate` only answers for times that are in the series data, and a session
 * boundary at 17:00 on a Friday is not a bar on a market that shut at 21:00. So band edges are
 * interpolated between the bars either side of them and converted through
 * `logicalToCoordinate`, which accepts fractions. A weekend collapses to a hairline, which is
 * correct: no bars means no width on a bar-indexed axis.
 */
function logicalFor(time) {
  const times = state.times;
  if (times.length === 0) return null;
  if (time <= times[0]) return 0;
  if (time >= times[times.length - 1]) return times.length - 1;

  let low = 0;
  let high = times.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (times[middle] <= time) low = middle;
    else high = middle;
  }
  const span = times[high] - times[low];
  return span > 0 ? low + (time - times[low]) / span : low;
}

function drawSessions() {
  const canvas = document.getElementById("sessions");
  const context = canvas.getContext("2d");
  const wrap = canvas.parentElement;
  context.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);

  if (!state.chart || !state.layers.sessions || state.bands.length === 0) return;

  const scale = state.chart.timeScale();
  const visible = scale.getVisibleLogicalRange();
  if (!visible) return;

  // Keep the shading inside the plot area: the price scale on the right and the time axis
  // below belong to the chart's chrome, and painting over them looks like a rendering fault.
  const plotWidth = wrap.clientWidth - state.chart.priceScale("right").width();
  const plotHeight = wrap.clientHeight - scale.height();

  context.save();
  context.beginPath();
  context.rect(0, 0, plotWidth, plotHeight);
  context.clip();

  for (const band of state.bands) {
    const from = logicalFor(band.start);
    const to = logicalFor(band.end);
    if (from === null || to === null) continue;

    const left = Math.max(from, visible.from);
    const right = Math.min(to, visible.to);
    if (right <= left) continue;

    const x1 = scale.logicalToCoordinate(left);
    const x2 = scale.logicalToCoordinate(right);
    if (x1 === null || x2 === null) continue;

    context.fillStyle = SESSION_FILL[band.session] || "rgba(255, 255, 255, 0.05)";
    context.fillRect(x1, 0, Math.max(x2 - x1, 1), plotHeight);
  }

  context.restore();
}

// --- feed ---------------------------------------------------------------------------

function renderNotes(container, notes) {
  clear(container);
  for (const note of notes || []) container.appendChild(el("p", null, note));
}

function renderGrant(grant) {
  const card = document.getElementById("grant");
  clear(card);

  const row = el("div", "grant__row");
  row.appendChild(el("span", `grant__state grant__state--${grant.state}`, grant.state));
  if (grant.symbols && grant.symbols.length) {
    row.appendChild(el("span", "badge", grant.symbols.join(" ")));
  }
  const countdown = el("span", "grant__countdown");
  countdown.id = "countdown";
  row.appendChild(countdown);
  card.appendChild(row);

  if (grant.granted_at) {
    card.appendChild(el("div", "grant__reason", `granted ${stamp(grant.granted_at)}`));
  }
  card.appendChild(el("div", "grant__reason", grant.reason));

  state.grantExpiry = grant.expires_at ? new Date(grant.expires_at) : null;
  tickCountdown();
}

function tickCountdown() {
  const node = document.getElementById("countdown");
  if (!node) return;
  if (!state.grantExpiry) {
    node.textContent = "";
    return;
  }
  const left = Math.floor((state.grantExpiry.getTime() - Date.now()) / 1000);
  if (left <= 0) {
    node.textContent = "EXPIRED";
    return;
  }
  const hours = String(Math.floor(left / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((left % 3600) / 60)).padStart(2, "0");
  const seconds = String(left % 60).padStart(2, "0");
  node.textContent = `expires in ${hours}:${minutes}:${seconds}`;
}

function voteRow(vote) {
  // Silent, gated and flat are three different facts and stay three different rows. Collapsing
  // them would throw away the only reason the consensus diagnostics are written down.
  const silent = vote.direction === null || vote.direction === undefined;
  const gated = !silent && !vote.participated;
  const row = el("tr", silent ? "vote--silent" : gated ? "vote--gated" : null);

  row.appendChild(el("td", null, vote.strategy));
  row.appendChild(el("td", null, fmt(vote.weight, 2)));
  row.appendChild(el("td", `dir--${vote.direction || "NONE"}`, vote.direction || "silent"));
  row.appendChild(el("td", null, vote.confidence === null ? "—" : fmt(vote.confidence, 2)));
  row.appendChild(el("td", null, vote.reason));
  return row;
}

function narrationSection(title, narration, extra) {
  const section = el("div", "section");
  section.appendChild(el("h4", null, title));
  section.appendChild(el("p", null, narration.text));
  if (extra) section.appendChild(extra);

  const bits = [narration.provider, narration.model, stamp(narration.generated_at)].filter(
    (bit) => bit && bit !== "—",
  );
  if (bits.length) section.appendChild(el("div", "provenance", bits.join(" · ")));
  return section;
}

function analoguesTable(analogues) {
  const table = el("table", "analogues");
  const head = el("tr");
  for (const label of ["window", "similarity", "resolved", "outcome"]) {
    head.appendChild(el("th", null, label));
  }
  table.appendChild(head);

  for (const analogue of analogues) {
    const row = el("tr");
    row.appendChild(el("td", null, `${analogue.symbol} ${stamp(analogue.timestamp)}`));
    row.appendChild(el("td", null, fmt(analogue.similarity, 3)));
    // Displayed because it is the point-in-time claim: an analogue that resolved after the bar
    // it was retrieved for should never have been retrievable.
    row.appendChild(el("td", null, stamp(analogue.resolved_at)));
    row.appendChild(
      el(
        "td",
        null,
        analogue.outcome_r === null || analogue.outcome_r === undefined
          ? analogue.outcome || "—"
          : `${fmt(analogue.outcome_r, 2)}R ${analogue.outcome || ""}`.trim(),
      ),
    );
    table.appendChild(row);
  }
  return table;
}

function renderEntry(entry) {
  const node = el("article", `entry${entry.fired ? " entry--fired" : ""}`);

  const head = el("div", "entry__head");
  head.appendChild(el("span", "entry__time", stamp(entry.timestamp)));
  head.appendChild(el("span", "badge", entry.symbol));
  head.appendChild(
    el("span", entry.fired ? "badge badge--fired" : "badge", entry.fired ? "FIRED" : "no trade"),
  );
  head.appendChild(el("span", "badge", `score ${fmt(entry.consensus_score, 2)}`));
  node.appendChild(head);

  node.appendChild(el("div", "reason", entry.reason));

  const regime = el("div", "regime");
  const sessions = entry.regime.sessions.length ? entry.regime.sessions.join(" + ") : "none";
  regime.appendChild(el("span", null, sessions));
  regime.appendChild(el("span", null, `ADX ${fmt(entry.regime.trend_strength, 1)}`));
  regime.appendChild(el("span", null, `vol ${fmt(entry.regime.volatility_percentile, 0)}%`));
  regime.appendChild(
    el(
      "span",
      null,
      entry.regime.is_trending ? "trending" : entry.regime.is_ranging ? "ranging" : "neither",
    ),
  );
  regime.appendChild(el("span", null, entry.regime.market_open ? "open" : "shut"));
  node.appendChild(regime);

  const table = el("table", "votes");
  const header = el("tr");
  for (const label of ["strategy", "weight", "vote", "conf", "why"]) {
    header.appendChild(el("th", null, label));
  }
  table.appendChild(header);
  for (const vote of entry.votes) table.appendChild(voteRow(vote));
  node.appendChild(table);

  if (entry.chartist) node.appendChild(narrationSection("CHARTIST", entry.chartist));

  if (entry.historian) {
    node.appendChild(
      narrationSection(
        "HISTORIAN",
        entry.historian,
        entry.analogues.length ? analoguesTable(entry.analogues) : null,
      ),
    );
  } else if (entry.analogues.length) {
    const section = el("div", "section");
    section.appendChild(el("h4", null, "ANALOGUES"));
    section.appendChild(analoguesTable(entry.analogues));
    node.appendChild(section);
  }

  if (entry.risk_officer) {
    const section = narrationSection("RISK OFFICER", entry.risk_officer);
    if (entry.risk_officer.proceed_recommendation) {
      section.appendChild(
        el(
          "div",
          "advisory-only",
          `recommendation: ${entry.risk_officer.proceed_recommendation} — advisory only. ` +
            "The deterministic permission layer gates execution, and this does not.",
        ),
      );
    }
    node.appendChild(section);
  }

  for (const pattern of entry.patterns) {
    const section = el("div", "section section--pattern");
    section.appendChild(el("h4", null, `FORMATION — ${pattern.name}`));
    // The label rides on the data, so a formation cannot reach the screen without it.
    section.appendChild(el("div", "context-only", pattern.label));
    section.appendChild(el("p", null, pattern.definition));
    if (pattern.bar_time) {
      section.appendChild(el("div", "provenance", `on the bar at ${stamp(pattern.bar_time)}`));
    }
    node.appendChild(section);
  }

  for (const trade of entry.trades) {
    const section = el("div", "section section--trade");
    section.appendChild(el("h4", null, `TRADE ${trade.direction} — ${trade.mode}`));
    section.appendChild(
      el(
        "p",
        null,
        `${trade.volume} @ ${trade.entry_price} · stop ${trade.stop_price} · target ${trade.target_price}`,
      ),
    );
    if (trade.exit_time) {
      section.appendChild(
        el(
          "div",
          "provenance",
          `exited ${stamp(trade.exit_time)} at ${trade.exit_price} (${trade.barrier_touched}) ` +
            `${trade.r_multiple === null ? "" : fmt(trade.r_multiple, 2) + "R"}`,
        ),
      );
    } else {
      section.appendChild(el("div", "provenance", "open"));
    }
    node.appendChild(section);
  }

  if (entry.discarded.length) {
    const section = el("div", "section section--discarded");
    section.appendChild(el("h4", null, "DISCARDED"));
    section.appendChild(
      el(
        "p",
        null,
        `${entry.discarded.join(", ")} failed validation and was discarded whole rather than ` +
          "partially rendered.",
      ),
    );
    node.appendChild(section);
  }

  return node;
}

function applyFeed(payload) {
  renderGrant(payload.grant);
  renderNotes(document.getElementById("feed-notes"), payload.notes);

  const feed = document.getElementById("feed");
  clear(feed);

  if (payload.entries.length === 0) {
    feed.appendChild(el("div", "empty", "No evaluations in this window."));
    return;
  }

  // Only genuinely new entries animate in, staggered. Re-animating the whole list on every
  // push would make a quiet market look busy, which is the opposite of what the panel is for.
  let staggered = 0;
  for (const entry of payload.entries) {
    const node = renderEntry(entry);
    if (state.seenEntries.has(entry.evaluation_id)) {
      node.style.animation = "none";
    } else {
      node.style.setProperty("--delay", `${Math.min(staggered++ * 40, 240)}ms`);
      state.seenEntries.add(entry.evaluation_id);
    }
    feed.appendChild(node);
  }
}

// --- drawers -----------------------------------------------------------------------

const narrow = window.matchMedia("(max-width: 1100px)");

function setDrawer(name, open) {
  state.drawers[name] = open;
  save();

  const drawer = document.getElementById(name === "feed" ? "feed-drawer" : "options-drawer");
  const button = document.getElementById(name === "feed" ? "toggle-feed" : "toggle-options");
  drawer.dataset.open = String(open);
  button.setAttribute("aria-expanded", String(open));
  updateScrim();
}

/** The scrim only exists where the drawers float over the chart. Docked, there is nothing to
 *  dismiss and dimming the chart would be dimming the thing you came to look at. */
function updateScrim() {
  const scrim = document.getElementById("scrim");
  const covering = narrow.matches && (state.drawers.options || state.drawers.feed);
  scrim.hidden = !covering;
  scrim.dataset.show = String(covering);
}

function wireDrawers() {
  for (const name of ["options", "feed"]) setDrawer(name, state.drawers[name]);

  document
    .getElementById("toggle-options")
    .addEventListener("click", () => setDrawer("options", !state.drawers.options));
  document
    .getElementById("toggle-feed")
    .addEventListener("click", () => setDrawer("feed", !state.drawers.feed));
  document.getElementById("close-options").addEventListener("click", () => setDrawer("options", false));

  document.getElementById("scrim").addEventListener("click", () => {
    setDrawer("options", false);
    setDrawer("feed", false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !narrow.matches) return;
    if (state.drawers.options) setDrawer("options", false);
    else if (state.drawers.feed) setDrawer("feed", false);
  });

  narrow.addEventListener("change", updateScrim);
}

// --- switchers and socket ------------------------------------------------------------

function currentView() {
  const symbols = document.getElementById("symbol-tabs");
  const timeframes = document.getElementById("timeframe-tabs");
  return {
    symbol: symbols.querySelector('.tab[aria-selected="true"]')?.dataset.key,
    timeframe: timeframes.querySelector('.tab[aria-selected="true"]')?.dataset.key,
  };
}

function setStatus(text, kind) {
  const node = document.getElementById("status");
  node.className = `status status--${kind}`;
  node.querySelector(".status__text").textContent = text;
}

function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  document.getElementById("source").textContent = snapshot.chart.source || "";
  document.getElementById("generated").textContent = stamp(snapshot.generated_at);
  applyChart(snapshot.chart);
  applyFeed(snapshot.feed);
}

/**
 * Stay current, by whichever means this deployment supports.
 *
 * The server answers that at `/api/config` rather than the client guessing, because a client
 * cannot tell a host that refuses WebSocket upgrades from one that is briefly down, and those
 * two want opposite responses. See `dashboard/transport.py`.
 */
function connect() {
  stopPolling();
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close();
    state.socket = null;
  }

  const view = currentView();
  if (!view.symbol) return;

  state.seenEntries.clear();
  state.revision = null;

  if (state.transport === "poll") startPolling(view);
  else connectSocket(view);
}

function viewQuery(view) {
  return (
    `symbol=${encodeURIComponent(view.symbol)}` +
    `&timeframe=${encodeURIComponent(view.timeframe)}`
  );
}

// --- socket ---------------------------------------------------------------------

function connectSocket(view) {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}/ws?${viewQuery(view)}`;

  setStatus("connecting", "connecting");
  const socket = new WebSocket(url);
  state.socket = socket;
  let delivered = false;

  socket.onopen = () => {
    state.reconnectDelay = 1000;
    setStatus("live", "live");
  };

  socket.onmessage = (event) => {
    const envelope = JSON.parse(event.data);
    if (envelope.type === "snapshot" && envelope.snapshot) {
      delivered = true;
      state.socketFailures = 0;
      applySnapshot(envelope.snapshot);
      setStatus("live", "live");
    } else if (envelope.type === "error") {
      setStatus("store error", "down");
      renderNotes(document.getElementById("chart-notes"), [envelope.message]);
    }
  };

  socket.onclose = () => {
    // A socket that closes without ever delivering anything is not a flaky connection: it is a
    // host that will not carry one — a proxy stripping the upgrade, or a serverless runtime
    // that never could. Twice is enough to stop asking and fall back to something that works.
    if (!delivered && ++state.socketFailures >= 2) {
      renderNotes(document.getElementById("chart-notes"), [
        "This host is not carrying the WebSocket, so the panel has fallen back to polling " +
          `every ${state.pollSeconds}s. Updates will lag by up to that long.`,
      ]);
      state.transport = "poll";
      connect();
      return;
    }

    // The server pushes only on change, so silence is normal and a closed socket is not.
    // Back off to 15s so a restarted container is picked up without hammering it.
    setStatus("reconnecting", "down");
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 15000);
  };

  socket.onerror = () => setStatus("socket error", "down");
}

// --- polling --------------------------------------------------------------------

/* Bumped on every stop, so a request already in flight for the previous view lands, finds
   itself stale, and neither renders nor arms another timer. */
let pollGeneration = 0;

function stopPolling() {
  pollGeneration += 1;
  if (state.pollTimer !== null) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

/**
 * Conditional polling: ask only for what changed.
 *
 * The revision the client is holding goes up with the request, and an unchanged view comes
 * back as `304` with no body. A quiet market therefore costs an empty round trip per interval
 * rather than 150KB of identical JSON.
 */
function startPolling(view) {
  const query = viewQuery(view);
  const generation = pollGeneration;

  const tick = async () => {
    if (generation !== pollGeneration) return;
    try {
      const since = state.revision ? `&since=${encodeURIComponent(state.revision)}` : "";
      const response = await fetch(`/api/snapshot?${query}${since}`, { cache: "no-store" });

      if (response.status === 304) {
        setStatus("polling", "live");
      } else if (response.ok) {
        const snapshot = await response.json();
        state.revision = snapshot.revision;
        applySnapshot(snapshot);
        setStatus("polling", "live");
      } else {
        const detail = await response.json().catch(() => ({}));
        setStatus("store error", "down");
        renderNotes(document.getElementById("chart-notes"), [
          detail.error || `The server answered ${response.status}.`,
        ]);
      }
    } catch (error) {
      setStatus("cannot reach the server", "down");
      renderNotes(document.getElementById("chart-notes"), [String(error)]);
    }
    if (generation === pollGeneration) {
      state.pollTimer = setTimeout(tick, state.pollSeconds * 1000);
    }
  };

  setStatus("connecting", "connecting");
  tick();
}

function populateSwitchers(options) {
  const symbolTabs = document.getElementById("symbol-tabs");
  const timeframeTabs = document.getElementById("timeframe-tabs");
  const wanted = new URLSearchParams(location.search);

  const symbols = [...new Set(options.map((option) => option.symbol))].sort();
  let symbol = wanted.get("symbol");
  if (!symbols.includes(symbol)) symbol = symbols[0];

  function timeframesFor(forSymbol) {
    return [
      ...new Set(
        options.filter((o) => o.symbol === forSymbol).map((option) => option.timeframe),
      ),
    ].sort();
  }

  let timeframe = wanted.get("timeframe");
  const available = timeframesFor(symbol);
  if (!available.includes(timeframe)) timeframe = available.includes("H1") ? "H1" : available[0];

  function drawTimeframes(forSymbol, selected) {
    renderTabs(
      timeframeTabs,
      timeframesFor(forSymbol).map((key) => ({ key, label: key })),
      selected,
      (key) => {
        selectTab(timeframeTabs, key);
        // Switching view closes the socket and opens another, so one socket is always exactly
        // one view — see the note on `/ws` in app.py.
        connect();
      },
    );
  }

  renderTabs(
    symbolTabs,
    symbols.map((key) => ({ key, label: key })),
    symbol,
    (key) => {
      selectTab(symbolTabs, key);
      const list = timeframesFor(key);
      const keep = currentView().timeframe;
      drawTimeframes(key, list.includes(keep) ? keep : list[0]);
      connect();
    },
  );

  drawTimeframes(symbol, timeframe);
}

/**
 * Put a script error on the panel, not only in a console nobody has open.
 *
 * This screen is meant to be watched from across a room. A silent JavaScript failure would
 * leave the last snapshot frozen on it looking perfectly current, which is the one failure
 * mode an instrument panel must not have — a stale chart that still says "live" is worse than
 * a blank one.
 */
function surfaceErrors() {
  const report = (detail) => {
    setStatus("front-end error", "down");
    renderNotes(document.getElementById("chart-notes"), [
      `The panel hit a script error and may now be showing stale data: ${detail}`,
    ]);
  };

  window.addEventListener("error", (event) => report(event.message || String(event.error)));
  window.addEventListener("unhandledrejection", (event) => report(String(event.reason)));
}

async function start() {
  load();
  surfaceErrors();

  renderTabs(document.getElementById("type-tabs"), CHART_TYPES, state.chartType, setChartType);
  wireDrawers();
  setInterval(tickCountdown, 1000);
  window.addEventListener("resize", () => {
    document.querySelectorAll(".tabs").forEach(movePill);
  });

  if (!LWC) {
    renderNotes(document.getElementById("chart-notes"), [
      "The charting library is not vendored. Run `uv run python scripts/vendor_lightweight_charts.py` " +
        "and rebuild the container. The feed below still works.",
    ]);
  } else {
    buildChart();
  }

  try {
    const config = await fetch("/api/config").then((r) => r.json());
    state.transport = config.transport === "poll" ? "poll" : "ws";
    state.pollSeconds = Number(config.poll_seconds) || 15;

    const response = await fetch("/api/options");
    const options = await response.json();
    if (!Array.isArray(options) || options.length === 0) {
      renderNotes(document.getElementById("chart-notes"), [
        "The store holds no bars yet, so there is nothing to switch between.",
      ]);
      setStatus("no data", "down");
      return;
    }
    populateSwitchers(options);
    connect();
  } catch (error) {
    setStatus("cannot reach the server", "down");
    renderNotes(document.getElementById("chart-notes"), [String(error)]);
  }
}

start();
