/*
 * The panel's front end. Vanilla, deliberately.
 *
 * It does three things: keep one WebSocket open, redraw the chart when a snapshot arrives, and
 * render the feed. It computes nothing. Every price, every level, every session boundary and
 * every marker is calculated on the server from the same code the strategies read — this file
 * would be the easiest place in the system to introduce a second, subtly different definition
 * of "the London session" or "the 20-period EMA", so it is not allowed to have one.
 *
 * The single exception is the grant countdown, which ticks locally against `expires_at`. A
 * countdown pushed over a socket would measure network latency as well as time remaining.
 */

"use strict";

const LWC = window.LightweightCharts;

const state = {
  socket: null,
  snapshot: null,
  chart: null,
  candles: null,
  overlays: new Map(), // key -> ISeriesApi
  tradeLines: [], // { series, id }
  times: [], // candle times, ascending — the index the session bands are positioned from
  bands: [],
  grantExpiry: null,
  reconnectDelay: 1000,
};

const SESSION_FILL = {
  TOKYO: "rgba(120, 160, 220, 0.10)",
  LONDON: "rgba(230, 170, 80, 0.10)",
  NEW_YORK: "rgba(150, 120, 210, 0.10)",
  OVERLAP: "rgba(230, 120, 120, 0.12)",
};

// --- tiny DOM helpers --------------------------------------------------------
// Everything user-visible goes through textContent. Agent narration is prose from a language
// model and trade reasons are stored strings; neither is ever interpolated into markup.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmt(value, digits) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

/** ISO instant -> "2026-08-15 08:00Z". The panel is a UTC instrument; local time would make
 *  two people reading the same screen in two places disagree about when a bar was. */
function stamp(iso) {
  if (!iso) return "—";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  return at.toISOString().replace("T", " ").slice(0, 16) + "Z";
}

// --- chart -------------------------------------------------------------------

function buildChart() {
  const container = document.getElementById("chart");
  const chart = LWC.createChart(container, {
    layout: { background: { color: "#14161a" }, textColor: "#8a909b" },
    grid: {
      vertLines: { color: "rgba(43, 48, 56, 0.6)" },
      horzLines: { color: "rgba(43, 48, 56, 0.6)" },
    },
    rightPriceScale: { borderColor: "#2b3038" },
    timeScale: { borderColor: "#2b3038", timeVisible: true, secondsVisible: false },
    crosshair: { mode: LWC.CrosshairMode.Normal },
    localization: {
      // Axis labels in UTC, matching everything else on the page.
      timeFormatter: (t) => new Date(t * 1000).toISOString().slice(11, 16) + "Z",
    },
  });

  const candles = chart.addCandlestickSeries({
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderUpColor: "#26a69a",
    borderDownColor: "#ef5350",
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
  });

  state.chart = chart;
  state.candles = candles;

  chart.timeScale().subscribeVisibleLogicalRangeChange(drawSessions);
  window.addEventListener("resize", () => {
    resizeCanvas();
    drawSessions();
  });
}

function applyChart(payload) {
  const precision = payload.price_precision;
  state.candles.applyOptions({
    priceFormat: { type: "price", precision, minMove: Math.pow(10, -precision) },
  });
  state.candles.setData(payload.candles);
  state.times = payload.candles.map((candle) => candle.time);
  state.bands = payload.session_bands;

  applyOverlays(payload.overlays);
  applyTradeLines(payload.trades, precision);
  state.candles.setMarkers(
    payload.markers.map((marker) => ({
      time: marker.time,
      position: marker.position,
      shape: marker.shape,
      color: marker.colour,
      text: marker.text,
    })),
  );

  renderLegend(payload);
  renderNotes(document.getElementById("chart-notes"), payload.notes);
  resizeCanvas();
  drawSessions();
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

/** Entry, stop and target for each trade, as three two-point lines spanning the position.
 *  Recreated wholesale: trades are few, and a stop that moved would otherwise need diffing. */
function applyTradeLines(trades, precision) {
  for (const line of state.tradeLines) state.chart.removeSeries(line);
  state.tradeLines = [];

  for (const trade of trades) {
    const levels = [
      { price: trade.entry_price, colour: "#d8dce3", dash: LWC.LineStyle.Solid },
      { price: trade.stop_price, colour: "#ef5350", dash: LWC.LineStyle.Dashed },
      { price: trade.target_price, colour: "#26a69a", dash: LWC.LineStyle.Dashed },
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

function renderLegend(payload) {
  const legend = document.getElementById("legend");
  clear(legend);

  legend.appendChild(el("span", null, `${payload.symbol} ${payload.timeframe}`));

  for (const overlay of payload.overlays) {
    const item = el("span");
    const swatch = el("span", "swatch");
    swatch.style.background = overlay.colour;
    item.appendChild(swatch);
    item.appendChild(el("span", null, overlay.label));
    legend.appendChild(item);
  }

  const strategies = new Map();
  for (const marker of payload.markers) {
    if (marker.strategy && !strategies.has(marker.strategy)) {
      strategies.set(marker.strategy, marker.colour);
    }
  }
  for (const [strategy, colour] of strategies) {
    const item = el("span");
    const dot = el("span", "dot");
    dot.style.background = colour;
    item.appendChild(dot);
    item.appendChild(el("span", null, strategy));
    legend.appendChild(item);
  }

  for (const [session, fill] of Object.entries(SESSION_FILL)) {
    const item = el("span");
    const swatch = el("span", "swatch");
    swatch.style.background = fill.replace(/0\.1\d?\)/, "0.5)");
    item.appendChild(swatch);
    item.appendChild(el("span", null, session));
    legend.appendChild(item);
  }
}

// --- session shading ---------------------------------------------------------

function resizeCanvas() {
  const canvas = document.getElementById("sessions");
  const wrap = canvas.parentElement;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(wrap.clientWidth * ratio);
  canvas.height = Math.floor(wrap.clientHeight * ratio);
  canvas.style.width = wrap.clientWidth + "px";
  canvas.style.height = wrap.clientHeight + "px";
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}

/**
 * Position a UTC instant on the chart's logical (bar-index) axis.
 *
 * `timeToCoordinate` only answers for times that are in the series data, and a session
 * boundary at 17:00 on a Friday is not a bar on a market that shut at 21:00. So the band edges
 * are interpolated between the bars either side of them and converted through
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

  if (!state.chart || state.bands.length === 0) return;

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

// --- feed --------------------------------------------------------------------

function renderNotes(container, notes) {
  clear(container);
  for (const note of notes || []) container.appendChild(el("p", null, note));
}

function renderGrant(grant) {
  const card = document.getElementById("grant");
  clear(card);

  const head = el("div");
  head.appendChild(el("span", `state state--${grant.state}`, grant.state));
  if (grant.symbols && grant.symbols.length) {
    head.appendChild(el("span", "badge", grant.symbols.join(" ")));
  }
  const countdown = el("span", "countdown");
  countdown.id = "countdown";
  head.appendChild(countdown);
  card.appendChild(head);

  if (grant.granted_at) {
    card.appendChild(el("div", "reason", `granted ${stamp(grant.granted_at)}`));
  }
  card.appendChild(el("div", "reason", grant.reason));

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
  const node = el("article", "entry");

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
  regime.appendChild(el("span", null, `session ${sessions}`));
  regime.appendChild(el("span", null, `ADX ${fmt(entry.regime.trend_strength, 1)}`));
  regime.appendChild(el("span", null, `vol pct ${fmt(entry.regime.volatility_percentile, 0)}`));
  regime.appendChild(
    el(
      "span",
      null,
      entry.regime.is_trending ? "trending" : entry.regime.is_ranging ? "ranging" : "neither",
    ),
  );
  regime.appendChild(el("span", null, entry.regime.market_open ? "market open" : "market shut"));
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
    const section = el("div", "section");
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
  for (const entry of payload.entries) feed.appendChild(renderEntry(entry));
}

// --- switchers and socket ----------------------------------------------------

function currentView() {
  return {
    symbol: document.getElementById("symbol").value,
    timeframe: document.getElementById("timeframe").value,
  };
}

function setStatus(text, kind) {
  const node = document.getElementById("status");
  node.textContent = text;
  node.className = `status status--${kind}`;
}

function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  document.getElementById("source").textContent = `source ${snapshot.chart.source || "—"}`;
  document.getElementById("generated").textContent = `built ${stamp(snapshot.generated_at)}`;
  applyChart(snapshot.chart);
  applyFeed(snapshot.feed);
}

function connect() {
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close();
  }

  const view = currentView();
  if (!view.symbol) return;

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}/ws?symbol=${encodeURIComponent(view.symbol)}&timeframe=${encodeURIComponent(view.timeframe)}`;

  setStatus("connecting", "connecting");
  const socket = new WebSocket(url);
  state.socket = socket;

  socket.onopen = () => {
    state.reconnectDelay = 1000;
    setStatus("live", "live");
  };

  socket.onmessage = (event) => {
    const envelope = JSON.parse(event.data);
    if (envelope.type === "snapshot" && envelope.snapshot) {
      applySnapshot(envelope.snapshot);
      setStatus("live", "live");
    } else if (envelope.type === "error") {
      setStatus("store error", "down");
      renderNotes(document.getElementById("chart-notes"), [envelope.message]);
    }
  };

  socket.onclose = () => {
    // The server pushes only on change, so silence is normal and a closed socket is not.
    // Back off to 15s so a restarted container is picked up without hammering it.
    setStatus("reconnecting", "down");
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 15000);
  };

  socket.onerror = () => setStatus("socket error", "down");
}

function populateSwitchers(options) {
  const symbolSelect = document.getElementById("symbol");
  const timeframeSelect = document.getElementById("timeframe");

  const wanted = new URLSearchParams(location.search);
  const symbols = [...new Set(options.map((option) => option.symbol))].sort();

  clear(symbolSelect);
  for (const symbol of symbols) {
    symbolSelect.appendChild(new Option(symbol, symbol));
  }
  if (wanted.get("symbol")) symbolSelect.value = wanted.get("symbol");

  function refreshTimeframes() {
    const available = [
      ...new Set(
        options
          .filter((option) => option.symbol === symbolSelect.value)
          .map((option) => option.timeframe),
      ),
    ].sort();
    const previous = timeframeSelect.value;
    clear(timeframeSelect);
    for (const timeframe of available) {
      timeframeSelect.appendChild(new Option(timeframe, timeframe));
    }
    if (available.includes(previous)) timeframeSelect.value = previous;
    else if (available.includes("H1")) timeframeSelect.value = "H1";
  }

  refreshTimeframes();
  if (wanted.get("timeframe")) timeframeSelect.value = wanted.get("timeframe");

  symbolSelect.onchange = () => {
    refreshTimeframes();
    connect();
  };
  // Switching view closes the socket and opens another, so one socket is always exactly one
  // view — see the note on `/ws` in app.py.
  timeframeSelect.onchange = connect;
}

async function start() {
  if (!LWC) {
    renderNotes(document.getElementById("chart-notes"), [
      "The charting library is not vendored. Run `uv run python scripts/vendor_lightweight_charts.py` " +
        "and rebuild the container. The feed below still works.",
    ]);
  } else {
    buildChart();
  }

  setInterval(tickCountdown, 1000);

  try {
    const response = await fetch("/api/options");
    const options = await response.json();
    if (options.length === 0) {
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
