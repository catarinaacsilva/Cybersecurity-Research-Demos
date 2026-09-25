// Shared client helpers: REST for setup, a latest-wins WebSocket channel for live updates,
// the latency footer, the hub link and the colour ramp.

// Live channel: one WebSocket per page, one operation name per kind of pointer update.
// For each operation at most one request is in flight; while it runs, newer arguments
// replace older ones (latest wins), so pointer storms never queue up on the server.
export function connectLive() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const ops = new Map();
  let sock = null, seq = 0;

  function send(name) {
    const op = ops.get(name);
    if (!op.hasPending || op.busy || !sock || sock.readyState !== WebSocket.OPEN) return;
    op.busy = true; op.hasPending = false; op.args = op.pending; op.t0 = performance.now(); op.seq = ++seq;
    sock.send(JSON.stringify({ op: name, seq: op.seq, args: op.args }));
  }
  function open() {
    sock = new WebSocket(url);
    sock.onopen = () => { setStatus("live channel open"); for (const name of ops.keys()) send(name); };
    sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data), op = ops.get(msg.op);
      if (!op || msg.seq !== op.seq) return;
      op.busy = false;
      if (msg.error) console.warn(`${msg.op}: ${msg.error}`);
      else op.onResult(msg.data, op.args);
      status(msg.ms, performance.now() - op.t0);
      send(msg.op);
    };
    sock.onclose = () => {
      setStatus("live channel closed, reconnecting…");
      for (const op of ops.values()) if (op.busy) { op.busy = false; op.pending = op.args; op.hasPending = true; }
      setTimeout(open, 1000);
    };
  }
  open();
  return (name, onResult) => {
    ops.set(name, { onResult, busy: false, pending: null, hasPending: false, args: null, t0: 0, seq: 0 });
    return (args) => { const op = ops.get(name); op.pending = args; op.hasPending = true; send(name); };
  };
}

export async function getJSON(url, params = {}) {
  const q = new URLSearchParams(params).toString();
  const r = await fetch(q ? `${url}?${q}` : url);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(`${url}: ${detail}`);
  }
  return r.json();
}

const samples = [];
function setStatus(text) {
  const el = document.getElementById("status");
  if (el) el.textContent = text;
}
function status(serverMs, rttMs) {
  samples.push(rttMs);
  if (samples.length > 30) samples.shift();
  const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
  setStatus(`websocket · server ${serverMs} ms · round-trip ${rttMs.toFixed(1)} ms (avg ${avg.toFixed(1)})`);
}

// Link back to the hub: ?hub=URL overrides, default is the same host on port 8000.
export function hubLink() {
  const q = new URLSearchParams(location.search).get("hub");
  const href = q && /^https?:\/\//i.test(q) ? q : `${location.protocol}//${location.hostname}:8000/`; // no javascript: URLs
  for (const a of document.querySelectorAll("#hub, [data-hub]")) a.href = href;
}

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Colour ramp green -> yellow -> orange -> red for t in [0, 1].
export function ramp(t) {
  const stops = [[163, 190, 140], [235, 203, 139], [255, 129, 69], [191, 97, 106]];
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
}
