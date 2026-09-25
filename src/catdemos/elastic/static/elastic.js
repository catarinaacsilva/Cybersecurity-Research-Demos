import { connectLive, getJSON, hubLink, css } from "/common/client.js";

hubLink();
const $ = (id) => document.getElementById(id);
const meta = await getJSON("/api/meta");
const live = connectLive();
const L = window.L;

// ---------------------------------------------------------------- map
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.5 });
map.fitBounds([meta.fabrica, meta.campus], { padding: [120, 120] }); // same framing as ↺ Reset
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const pin = (ll, label) => L.circleMarker(ll, { radius: 6, color: css("--navy"), fillColor: css("--orange"), fillOpacity: 1, weight: 2 })
  .bindTooltip(label, { permanent: true, direction: "top", offset: [0, -6] }).addTo(map);
pin(meta.fabrica, "Fábrica Centro Ciência Viva");
pin(meta.campus, "UA Campus de Santiago");

const renderer = L.canvas({ padding: 0.3 });
const rawLine = L.polyline([], { color: css("--navy"), weight: 3, renderer }).addTo(map);
const released = L.layerGroup().addTo(map);
const membrane = L.circle(meta.fabrica, { radius: 0, weight: 2, fillOpacity: 0.15 }).addTo(map);
const cursorDot = L.circleMarker(meta.fabrica, { radius: 5, color: css("--navy"), fillColor: "#fff", fillOpacity: 1, renderer }).addTo(map);

// ------------------------------------------------------------- state
let drawMode = true, drawing = false, points = [], last = null, timer = null, res = null, t = 0, follow = true;
let tOffset = 0; // shifts a new stroke so it continues 1 s after the previous one (pauses between strokes are not dwell)

let replayNext = false; // a loaded sample replays itself once its result arrives
const walkQ = live("walk", (d) => {
  res = d;
  if (follow) t = d.seconds - 1;
  render();
  if (replayNext) { replayNext = false; replay(); }
});
const send = () => { if (points.length >= 2) walkQ({ points: points.slice() }); };

const MAX_POINTS = 20000; // server limit; beyond it only the most recent part is sent
function addPoint(ll) {
  points.push([ll.lat, ll.lng, performance.now() - tOffset]);
  if (points.length > MAX_POINTS) points.splice(0, points.length - MAX_POINTS);
  rawLine.addLatLng(ll);
}

map.on("mousedown", (e) => {
  if (!drawMode) return;
  routineOnly = false;
  map.dragging.disable();
  stopReplay();
  drawing = true; follow = true;
  tOffset = points.length ? performance.now() - (points.at(-1)[2] + 1000) : 0;
  last = e.latlng; addPoint(e.latlng);
  // While the button is held, keep sampling: standing still becomes dwelling.
  timer = setInterval(() => { addPoint(last); send(); }, 120);
});
map.on("mousemove", (e) => {
  if (!drawing) return;
  if (e.originalEvent.buttons === 0) { stopDrawing(); return; } // the release happened elsewhere
  last = e.latlng; addPoint(e.latlng); send();
});
// Stop on any way the press can end, so a lost mouseup never leaves the dwell timer running.
function stopDrawing(silent = false) {
  if (!drawing) return;
  drawing = false; clearInterval(timer); map.dragging.enable();
  if (!silent) send();
}
map.on("mouseup", () => stopDrawing());
for (const ev of ["mouseup", "pointerup", "pointercancel", "blur"]) addEventListener(ev, () => stopDrawing());

$("draw").addEventListener("click", () => {
  drawMode = !drawMode;
  $("draw").classList.toggle("active", drawMode);
  $("map").classList.toggle("drawing", drawMode);
});
const home = () => map.fitBounds([meta.fabrica, meta.campus], { padding: [120, 120] }); // same framing as ↺ Reset
$("clear").addEventListener("click", () => { clearAll(); home(); });
function clearAll() {
  stopDrawing(true);
  stopReplay();
  points = []; res = null; rawLine.setLatLngs([]); released.clearLayers(); membrane.setRadius(0);
  for (const id of ["t", "radius", "score", "shield", "aerr"]) $(id).textContent = "–";
  $("alertsec").textContent = "";
  drawSpark(); $("state").textContent = "Draw a walk"; $("state").className = "state calm";
}
// Simulated walks on real Aveiro streets; each click draws a different route.
let sampleSeed = 1, routineOnly = false;
async function loadSample(crisis) {
  clearAll();
  routineOnly = !crisis;
  const d = await getJSON("/api/sample", { seed: sampleSeed++, crisis });
  points = d.points;
  rawLine.setLatLngs(points.map((p) => [p[0], p[1]]));
  map.fitBounds(rawLine.getBounds(), { padding: [60, 60] });
  follow = true; replayNext = true; send();
}
$("routine").addEventListener("click", () => loadSample(false));
$("sample").addEventListener("click", () => loadSample(true));

// ------------------------------------------------------------ render
function render() {
  if (!res) return;
  rawLine.setLatLngs(res.raw);
  released.clearLayers();
  const step = Math.max(1, Math.floor(res.seconds / 600)); // cap markers for smoothness
  for (let i = 0; i < res.seconds; i += step) {
    L.circleMarker(res.obf[i], { radius: 2.5, weight: 0, fillOpacity: 0.75, renderer,
      fillColor: res.alert[i] ? css("--red") : css("--orange") }).addTo(released);
  }
  const s = res.summary;
  $("shield").textContent = s.shield_m == null ? "–" : `${s.shield_m} m`;
  $("aerr").textContent = s.alert_error_m == null ? "–" : `${s.alert_error_m} m`;
  $("alertsec").textContent = `${res.seconds} s walked, ${s.alert_seconds} s under crisis alert.`
    + (routineOnly ? " This walk has no crisis: every alert here is a false alarm." : "");
  scrub(t);
}

function scrub(i) {
  if (!res) return;
  t = Math.max(0, Math.min(res.seconds - 1, Math.round(i)));
  const alert = res.alert[t], r = res.radius[t];
  const color = alert ? css("--red") : css("--frost-2");
  membrane.setLatLng(res.raw[t]).setRadius(r).setStyle({ color, fillColor: color, weight: 3, fillOpacity: 0.12 });
  cursorDot.setLatLng(res.raw[t]);
  $("t").textContent = t;
  $("radius").textContent = `${r.toFixed(0)} m`;
  $("score").textContent = res.score[t].toFixed(2);
  $("state").textContent = !alert ? "Routine: location blurred"
    : routineOnly ? "False alarm: location kept precise" : "Crisis: location kept precise";
  $("state").className = `state ${alert ? "alert" : "calm"}`;
  drawSpark();
}

// -------------------------------------------------------------- replay
// Animates the membrane along the walk in ~8 s; any scrub or new drawing stops it.
let playing = 0;
function replay() {
  if (!res) return;
  cancelAnimationFrame(playing);
  follow = false;
  const t0 = performance.now(), n = res.seconds;
  const step = (now) => {
    const i = Math.min(n - 1, ((now - t0) / 8000) * n);
    scrub(i);
    if (i < n - 1) playing = requestAnimationFrame(step);
  };
  playing = requestAnimationFrame(step);
}
const stopReplay = () => cancelAnimationFrame(playing);
$("replay").addEventListener("click", replay);

// ---------------------------------------------------------- sparkline
const spark = $("spark");
function drawSpark() {
  const rect = spark.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  spark.width = rect.width * dpr; spark.height = rect.height * dpr;
  const ctx = spark.getContext("2d"), W = spark.width, H = spark.height;
  ctx.clearRect(0, 0, W, H);
  if (!res) return;
  const n = res.seconds, X = (i) => (i / Math.max(1, n - 1)) * W;
  for (let i = 0; i < n; i++) if (res.alert[i]) {
    ctx.fillStyle = "rgba(191, 97, 106, 0.18)"; ctx.fillRect(X(i), 0, Math.max(1, W / n), H);
  }
  const line = (arr, max, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr; ctx.beginPath();
    arr.forEach((v, i) => { const y = H - (Math.min(v, max) / max) * (H - 4 * dpr) - 2 * dpr; i ? ctx.lineTo(X(i), y) : ctx.moveTo(X(i), y); });
    ctx.stroke();
  };
  line(res.radius, meta.r_max, css("--frost-2"));
  line(res.score, 4, css("--orange"));
  ctx.strokeStyle = css("--navy"); ctx.lineWidth = 2 * dpr;
  ctx.beginPath(); ctx.moveTo(X(t), 0); ctx.lineTo(X(t), H); ctx.stroke();
}
let scrubbing = false;
const scrubAt = (e) => {
  stopReplay();
  const r = spark.getBoundingClientRect();
  follow = false;
  scrub(((e.clientX - r.left) / r.width) * ((res?.seconds ?? 1) - 1));
};
spark.addEventListener("pointerdown", (e) => { scrubbing = true; spark.setPointerCapture(e.pointerId); scrubAt(e); });
spark.addEventListener("pointermove", (e) => scrubbing && scrubAt(e));
spark.addEventListener("pointerup", () => { scrubbing = false; });
addEventListener("resize", drawSpark);

// ------------------------------------------------------------ bench
const b = meta.bench;
$("b-shield").textContent = `${b.shield_m} m`; $("p-shield").textContent = `${b.paper.shield_m} m`;
$("b-crisis").textContent = `${b.crisis_error_m} m`; $("p-crisis").textContent = `${b.paper.crisis_error_m} m`;
$("b-det").textContent = `${Math.round(b.crisis_detected * 100)} %`;
$("b-fa").textContent = `${Math.round(b.false_alert * 100)} %`;
$("status").textContent = `street graph: ${meta.graph} · ready`;
