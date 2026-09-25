import { connectLive, getJSON, hubLink, css } from "/common/client.js";

hubLink();
const $ = (id) => document.getElementById(id);
const meta = await getJSON("/api/meta");
const live = connectLive();
const state = { target: "married", hidden: new Set(meta.released.married), curve: new Map(), last: null,
  epsIndex: Math.round(meta.eps_grid.length * 0.72), hovering: false };
const key = () => `${state.target}|${[...state.hidden].sort().join(",")}`;

// ------------------------------------------------------------ controls
function renderControls() {
  $("targets").innerHTML = Object.entries(meta.targets).map(([k, v]) =>
    `<button class="ghost ${k === state.target ? "active" : ""}" data-t="${k}">${v}</button>`).join("");
  $("fields").innerHTML = meta.released[state.target].map((f) =>
    `<span class="field ${state.hidden.has(f) ? "hidden" : ""}" data-f="${f}">${f}</span>`).join("");
}
function changed() { state.curve = new Map(); state.last = null; renderControls(); draw(); ask(state.epsIndex); }

$("targets").addEventListener("click", (e) => {
  const t = e.target.dataset.t;
  if (!t || t === state.target) return;
  state.target = t;
  state.hidden = new Set([...state.hidden].filter((f) => meta.released[t].includes(f)));
  changed();
});
$("fields").addEventListener("click", (e) => {
  const f = e.target.closest(".field")?.dataset.f;
  if (!f) return;
  state.hidden.has(f) ? state.hidden.delete(f) : state.hidden.add(f);
  changed();
});
$("all").addEventListener("click", () => { state.hidden = new Set(meta.released[state.target]); changed(); });
$("none").addEventListener("click", () => { state.hidden = new Set(); changed(); });

// --------------------------------------------------------------- chart
const cv = $("chart");
const L = Math.log10(meta.eps_grid[0]), R = Math.log10(meta.eps_grid.at(-1));
const PAD = 34;
function draw() {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = r.height * dpr;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, p = PAD * dpr, pr = 110 * dpr;
  const X = (eps) => p + ((Math.log10(eps) - L) / (R - L)) * (W - p - pr);
  const Y = (v) => H - p - v * (H - 1.6 * p);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted"); ctx.font = `${11 * dpr}px monospace`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath(); ctx.moveTo(p, Y(v)); ctx.lineTo(W - pr, Y(v)); ctx.stroke(); ctx.fillText(`${v * 100}%`, 2, Y(v) + 4);
  }
  for (const e of [0.01, 0.1, 1, 10]) ctx.fillText(`ε=${e}`, X(e) - 14 * dpr, H - 8 * dpr);
  const d = state.last;
  if (!d) return;
  const hline = (v, color, dash) => {
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.moveTo(p, Y(v)); ctx.lineTo(W - pr, Y(v)); ctx.stroke(); ctx.setLineDash([]);
  };
  hline(d.baseline, css("--orange"), [6 * dpr, 4 * dpr]);
  hline(d.no_dp.profiling, css("--orange"), [2 * dpr, 3 * dpr]);
  hline(d.no_dp.reid, css("--frost-2"), [2 * dpr, 3 * dpr]);
  const pts = [...state.curve.values()].sort((a, b) => a.eps - b.eps);
  ctx.font = `600 ${12 * dpr}px sans-serif`;
  for (const [k, c, label] of [["profiling", css("--orange"), "profiling"], ["reid", css("--frost-2"), "re-identification"]]) {
    ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 3 * dpr;
    ctx.beginPath(); pts.forEach((q, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(q.eps), Y(q[k]))); ctx.stroke();
    for (const q of pts) { ctx.beginPath(); ctx.arc(X(q.eps), Y(q[k]), 2.5 * dpr, 0, 7); ctx.fill(); }
    if (pts.length) { const q = pts.at(-1); ctx.fillText(label, X(q.eps) + 8 * dpr, Y(q[k]) + 4 * dpr); }
  }
  const e = meta.eps_grid[state.epsIndex];
  ctx.strokeStyle = css("--navy"); ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.moveTo(X(e), p / 2); ctx.lineTo(X(e), H - p); ctx.stroke();
}

// ---------------------------------------------------------------- live
const pct = (x) => `${Math.round(x * 100)} %`;
function onPoint(d, arg) {
  if (arg.k !== key()) return; // reply for a previous choice of fields
  state.last = d;
  for (const q of d.curve) state.curve.set(q.eps, q);
  if (arg.i === state.epsIndex) {
    $("eps").textContent = d.eps < 1 ? d.eps.toFixed(2) : d.eps.toFixed(1);
    $("keepP").textContent = pct(d.kept.profiling);
    $("keepR").textContent = pct(d.kept.reid);
    $("prof").textContent = pct(d.profiling);
    $("reid").textContent = pct(d.reid);
    $("verdict").textContent = !d.hidden.length ? "Nothing is hidden: DP has no effect."
      : d.kept.reid < 0.1 && d.kept.profiling >= 0.5 ? "Linkage is defeated, profiling is not: the paper's finding."
      : d.kept.profiling < 0.1 ? "So much noise that the data is useless, even for honest analysis."
      : "";
  }
  draw();
  // While the pointer is elsewhere, fill in the rest of the curve one epsilon at a time.
  if (!state.hovering) {
    const miss = meta.eps_grid.findIndex((e) => !state.curve.has(e));
    if (miss >= 0) setTimeout(() => !state.hovering && ask(miss), 20);
  }
}
const pointQ = live("point", onPoint);
const ask = (i) => pointQ({ target: state.target, hidden: [...state.hidden], i, k: key() });

cv.addEventListener("pointerenter", () => { state.hovering = true; });
cv.addEventListener("pointerleave", () => { state.hovering = false; ask(state.epsIndex); });
cv.addEventListener("pointermove", (e) => {
  const r = cv.getBoundingClientRect();
  const t = Math.min(1, Math.max(0, (e.clientX - r.left - PAD) / (r.width - PAD - 110)));
  const i = Math.round(t * (meta.eps_grid.length - 1));
  if (i !== state.epsIndex) { state.epsIndex = i; draw(); ask(i); }
});
addEventListener("resize", draw);

renderControls(); draw(); ask(state.epsIndex);
