import { connectLive, getJSON, hubLink, css, ramp } from "/common/client.js";

hubLink();
const $ = (id) => document.getElementById(id);
const mapCv = $("map"), histCv = $("hist");
let ov = null, bg = null, cursor = null, thr = 0;

function fit(cv) {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  return dpr;
}

// Background heat map: one pixel per grid cell, scaled up with smoothing.
function buildBackground() {
  const n = ov.grid_n, img = new ImageData(n, n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const [R, G, B] = ramp(ov.grid[r][c]), k = 4 * (r * n + c);
    img.data.set([R, G, B, 110], k);
  }
  bg = document.createElement("canvas"); bg.width = bg.height = n;
  bg.getContext("2d").putImageData(img, 0, 0);
}

const toPx = (v, size) => ((v + ov.lim) / (2 * ov.lim)) * size;

function drawMap() {
  const dpr = fit(mapCv), ctx = mapCv.getContext("2d"), W = mapCv.width, H = mapCv.height;
  ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(bg, 0, 0, W, H);
  const cb = css("--frost-2"), cm = css("--red"), r = 1.6 * dpr;
  for (const [x, y, lab] of ov.dots) {
    ctx.fillStyle = lab ? cm : cb;
    ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.arc(toPx(x, W), H - toPx(y, H), r, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (cursor) {
    const px = toPx(cursor.x, W), py = H - toPx(cursor.y, H);
    ctx.strokeStyle = css("--navy"); ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.arc(px, py, 9 * dpr, 0, 7); ctx.stroke();
    ctx.strokeStyle = css("--orange"); ctx.beginPath(); ctx.arc(px, py, 6 * dpr, 0, 7); ctx.stroke();
  }
}

function drawHist() {
  const dpr = fit(histCv), ctx = histCv.getContext("2d"), W = histCv.width, H = histCv.height;
  const { edges, benign, malicious } = ov.hist, n = benign.length;
  const max = Math.max(...benign, ...malicious), pad = 18 * dpr, bw = W / n;
  ctx.clearRect(0, 0, W, H);
  const bar = (arr, color) => {
    ctx.fillStyle = color; ctx.globalAlpha = 0.65;
    arr.forEach((v, i) => { const h = (v / max) * (H - pad); ctx.fillRect(i * bw, H - pad - h, bw - 1, h); });
  };
  bar(benign, css("--frost-2")); bar(malicious, css("--red")); ctx.globalAlpha = 1;
  const x = ((thr - edges[0]) / (edges[n] - edges[0])) * W;
  ctx.strokeStyle = css("--orange"); ctx.lineWidth = 3 * dpr;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - pad); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = `${11 * dpr}px monospace`;
  ctx.fillText("← looks benign            anomaly score            looks malicious →", 4 * dpr, H - 4 * dpr);
}

const GROUPS = [[1, "Direct personal data"], [2, "Behavioural"], [3, "Structural"]];
function showProbe(d) {
  $("label").textContent = d.label; $("label").className = `badge lv-${d.label}`;
  $("verdict").textContent = d.flagged ? "Flagged as privacy-harmful" : "Looks benign";
  $("verdict").className = `verdict ${d.flagged ? "bad" : "ok"}`;
  $("pm").textContent = d.p_malicious.toFixed(2); $("pm-bar").style.width = `${d.p_malicious * 100}%`;
  $("exp").textContent = d.exposure.toFixed(2); $("exp-bar").style.width = `${d.exposure * 100}%`;
  $("nn").textContent = `${d.neighbours.malicious} of ${d.neighbours.k}`;
  $("tiers").textContent = `(${d.n_perms})`;
  $("perms").innerHTML = GROUPS.map(([t, name]) => {
    const ps = d.perms.filter((p) => p.tier === t);
    return `<div class="group t${t}"><h4>${name} · ${ps.length}</h4>${ps.length
      ? ps.map((p) => `<span class="chip" title="${p.full}">${p.name}</span>`).join("")
      : `<span class="none">none</span>`}</div>`;
  }).join("");
}

function showThreshold(d) {
  for (const k of ["tp", "fp", "fn", "tn"]) $(k).textContent = d[k];
  $("mcc").textContent = d.mcc.toFixed(3);
  $("rates").textContent = `${Math.round(d.tpr * 100)} % / ${Math.round(d.fpr * 100)} %`;
  $("thrnote").textContent = Math.abs(thr - ov.threshold) < 1e-9
    ? "dynamic threshold: the best MCC on a validation split" : "your threshold";
}

const live = connectLive();
const probe = live("probe", showProbe);
const setThr = live("threshold", showThreshold);

mapCv.addEventListener("pointermove", (e) => {
  const r = mapCv.getBoundingClientRect();
  const x = ((e.clientX - r.left) / r.width) * 2 * ov.lim - ov.lim;
  const y = ov.lim - ((e.clientY - r.top) / r.height) * 2 * ov.lim;
  cursor = { x: Math.max(-ov.lim, Math.min(ov.lim, x)), y: Math.max(-ov.lim, Math.min(ov.lim, y)) };
  requestAnimationFrame(drawMap);
  probe({ x: cursor.x, y: cursor.y });
});

let dragging = false;
function histPointer(e) {
  const r = histCv.getBoundingClientRect(), e0 = ov.hist.edges[0], e1 = ov.hist.edges.at(-1);
  thr = e0 + ((e.clientX - r.left) / r.width) * (e1 - e0);
  requestAnimationFrame(drawHist);
  setThr({ t: thr });
}
histCv.addEventListener("pointerdown", (e) => { dragging = true; histCv.setPointerCapture(e.pointerId); histPointer(e); });
histCv.addEventListener("pointermove", (e) => dragging && histPointer(e));
histCv.addEventListener("pointerup", () => { dragging = false; });
$("reset").addEventListener("click", () => { thr = ov.threshold; drawHist(); setThr({ t: thr }); });
addEventListener("resize", () => { drawMap(); drawHist(); });

ov = await getJSON("/api/overview");
thr = ov.threshold;
buildBackground(); drawMap(); drawHist(); showThreshold(ov.at_threshold);
probe({ x: 0, y: 0 });
