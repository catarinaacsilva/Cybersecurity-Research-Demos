import { connectLive, getJSON, hubLink } from "/common/client.js";

hubLink();
const $ = (id) => document.getElementById(id);
const meta = await getJSON("/api/meta");
const live = connectLive();
const state = { fields: new Set(), frequency: "daily", retention: "1 year", granularity: "exact", result: null };
const LABEL = { identity: "Identity", "socio-economic": "Socio-economic", sensitive: "Sensitive" };

// ------------------------------------------------------------- palette
function renderPalette() {
  $("palette").innerHTML = meta.fields.map((f) =>
    `<div class="src ${state.fields.has(f.name) ? "used" : ""}" data-f="${f.name}"
      title="${f.desc}">${f.name}<small>${f.kind === "num" ? "number, e.g. " : ""}${f.values.join(", ")}</small></div>`).join("");
}
// Pointer-based drag (mouse, touch and pen alike): a ghost chip follows the pointer and is
// dropped into the tree; a press without movement is a click and adds the field directly.
const tree = $("tree");
let drag = null;
$("palette").addEventListener("pointerdown", (e) => {
  const src = e.target.closest(".src");
  if (!src || src.classList.contains("used")) return;
  e.preventDefault();
  drag = { f: src.dataset.f, x: e.clientX, y: e.clientY, ghost: null };
});
addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (!drag.ghost && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 6) {
    drag.ghost = Object.assign(document.createElement("div"), { className: "src ghost", textContent: drag.f });
    document.body.append(drag.ghost);
  }
  if (drag.ghost) {
    drag.ghost.style.transform = `translate(${e.clientX + 8}px, ${e.clientY + 8}px)`;
    tree.classList.toggle("over", overTree(e));
  }
});
addEventListener("pointerup", (e) => {
  if (!drag) return;
  if (!drag.ghost || overTree(e)) add(drag.f);
  drag.ghost?.remove(); drag = null; tree.classList.remove("over");
});
function overTree(e) {
  const r = tree.getBoundingClientRect();
  return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
}
tree.addEventListener("click", (e) => { const f = e.target.dataset.remove; if (f) { state.fields.delete(f); update(); } });

function add(f) { if (f && !state.fields.has(f)) { state.fields.add(f); update(); } }

// ---------------------------------------------------------- structural
function renderStructural() {
  const seg = (key, opts) => `<span class="seg" data-k="${key}">${opts.map((o) =>
    `<button class="ghost ${state[key] === o ? "active" : ""}" data-v="${o}">${o}</button>`).join("")}</span>`;
  $("structural").innerHTML = `<span>collected</span>${seg("frequency", meta.frequency)}<span></span>
    <span>kept for</span>${seg("retention", meta.retention)}<span></span>
    <span>precision</span>${seg("granularity", meta.granularity)}<span class="hint" style="margin:0">coarse = binned values</span>`;
}
$("structural").addEventListener("click", (e) => {
  const v = e.target.dataset.v, k = e.target.closest(".seg")?.dataset.k;
  if (v && k) { state[k] = v; update(); }
});

// ---------------------------------------------------------------- live
const modelQ = live("model", (d) => { state.result = d; renderResult(); });
function update() {
  renderPalette(); renderStructural();
  modelQ({ fields: [...state.fields], frequency: state.frequency, retention: state.retention, granularity: state.granularity });
}

const pct = (x) => `${Math.round(x * 100)}%`;
function renderResult() {
  const d = state.result;
  $("lanes").innerHTML = meta.categories.map((c) => `<div class="lane ${c}"><h4>${LABEL[c]}</h4>${
    d.fields.filter((f) => f.category === c).map((f) => `<div class="item" title="expert: ${f.expert}">
      <span>${f.name}</span><span class="c">${Math.round(f.confidence * 100)}%${f.expert === f.category ? "" : " ≠ expert"}
      <button data-remove="${f.name}" title="remove">×</button></span></div>`).join("")}</div>`).join("");
  const tags = d.tags.map((t) => t.collected
    ? `<div class="tag"><span>${t.label}</span><span class="hint" style="margin:0">collected directly</span><span></span></div>`
    : `<div class="tag"><span>infers <b>${t.label}</b></span><span class="meter"><i style="width:${pct(t.accuracy)}"></i>
        <b style="left:${pct(t.baseline)}"></b></span><span class="c mono">${pct(t.accuracy)} (guess ${pct(t.baseline)})</span></div>`);
  tags.push(`<div class="tag"><span>makes people <b>unique</b></span><span class="meter"><i style="width:${pct(d.unique)}"></i></span>
    <span class="mono">${pct(d.unique)} of 9,000</span></div>`);
  $("tags").innerHTML = d.fields.length ? tags.join("") : `<p class="hint">Add fields to see what they reveal.</p>`;
  $("score").textContent = d.score.toFixed(2);
  const part = (label, v) => `<div class="part"><span>${label}</span><span class="meter"><i style="width:${pct(v)}"></i></span>
    <span class="mono">${v.toFixed(2)}</span></div>`;
  $("parts").innerHTML = part("direct data", d.parts.direct) + part("dynamic tags", d.parts.dynamic)
    + part("structural", d.parts.structural);
}

$("agree").textContent = pct(meta.agreement);
for (const f of ["age", "sex", "occupation"]) state.fields.add(f); // a small starting model
update();
