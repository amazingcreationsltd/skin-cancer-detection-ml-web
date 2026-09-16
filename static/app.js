/* ============================================================================
   Skin Cancer Detection — frontend
   ----------------------------------------------------------------------------
   API_BASE: where the Python backend lives.
     • Same-origin deploy (Render / Railway / Docker / localhost): leave as "".
     • GitHub Pages (static frontend) + backend elsewhere: set to your API URL,
       e.g.  const API_BASE = "https://skin-cancer-api.onrender.com";
   ========================================================================== */
const API_BASE = "";

const $ = (id) => document.getElementById(id);
const api = (path) => `${API_BASE}${path}`;

let sessionId = null;
let hasImage = false;

/* ---------------- health / backend badge ---------------- */
fetch(api("/api/health"))
  .then((r) => r.json())
  .then((h) => {
    $("backendBadge").innerHTML =
      `engine: <b>${h.backend === "torch" ? "PyTorch" : "ONNX Runtime"}</b>` +
      (h.gradcam ? " · Grad-CAM ✓" : " · Grad-CAM n/a");
  })
  .catch(() => { $("backendBadge").textContent = "backend offline"; });

/* ---------------- upload ---------------- */
const dz = $("dropzone"), fi = $("fileInput");
dz.addEventListener("click", () => fi.click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fi.click(); } });
["dragover", "dragenter"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (ev) => { if (ev.dataTransfer.files[0]) uploadFile(ev.dataTransfer.files[0]); });
fi.addEventListener("change", () => { if (fi.files[0]) uploadFile(fi.files[0]); });

function uploadFile(file) {
  showError("");
  const fd = new FormData();
  fd.append("image", file);
  if (sessionId) fd.append("session_id", sessionId);
  $("progress").classList.remove("hidden");
  $("progressText").textContent = "Uploading to temp folder…";
  fetch(api("/api/upload"), { method: "POST", body: fd })
    .then(async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || "Upload failed"); return j; })
    .then((j) => {
      sessionId = j.session_id;
      hasImage = true;
      $("previewImg").src = j.preview;
      $("previewDims").textContent = `${j.width} × ${j.height}px`;
      $("sessionTag").textContent = `temp session ${sessionId.slice(0, 8)}…`;
      $("previewWrap").classList.remove("hidden");
      $("diagnoseBtn").disabled = false;
    })
    .catch((e) => showError(e.message))
    .finally(() => $("progress").classList.add("hidden"));
}

/* ---------------- diagnose ---------------- */
$("diagnoseBtn").addEventListener("click", () => {
  if (!hasImage || !sessionId) return;
  showError("");
  const mode = document.querySelector('input[name="mode"]:checked').value;
  $("progress").classList.remove("hidden");
  $("progressText").textContent = mode === "quick"
    ? "Running model (quick)…" : "Running model + patch analysis + heatmaps… (may take ~30–90s on CPU)";
  $("diagnoseBtn").disabled = true;
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("mode", mode);
  fetch(api("/api/diagnose"), { method: "POST", body: fd })
    .then(async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || "Diagnosis failed"); return j; })
    .then(renderResults)
    .catch((e) => showError(e.message))
    .finally(() => { $("progress").classList.add("hidden"); $("diagnoseBtn").disabled = false; });
});

function showError(msg) {
  const box = $("errorBox");
  if (!msg) { box.classList.add("hidden"); return; }
  box.textContent = msg;
  box.classList.remove("hidden");
}

/* ---------------- results ---------------- */
const GALLERY = [
  ["gradcam", "Grad-CAM (whole image)"],
  ["gradcam_overlay", "Grad-CAM overlay"],
  ["heatmap", "Cancer probability heatmap"],
  ["heatmap_overlay", "Heatmap overlay"],
  ["risk_map", "Patch-wise cancer risk map"],
  ["clinical", "AI clinical summary"],
];

function renderResults(r) {
  $("emptyState").classList.add("hidden");
  $("results").classList.remove("hidden");

  // verdict
  const risk = (r.risk || "Low").toLowerCase();
  $("verdictCard").className = "verdict " + risk;
  $("predName").textContent = (r.prediction || "—").replace(/_/g, " ");
  $("confVal").textContent = pct(r.confidence);
  requestAnimationFrame(() => { $("confFill").style.width = (r.confidence * 100).toFixed(1) + "%"; });

  // gauge
  $("scoreVal").textContent = (r.cancer_score * 100).toFixed(1) + "%";
  const arc = $("gaugeArc");
  arc.style.strokeDashoffset = 157 - 157 * Math.min(1, r.cancer_score);
  arc.style.stroke = risk === "high" ? "var(--danger)" : risk === "moderate" ? "var(--warn)" : "var(--accent)";
  const rb = $("riskBadge");
  rb.textContent = r.risk + " risk";
  rb.className = "risk " + risk;

  // meta
  $("modalityVal").textContent = r.modality_hint || "—";
  $("modalityAlpha").textContent = r.modality_alpha != null ? `(α=${r.modality_alpha.toFixed(2)})` : "";
  $("engineVal").textContent = r.backend === "torch" ? "PyTorch" : "ONNX Runtime";
  $("timeVal").textContent = r.timings && r.timings.total != null ? `${r.timings.total}s total` : "—";

  // probabilities
  const bars = $("probBars");
  bars.innerHTML = "";
  const topIdx = r.probabilities.indexOf(r.probabilities.reduce((a, b) => (a.prob > b.prob ? a : b)));
  r.probabilities.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "prow" + (i === topIdx ? " top" : "");
    row.innerHTML = `<span class="nm${p.malignant ? " mal" : ""}" title="${p.malignant ? "malignant class" : "benign class"}">${p.malignant ? "⚠ " : ""}${p.class.replace(/_/g, " ")}</span>
      <div class="tr"><div class="fl"></div></div><span class="pc">${pct(p.prob)}</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { row.querySelector(".fl").style.width = (p.prob * 100).toFixed(1) + "%"; }));
  });

  // patch stats
  const ps = $("patchStats");
  if (r.patch_stats) {
    const s = r.patch_stats;
    ps.classList.remove("hidden");
    ps.innerHTML = `
      <div class="stat"><b>${s.total}</b><span>patches (${s.patch_size}px / stride ${s.stride})</span></div>
      <div class="stat"><b style="color:var(--danger)">${s.high}</b><span>high-risk</span></div>
      <div class="stat"><b style="color:var(--warn)">${s.medium}</b><span>medium-risk</span></div>
      <div class="stat"><b style="color:var(--accent)">${s.low}</b><span>low-risk</span></div>
      <div class="stat"><b>${(s.avg_cancer_score * 100).toFixed(1)}%</b><span>avg cancer score</span></div>`;
  } else ps.classList.add("hidden");

  // gallery
  const gal = $("gallery");
  gal.innerHTML = "";
  let n = 0;
  GALLERY.forEach(([key, cap]) => {
    if (r.images && r.images[key]) {
      n++;
      const d = document.createElement("div");
      d.className = "gitem";
      d.innerHTML = `<img src="${r.images[key]}" alt="${cap}" loading="lazy" /><p>${cap}</p>`;
      d.querySelector("img").addEventListener("click", (e) => openLB(e.target.src, cap));
      gal.appendChild(d);
    }
  });
  $("galleryTitle").classList.toggle("hidden", n === 0);

  // top patches
  const tp = $("topPatches");
  tp.innerHTML = "";
  (r.top_patches || []).forEach((p) => {
    const cls = p.cancer_score >= 0.85 ? "sc-high" : p.cancer_score >= 0.6 ? "sc-med" : "sc-low";
    const d = document.createElement("div");
    d.className = "pitem";
    d.innerHTML = `<img src="${p.image}" alt="patch ${p.rank}" loading="lazy" />
      <div class="pi"><b>#${p.rank} · ${p.prediction.replace(/_/g, " ")}</b>
      <span class="${cls}">score ${(p.cancer_score * 100).toFixed(1)}%</span> · conf ${pct(p.confidence)}</div>`;
    d.querySelector("img").addEventListener("click", (e) => openLB(e.target.src, `Patch #${p.rank} — ${p.prediction}`));
    tp.appendChild(d);
  });
  $("topTitle").classList.toggle("hidden", !(r.top_patches || []).length);

  // regions
  const rl = $("regionList");
  rl.innerHTML = "";
  (r.regions || []).forEach((g, i) => {
    const d = document.createElement("div");
    d.className = "region";
    d.innerHTML = `<span class="dot ${g.label}"></span><span><strong>Region ${i + 1}</strong> — ${g.label} risk · score ${(g.score * 100).toFixed(1)}% · box (${g.x}, ${g.y}, ${g.w}×${g.h})</span>`;
    rl.appendChild(d);
  });
  $("regionTitle").classList.toggle("hidden", !(r.regions || []).length);

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function pct(x) { return x == null ? "—" : (x * 100).toFixed(1) + "%"; }

/* ---------------- lightbox ---------------- */
function openLB(src, cap) {
  $("lbImg").src = src; $("lbCap").textContent = cap;
  $("lightbox").classList.remove("hidden");
}
$("lbClose").addEventListener("click", () => $("lightbox").classList.add("hidden"));
$("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox" || e.target.classList.contains("lb-backdrop")) $("lightbox").classList.add("hidden"); });
window.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").classList.add("hidden"); });

/* ---------------- cleanup temp session when the site is closed ---------------- */
function cleanupSession() {
  if (!sessionId) return;
  const payload = JSON.stringify({ session_id: sessionId });
  try {
    if (navigator.sendBeacon) {
      navigator.sendBeacon(api("/api/cleanup"), new Blob([payload], { type: "application/json" }));
    } else {
      fetch(api("/api/cleanup"), { method: "POST", body: payload, keepalive: true,
        headers: { "Content-Type": "application/json" } });
    }
  } catch (e) { /* tab is closing — nothing more to do */ }
}
window.addEventListener("pagehide", cleanupSession);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") cleanupSession(); });
