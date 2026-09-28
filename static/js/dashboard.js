const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileInput = document.getElementById("file-input");
const analyzeBtn = document.getElementById("analyze-btn");
const uploadStatus = document.getElementById("upload-status");
const modelStatus = document.getElementById("model-status");
const violationsBody = document.getElementById("violations-body");
const tracksBody = document.getElementById("tracks-body");
const trackFilter = document.getElementById("track-filter");
const evidenceModal = document.getElementById("evidence-modal");
const evidenceImage = document.getElementById("evidence-image");
const evidenceUnavailable = document.getElementById("evidence-unavailable");
const evidenceMetadata = document.getElementById("evidence-metadata");
const evidenceImageWrap = document.getElementById("evidence-image-wrap");
let evidenceOpener = null;
let evidenceZoom = 1;

let selectedFile = null;
let currentSessionId = null;

// ---------------------------------------------------------------- upload UI
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    selectedFile = fileInput.files[0];
    dropzoneText.textContent = selectedFile.name;
    analyzeBtn.disabled = false;
  }
});

["dragover", "dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.toggle("dragover", evt === "dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = e.dataTransfer.files;
  selectedFile = file;
  dropzoneText.textContent = file.name;
  analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  uploadStatus.textContent = "Uploading file…";
  uploadStatus.className = "upload-status";
  const progressTimer = setInterval(refreshProcessingProgress, 800);

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");

    currentSessionId = data.session_id;
    uploadStatus.textContent = `Analysis complete — ${data.tracks_found} tracked people: ${data.helmeted} with helmet evidence, ${data.no_helmet} classified no-helmet, ${data.unknown} unknown; ${data.violations} confirmed violation event(s). Unknown means the available model could not determine helmet status.`;
    uploadStatus.className = "upload-status is-ok";
    refreshStats();
    refreshTracks();
    refreshViolations();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = "upload-status is-error";
  } finally {
    clearInterval(progressTimer);
    analyzeBtn.disabled = false;
  }
});

async function refreshProcessingProgress() {
  try {
    const res = await fetch("/api/progress", { cache: "no-store" });
    const progress = await res.json();
    if (!progress.active) return;
    if (progress.phase === "Analyzing video") {
      uploadStatus.textContent = progress.percent == null
        ? `Analyzing video… frame ${progress.frame}`
        : `Analyzing video… ${progress.percent}% (frame ${progress.frame} of ${progress.total_frames})`;
    } else {
      uploadStatus.textContent = `${progress.phase}…`;
    }
  } catch { /* The upload request remains authoritative if progress polling fails. */ }
}

// ---------------------------------------------------------------- status pill
async function refreshModelStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.helmet_model_loaded) {
      modelStatus.textContent = data.direct_two_class
        ? "● 2-class model loaded"
        : "● single-class model · heuristic fallback";
      modelStatus.className = data.direct_two_class
        ? "status-pill status-pill--ok" : "status-pill status-pill--warn";
    } else {
      modelStatus.textContent = data.error || "Detection model unavailable";
      modelStatus.className = "status-pill status-pill--warn";
    }
  } catch {
    modelStatus.textContent = "● status unavailable";
  }
}

// ---------------------------------------------------------------- stats
function animateValue(el, to, suffixEl) {
  const from = parseInt(el.textContent, 10) || 0;
  const duration = 500;
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const value = Math.round(from + (to - from) * p);
    el.firstChild.textContent = value;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function sessionQuery() {
  return currentSessionId ? `?session_id=${encodeURIComponent(currentSessionId)}` : "";
}

async function refreshStats() {
  const res = await fetch(`/api/stats${sessionQuery()}`);
  const s = await res.json();
  setStatText("stat-scanned", s.total_scanned);
  setStatText("stat-violations", s.violations);
  setStatText("stat-compliance", s.compliance_rate == null ? "N/A" : s.compliance_rate);
  setStatText("stat-plates", s.plates_read);
  setStatText("stat-helmeted", s.helmeted);
  setStatText("stat-no-helmet", s.no_helmet);
  setStatText("stat-unknown", s.unknown);
  setStatText("stat-plates-detected", s.plates_detected);
  setStatText("stat-plate-rate", `${s.plate_read_rate}%`);
  setStatText("stat-ocr-failures", s.ocr_failures);
}

function setStatText(id, value) {
  const el = document.getElementById(id);
  if (el.firstChild) el.firstChild.textContent = value;
  else el.textContent = value;
}

trackFilter.addEventListener("change", refreshTracks);

async function refreshTracks() {
  const params = new URLSearchParams();
  if (currentSessionId) params.set("session_id", currentSessionId);
  if (trackFilter.value) params.set("status", trackFilter.value);
  const query = params.toString();
  const res = await fetch(`/api/tracks${query ? `?${query}` : ""}`);
  const tracks = await res.json();
  if (!tracks.length) {
    const message = currentSessionId
      ? "No people match this filter in the current upload."
      : "Upload footage to create a track record.";
    tracksBody.innerHTML = `<tr class="empty-row"><td colspan="6">${message}</td></tr>`;
    return;
  }

  const labels = { helmet: "Helmet detected", no_helmet: "No helmet (model-confirmed)", unknown: "Unknown" };
  tracksBody.innerHTML = tracks.map((track) => {
    const state = labels[track.helmet_status] || "Unknown";
    const stateClass = track.helmet_status === "helmet" ? "helmet"
      : track.helmet_status === "no_helmet" ? "no-helmet" : "unknown";
    const score = Number.isFinite(track.helmet_confidence) && track.helmet_confidence >= 0 && track.helmet_confidence <= 1
      ? `${Math.round(track.helmet_confidence * 100)}%` : "N/A";
    const snapshot = track.snapshot_url
      ? `<a class="track-snapshot-link" href="${escapeHtml(track.snapshot_url)}" target="_blank" rel="noopener" aria-label="Open person snapshot for track ${escapeHtml(String(track.tracking_id))}"><img class="track-thumb" src="${escapeHtml(track.snapshot_url)}" alt="Person snapshot for track ${escapeHtml(String(track.tracking_id))}"></a>`
      : '<span class="thumb-missing">Unavailable</span>';
    return `<tr>
      <td>${snapshot}</td>
      <td>${escapeHtml(String(track.tracking_id ?? "N/A"))}</td>
      <td><span class="track-status track-status--${stateClass}">${state}</span></td>
      <td>${score}</td>
      <td>${escapeHtml(track.source_filename || "N/A")}</td>
      <td>${track.frame_number == null ? "Image" : escapeHtml(String(track.frame_number))}</td>
    </tr>`;
  }).join("");
}

tracksBody.addEventListener("error", (event) => {
  if (event.target.matches(".track-thumb")) {
    event.target.replaceWith(Object.assign(document.createElement("span"), {
      className: "thumb-missing", textContent: "Snapshot unavailable"
    }));
  }
}, true);

// ---------------------------------------------------------------- violations table
function formatTime(iso) {
  if (!iso) return "N/A";
  const d = new Date(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(d.getTime())) return "N/A";
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

async function refreshViolations() {
  const query = currentSessionId ? `&session_id=${encodeURIComponent(currentSessionId)}` : "";
  const res = await fetch(`/api/violations?limit=50${query}`);
  const rows = await res.json();

  if (!rows.length) {
    violationsBody.innerHTML = '<tr class="empty-row"><td colspan="7">No violations logged yet.</td></tr>';
    return;
  }

  violationsBody.innerHTML = rows
    .map((r) => {
      const thumb = r.snapshot_url
        ? `<button class="thumb-button" type="button" data-evidence="${escapeHtml(r.snapshot_url)}" data-row="${escapeHtml(JSON.stringify(r))}" aria-label="Open snapshot evidence"><img class="thumb" src="${escapeHtml(r.snapshot_url)}" alt="Violation snapshot"></button>`
        : '<span class="thumb-missing">Unavailable</span>';
      const plate = `<span class="plate-chip ${r.plate_matched_template ? "" : "unmatched"}">${escapeHtml(r.plate_clean || "Unreadable")}</span>`;
      const conf = Number.isFinite(r.helmet_confidence) && r.helmet_confidence >= 0 && r.helmet_confidence <= 1
        ? `${Math.round(r.helmet_confidence * 100)}%` : "N/A";
      const noHelmetConf = Number.isFinite(r.no_helmet_confidence)
        ? `${Math.round(Math.max(0, Math.min(1, r.no_helmet_confidence)) * 100)}%` : "N/A";
      const plateConf = Number.isFinite(r.plate_confidence)
        ? `${Math.round(Math.max(0, Math.min(1, r.plate_confidence)) * 100)}%` : "N/A";
      const source = r.source_filename || "N/A";
      const plateStatus = r.plate_status || "unknown";
      return `<tr>
        <td>${thumb}</td>
        <td>${formatTime(r.captured_at)}</td>
        <td>${plate}</td>
        <td><span class="conf-bar">No-Helmet model ${noHelmetConf} · Helmet ${conf}</span></td>
        <td>${plateConf} (${escapeHtml(r.plate_status || "unknown")})</td>
        <td>${escapeHtml(source)}</td>
        <td>${r.frame_number == null ? "—" : r.frame_number}</td>
      </tr>`;
    })
    .join("");
}

function confidenceText(value) {
  return Number.isFinite(value) ? `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` : "N/A";
}

function setEvidenceZoom(value) {
  evidenceZoom = Math.max(0.5, Math.min(4, value));
  evidenceImage.style.transform = `scale(${evidenceZoom})`;
}

function openEvidence(button, row) {
  evidenceOpener = button;
  evidenceImage.hidden = true;
  evidenceUnavailable.hidden = true;
  evidenceImageWrap.classList.remove("is-missing");
  evidenceImage.onload = () => { evidenceImage.hidden = false; evidenceUnavailable.hidden = true; };
  evidenceImage.onerror = () => { evidenceImage.hidden = true; evidenceUnavailable.hidden = false; evidenceImageWrap.classList.add("is-missing"); };
  evidenceImage.src = button.dataset.evidence;
  evidenceMetadata.innerHTML = [
    ["Violation", "No-Helmet"], ["Time", formatTime(row.captured_at)],
    ["Source", row.source_filename || "N/A"], ["Frame", row.frame_number == null ? "N/A" : row.frame_number],
    ["Tracking ID", row.tracking_id == null ? "N/A" : row.tracking_id],
    ["Helmet status", "No-Helmet"], ["Helmet confidence", confidenceText(row.helmet_confidence)],
    ["No-Helmet confidence", confidenceText(row.no_helmet_confidence)],
    ["Plate status", row.plate_status || "unknown"], ["Plate confidence", confidenceText(row.plate_confidence)],
    ["Vehicle number", row.plate_clean || "Unreadable"]
  ].map(([label, value]) => `<div><dt>${escapeHtml(String(label))}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join("");
  setEvidenceZoom(1);
  evidenceModal.hidden = false;
  evidenceModal.setAttribute("aria-hidden", "false");
  document.body.classList.add("evidence-open");
  document.getElementById("evidence-close").focus();
}

function closeEvidence() {
  if (evidenceModal.hidden) return;
  evidenceModal.hidden = true;
  evidenceModal.setAttribute("aria-hidden", "true");
  evidenceImage.removeAttribute("src");
  document.body.classList.remove("evidence-open");
  if (evidenceOpener && evidenceOpener.isConnected) evidenceOpener.focus();
}

violationsBody.addEventListener("click", (event) => {
  const button = event.target.closest(".thumb-button");
  if (!button) return;
  try { openEvidence(button, JSON.parse(button.dataset.row)); }
  catch { openEvidence(button, {}); }
});

violationsBody.addEventListener("error", (event) => {
  if (event.target.matches(".thumb")) {
    const button = event.target.closest(".thumb-button");
    if (button) button.innerHTML = '<span class="thumb-missing">Snapshot unavailable</span>';
  }
}, true);

document.getElementById("evidence-close").addEventListener("click", closeEvidence);
evidenceModal.addEventListener("click", (event) => { if (event.target === evidenceModal) closeEvidence(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeEvidence(); });
document.getElementById("zoom-in").addEventListener("click", () => setEvidenceZoom(evidenceZoom + 0.25));
document.getElementById("zoom-out").addEventListener("click", () => setEvidenceZoom(evidenceZoom - 0.25));
document.getElementById("zoom-reset").addEventListener("click", () => setEvidenceZoom(1));
document.getElementById("zoom-fit").addEventListener("click", () => { evidenceImageWrap.scrollTo(0, 0); setEvidenceZoom(1); });

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------- init
refreshModelStatus();
refreshStats();
refreshTracks();
refreshViolations();
