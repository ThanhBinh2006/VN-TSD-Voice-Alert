/* Giao diện web TSD Voice Alert (bản 3).   Phụ trách: Thành viên 3
   JavaScript thuần, không thư viện. Chỉ gọi các API trong app/main.py:
     GET /api/health, POST /api/jobs, GET /api/jobs, GET /api/jobs/<id>,
     GET /api/jobs/<id>/voice-events, GET /api/jobs/<id>/detections
   Điều hướng bằng hash: "#/job/12" mở chi tiết job 12, "#/" là màn hình tải file lên. */
"use strict";

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

const MODEL_LABEL = { yolo: "YOLO26m", faster_rcnn: "Faster R-CNN" };
const POSITION_LABEL = { left: "Bên trái", front: "Phía trước", right: "Bên phải" };
const STATUS_LABEL = { queued: "Đang chờ", processing: "Đang xử lý", done: "Xong", failed: "Lỗi" };
const STAGE_LABEL = {
  queued: "Đang chờ tới lượt",
  loading_model: "Đang nạp mô hình (chỉ lần đầu)",
  processing: "Đang nhận diện biển báo",
  finalizing: "Đang ghép giọng nói vào video",
  done: "Đã xử lý xong",
  failed: "Xử lý thất bại",
};
const CAPTION_SECONDS = 3;        // phụ đề cảnh báo hiện trong 3 giây kể từ lúc phát
const POLL_MS = 1000;
const DENSITY_BINS = 30;          // số cột của biểu đồ "Mật độ biển báo"
const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.4 6z" fill="currentColor"/></svg>';

const state = { jobId: null, job: null, pollTimer: null, events: [], detections: [], duration: 0, maxUploadMb: 200 };

/* ------------------------------------------------------------------ tiện ích */

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Lỗi ${res.status}`);
  return body;
}

function formatTime(sec) {
  if (!Number.isFinite(sec)) return "–";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s < 10 ? "0" : ""}${s.toFixed(1)}`;
}

function formatMs(ms) {
  if (ms == null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} giây` : `${Math.round(ms)} ms`;
}

function formatClock(epochSec) {
  if (!epochSec) return "";
  return new Date(epochSec * 1000).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
}

/** Bỏ tiền tố ngẫu nhiên "a1b2c3d4_" mà server thêm vào tên file đã lưu. */
function displayName(job) {
  return (job.input_name || "").replace(/^[0-9a-f]{8}_/, "") || `Job ${job.id}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;          // chỉ dùng cho icon SVG cố định trong file này
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

/* ------------------------------------------------------------------ trạng thái server */

async function loadHealth() {
  const chip = $("server-status");
  const label = chip.querySelector("span");
  try {
    const h = await api("/api/health");
    state.maxUploadMb = h.max_upload_mb;
    const exts = (h.accept || []).map((e) => e.slice(1)).join(", ");
    $("accept-hint").textContent = `Hỗ trợ ${exts}. Tối đa ${h.max_upload_mb} MB.`;
    $("file-input").setAttribute("accept", (h.accept || []).join(","));
    if (h.fake) {
      label.textContent = "Chế độ thử: dữ liệu giả";
      chip.className = "status-chip fake";
      chip.title = `Đang dùng bản giả cho: ${h.fake_parts.join(", ")}`;
    } else {
      label.textContent = "Đang dùng mô hình thật";
      chip.className = "status-chip real";
    }
  } catch {
    label.textContent = "Không kết nối được server";
    chip.className = "status-chip down";
  }
}

/* ------------------------------------------------------------------ tải file lên */

function setupUpload() {
  const form = $("upload-form");
  const input = $("file-input");
  const drop = $("drop-zone");

  const showFile = () => {
    const file = input.files[0];
    $("file-name").hidden = !file;
    $("file-name").textContent = file ? `${file.name}, ${(file.size / 1048576).toFixed(1)} MB` : "";
    $("form-error").textContent = "";
  };
  input.addEventListener("change", showFile);

  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("dragging"); });
  }
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      showFile();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const file = input.files[0];
    if (!file) {
      $("form-error").textContent = "Chọn một video hoặc ảnh trước khi bấm Bắt đầu xử lý.";
      return;
    }
    if (file.size > state.maxUploadMb * 1048576) {
      $("form-error").textContent = `File nặng ${(file.size / 1048576).toFixed(0)} MB, vượt giới hạn ${state.maxUploadMb} MB. Hãy cắt ngắn video rồi thử lại.`;
      return;
    }
    uploadFile(new FormData(form));
  });
}

/** Dùng XMLHttpRequest (không phải fetch) để hiện % tải lên với file video lớn. */
function uploadFile(formData) {
  const xhr = new XMLHttpRequest();
  const button = $("submit-btn");
  button.disabled = true;
  $("form-error").textContent = "";
  $("upload-progress").hidden = false;

  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((100 * e.loaded) / e.total);
    $("upload-pct").textContent = `${pct}%`;
    $("upload-fill").style.width = `${pct}%`;
  });
  xhr.addEventListener("loadend", () => {
    button.disabled = false;
    $("upload-progress").hidden = true;
    $("upload-fill").style.width = "0";
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* bỏ qua */ }
    if (xhr.status === 202) {
      $("upload-form").reset();
      $("file-name").hidden = true;
      location.hash = `#/job/${body.job_id}`;
      loadHistory();
    } else if (xhr.status === 0) {
      $("form-error").textContent = "Mất kết nối tới server. Kiểm tra cửa sổ đang chạy python -m app.main.";
    } else {
      $("form-error").textContent = body.error || `Tải lên thất bại (mã ${xhr.status}).`;
    }
  });
  xhr.open("POST", "/api/jobs");
  xhr.send(formData);
}

/* ------------------------------------------------------------------ lịch sử */

async function loadHistory() {
  let jobs = [];
  try { jobs = await api("/api/jobs?limit=100"); } catch { return; }
  const list = $("history-list");
  list.replaceChildren();
  $("history-empty").hidden = jobs.length > 0;
  for (const job of jobs) {
    const button = el("button", { class: "history-item", type: "button", title: displayName(job),
      onclick: () => { location.hash = `#/job/${job.id}`; } },
      el("span", { class: "history-name" }, displayName(job)),
      el("span", { class: "history-sub" }, `${MODEL_LABEL[job.model_name] || job.model_name}, ${formatClock(job.created_at)}`),
      el("span", { class: `badge ${job.status}` }, STATUS_LABEL[job.status] || job.status));
    if (job.id === state.jobId) button.setAttribute("aria-current", "true");
    list.append(el("li", {}, button));
  }
}

/* ------------------------------------------------------------------ chi tiết job */

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function showUploadView() {
  stopPolling();
  state.jobId = null;
  $("job-view").hidden = true;
  $("home").hidden = false;
  const video = $("result-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  loadHistory();
}

function showJobView(jobId) {
  stopPolling();
  Object.assign(state, { jobId, job: null, events: [], detections: [], duration: 0 });
  $("home").hidden = true;
  $("job-view").hidden = false;
  for (const id of ["result", "insights", "events-card", "error-panel", "download-link"]) $(id).hidden = true;
  $("progress-panel").hidden = false;
  $("job-title").textContent = `Job ${jobId}`;
  $("job-meta").replaceChildren();
  $("stage-text").textContent = "Đang tải thông tin…";
  $("progress-pct").textContent = "";
  $("progress-detail").textContent = "";
  $("progress-fill").className = "track-fill indeterminate";
  window.scrollTo(0, 0);
  pollJob(jobId);
}

async function pollJob(jobId) {
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    if (state.jobId !== jobId) return;
    $("stage-text").textContent = err.message;
    $("progress-fill").className = "track-fill failed";
    return;
  }
  if (state.jobId !== jobId) return;           // người dùng đã chuyển sang job khác
  renderProgress(job);
  loadHistory();                                // cập nhật trạng thái trong cột lịch sử
  if (job.status === "done") {
    renderResult(job);
  } else if (job.status === "failed") {
    $("progress-panel").hidden = true;
    $("error-message").textContent = job.error_message || "Không rõ nguyên nhân.";
    $("error-panel").hidden = false;
  } else {
    state.pollTimer = setTimeout(() => pollJob(jobId), POLL_MS);
  }
}

function renderProgress(job) {
  const p = job.progress || {};
  $("job-title").textContent = displayName(job);
  $("job-meta").replaceChildren(
    el("li", {}, MODEL_LABEL[job.model_name] || job.model_name),
    el("li", {}, job.input_type === "image" ? "Ảnh" : "Video"),
    el("li", {}, `Gửi lúc ${formatClock(job.created_at)}`));
  $("stage-text").textContent = STAGE_LABEL[p.stage] || STATUS_LABEL[job.status] || job.status;

  const fill = $("progress-fill");
  if (p.percent == null && job.status !== "failed") {
    fill.className = "track-fill indeterminate";
    $("progress-pct").textContent = "";
  } else {
    fill.className = `track-fill ${job.status === "done" ? "done" : job.status === "failed" ? "failed" : ""}`;
    fill.style.width = `${p.percent ?? 100}%`;
    $("progress-pct").textContent = p.percent != null ? `${Math.floor(p.percent)}%` : "";
  }

  const parts = [];
  if (p.frame_count && job.input_type === "video") parts.push(`Khung ${p.frames_done}/${p.frame_count}`);
  if (p.fps) parts.push(`${p.fps.toFixed(1)} khung/giây`);
  if (p.eta_sec != null) parts.push(`còn khoảng ${Math.ceil(p.eta_sec)} giây`);
  $("progress-detail").textContent = parts.join(", ");
}

async function renderResult(job) {
  const [events, detections] = await Promise.all([
    api(`/api/jobs/${job.id}/voice-events`).catch(() => []),
    api(`/api/jobs/${job.id}/detections`).catch(() => []),
  ]);
  if (state.jobId !== job.id) return;
  state.job = job;
  state.events = events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  state.detections = detections;

  $("progress-panel").hidden = true;
  $("result").hidden = false;
  $("insights").hidden = false;
  $("events-card").hidden = false;
  $("download-link").hidden = false;
  $("download-link").href = job.download_url;

  const video = $("result-video");
  const image = $("result-image");
  const isVideo = job.input_type === "video";
  video.hidden = !isVideo;
  image.hidden = isVideo;
  $("road-wrap").hidden = !isVideo;
  $("caption").hidden = true;
  if (isVideo) {
    video.src = job.result_url;
  } else {
    video.removeAttribute("src");
    image.src = `${job.result_url}?t=${Date.now()}`;
  }

  renderHud(job);
  renderEvents(isVideo);
  renderInsights();
}

/** Các viên thông số ở góc trên video. */
function renderHud(job) {
  const p = job.progress || {};
  const items = [el("li", { class: "live" }, MODEL_LABEL[job.model_name] || job.model_name)];
  if (p.fps) items.push(el("li", {}, el("b", {}, p.fps.toFixed(1)), " khung/giây"));
  items.push(el("li", {}, el("b", {}, String(job.unique_sign_count ?? state.events.length)), " biển báo"));
  items.push(el("li", {}, el("b", {}, String(job.voice_play_count ?? state.events.length)), " lần đọc cảnh báo"));
  $("hud").replaceChildren(...items);
}

function seekTo(sec) {
  const video = $("result-video");
  video.currentTime = Math.max(0, sec - 0.3);   // lùi 0,3 giây để kịp thấy biển xuất hiện
  video.play().catch(() => {});
}

/** Dòng thời gian các lần cảnh báo: thời điểm, chấm biển báo, câu đọc, làn, nút nghe. */
function renderEvents(isVideo) {
  const list = $("events");
  list.replaceChildren();
  $("events-empty").hidden = state.events.length > 0;
  list.hidden = state.events.length === 0;

  state.events.forEach((ev, i) => {
    const time = isVideo
      ? el("button", { class: "tl-time", type: "button", onclick: () => seekTo(ev.timestamp_sec),
          "aria-label": `Tua video tới ${formatTime(ev.timestamp_sec)}` }, formatTime(ev.timestamp_sec))
      : el("span", { class: "tl-time" }, "–");
    const play = el("button", { class: "tl-play", type: "button", title: "Nghe câu cảnh báo",
      "aria-label": `Nghe: ${ev.spoken_text}`, html: PLAY_ICON,
      onclick: () => new Audio(ev.audio_url).play().catch(() => {}) });
    list.append(el("li", { "data-index": i },
      time,
      el("span", { class: "tl-dot", "aria-hidden": "true" }),
      el("span", { class: "tl-text" }, ev.spoken_text),
      el("span", { class: "tl-lane" }, POSITION_LABEL[laneOf(ev)]),
      play));
  });
  if (isVideo) drawMarkers();
}

/* ------------------------------------------------------------------ dải 4 số liệu */

function renderInsights() {
  const job = state.job;
  if (!job) return;
  renderSpeed(job);
  renderPositions();
  renderTiming(job);
  renderDensity(job);
  renderHud(job);
}

/** Đồng hồ đo: tốc độ xử lý so với tốc độ phát của video (≥ 1 lần là kịp thời gian thực). */
function renderSpeed(job) {
  const box = $("ins-speed");
  box.replaceChildren();
  const p = job.progress || {};
  const videoFps = job.input_type === "video" && state.duration && p.frame_count ? p.frame_count / state.duration : null;
  const ratio = p.fps && videoFps ? p.fps / videoFps : null;

  box.append(el("div", {},
    el("div", { class: "big" }, ratio != null ? `${ratio.toFixed(1)}×` : "–"),
    el("p", { class: "caption-sm" }, "so với thời gian thực")));

  const g = svg("svg", { viewBox: "0 0 120 64", role: "img",
    "aria-label": ratio ? `Xử lý nhanh gấp ${ratio.toFixed(1)} lần thời gian thực` : "Chưa đo được tỉ lệ thời gian thực" });
  const arc = "M10 58 A50 50 0 0 1 110 58";
  g.append(svg("path", { d: arc, class: "gauge-track", pathLength: 100 }));
  if (ratio != null) {
    const fill = Math.min(100, (ratio / 2) * 100);           // nửa vòng = gấp 2 lần thời gian thực
    g.append(svg("path", { d: arc, class: `gauge-value${ratio < 1 ? " slow" : ""}`, pathLength: 100,
      "stroke-dasharray": `${fill} 100` }));
  }
  g.append(svg("line", { x1: 60, y1: 2, x2: 60, y2: 14, class: "gauge-tick" }));   // mốc 1× = thời gian thực
  box.append(g);

  const note = ratio == null ? (job.input_type === "image" ? "Ảnh tĩnh, không tính thời gian thực." : "Đang đọc độ dài video…")
    : ratio >= 1 ? `Kịp thời gian thực, ${p.fps.toFixed(1)} khung/giây.` : `Chậm hơn thời gian thực, ${p.fps.toFixed(1)} khung/giây.`;
  box.append(el("p", { class: "caption-sm" }, note));
}

/** Quạt tầm nhìn người lái chia 3 vùng: số biển báo (sau chống lặp) ở mỗi vùng. */
function renderPositions() {
  const box = $("ins-position");
  box.replaceChildren();
  const counts = { left: 0, front: 0, right: 0 };
  state.events.forEach((ev) => { counts[laneOf(ev)] += 1; });
  const max = Math.max(1, ...Object.values(counts));

  const cx = 60, cy = 70, r = 58;
  const sectors = [["left", 160, 112], ["front", 112, 68], ["right", 68, 20]];
  const g = svg("svg", { viewBox: "0 0 120 80", role: "img",
    "aria-label": `Bên trái ${counts.left}, phía trước ${counts.front}, bên phải ${counts.right} biển báo` });
  const pt = (deg, rad) => [cx + rad * Math.cos((deg * Math.PI) / 180), cy - rad * Math.sin((deg * Math.PI) / 180)];
  for (const [key, a1, a2] of sectors) {
    const [x1, y1] = pt(a1, r);
    const [x2, y2] = pt(a2, r);
    g.append(svg("path", { d: `M${cx} ${cy} L${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2} Z`, class: "fan-sector",
      "fill-opacity": (0.1 + 0.8 * (counts[key] / max)).toFixed(2) }));
  }
  g.append(svg("rect", { x: cx - 6, y: cy - 6, width: 12, height: 12, rx: 3, class: "fan-car" }));
  box.append(g);

  box.append(el("div", { class: "figure-row" },
    ...Object.keys(counts).map((k) => el("div", {},
      el("div", { class: "big" }, String(counts[k])),
      el("p", { class: "caption-sm" }, POSITION_LABEL[k])))));
}

/** Thanh chia tổng thời gian: nhận diện và phần còn lại / vẽ khung / tạo giọng nói. */
function renderTiming(job) {
  const box = $("ins-timing");
  box.replaceChildren();
  const total = job.total_time_ms || 0;
  const layer = job.layer_time_ms || 0;
  const tts = job.tts_time_ms || 0;
  const detect = Math.max(0, total - layer - tts);
  const parts = [["c-detect", "Nhận diện", detect], ["c-layer", "Vẽ khung", layer], ["c-tts", "Giọng nói", tts]];

  box.append(el("div", {},
    el("div", { class: "big" }, total >= 1000 ? (total / 1000).toFixed(1) : String(Math.round(total)),
      el("small", {}, total >= 1000 ? "giây" : "ms")),
    el("p", { class: "caption-sm" }, "tổng thời gian xử lý")));
  const split = el("div", { class: "split", role: "img",
    "aria-label": parts.map(([, label, ms]) => `${label} ${formatMs(ms)}`).join(", ") });
  for (const [cls, , ms] of parts) {
    const seg = el("span", { class: cls });
    seg.style.width = `${total ? (100 * ms) / total : 0}%`;
    split.append(seg);
  }
  box.append(split);
  box.append(el("ul", { class: "legend" },
    ...parts.map(([cls, label, ms]) => el("li", {}, el("i", { class: cls }), label, el("b", {}, formatMs(ms))))));
}

/** Số biển báo trong khung hình theo thời gian video; bấm vào cột để tua tới đoạn đó. */
function renderDensity(job) {
  const box = $("ins-density");
  box.replaceChildren();
  const dets = state.detections;
  if (job.input_type !== "video") {
    box.append(el("div", {}, el("div", { class: "big" }, String(dets.length)),
      el("p", { class: "caption-sm" }, "biển báo trong ảnh")));
    return;
  }
  const duration = state.duration || Math.max(0, ...dets.map((d) => d.timestamp_sec)) || 1;
  const bins = new Array(DENSITY_BINS).fill(0);
  for (const d of dets) bins[Math.min(DENSITY_BINS - 1, Math.floor((d.timestamp_sec / duration) * DENSITY_BINS))] += 1;
  const max = Math.max(1, ...bins);
  const framesWithSign = new Set(dets.map((d) => d.frame_index)).size;
  const frameCount = (job.progress || {}).frame_count;

  box.append(el("div", {},
    el("div", { class: "big" }, frameCount ? `${Math.round((100 * framesWithSign) / frameCount)}%` : String(framesWithSign)),
    el("p", { class: "caption-sm" }, `khung hình có biển, ${dets.length} lượt phát hiện`)));
  const chart = el("div", { class: "density" });
  bins.forEach((n, i) => {
    const start = (i / DENSITY_BINS) * duration;
    const bar = el("button", { type: "button", class: n ? "" : "empty",
      title: `${formatTime(start)}: ${n} lượt phát hiện`, "aria-label": `Tua tới ${formatTime(start)}, ${n} lượt phát hiện`,
      onclick: () => seekTo(start + 0.3) });
    bar.style.height = `${Math.max(4, (100 * n) / max)}%`;
    chart.append(bar);
  });
  box.append(chart);
  box.append(el("div", { class: "density-axis" }, el("span", {}, "0:00"), el("span", {}, formatTime(duration))));
}

/* ------------------------------------------------------------------ dải đường 3 làn */

/** Làn của một cảnh báo: ưu tiên "position" server trả về; nếu thiếu thì đoán từ câu đọc. */
function laneOf(ev) {
  if (POSITION_LABEL[ev.position]) return ev.position;
  const text = ev.spoken_text.toLowerCase();
  if (text.includes("bên trái")) return "left";
  if (text.includes("bên phải")) return "right";
  return "front";
}

function drawMarkers() {
  const video = $("result-video");
  const road = $("road");
  road.querySelectorAll(".marker").forEach((m) => m.remove());
  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) return;   // vẽ lại khi có loadedmetadata
  state.duration = duration;
  $("road-end").textContent = formatTime(duration);

  state.events.forEach((ev, i) => {
    const lane = road.querySelector(`.lane[data-lane="${laneOf(ev)}"]`);
    const marker = el("button", { class: "marker", type: "button", "data-index": i,
      title: `${formatTime(ev.timestamp_sec)}: ${ev.spoken_text}`,
      "aria-label": `${formatTime(ev.timestamp_sec)}, ${ev.spoken_text}`,
      onclick: (e) => { e.stopPropagation(); seekTo(ev.timestamp_sec); } });
    marker.style.left = `${Math.min(100, (100 * ev.timestamp_sec) / duration)}%`;
    lane.append(marker);
  });
}

function setupPlayer() {
  const video = $("result-video");
  const road = $("road");
  // Có độ dài video thì vẽ được mốc trên dải đường và tính tỉ lệ thời gian thực.
  video.addEventListener("loadedmetadata", () => { drawMarkers(); renderInsights(); });

  video.addEventListener("timeupdate", () => {
    const t = video.currentTime;
    if (state.duration) road.querySelector(".playhead").style.setProperty("--p", Math.min(1, t / state.duration));

    // Câu cảnh báo gần nhất đã phát trong CAPTION_SECONDS giây vừa qua.
    let current = -1;
    state.events.forEach((ev, i) => {
      if (ev.timestamp_sec <= t && t - ev.timestamp_sec < CAPTION_SECONDS) current = i;
    });
    const caption = $("caption");
    caption.hidden = current < 0;
    if (current >= 0) caption.textContent = state.events[current].spoken_text;
    road.querySelectorAll(".marker").forEach((m) => m.classList.toggle("active", Number(m.dataset.index) === current));
    $("events").querySelectorAll("li").forEach((r) => r.classList.toggle("active", Number(r.dataset.index) === current));
  });

  // Bấm vào bất kỳ đâu trên dải đường để tua tới thời điểm tương ứng.
  road.addEventListener("click", (e) => {
    const lane = road.querySelector(".lane").getBoundingClientRect();
    if (!state.duration || e.clientX < lane.left) return;
    seekTo(((e.clientX - lane.left) / lane.width) * state.duration + 0.3);
  });
}

/* ------------------------------------------------------------------ điều hướng */

function route() {
  const match = location.hash.match(/^#\/job\/(\d+)$/);
  if (match) showJobView(Number(match[1]));
  else showUploadView();
}

$("new-btn").addEventListener("click", () => { location.hash = "#/"; });
window.addEventListener("hashchange", route);

setupUpload();
setupPlayer();
loadHealth();
route();
