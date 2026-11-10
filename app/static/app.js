/* Giao diện web TSD Voice Alert (bản 3).   Phụ trách: Thành viên 3
   JavaScript thuần, không thư viện. Chỉ gọi các API trong app/main.py:
     GET /api/health, POST /api/jobs, GET /api/jobs, GET /api/jobs/<id>,
     GET /api/jobs/<id>/voice-events, GET /api/jobs/<id>/detections,
     GET /api/jobs/<id>/stream
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
const POLL_MS_QUEUE = 750;        // chu kỳ poll khi đang chờ / nạp model
const POLL_MS_LIVE = 400;         // chu kỳ poll khi đang phát stream trực tiếp
const DENSITY_BINS = 30;          // số cột của biểu đồ "Mật độ biển báo"
const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.4 6z" fill="currentColor"/></svg>';

const state = {
  jobId: null,
  job: null,
  pollTimer: null,
  pollInFlight: false,
  events: [],
  detections: [],
  duration: 0,
  maxUploadMb: 200,
  defaultFrameStride: 30,

  // Luồng live streaming & âm thanh trực tiếp
  activeStreamJobId: null,
  liveAudioEnabled: true,
  liveAudioBlocked: false,
  audioUnlocked: false,
  audioCtx: null,
  audioQueue: [],
  currentAudio: null,
  seenEventIds: new Set(),
  eventsInitializedForJob: null,
  autoPlayLiveJobId: null,
  captionTimer: null,
};

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

/* ------------------------------------------------------------------ âm thanh trực tiếp (Audio Autoplay & Queue) */

/** Mở khóa quyền autoplay âm thanh từ tương tác người dùng. */
function unlockAudio() {
  state.audioUnlocked = true;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      if (!state.audioCtx) state.audioCtx = new AudioCtx();
      if (state.audioCtx.state === "suspended") {
        state.audioCtx.resume().catch(() => {});
      }
    }
    const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");
    silent.volume = 0;
    const p = silent.play();
    if (p && p.then) p.catch(() => {});
  } catch (_) {}
}

/** Phát tuần tự từng câu cảnh báo trong hàng đợi âm thanh trực tiếp, không phát đè nhau. */
function playNextAudio() {
  if (!state.liveAudioEnabled) {
    state.audioQueue = [];
    return;
  }
  if (state.currentAudio != null) return;
  if (state.audioQueue.length === 0) return;

  const ev = state.audioQueue.shift();
  if (!ev || !ev.audio_url) {
    playNextAudio();
    return;
  }

  const audio = new Audio(ev.audio_url);
  state.currentAudio = audio;

  const finish = () => {
    if (state.currentAudio === audio) {
      state.currentAudio = null;
    }
    playNextAudio();
  };

  audio.addEventListener("ended", finish, { once: true });
  audio.addEventListener("error", finish, { once: true });

  audio.play().catch((err) => {
    const isAutoplayBlocked = err && (
      err.name === "NotAllowedError" ||
      err.name === "AbortError" ||
      /play\(\) failed because the user didn't interact/i.test(err.message || "")
    );

    if (isAutoplayBlocked) {
      console.warn("Autoplay audio bị chặn:", err);
      state.liveAudioBlocked = true;
      // Đưa lại câu cảnh báo vào đầu hàng đợi, cắt bớt nếu tồn đọng quá 8 câu
      state.audioQueue.unshift(ev);
      if (state.audioQueue.length > 8) {
        state.audioQueue = state.audioQueue.slice(-8);
      }
      state.currentAudio = null;
      updateLiveAudioBtn(true, true);
      return; // Dừng tại đây, chờ người dùng bấm nút mở khóa
    }

    // Các lỗi khác (network / file hỏng): bỏ qua câu này và phát câu tiếp theo
    console.warn("Lỗi phát audio:", err);
    finish();
  });
}

/** Dừng hẳn việc phát âm thanh trực tiếp và xóa hàng đợi. */
function stopLiveAudio() {
  if (state.currentAudio) {
    try {
      state.currentAudio.pause();
      state.currentAudio.removeAttribute("src");
    } catch (_) {}
    state.currentAudio = null;
  }
  state.audioQueue = [];
  state.liveAudioBlocked = false;
}

/** Cập nhật giao diện nút bật/tắt tiếng trực tiếp. */
function updateLiveAudioBtn(enabled, blocked = false) {
  const btn = $("live-audio-btn");
  if (!btn) return;
  btn.setAttribute("aria-pressed", enabled ? "true" : "false");
  btn.classList.toggle("muted", !enabled);

  const iconOn = btn.querySelector(".audio-icon.on");
  const iconOff = btn.querySelector(".audio-icon.off");
  if (iconOn && iconOff) {
    iconOn.hidden = !enabled;
    iconOff.hidden = enabled;
  }

  const label = $("live-audio-label");
  if (label) {
    if (blocked && enabled) {
      label.textContent = "Bật tiếng trực tiếp";
    } else {
      label.textContent = enabled ? "Giọng nói: Bật" : "Giọng nói: Tắt";
    }
  }
}

function toggleLiveAudio() {
  unlockAudio();
  if (state.liveAudioBlocked) {
    // Đang bị chặn autoplay: bấm nút là mở tiếng và phát ngay hàng đợi chờ
    state.liveAudioBlocked = false;
    state.liveAudioEnabled = true;
    updateLiveAudioBtn(true, false);
    playNextAudio();
    return;
  }

  state.liveAudioEnabled = !state.liveAudioEnabled;
  updateLiveAudioBtn(state.liveAudioEnabled, false);
  if (!state.liveAudioEnabled) {
    stopLiveAudio();
  } else {
    playNextAudio();
  }
}

/** Giải phóng luồng MJPEG để trình duyệt đóng socket ngay lập tức. */
function teardownLiveStream() {
  const liveImg = $("live-stream");
  if (liveImg) {
    liveImg.hidden = true;
    liveImg.removeAttribute("src");
  }
  state.activeStreamJobId = null;
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

    // Tự điền frame_stride từ cấu hình server nếu người dùng chưa sửa tay
    if (h.frame_stride != null) {
      state.defaultFrameStride = h.frame_stride;
      const strideInput = $("frame-stride-input");
      if (strideInput && !strideInput.dataset.userEdited) {
        strideInput.value = String(h.frame_stride);
      }
    }

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
  const strideInput = $("frame-stride-input");

  if (strideInput) {
    strideInput.addEventListener("input", () => {
      strideInput.dataset.userEdited = "true";
    });
  }

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

    if (strideInput) {
      const rawStride = strideInput.value.trim();
      if (!/^\d+$/.test(rawStride)) {
        $("form-error").textContent = "Số frame đoán 1 lần (frame_stride) phải là số nguyên từ 1 đến 300.";
        return;
      }
      const strideVal = parseInt(rawStride, 10);
      if (strideVal < 1 || strideVal > 300) {
        $("form-error").textContent = "Số frame đoán 1 lần (frame_stride) phải là số nguyên từ 1 đến 300.";
        return;
      }
    }

    // Mở khóa âm thanh từ sự kiện submit của người dùng
    unlockAudio();
    uploadFile(new FormData(form));
  });
}

/** Dùng XMLHttpRequest (không phải fetch) để hiện % tải lên với file video lớn. */
function uploadFile(formData) {
  const xhr = new XMLHttpRequest();
  const button = $("submit-btn");
  const strideInput = $("frame-stride-input");
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
      if (strideInput) {
        delete strideInput.dataset.userEdited;
        strideInput.value = String(state.defaultFrameStride ?? 30);
      }
      if (body.job_id) {
        state.autoPlayLiveJobId = body.job_id;
      }
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

/* ------------------------------------------------------------------ dọn dẹp & chi tiết job */

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function clearJobArtifacts() {
  stopPolling();
  teardownLiveStream();
  stopLiveAudio();
  clearTimeout(state.captionTimer);
  state.captionTimer = null;

  const video = $("result-video");
  video.pause();
  video.removeAttribute("src");
  video.load();

  const liveImg = $("live-stream");
  if (liveImg) {
    liveImg.hidden = true;
    liveImg.removeAttribute("src");
  }

  const liveBar = $("live-bar");
  if (liveBar) liveBar.hidden = true;

  const caption = $("caption");
  if (caption) {
    caption.hidden = true;
    caption.textContent = "";
  }

  // Reset đầu đọc playhead về mốc 0:00
  const playhead = $("road") ? $("road").querySelector(".playhead") : null;
  if (playhead) playhead.style.setProperty("--p", 0);
}

function showUploadView() {
  clearJobArtifacts();
  state.jobId = null;
  state.job = null;
  $("job-view").hidden = true;
  $("home").hidden = false;
  loadHistory();
}

function showJobView(jobId) {
  clearJobArtifacts();
  Object.assign(state, {
    jobId,
    job: null,
    events: [],
    detections: [],
    duration: 0,
    pollInFlight: false,
    activeStreamJobId: null,
    seenEventIds: new Set(),
    eventsInitializedForJob: null,
  });

  updateLiveAudioBtn(state.liveAudioEnabled, false);

  $("home").hidden = true;
  $("job-view").hidden = false;
  for (const id of ["result", "insights", "events-card", "error-panel", "download-link", "live-bar"]) {
    const elNode = $(id);
    if (elNode) elNode.hidden = true;
  }
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

/* ------------------------------------------------------------------ vòng lặp poll & stream */

async function pollJob(jobId) {
  if (state.jobId !== jobId || state.pollInFlight) return;
  state.pollInFlight = true;

  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    state.pollInFlight = false;
    if (state.jobId !== jobId) return;
    $("stage-text").textContent = err.message;
    $("progress-fill").className = "track-fill failed";
    return;
  }

  if (state.jobId !== jobId) {
    state.pollInFlight = false;
    return;
  }

  loadHistory();

  const p = job.progress || {};
  const stage = p.stage || job.status;
  const isVideo = job.input_type === "video";
  const isLiveProcessing = isVideo && (stage === "processing" || stage === "finalizing") && job.status !== "done" && job.status !== "failed";

  if (job.status === "done") {
    state.autoPlayLiveJobId = null;
    state.pollInFlight = false;
    stopPolling();
    teardownLiveStream();
    stopLiveAudio();
    if ($("live-bar")) $("live-bar").hidden = true;
    await renderResult(job);
    return;
  }

  if (job.status === "failed") {
    state.autoPlayLiveJobId = null;
    state.pollInFlight = false;
    stopPolling();
    teardownLiveStream();
    stopLiveAudio();
    if ($("live-bar")) $("live-bar").hidden = true;
    $("progress-panel").hidden = true;
    $("error-message").textContent = job.error_message || "Không rõ nguyên nhân.";
    $("error-panel").hidden = false;
    return;
  }

  if (isLiveProcessing) {
    await renderLiveProgress(job);
  } else {
    renderProgress(job);
  }

  state.pollInFlight = false;
  if (state.jobId === jobId) {
    const nextInterval = isLiveProcessing ? POLL_MS_LIVE : POLL_MS_QUEUE;
    state.pollTimer = setTimeout(() => pollJob(jobId), nextInterval);
  }
}

/** Hiển thị màn hình chờ khi job ở queued hoặc loading_model (hoặc ảnh tĩnh). */
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

/** Chế độ Live Streaming: vừa nhận diện, vừa phát luồng trực tiếp và đọc cảnh báo. */
async function renderLiveProgress(job) {
  state.job = job;
  const p = job.progress || {};
  const stage = p.stage || job.status;

  $("job-title").textContent = displayName(job);
  const metaItems = [
    el("li", {}, MODEL_LABEL[job.model_name] || job.model_name),
    el("li", {}, "Video"),
    el("li", {}, `Gửi lúc ${formatClock(job.created_at)}`),
  ];
  const strideVal = job.frame_stride || p.frame_stride;
  if (strideVal) {
    metaItems.push(el("li", { class: "num" }, `Mỗi ${strideVal} frame`));
  }
  $("job-meta").replaceChildren(...metaItems);

  // Ẩn panel chờ, hiện ngay khu vực kết quả & dải số liệu
  $("progress-panel").hidden = true;
  $("result").hidden = false;
  $("insights").hidden = false;
  $("events-card").hidden = false;
  $("download-link").hidden = true;

  const video = $("result-video");
  const image = $("result-image");
  const liveImg = $("live-stream");

  video.hidden = true;
  video.pause();
  image.hidden = true;

  // Gán stream_url MỘT LẦN DUY NHẤT để tránh browser ngắt/nối lại MJPEG
  if (state.activeStreamJobId !== job.id) {
    state.activeStreamJobId = job.id;
    liveImg.hidden = false;
    liveImg.src = job.stream_url || `/api/jobs/${job.id}/stream`;
  } else if (liveImg.getAttribute("src")) {
    liveImg.hidden = false;
  }

  // Dải điều khiển & tiến độ trực tiếp
  const liveBar = $("live-bar");
  if (liveBar) {
    liveBar.hidden = false;
    const statusText = $("live-status-text");
    if (statusText) {
      statusText.textContent = stage === "finalizing"
        ? "Đang ghép âm thanh vào video…"
        : "Đang xử lý & phát trực tiếp";
    }

    const liveFill = $("live-fill");
    if (liveFill) {
      if (p.percent == null) {
        liveFill.className = "track-fill indeterminate";
      } else {
        liveFill.className = "track-fill";
        liveFill.style.width = `${Math.min(100, Math.floor(p.percent))}%`;
      }
    }

    const liveDetail = $("live-detail");
    if (liveDetail) {
      const parts = [];
      if (p.frames_done != null && p.frame_count) parts.push(`Khung ${p.frames_done}/${p.frame_count}`);
      if (p.percent != null) parts.push(`${Math.floor(p.percent)}%`);
      if (p.fps) parts.push(`${p.fps.toFixed(1)} fps`);
      if (p.eta_sec != null) parts.push(`còn ~${Math.ceil(p.eta_sec)}s`);
      if (strideVal) parts.push(`Mỗi ${strideVal} frame đoán 1 lần`);
      liveDetail.textContent = parts.join(" · ");
    }
  }

  // Gọi đồng thời cả 2 API để cập nhật dần dữ liệu ngay trong lúc đang xử lý
  const [events, detections] = await Promise.all([
    api(`/api/jobs/${job.id}/voice-events`).catch(() => []),
    api(`/api/jobs/${job.id}/detections`).catch(() => []),
  ]);
  if (state.jobId !== job.id) return;

  state.events = events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  state.detections = detections;

  // Cập nhật độ dài video dự tính
  state.duration = p.duration_sec || (video.duration > 0 ? video.duration : 0);
  if (state.duration) {
    $("road-end").textContent = formatTime(state.duration);
  }

  // Cập nhật đầu đọc playhead trực tiếp theo timestamp hiện tại của frame đang xử lý
  const curSec = p.current_timestamp_sec || 0;
  if (state.duration > 0) {
    const playhead = $("road") ? $("road").querySelector(".playhead") : null;
    if (playhead) {
      playhead.style.setProperty("--p", Math.min(1, curSec / state.duration));
    }
  }

  // Phát hiện và xử lý các sự kiện cảnh báo giọng nói mới xuất hiện
  const isInitialFetch = (state.eventsInitializedForJob !== job.id);
  const canPlay = (state.autoPlayLiveJobId === job.id || !isInitialFetch);

  state.events.forEach((ev) => {
    if (!state.seenEventIds.has(ev.id)) {
      state.seenEventIds.add(ev.id);
      if (canPlay) {
        showLiveCaption(ev.spoken_text);
        state.audioQueue.push(ev);
        playNextAudio();
      }
    }
  });

  if (isInitialFetch) {
    state.eventsInitializedForJob = job.id;
  }

  renderHud(job);
  renderEvents(true);
  renderInsights();
  drawMarkers();
}

/** Hiện câu cảnh báo trên bảng LED trong 3 giây rồi dọn sạch. */
function showLiveCaption(text) {
  const caption = $("caption");
  if (!caption) return;
  caption.hidden = false;
  caption.textContent = text;
  clearTimeout(state.captionTimer);
  state.captionTimer = setTimeout(() => {
    if (state.activeStreamJobId) {
      caption.hidden = true;
      caption.textContent = "";
    }
  }, CAPTION_SECONDS * 1000);
}

/** Chuyển sang kết quả hoàn chỉnh sau khi xử lý xong (done). */
async function renderResult(job) {
  state.autoPlayLiveJobId = null;
  clearTimeout(state.captionTimer);
  state.captionTimer = null;
  const playhead = $("road") ? $("road").querySelector(".playhead") : null;
  if (playhead) playhead.style.setProperty("--p", 0);

  teardownLiveStream();
  stopLiveAudio();
  if ($("live-bar")) $("live-bar").hidden = true;

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
  const signCount = job.unique_sign_count ?? p.unique_sign_count ?? state.events.length;
  const voiceCount = job.voice_play_count ?? p.voice_play_count ?? state.events.length;
  items.push(el("li", {}, el("b", {}, String(signCount)), " biển báo"));
  items.push(el("li", {}, el("b", {}, String(voiceCount)), " lần đọc cảnh báo"));
  $("hud").replaceChildren(...items);
}

function seekTo(sec) {
  if (state.activeStreamJobId) return;   // Luồng trực tiếp MJPEG không tua được cho tới khi done
  const video = $("result-video");
  if (!video.src) return;
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
  const videoFps = p.video_fps || (job.input_type === "video" && state.duration && p.frame_count ? p.frame_count / state.duration : null);
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
  const p = job.progress || {};
  const total = job.total_time_ms ?? p.elapsed_ms ?? 0;
  const layer = job.layer_time_ms ?? p.layer_time_ms ?? 0;
  const tts = job.tts_time_ms ?? p.tts_time_ms ?? 0;
  const detect = Math.max(0, total - layer - tts);
  const parts = [["c-detect", "Nhận diện", detect], ["c-layer", "Vẽ khung", layer], ["c-tts", "Giọng nói", tts]];
  const isLive = job.status !== "done";

  box.append(el("div", {},
    el("div", { class: "big" }, total >= 1000 ? (total / 1000).toFixed(1) : String(Math.round(total)),
      el("small", {}, total >= 1000 ? "giây" : "ms")),
    el("p", { class: "caption-sm" }, isLive ? "thời gian đã chạy" : "tổng thời gian xử lý")));
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
  const p = job.progress || {};
  const duration = state.duration || p.duration_sec || Math.max(0, ...dets.map((d) => d.timestamp_sec)) || 1;
  const bins = new Array(DENSITY_BINS).fill(0);
  for (const d of dets) bins[Math.min(DENSITY_BINS - 1, Math.floor((d.timestamp_sec / duration) * DENSITY_BINS))] += 1;
  const max = Math.max(1, ...bins);
  const framesWithSign = new Set(dets.map((d) => d.frame_index)).size;
  const frameCount = p.frame_count;

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

  const p = (state.job || {}).progress || {};
  const duration = (Number.isFinite(video.duration) && video.duration > 0)
    ? video.duration
    : (p.duration_sec || state.duration);

  if (!Number.isFinite(duration) || duration <= 0) return;
  state.duration = duration;
  $("road-end").textContent = formatTime(duration);

  state.events.forEach((ev, i) => {
    const lane = road.querySelector(`.lane[data-lane="${laneOf(ev)}"]`);
    if (!lane) return;
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
  const liveImg = $("live-stream");

  // Xử lý khi luồng MJPEG gặp lỗi mạng hoặc đứt kết nối để tránh hiện icon ảnh vỡ
  if (liveImg) {
    liveImg.addEventListener("error", () => {
      if (state.activeStreamJobId === state.jobId) {
        console.warn("Luồng MJPEG stream bị ngắt hoặc lỗi tải ảnh.");
        liveImg.hidden = true;
        liveImg.removeAttribute("src");
      }
    });
  }

  video.addEventListener("loadedmetadata", () => { drawMarkers(); renderInsights(); });

  video.addEventListener("timeupdate", () => {
    const t = video.currentTime;
    if (state.duration) road.querySelector(".playhead").style.setProperty("--p", Math.min(1, t / state.duration));

    // Câu cảnh báo gần nhất đã phát trong CAPTION_SECONDS giây vừa qua
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

  // Bấm vào bất kỳ đâu trên dải đường để tua tới thời điểm tương ứng
  road.addEventListener("click", (e) => {
    const lane = road.querySelector(".lane").getBoundingClientRect();
    if (!state.duration || e.clientX < lane.left) return;
    seekTo(((e.clientX - lane.left) / lane.width) * state.duration + 0.3);
  });

  // Gắn sự kiện nút âm thanh trực tiếp
  const liveAudioBtn = $("live-audio-btn");
  if (liveAudioBtn) {
    liveAudioBtn.addEventListener("click", toggleLiveAudio);
  }
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
