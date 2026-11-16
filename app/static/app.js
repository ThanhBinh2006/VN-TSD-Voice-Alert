/* Giao diện web TSD Voice Alert (Bản đồng bộ mượt mà 30 FPS + Bộ đệm 5s + Đồng bộ tuyệt đối Giọng nói & Bounding Box).
   Phụ trách: Antigravity Flash
   JavaScript thuần, an toàn bộ nhớ & DOM, không thư viện bên ngoài.
   Hỗ trợ:
     - Video HTML5 mượt mà (30/60 FPS) giải mã phần cứng Native
     - Bounding box vẽ chính xác qua Canvas Overlay (OverlayCanvasRenderer)
     - Bộ đệm 5 giây (buffer_ahead_sec = 5.0, mode = buffered) loại bỏ hiện tượng giật cục
     - Đồng bộ tuyệt đối: Giọng nói và Bounding Box xuất hiện cùng 1 thời điểm khi video chạy đến biển báo
     - Thanh kéo tua video (Scrubber) đồng bộ dải đường và đệm 5s khi tua
     - Tốc độ phát video linh hoạt (0.25× - 1.5×, mặc định 1.0×)
     - Tốc độ đọc giọng nói linh hoạt (1.0× - 1.8×, mặc định 1.4×) với preservesPitch
     - Tự động ngắt giọng nói khi xe đi qua biển báo (cutoff on pass)
     - Kế thừa luồng live stream /stream
   Các API chính trong app/main.py:
     GET /api/health, POST /api/jobs, GET /api/jobs, GET /api/jobs/<id>,
     GET /api/jobs/<id>/voice-events, GET /api/jobs/<id>/detections,
     GET /api/jobs/<id>/stream, GET /api/jobs/<id>/events,
     POST /api/jobs/<id>/playback, POST /api/jobs/<id>/cancel
*/
"use strict";

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

const MODEL_LABEL = { yolo: "YOLO26m", faster_rcnn: "Faster R-CNN" };
const POSITION_LABEL = { left: "Bên trái", front: "Phía trước", right: "Bên phải" };
const STATUS_LABEL = {
  queued: "Đang chờ",
  processing: "Đang xử lý",
  done: "Xong",
  failed: "Lỗi",
  cancelled: "Đã hủy",
  stopped: "Đã dừng",
};
const STAGE_LABEL = {
  queued: "Đang chờ tới lượt",
  loading_model: "Đang nạp mô hình",
  processing: "Đang nhận diện biển báo",
  finalizing: "Đang hoàn tất xử lý",
  done: "Đã xử lý xong",
  failed: "Xử lý thất bại",
};

const CAPTION_SECONDS = 3;
const POLL_MS_QUEUE = 750;
const POLL_MS_LEGACY_LIVE = 400;
const DENSITY_BINS = 30;
const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.4 6z" fill="currentColor"/></svg>';

/* ------------------------------------------------------------------ trạng thái toàn cục */

const state = {
  jobId: null,
  job: null,
  pollTimer: null,
  pollInFlight: false,
  eventsSource: null,
  events: [],
  detections: [],
  duration: 0,
  maxUploadMb: 200,
  defaultFrameStride: 30,

  // Cơ chế đệm trước 5 giây
  bufferAheadSec: 5.0,
  bufferMaxTimestamp: 0,
  isBuffering: false,
  bufferSafetyTimer: null,
  targetPlayTime: 0,
  pendingAlerts: [], // { id, alert_id, track_id, spoken_text, timestamp_sec, position, audio_url, played }
  isScrubbing: false,

  // Tốc độ phát video & âm thanh video gốc
  videoPlaybackRate: 1.0,
  userCustomVideoSpeed: false,
  videoMuted: false,

  // Tốc độ giọng nói và chế độ ngắt khi qua biển
  voicePlaybackRate: 1.4,
  userCustomSpeed: false,
  cutoffOnPass: true,
  userCustomCutoff: false,
  cutoffGraceSec: 0.25,

  // Quản lý theo dõi track để ngắt khi qua biển
  videoTrackState: {
    lostTracks: new Map(), // track_id -> { lostTimestamp, generation }
    seenActiveTracks: new Map(), // track_id -> lastSeenMediaTime
    lastOverlayTime: null,
  },

  // Đồng bộ phát lại & overlay canvas
  overlayRenderer: null,
  playbackSyncAbort: null,
  playbackSyncTimer: null,
  lastPlaybackSyncTime: 0,
  activePlaybackMode: "buffered", // "buffered" | "realtime"

  // Luồng âm thanh & caption
  liveAudioEnabled: true,
  liveAudioBlocked: false,
  audioUnlocked: false,
  audioCtx: null,
  audioQueue: [],
  currentAudio: null,
  currentAudioMeta: null, // { track_id, timestamp_sec, alert_id }
  seenAlertIds: new Set(),
  interruptedAlertIds: new Set(),
  manualAudio: null,
  captionTimer: null,
  liveCaptionUntil: 0,

  // Legacy stream
  activeStreamJobId: null,
};

/* ------------------------------------------------------------------ tiện ích */

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Lỗi ${res.status}`);
  return body;
}

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
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

function displayName(job) {
  return (job.input_name || "").replace(/^[0-9a-f]{8}_/, "") || `Job ${job.id}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child);
  }
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

/* ------------------------------------------------------------------ điều khiển đệm 5s */

function setBufferingState(isBuffering, text) {
  clearTimeout(state.bufferSafetyTimer);
  state.bufferSafetyTimer = null;
  state.isBuffering = isBuffering;

  const notice = $("buffer-notice");
  const noticeText = $("buffer-text");
  if (!notice) return;

  if (isBuffering) {
    notice.hidden = false;
    if (noticeText) noticeText.textContent = text || `Đang đệm dữ liệu video…`;

    // Cơ chế an toàn (Safety fallback): sau tối đa 2.2 giây tự giải phóng spinner nếu trễ mạng hoặc tới cuối video
    state.bufferSafetyTimer = setTimeout(() => {
      if (state.isBuffering) {
        state.isBuffering = false;
        notice.hidden = true;
        const video = $("result-video");
        if (video && video.paused) {
          video.play().catch(() => {});
        }
      }
    }, 2200);
  } else {
    notice.hidden = true;
  }
}

function checkBufferReady() {
  if (!state.isBuffering) return;
  const video = $("result-video");
  const duration = (video && Number.isFinite(video.duration) && video.duration > 0)
    ? video.duration
    : (state.duration || 0);
  const curTime = state.targetPlayTime || (video ? video.currentTime : 0);
  const targetBuffer = duration > 0
    ? Math.min(duration, curTime + state.bufferAheadSec)
    : (curTime + state.bufferAheadSec);
  const isDone = (state.job && state.job.status === "done") || !state.eventsSource;
  const hasEnoughBuffer = state.bufferMaxTimestamp >= targetBuffer - 0.25;

  if (hasEnoughBuffer || isDone) {
    setBufferingState(false);
    if (video && video.paused) {
      video.play().catch((err) => {
        console.warn("Autoplay video sau khi đệm bị chặn:", err);
      });
    }
  }
}

/* ------------------------------------------------------------------ điều khiển tốc độ phát video & âm thanh video */

function setVideoMuted(muted) {
  state.videoMuted = Boolean(muted);
  const video = $("result-video");
  if (video && video.muted !== state.videoMuted) {
    video.muted = state.videoMuted;
  }

  const btn = $("video-mute-btn");
  if (btn) {
    btn.setAttribute("aria-pressed", state.videoMuted ? "true" : "false");
    btn.classList.toggle("muted", state.videoMuted);
    const iconOn = btn.querySelector(".audio-icon.on");
    const iconOff = btn.querySelector(".audio-icon.off");
    if (iconOn && iconOff) {
      iconOn.hidden = state.videoMuted;
      iconOff.hidden = !state.videoMuted;
    }
    const label = $("video-mute-label");
    if (label) {
      label.textContent = state.videoMuted ? "Âm video: Tắt" : "Âm video";
    }
  }

  const liveBtn = $("live-video-mute-btn");
  if (liveBtn) {
    liveBtn.setAttribute("aria-pressed", state.videoMuted ? "true" : "false");
    liveBtn.classList.toggle("muted", state.videoMuted);
    const span = liveBtn.querySelector("span");
    if (span) {
      span.textContent = state.videoMuted ? "Âm video: Tắt" : "Âm video: Bật";
    }
  }

  const muteCheck = $("video-mute-default");
  if (muteCheck && muteCheck.checked !== state.videoMuted) {
    muteCheck.checked = state.videoMuted;
  }
}

function toggleVideoMuted() {
  setVideoMuted(!state.videoMuted);
}

function applyVideoPlaybackRate(videoEl, rate = state.videoPlaybackRate) {
  if (!videoEl) return;
  const r = Math.min(4.0, Math.max(0.1, Number(rate) || 1.0));
  try {
    videoEl.defaultPlaybackRate = r;
    videoEl.playbackRate = r;
    videoEl.preservesPitch = true;
    videoEl.mozPreservesPitch = true;
    videoEl.webkitPreservesPitch = true;
  } catch (_) { }
}

function updateVideoSpeedButtons(rate) {
  const r = Number(rate) || 1.0;
  const group = $("video-speed-group");
  if (group) {
    group.querySelectorAll(".video-speed-btn").forEach((btn) => {
      const btnRate = parseFloat(btn.dataset.rate || "1.0");
      const isActive = Math.abs(btnRate - r) < 0.05;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }
  const defSel = $("video-speed-default");
  if (defSel) {
    for (const opt of defSel.options) {
      if (Math.abs(parseFloat(opt.value) - r) < 0.05) {
        defSel.value = opt.value;
        break;
      }
    }
  }
}

function syncVideoSpeedBarVisibility() {
  const bar = $("video-speed-bar");
  if (!bar) return;
  const result = $("result");
  const video = $("result-video");
  const isResultVisible = result && !result.hidden;
  const isVideoVisible = video && !video.hidden;
  const isStreamActive = Boolean(state.activeStreamJobId);
  const shouldShow = isResultVisible && isVideoVisible && !isStreamActive;
  bar.hidden = !shouldShow;
}

function setVideoPlaybackRate(rate, syncBackend = true) {
  const r = Math.min(4.0, Math.max(0.1, Number(rate) || 1.0));
  state.videoPlaybackRate = r;
  state.userCustomVideoSpeed = true;
  updateVideoSpeedButtons(r);

  const video = $("result-video");
  const prevRate = video ? (video.playbackRate || 1.0) : null;
  if (video) {
    applyVideoPlaybackRate(video, r);
  }

  const handledByRateChange = Boolean(
    video &&
    prevRate !== null &&
    Math.abs((video.playbackRate || 1.0) - prevRate) > 1e-4 &&
    state.playbackSyncAbort
  );
  if (syncBackend && !handledByRateChange && state.jobId && state.job && (state.job.status === "processing" || state.job.status === "queued")) {
    fetch(`/api/jobs/${state.jobId}/playback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playback_rate: r }),
    }).catch(() => {});
  }
}

/* ------------------------------------------------------------------ điều khiển tốc độ & tự ngắt khi qua biển */

function updateSpeedSelects(rate) {
  for (const id of ["live-speed-select", "camera-speed-select"]) {
    const sel = $(id);
    if (!sel) continue;
    for (const opt of sel.options) {
      if (Math.abs(Number(opt.value) - rate) < 0.05) {
        sel.value = opt.value;
        break;
      }
    }
  }
}

function setVoicePlaybackRate(rate) {
  const numRate = Number(rate) || 1.4;
  state.voicePlaybackRate = numRate;
  state.userCustomSpeed = true;
  updateSpeedSelects(numRate);

  if (state.currentAudio) {
    try { state.currentAudio.playbackRate = numRate; } catch (_) { }
  }
}

function updateCutoffBtns(enabled) {
  for (const id of ["live-cutoff-btn", "camera-cutoff-btn"]) {
    const btn = $(id);
    if (!btn) continue;
    btn.setAttribute("aria-pressed", enabled ? "true" : "false");
    btn.classList.toggle("active", enabled);
    const span = btn.querySelector("span");
    if (span) {
      span.textContent = enabled ? "Qua biển: Tự ngắt" : "Qua biển: Đọc hết";
    }
  }
}

function toggleCutoffOnPass() {
  state.cutoffOnPass = !state.cutoffOnPass;
  state.userCustomCutoff = true;
  updateCutoffBtns(state.cutoffOnPass);

  if (state.cutoffOnPass) {
    const video = $("result-video");
    const mediaTime = video ? (video.currentTime || 0) : 0;
    pruneAndInterruptVideoAudio(mediaTime);
  }
}

function resetVideoTrackState() {
  state.videoTrackState = {
    lostTracks: new Map(),
    seenActiveTracks: new Map(),
    lastOverlayTime: null,
  };
}

/* ------------------------------------------------------------------ âm thanh & mở khóa autoplay */

function unlockAudio() {
  state.audioUnlocked = true;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      if (!state.audioCtx) state.audioCtx = new AudioCtx();
      if (state.audioCtx.state === "suspended") {
        state.audioCtx.resume().catch(() => { });
      }
    }
    const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");
    silent.volume = 0;
    const p = silent.play();
    if (p && p.then) p.catch(() => { });
  } catch (_) { }
}

function applyAudioPlaybackRate(audio) {
  if (!audio) return;
  const rate = state.voicePlaybackRate || 1.4;
  audio.playbackRate = rate;
  audio.preservesPitch = true;
  audio.mozPreservesPitch = true;
  audio.webkitPreservesPitch = true;
}

/* ------------------------------------------------------------------ logic ngắt khi qua biển (Video) */

function isVideoTrackPassed(trackId, timestampSec, mediaTime, bestKf, visibleTrackIds) {
  if (!state.cutoffOnPass) return false;

  // Nếu không có track_id: fallback kiểm tra trôi quá 3.5s
  if (trackId == null) {
    const video = $("result-video");
    return Boolean(video && !video.paused && (mediaTime - (timestampSec || 0) > 3.5));
  }

  // Điều kiện a: track_lost event đã đến từ backend
  if (state.videoTrackState.lostTracks.has(trackId)) {
    const lostEntry = state.videoTrackState.lostTracks.get(trackId);
    const genMatch = (lostEntry.generation == null || !state.overlayRenderer || lostEntry.generation === state.overlayRenderer.generation);
    if (genMatch) {
      if (lostEntry.lostTimestamp == null || mediaTime >= lostEntry.lostTimestamp) {
        return true;
      }
    }
  }

  // Điều kiện b: track đã từng nhìn thấy nhưng biến mất khỏi frame hiển thị quá cutoffGraceSec
  if (state.videoTrackState.seenActiveTracks.has(trackId)) {
    const kf = bestKf || (state.overlayRenderer ? state.overlayRenderer.lastRenderedKf : null);
    const vIds = visibleTrackIds || (state.overlayRenderer ? state.overlayRenderer.lastVisibleTrackIds : null);
    if (kf && Math.abs(mediaTime - kf.source_timestamp) <= 0.8) {
      if (vIds && !vIds.has(trackId)) {
        const lastSeen = state.videoTrackState.seenActiveTracks.get(trackId);
        if (mediaTime - lastSeen >= state.cutoffGraceSec) {
          return true;
        }
      }
    }
  }

  return false;
}

function pruneAndInterruptVideoAudio(mediaTime) {
  if (!state.cutoffOnPass) return;

  const bestKf = state.overlayRenderer ? state.overlayRenderer.lastRenderedKf : null;
  const visibleTrackIds = state.overlayRenderer ? state.overlayRenderer.lastVisibleTrackIds : null;

  // 1. Lọc bỏ các câu trong hàng đợi đã bị qua biển
  if (state.audioQueue.length > 0) {
    state.audioQueue = state.audioQueue.filter((item) => {
      const passed = isVideoTrackPassed(item.track_id, item.timestamp_sec, mediaTime, bestKf, visibleTrackIds);
      return !passed;
    });
  }

  // 2. Ngắt câu đang phát nếu track tương ứng đã qua
  if (state.currentAudio && state.currentAudioMeta) {
    const passed = isVideoTrackPassed(
      state.currentAudioMeta.track_id,
      state.currentAudioMeta.timestamp_sec,
      mediaTime,
      bestKf,
      visibleTrackIds
    );
    if (passed) {
      interruptCurrentVideoAudio();
    }
  }
}

function interruptCurrentVideoAudio() {
  const audio = state.currentAudio;
  if (state.currentAudioMeta && state.currentAudioMeta.alert_id != null) {
    state.interruptedAlertIds.add(state.currentAudioMeta.alert_id);
  }
  state.currentAudio = null;
  state.currentAudioMeta = null;
  state.liveCaptionUntil = 0;

  const caption = $("caption");
  if (caption) {
    caption.hidden = true;
    caption.textContent = "";
  }

  if (audio) {
    audio.onended = null;
    audio.onerror = null;
    try {
      audio.pause();
      audio.removeAttribute("src");
    } catch (_) { }
  }

  playNextAudio();
}

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

  const video = $("result-video");
  const mediaTime = video && !video.hidden ? (video.currentTime || 0) : (ev.timestamp_sec || 0);
  if (isVideoTrackPassed(ev.track_id, ev.timestamp_sec, mediaTime)) {
    playNextAudio();
    return;
  }

  const audio = new Audio(ev.audio_url);
  applyAudioPlaybackRate(audio);

  state.currentAudio = audio;
  state.currentAudioMeta = {
    track_id: ev.track_id,
    timestamp_sec: ev.timestamp_sec,
    alert_id: ev.id,
  };

  let finalized = false;
  const finish = () => {
    if (finalized) return;
    finalized = true;
    audio.onended = null;
    audio.onerror = null;
    if (state.currentAudio === audio) {
      state.currentAudio = null;
      state.currentAudioMeta = null;
    }
    playNextAudio();
  };

  audio.onended = finish;
  audio.onerror = finish;

  audio.play().then(() => {
    if (state.currentAudio === audio) {
      showLiveCaption(ev.spoken_text);
    }
  }).catch((err) => {
    const isBlocked = err && (
      err.name === "NotAllowedError" ||
      err.name === "AbortError" ||
      /play\(\) failed because the user didn't interact/i.test(err.message || "")
    );
    if (isBlocked) {
      console.warn("Autoplay audio bị chặn:", err);
      state.liveAudioBlocked = true;
      state.audioQueue.unshift(ev);
      if (state.audioQueue.length > 8) {
        state.audioQueue = state.audioQueue.slice(-8);
      }
      finalized = true;
      audio.onended = null;
      audio.onerror = null;
      state.currentAudio = null;
      state.currentAudioMeta = null;
      const caption = $("caption");
      if (caption) {
        caption.hidden = true;
        caption.textContent = "";
      }
      updateLiveAudioBtn(true, true);
      return;
    }
    console.warn("Lỗi phát audio:", err);
    finish();
  });
}

function stopLiveAudio() {
  if (state.currentAudio) {
    state.currentAudio.onended = null;
    state.currentAudio.onerror = null;
    try {
      state.currentAudio.pause();
      state.currentAudio.removeAttribute("src");
    } catch (_) { }
    state.currentAudio = null;
    state.currentAudioMeta = null;
  }
  state.audioQueue = [];
  state.liveAudioBlocked = false;
}

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

function showLiveCaption(text) {
  const caption = $("caption");
  if (!caption || !text) return;
  caption.hidden = false;
  caption.textContent = text;
  state.liveCaptionUntil = Date.now() + CAPTION_SECONDS * 1000;
  clearTimeout(state.captionTimer);
  state.captionTimer = setTimeout(() => {
    if (Date.now() >= state.liveCaptionUntil) {
      caption.hidden = true;
      caption.textContent = "";
    }
  }, CAPTION_SECONDS * 1000);
}

/* ------------------------------------------------------------------ bộ vẽ Canvas Overlay đồng bộ */

class OverlayCanvasRenderer {
  constructor(videoElement, canvasElement, options = {}) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d", { alpha: true });
    this.isLive = Boolean(options.isLive);
    this.onRender = typeof options.onRender === "function" ? options.onRender : null;
    this.keyframes = [];
    this.generation = 0;
    this.active = false;
    this.rvfcHandle = null;
    this.rafHandle = null;
    this.resizeObserver = null;
    this.lastRenderedKf = null;
    this.lastVisibleTrackIds = null;

    this.onFrame = this.onFrame.bind(this);
    this.initResizeObserver();
  }

  initResizeObserver() {
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.updateDimensions();
        this.renderCurrent();
      });
      if (this.canvas.parentElement) {
        this.resizeObserver.observe(this.canvas.parentElement);
      }
    }
  }

  start() {
    if (this.active) return;
    this.active = true;
    this.scheduleNextFrame();
  }

  stop() {
    this.active = false;
    if (this.rvfcHandle != null && this.video.cancelVideoFrameCallback) {
      this.video.cancelVideoFrameCallback(this.rvfcHandle);
      this.rvfcHandle = null;
    }
    if (this.rafHandle != null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    this.clear();
  }

  destroy() {
    this.stop();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    this.keyframes = [];
    this.lastRenderedKf = null;
    this.lastVisibleTrackIds = null;
  }

  setGeneration(gen) {
    if (this.generation !== gen) {
      this.generation = gen;
      this.keyframes = this.keyframes.filter((kf) => kf.generation === gen);
      this.lastRenderedKf = null;
      this.lastVisibleTrackIds = null;
      this.clear();
    }
  }

  addKeyframe(kf) {
    const owned = {
      ...kf,
      receivedAtMs: performance.now(),
    };
    if (owned.generation != null && owned.generation !== this.generation) {
      if (owned.generation > this.generation) {
        this.setGeneration(owned.generation);
      } else {
        return;
      }
    }
    this.keyframes.push(owned);
    if (this.keyframes.length > 1500) {
      this.keyframes = this.keyframes.slice(-1000);
    }
    this.keyframes.sort((a, b) => (a.source_timestamp || 0) - (b.source_timestamp || 0));
  }

  clear() {
    if (!this.ctx) return;
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

  scheduleNextFrame() {
    if (!this.active) return;
    if ("requestVideoFrameCallback" in this.video) {
      this.rvfcHandle = this.video.requestVideoFrameCallback((now, metadata) => {
        this.onFrame(metadata ? metadata.mediaTime : this.video.currentTime);
        this.scheduleNextFrame();
      });
    } else {
      this.rafHandle = requestAnimationFrame(() => {
        this.onFrame(this.video.currentTime);
        this.scheduleNextFrame();
      });
    }
  }

  updateDimensions() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.max(1, Math.round(rect.width * dpr));
    const targetH = Math.max(1, Math.round(rect.height * dpr));

    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
  }

  onFrame(mediaTime) {
    if (!this.active) return;
    this.renderAtTime(mediaTime);
  }

  renderCurrent() {
    this.renderAtTime(this.video.currentTime || 0);
  }

  renderAtTime(mediaTime) {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) {
      this.clear();
      return;
    }

    this.updateDimensions();

    const vRect = this.video.getBoundingClientRect();
    const cRect = this.canvas.getBoundingClientRect();
    if (vRect.width <= 0 || vRect.height <= 0 || cRect.width <= 0 || cRect.height <= 0) {
      this.clear();
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const ctx = this.ctx;

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.scale(dpr, dpr);

    const vx = vRect.left - cRect.left;
    const vy = vRect.top - cRect.top;
    const scale = Math.min(vRect.width / vw, vRect.height / vh);
    const offsetX = vx + (vRect.width - vw * scale) / 2;
    const offsetY = vy + (vRect.height - vh * scale) / 2;

    if (this.keyframes.length === 0) {
      ctx.restore();
      return;
    }

    let bestKf = null;
    let dt = 0;

    if (this.isLive) {
      bestKf = this.keyframes[this.keyframes.length - 1];
      const ageSec = (performance.now() - (bestKf.receivedAtMs || performance.now())) / 1000;
      dt = ageSec;
    } else {
      let minAbsDiff = Infinity;
      for (let i = this.keyframes.length - 1; i >= 0; i--) {
        const kf = this.keyframes[i];
        if (kf.generation !== this.generation) continue;
        const diff = mediaTime - kf.source_timestamp;
        const absDiff = Math.abs(diff);
        if (diff >= -0.25 && diff <= 1.5) {
          if (absDiff < minAbsDiff || (absDiff === minAbsDiff && diff >= 0)) {
            minAbsDiff = absDiff;
            bestKf = kf;
          }
        }
      }
      if (bestKf) {
        dt = mediaTime - bestKf.source_timestamp;
      }
    }

    if (!bestKf || !bestKf.detections || bestKf.detections.length === 0) {
      this.lastRenderedKf = bestKf;
      this.lastVisibleTrackIds = new Set();
      if (typeof this.onRender === "function") {
        this.onRender(bestKf, this.lastVisibleTrackIds, mediaTime);
      }
      ctx.restore();
      return;
    }

    const visibleTrackIds = new Set();

    for (const det of bestKf.detections) {
      if (det.state === "lost") continue;
      const ttl = det.ttl_sec != null ? det.ttl_sec : 1.2;

      if (this.isLive) {
        if (dt > ttl) continue;
      } else {
        if (dt < -0.25 || dt > ttl) continue;
      }

      let [x1, y1, x2, y2] = det.bbox || [det.x1, det.y1, det.x2, det.y2];

      if (dt >= 0 && dt <= Math.min(0.45, ttl)) {
        if (det.vx != null && det.vy != null && Number.isFinite(det.vx) && Number.isFinite(det.vy)) {
          x1 += det.vx * dt;
          x2 += det.vx * dt;
          y1 += det.vy * dt;
          y2 += det.vy * dt;
        }
      }

      const bx = offsetX + x1 * scale;
      const by = offsetY + y1 * scale;
      const bw = (x2 - x1) * scale;
      const bh = (y2 - y1) * scale;

      if (bw <= 0 || bh <= 0) continue;

      if (det.track_id != null) {
        visibleTrackIds.add(det.track_id);
      }

      const isConfirmed = det.state === "active" || det.state === "confirmed";
      const strokeColor = isConfirmed ? "#2E9E57" : "#FFB21E";
      const fillColor = isConfirmed ? "rgba(46, 158, 87, 0.12)" : "rgba(255, 178, 30, 0.12)";

      ctx.fillStyle = fillColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;

      const radius = 4;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, radius);
      ctx.fill();
      ctx.stroke();

      const posText = POSITION_LABEL[det.position] || det.position || "";
      const confPct = Math.round((det.confidence || 0) * 100);
      const trackStr = det.track_id != null ? `#${det.track_id}` : "";
      const labelParts = [det.label || "Biển báo"];
      if (posText) labelParts.push(posText);
      labelParts.push(`${confPct}%`);
      if (trackStr) labelParts.push(trackStr);
      const labelText = labelParts.join(" · ");

      ctx.font = "600 11px Inter, system-ui, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const badgeHeight = 18;
      const badgeWidth = textWidth + 12;
      const badgeX = Math.max(offsetX, Math.min(bx, offsetX + vw * scale - badgeWidth));
      const badgeY = by - badgeHeight - 4 >= offsetY ? by - badgeHeight - 4 : by + 4;

      ctx.fillStyle = "rgba(14, 17, 21, 0.88)";
      ctx.beginPath();
      ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, 3);
      ctx.fill();

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#F4F5F2";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, badgeX + 6, badgeY + badgeHeight / 2);
    }

    this.lastRenderedKf = bestKf;
    this.lastVisibleTrackIds = visibleTrackIds;

    if (typeof this.onRender === "function") {
      this.onRender(bestKf, visibleTrackIds, mediaTime);
    }

    ctx.restore();
  }
}

/* ------------------------------------------------------------------ đồng bộ âm thanh & box theo mediaTime */

function checkAndTriggerSynchronizedAlerts(mediaTime) {
  if (!state.liveAudioEnabled && !state.cutoffOnPass) return;

  for (let i = 0; i < state.pendingAlerts.length; i++) {
    const item = state.pendingAlerts[i];
    if (item.played) continue;

    // Video đã đến hoặc vừa chạm mốc thời gian của biển báo (sai số trong khoảng -0.08s đến +0.40s)
    const dt = mediaTime - item.timestamp_sec;
    if (dt >= -0.08 && dt <= 0.40) {
      item.played = true;

      const isPassed = isVideoTrackPassed(item.track_id, item.timestamp_sec, mediaTime);
      if (!isPassed && state.liveAudioEnabled) {
        state.audioQueue.push(item);
        playNextAudio();
      }
      showLiveCaption(item.spoken_text);
    } else if (dt > 0.40) {
      // Đã trôi qua mốc (ví dụ khi người dùng tua video nhảy cóc)
      item.played = true;
    }
  }
}

/* ------------------------------------------------------------------ giải phóng và dọn dẹp */

function teardownOverlayRenderer() {
  if (state.playbackSyncAbort) {
    state.playbackSyncAbort.abort();
    state.playbackSyncAbort = null;
  }
  if (state.overlayRenderer) {
    state.overlayRenderer.destroy();
    state.overlayRenderer = null;
  }
  clearInterval(state.playbackSyncTimer);
  state.playbackSyncTimer = null;
}

function teardownEventSource() {
  if (state.eventsSource) {
    state.eventsSource.close();
    state.eventsSource = null;
  }
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function teardownLiveStream() {
  const liveImg = $("live-stream");
  if (liveImg) {
    liveImg.hidden = true;
    liveImg.removeAttribute("src");
  }
  state.activeStreamJobId = null;
}

function clearJobArtifacts() {
  stopPolling();
  teardownEventSource();
  teardownOverlayRenderer();
  teardownLiveStream();
  stopLiveAudio();
  resetVideoTrackState();
  state.interruptedAlertIds.clear();
  state.pendingAlerts = [];
  clearTimeout(state.bufferSafetyTimer);
  state.bufferSafetyTimer = null;
  setBufferingState(false);

  if (state.manualAudio) {
    try {
      state.manualAudio.pause();
      state.manualAudio.removeAttribute("src");
    } catch (_) { }
    state.manualAudio = null;
  }

  clearTimeout(state.captionTimer);
  state.captionTimer = null;
  state.liveCaptionUntil = 0;

  const video = $("result-video");
  if (video) {
    video.pause();
    video.removeAttribute("src");
    video.defaultPlaybackRate = 1.0;
    video.playbackRate = 1.0;
    video.load();
  }

  const canvas = $("overlay-canvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

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

  const cancelBtn = $("cancel-job-btn");
  if (cancelBtn) cancelBtn.hidden = true;

  const modeBtn = $("live-mode-toggle-btn");
  if (modeBtn) modeBtn.hidden = true;

  const speedBar = $("video-speed-bar");
  if (speedBar) speedBar.hidden = true;

  const scrubber = $("video-scrubber");
  if (scrubber) scrubber.value = "0";
  const curText = $("current-time-text");
  if (curText) curText.textContent = "0:00";
  const durText = $("duration-time-text");
  if (durText) durText.textContent = "0:00";

  const playhead = $("road") ? $("road").querySelector(".playhead") : null;
  if (playhead) playhead.style.setProperty("--p", 0);

  const muteCheck = $("video-mute-default");
  setVideoMuted(muteCheck ? muteCheck.checked : false);
}

/* ------------------------------------------------------------------ trạng thái server */

async function loadHealth() {
  const chip = $("server-status");
  const label = chip ? chip.querySelector("span") : null;
  try {
    const h = await api("/api/health");
    state.maxUploadMb = h.max_upload_mb;
    const exts = (h.accept || []).map((e) => e.slice(1)).join(", ");
    const acceptHint = $("accept-hint");
    if (acceptHint) acceptHint.textContent = `Hỗ trợ ${exts}. Tối đa ${h.max_upload_mb} MB.`;
    const fileInput = $("file-input");
    if (fileInput) fileInput.setAttribute("accept", (h.accept || []).join(","));

    if (h.frame_stride != null) {
      state.defaultFrameStride = h.frame_stride;
      const strideInput = $("frame-stride-input");
      if (strideInput && !strideInput.dataset.userEdited) {
        strideInput.value = String(h.frame_stride);
      }
    }

    // Đọc cấu hình TTS từ health nếu người dùng chưa tự chỉnh
    const ttsCfg = (h.realtime && h.realtime.tts) || {};
    if (!state.userCustomSpeed && ttsCfg.playback_rate != null) {
      state.voicePlaybackRate = Number(ttsCfg.playback_rate) || 1.4;
      updateSpeedSelects(state.voicePlaybackRate);
    }
    if (!state.userCustomCutoff && ttsCfg.cutoff_on_pass != null) {
      state.cutoffOnPass = Boolean(ttsCfg.cutoff_on_pass);
      updateCutoffBtns(state.cutoffOnPass);
    }
    if (ttsCfg.cutoff_grace_sec != null) {
      state.cutoffGraceSec = Number(ttsCfg.cutoff_grace_sec) || 0.25;
    }

    // Đọc cấu hình tốc độ video từ health nếu người dùng chưa tự chỉnh
    const videoRateHealth = (h.realtime && h.realtime.video_playback_rate != null)
      ? h.realtime.video_playback_rate
      : (ttsCfg.video_playback_rate != null ? ttsCfg.video_playback_rate : null);
    if (!state.userCustomVideoSpeed && videoRateHealth != null) {
      state.videoPlaybackRate = Number(videoRateHealth) || 1.0;
      updateVideoSpeedButtons(state.videoPlaybackRate);
    }

    if (label && chip) {
      if (h.fake) {
        label.textContent = "Chế độ thử: dữ liệu giả";
        chip.className = "status-chip fake";
        chip.title = `Đang dùng bản giả cho: ${(h.fake_parts || []).join(", ")}`;
      } else {
        label.textContent = "Đang dùng mô hình thật";
        chip.className = "status-chip real";
      }
    }
  } catch {
    if (label && chip) {
      label.textContent = "Không kết nối được server";
      chip.className = "status-chip down";
    }
  }
}

/* ------------------------------------------------------------------ upload và xử lý file */

function setupUpload() {
  const form = $("upload-form");
  const input = $("file-input");
  const drop = $("drop-zone");
  const strideInput = $("frame-stride-input");

  if (!form || !input || !drop) return;

  if (strideInput) {
    strideInput.addEventListener("input", () => {
      strideInput.dataset.userEdited = "true";
    });
  }

  const showFile = () => {
    const file = input.files[0];
    const fileNameEl = $("file-name");
    if (fileNameEl) {
      fileNameEl.hidden = !file;
      fileNameEl.textContent = file ? `${file.name}, ${(file.size / 1048576).toFixed(1)} MB` : "";
    }
    const formError = $("form-error");
    if (formError) formError.textContent = "";
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
    const formError = $("form-error");
    if (!file) {
      if (formError) formError.textContent = "Chọn một video hoặc ảnh trước khi bấm Bắt đầu xử lý.";
      return;
    }
    if (file.size > state.maxUploadMb * 1048576) {
      if (formError) formError.textContent = `File nặng ${(file.size / 1048576).toFixed(0)} MB, vượt giới hạn ${state.maxUploadMb} MB. Hãy cắt ngắn video rồi thử lại.`;
      return;
    }

    if (strideInput) {
      const rawStride = strideInput.value.trim();
      if (!/^\d+$/.test(rawStride)) {
        if (formError) formError.textContent = "Số frame đọc mỗi lần (frame_stride) phải là số nguyên từ 1 đến 300.";
        return;
      }
      const strideVal = parseInt(rawStride, 10);
      if (strideVal < 1 || strideVal > 300) {
        if (formError) formError.textContent = "Số frame đọc mỗi lần (frame_stride) phải là số nguyên từ 1 đến 300.";
        return;
      }
    }

    const defSpeedSel = $("video-speed-default");
    if (defSpeedSel) {
      const chosenRate = parseFloat(defSpeedSel.value);
      if (Number.isFinite(chosenRate) && chosenRate > 0) {
        state.videoPlaybackRate = chosenRate;
        state.userCustomVideoSpeed = true;
      }
    }

    const muteCheck = $("video-mute-default");
    if (muteCheck) {
      state.videoMuted = Boolean(muteCheck.checked);
    }

    unlockAudio();
    const formData = new FormData(form);
    if (strideInput && strideInput.value.trim()) {
      formData.set("frame_stride", strideInput.value.trim());
    }
    formData.set("playback_rate", String(state.videoPlaybackRate || 1.0));
    formData.set("analysis_mode", "realtime_overlay");
    formData.set("playback_mode", "buffered");
    formData.set("buffer_ahead_sec", String(state.bufferAheadSec));
    uploadFile(formData);
  });
}

function uploadFile(formData) {
  const xhr = new XMLHttpRequest();
  const button = $("submit-btn");
  const strideInput = $("frame-stride-input");
  if (button) button.disabled = true;
  const formError = $("form-error");
  if (formError) formError.textContent = "";
  const uploadProg = $("upload-progress");
  if (uploadProg) uploadProg.hidden = false;

  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((100 * e.loaded) / e.total);
    const uploadPct = $("upload-pct");
    if (uploadPct) uploadPct.textContent = `${pct}%`;
    const uploadFill = $("upload-fill");
    if (uploadFill) uploadFill.style.width = `${pct}%`;
  });

  xhr.addEventListener("loadend", () => {
    if (button) button.disabled = false;
    if (uploadProg) uploadProg.hidden = true;
    const uploadFill = $("upload-fill");
    if (uploadFill) uploadFill.style.width = "0";

    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* bỏ qua */ }

    if (xhr.status === 202) {
      const form = $("upload-form");
      if (form) form.reset();
      const fileNameEl = $("file-name");
      if (fileNameEl) fileNameEl.hidden = true;
      if (strideInput) {
        delete strideInput.dataset.userEdited;
        strideInput.value = String(state.defaultFrameStride ?? 30);
      }
      updateVideoSpeedButtons(state.videoPlaybackRate || 1.0);
      location.hash = `#/job/${body.job_id}`;
      loadHistory();
    } else if (xhr.status === 0) {
      if (formError) formError.textContent = "Mất kết nối tới server. Kiểm tra cửa sổ đang chạy python -m app.main.";
    } else {
      if (formError) formError.textContent = body.error || `Tải lên thất bại (mã ${xhr.status}).`;
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
  if (!list) return;
  list.replaceChildren();
  const histEmpty = $("history-empty");
  if (histEmpty) histEmpty.hidden = jobs.length > 0;

  for (const job of jobs) {
    const button = el("button", {
      class: "history-item",
      type: "button",
      title: displayName(job),
      onclick: () => { location.hash = `#/job/${job.id}`; },
    },
      el("span", { class: "history-name" }, displayName(job)),
      el("span", { class: "history-sub" }, `${MODEL_LABEL[job.model_name] || job.model_name}, ${formatClock(job.created_at)}`),
      el("span", { class: `badge ${job.status}` }, STATUS_LABEL[job.status] || job.status));
    if (job.id === state.jobId) button.setAttribute("aria-current", "true");
    list.append(el("li", {}, button));
  }
}

/* ------------------------------------------------------------------ chuyển đổi màn hình */

function showUploadView() {
  clearJobArtifacts();
  state.jobId = null;
  state.job = null;
  const jobView = $("job-view");
  if (jobView) jobView.hidden = true;
  const home = $("home");
  if (home) home.hidden = false;
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
    seenAlertIds: new Set(),
    interruptedAlertIds: new Set(),
    pendingAlerts: [],
    manualAudio: null,
  });
  resetVideoTrackState();

  updateLiveAudioBtn(state.liveAudioEnabled, false);
  updateCutoffBtns(state.cutoffOnPass);
  updateSpeedSelects(state.voicePlaybackRate);

  const home = $("home");
  if (home) home.hidden = true;
  const jobView = $("job-view");
  if (jobView) jobView.hidden = false;

  for (const id of ["result", "insights", "events-card", "error-panel", "download-link", "live-bar", "export-btn"]) {
    const elNode = $(id);
    if (elNode) elNode.hidden = true;
  }

  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = false;
  const jobTitle = $("job-title");
  if (jobTitle) jobTitle.textContent = `Job ${jobId}`;
  const jobMeta = $("job-meta");
  if (jobMeta) jobMeta.replaceChildren();
  const stageText = $("stage-text");
  if (stageText) stageText.textContent = "Đang tải thông tin…";
  const progPct = $("progress-pct");
  if (progPct) progPct.textContent = "";
  const progDetail = $("progress-detail");
  if (progDetail) progDetail.textContent = "";
  const progFill = $("progress-fill");
  if (progFill) progFill.className = "track-fill indeterminate";

  window.scrollTo(0, 0);
  initJobWorkflow(jobId);
}

/* ------------------------------------------------------------------ luồng xử lý job */

async function initJobWorkflow(jobId) {
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    if (state.jobId !== jobId) return;
    const stageText = $("stage-text");
    if (stageText) stageText.textContent = err.message;
    const progFill = $("progress-fill");
    if (progFill) progFill.className = "track-fill failed";
    return;
  }

  if (state.jobId !== jobId) return;
  state.job = job;
  loadHistory();

  const isVideo = job.input_type === "video";
  if (isVideo) {
    state.videoPlaybackRate = Number(job.playback_rate) || 1.0;
    updateVideoSpeedButtons(state.videoPlaybackRate);
  }

  if (!isVideo) {
    if (job.status === "done") {
      await renderResult(job);
    } else {
      pollLegacyJob(jobId);
    }
    return;
  }

  if (job.status === "done") {
    await renderRealtimeDone(job);
    return;
  }

  if (job.status === "failed") {
    renderFailed(job);
    return;
  }

  await startRealtimeVideoSession(job);
}

/* ------------------------------------------------------------------ luồng realtime video mượt mà & đệm 5s */

async function startRealtimeVideoSession(job) {
  state.job = job;
  state.activePlaybackMode = "buffered";
  state.bufferMaxTimestamp = 0;
  state.targetPlayTime = 0;
  state.pendingAlerts = [];
  resetVideoTrackState();

  const jobTitle = $("job-title");
  if (jobTitle) jobTitle.textContent = displayName(job);
  renderJobMeta(job);

  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = true;
  const result = $("result");
  if (result) result.hidden = false;
  const insights = $("insights");
  if (insights) insights.hidden = false;
  const eventsCard = $("events-card");
  if (eventsCard) eventsCard.hidden = false;
  const downloadLink = $("download-link");
  if (downloadLink) downloadLink.hidden = true;

  const video = $("result-video");
  const image = $("result-image");
  const liveImg = $("live-stream");
  const canvas = $("overlay-canvas");
  const liveBar = $("live-bar");

  if (image) image.hidden = true;
  if (liveImg) {
    liveImg.hidden = true;
    liveImg.removeAttribute("src");
  }

  // Luôn hiển thị thẻ video và canvas để phát 30/60 FPS mượt mà
  if (video) {
    video.hidden = false;
    video.pause();
  }
  if (canvas) canvas.hidden = false;
  if (liveBar) liveBar.hidden = false;

  const statusText = $("live-status-text");
  if (statusText) statusText.textContent = "Đang nhận diện & đệm video 5s mượt mà";

  const cancelBtn = $("cancel-job-btn");
  if (cancelBtn) cancelBtn.hidden = false;

  if (job.playback_rate != null) {
    state.videoPlaybackRate = Number(job.playback_rate) || 1.0;
  }
  updateVideoSpeedButtons(state.videoPlaybackRate);
  syncVideoSpeedBarVisibility();

  updateCutoffBtns(state.cutoffOnPass);
  updateSpeedSelects(state.voicePlaybackRate);

  // Nạp source video HTML5
  const sourceUrl = job.source_url || `/api/jobs/${job.id}/source`;
  if (video) {
    applyVideoPlaybackRate(video, state.videoPlaybackRate);
    if (video.getAttribute("src") !== sourceUrl) {
      video.src = sourceUrl;
      applyVideoPlaybackRate(video, state.videoPlaybackRate);
      video.load();
    }
    setVideoMuted(state.videoMuted);
  }

  // Khởi tạo Canvas Overlay Renderer
  if (video && canvas && !state.overlayRenderer) {
    state.overlayRenderer = new OverlayCanvasRenderer(video, canvas, {
      isLive: false,
      onRender: (bestKf, visibleTrackIds, mediaTime) => {
        state.videoTrackState.lastOverlayTime = bestKf ? bestKf.source_timestamp : null;
        if (visibleTrackIds) {
          for (const tid of visibleTrackIds) {
            state.videoTrackState.seenActiveTracks.set(tid, mediaTime);
            const lostEntry = state.videoTrackState.lostTracks.get(tid);
            if (lostEntry && bestKf && bestKf.source_timestamp > (lostEntry.lostTimestamp ?? 0)) {
              state.videoTrackState.lostTracks.delete(tid);
            }
          }
        }
        pruneAndInterruptVideoAudio(mediaTime);

        // ĐỒNG BỘ TUYỆT ĐỐI GIỌNG NÓI VÀ BOUNDING BOX
        checkAndTriggerSynchronizedAlerts(mediaTime);
      },
    });
    state.overlayRenderer.start();
  }

  // Bắt đầu trạng thái đệm 5s ban đầu
  setBufferingState(true, `Đang đệm ${state.bufferAheadSec}s dữ liệu video…`);

  if (video) {
    setupPlaybackSync(job.id, video);
  }

  // Gửi thông báo bắt đầu đệm 5s tới backend
  fetch(`/api/jobs/${job.id}/playback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      timestamp_sec: 0,
      seeked: true,
      buffer_ahead_sec: state.bufferAheadSec,
      mode: "buffered",
      playback_rate: state.videoPlaybackRate || 1.0,
    }),
  }).catch(() => {});

  const eventsUrl = job.events_url || `/api/jobs/${job.id}/events`;
  teardownEventSource();

  const es = new EventSource(eventsUrl);
  state.eventsSource = es;

  // Lắng nghe sự kiện FRAME
  es.addEventListener("frame", (e) => {
    try {
      const frameData = JSON.parse(e.data);
      handleSSEFrame(frameData);
      if (frameData.timestamp_sec != null) {
        state.bufferMaxTimestamp = Math.max(state.bufferMaxTimestamp, frameData.timestamp_sec);
      }
      checkBufferReady();
    } catch (_) { }
  });

  es.addEventListener("status", (e) => {
    try {
      const statusData = JSON.parse(e.data);
      handleSSEStatus(job.id, statusData);
    } catch (_) { }
  });

  es.addEventListener("overlay", (e) => {
    try {
      const overlayData = JSON.parse(e.data);
      if (state.overlayRenderer) {
        state.overlayRenderer.addKeyframe(overlayData);
      }
      if (overlayData.source_timestamp != null) {
        state.bufferMaxTimestamp = Math.max(state.bufferMaxTimestamp, overlayData.source_timestamp);
      }
      checkBufferReady();
    } catch (_) { }
  });

  es.addEventListener("alert", (e) => {
    try {
      const alertData = JSON.parse(e.data);
      handleSSEAlert(alertData);
      const ts = alertData.source_timestamp != null ? alertData.source_timestamp : (alertData.timestamp_sec || 0);
      state.bufferMaxTimestamp = Math.max(state.bufferMaxTimestamp, ts);
      checkBufferReady();
    } catch (_) { }
  });

  es.addEventListener("track_lost", (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.track_id != null) {
        if (data.generation == null || !state.overlayRenderer || data.generation === state.overlayRenderer.generation) {
          state.videoTrackState.lostTracks.set(data.track_id, {
            lostTimestamp: data.lost_timestamp != null ? data.lost_timestamp : null,
            generation: data.generation,
          });
          const curTime = video ? video.currentTime : 0;
          pruneAndInterruptVideoAudio(curTime);
        }
      }
    } catch (_) { }
  });

  es.addEventListener("telemetry", (e) => {
    try {
      const telem = JSON.parse(e.data);
      handleSSETelemetry(job.id, telem);
    } catch (_) { }
  });

  es.addEventListener("done", () => {
    setBufferingState(false);
    checkBufferReady();
    teardownEventSource();
    api(`/api/jobs/${job.id}`).then((latest) => {
      renderRealtimeDone(latest);
    }).catch(() => { });
  });

  es.onerror = () => {
    if (state.jobId === job.id && state.job && state.job.status !== "done") {
      api(`/api/jobs/${job.id}`).then((latest) => {
        if (latest.status === "done") {
          renderRealtimeDone(latest);
        } else if (latest.status === "failed" || latest.status === "cancelled" || latest.status === "stopped") {
          renderFailed(latest);
        }
      }).catch(() => { });
    }
  };

  renderHud(job);
  renderInsights();
}

function handleSSEFrame(frame) {
  if (!frame) return;

  // Cập nhật tiến trình
  if (frame.progress_pct != null) {
    const liveFill = $("live-fill");
    if (liveFill) {
      liveFill.className = "track-fill";
      liveFill.style.width = `${Math.min(100, Math.floor(frame.progress_pct))}%`;
    }
  }

  // Cập nhật telemetry / detail
  const liveDetail = $("live-detail");
  if (liveDetail) {
    const parts = [];
    if (frame.frame_index != null) parts.push(`Khung ${frame.frame_index}`);
    if (frame.fps != null) parts.push(`${frame.fps.toFixed(1)} fps`);
    if (frame.progress_pct != null) parts.push(`${Math.floor(frame.progress_pct)}%`);
    liveDetail.textContent = parts.join(" · ");
  }

  // Nạp voice_alerts vào pendingAlerts để phát đồng thời khi video chạy tới
  if (Array.isArray(frame.voice_alerts) && frame.voice_alerts.length > 0) {
    for (const alert of frame.voice_alerts) {
      handleSSEAlert({
        ...alert,
        status: "ready",
        source_timestamp: frame.timestamp_sec,
        timestamp_sec: frame.timestamp_sec,
        spoken_text: alert.text,
      });
    }
  }
}

function handleSSEStatus(jobId, data) {
  if (state.jobId !== jobId) return;
  const p = data || {};
  const status = p.status || "processing";

  if (status === "done") {
    setBufferingState(false);
    checkBufferReady();
    teardownEventSource();
    api(`/api/jobs/${jobId}`).then((latest) => {
      renderRealtimeDone(latest);
    }).catch(() => { });
    return;
  }

  if (status === "failed" || status === "cancelled" || status === "stopped") {
    setBufferingState(false);
    teardownEventSource();
    api(`/api/jobs/${jobId}`).then((latest) => {
      renderFailed(latest);
    }).catch(() => { });
    return;
  }

  const liveFill = $("live-fill");
  if (liveFill && p.frame_count && p.frame_id) {
    const pct = Math.min(100, Math.floor((p.frame_id / p.frame_count) * 100));
    liveFill.className = "track-fill";
    liveFill.style.width = `${pct}%`;
  }
}

function handleSSEAlert(ev) {
  if (ev.status !== "ready" || !ev.audio_url) {
    return;
  }

  const alertId = ev.alert_id || ev.id || `${ev.source_timestamp || ev.timestamp_sec}_${ev.spoken_text || ev.text}`;
  if (state.seenAlertIds.has(alertId)) return;
  state.seenAlertIds.add(alertId);

  if (ev.track_lost && ev.track_id != null && !state.videoTrackState.lostTracks.has(ev.track_id)) {
    state.videoTrackState.lostTracks.set(ev.track_id, {
      lostTimestamp: ev.lost_timestamp != null ? ev.lost_timestamp : null,
      generation: ev.generation != null ? ev.generation : null,
    });
  }

  const eventObj = {
    id: ev.id || alertId,
    alert_id: alertId,
    track_id: ev.track_id != null ? ev.track_id : null,
    spoken_text: ev.spoken_text || ev.text || ev.class_name || ev.label || "",
    timestamp_sec: ev.source_timestamp != null ? ev.source_timestamp : (ev.timestamp_sec || 0),
    position: ev.position || "front",
    audio_url: ev.audio_url,
    played: false,
  };

  // Lưu lại sự kiện vào state.events để hiện trên timeline và dải đường
  state.events.push(eventObj);
  state.events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);

  // Lưu vào pendingAlerts để phát ĐỒNG THỜI với Bounding Box khi video chạy tới timestamp_sec
  state.pendingAlerts.push(eventObj);

  renderEvents(true);
  drawMarkers();
  renderInsights();
}

function handleSSETelemetry(jobId, telem) {
  if (state.jobId !== jobId) return;
  if (!state.job) state.job = { id: jobId, model_name: telem.model_name || "yolo" };
  if (!state.job.progress) state.job.progress = {};
  const p = state.job.progress;

  if (telem.infer_fps != null) p.fps = telem.infer_fps;
  if (telem.inference_fps != null) p.fps = telem.inference_fps;
  if (telem.unique_sign_count != null) p.unique_sign_count = telem.unique_sign_count;
  if (telem.voice_play_count != null) p.voice_play_count = telem.voice_play_count;
  p.lag_ms = telem.lag_ms;
  p.active_tracks = telem.active_tracks;

  renderHud(state.job);

  const liveDetail = $("live-detail");
  if (liveDetail) {
    const parts = [];
    if (telem.infer_fps != null) parts.push(`Infer: ${telem.infer_fps.toFixed(1)} fps`);
    if (telem.lag_ms != null) parts.push(`Trễ: ${Math.round(telem.lag_ms)} ms`);
    if (telem.dropped_frames != null) parts.push(`Drop: ${telem.dropped_frames}`);
    if (telem.cpu_percent != null) parts.push(`CPU: ${Math.round(telem.cpu_percent)}%`);
    if (telem.rss_mb != null) parts.push(`RAM: ${Math.round(telem.rss_mb)} MB`);
    if (telem.late_alerts != null && telem.late_alerts > 0) parts.push(`Trễ TTS: ${telem.late_alerts}`);
    liveDetail.textContent = parts.join(" · ");
  }
}

function setupPlaybackSync(jobId, video) {
  clearInterval(state.playbackSyncTimer);
  if (state.playbackSyncAbort) {
    state.playbackSyncAbort.abort();
    state.playbackSyncAbort = null;
  }
  state.playbackSyncAbort = new AbortController();
  const signal = state.playbackSyncAbort.signal;

  const sendSync = (opts = {}) => {
    if (state.jobId !== jobId) return;
    if (!state.job || (state.job.status !== "processing" && state.job.status !== "queued")) return;

    const payload = {
      timestamp_sec: video.currentTime || 0,
      playing: !video.paused,
      playback_rate: video.playbackRate || 1.0,
      mode: "buffered",
      buffer_ahead_sec: state.bufferAheadSec,
      ...opts,
    };

    fetch(`/api/jobs/${jobId}/playback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }).then((res) => {
      if (res.ok) return res.json();
      return null;
    }).then((body) => {
      if (body && body.generation != null && state.overlayRenderer) {
        state.overlayRenderer.setGeneration(body.generation);
      }
    }).catch(() => { });
  };

  const onPlay = () => sendSync({ playing: true });
  const onPause = () => sendSync({ playing: false });
  const onRateChange = () => sendSync({ playback_rate: video.playbackRate || 1.0 });

  video.addEventListener("play", onPlay, { signal });
  video.addEventListener("pause", onPause, { signal });
  video.addEventListener("ratechange", onRateChange, { signal });

  state.playbackSyncTimer = setInterval(() => {
    if (!video.paused && state.job && (state.job.status === "processing" || state.job.status === "queued")) {
      const now = Date.now();
      if (now - state.lastPlaybackSyncTime >= 320) {
        state.lastPlaybackSyncTime = now;
        sendSync();
      }
    }
  }, 350);

  const cancelBtn = $("cancel-job-btn");
  if (cancelBtn) {
    cancelBtn.onclick = async () => {
      cancelBtn.disabled = true;
      cancelBtn.textContent = "Đang hủy…";
      try {
        await api(`/api/jobs/${jobId}/cancel`, { method: "POST" });
        clearJobArtifacts();
        showUploadView();
      } catch (err) {
        const liveDetail = $("live-detail");
        if (liveDetail) liveDetail.textContent = `Không thể hủy phiên: ${err.message}`;
        cancelBtn.disabled = false;
        cancelBtn.textContent = "Hủy phân tích";
      }
    };
  }
}

/* ------------------------------------------------------------------ hoàn tất job realtime */

async function renderRealtimeDone(job) {
  state.job = job;
  setBufferingState(false);
  teardownEventSource();
  teardownLiveStream();
  stopLiveAudio();

  const cancelBtn = $("cancel-job-btn");
  if (cancelBtn) cancelBtn.hidden = true;
  const liveBar = $("live-bar");
  if (liveBar) liveBar.hidden = true;

  const [events, detections] = await Promise.all([
    api(`/api/jobs/${job.id}/voice-events`).catch(() => []),
    api(`/api/jobs/${job.id}/detections`).catch(() => []),
  ]);
  if (state.jobId !== job.id) return;

  state.events = events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  state.detections = detections;

  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = true;
  const result = $("result");
  if (result) result.hidden = false;
  const insights = $("insights");
  if (insights) insights.hidden = false;
  const eventsCard = $("events-card");
  if (eventsCard) eventsCard.hidden = false;
  const jobTitle = $("job-title");
  if (jobTitle) jobTitle.textContent = displayName(job);
  renderJobMeta(job);

  const video = $("result-video");
  const image = $("result-image");
  const canvas = $("overlay-canvas");
  const downloadLink = $("download-link");

  if (image) image.hidden = true;
  if (video) video.hidden = false;
  if (canvas) canvas.hidden = false;

  if (downloadLink) {
    if (job.download_url && job.result_url) {
      downloadLink.hidden = false;
      downloadLink.href = job.download_url;
    } else {
      downloadLink.hidden = true;
    }
  }

  const sourceUrl = job.source_url || `/api/jobs/${job.id}/source`;
  if (video && video.getAttribute("src") !== sourceUrl) {
    video.src = sourceUrl;
  }

  if (video && canvas && !state.overlayRenderer) {
    state.overlayRenderer = new OverlayCanvasRenderer(video, canvas, { isLive: false });
    state.overlayRenderer.start();
  }

  if (state.overlayRenderer && state.overlayRenderer.keyframes.length === 0) {
    populateKeyframesFromDetections(detections);
  }

  const targetRate = state.videoPlaybackRate || Number(job.playback_rate) || 1.0;
  if (video) applyVideoPlaybackRate(video, targetRate);
  updateVideoSpeedButtons(targetRate);
  setVideoMuted(state.videoMuted);
  syncVideoSpeedBarVisibility();

  renderHud(job);
  renderEvents(true);
  renderInsights();
}

function populateKeyframesFromDetections(detections) {
  if (!state.overlayRenderer || !detections || detections.length === 0) return;
  const groups = new Map();
  const strideVal = (state.job && state.job.frame_stride) || 15;
  const ttlSec = Math.max(1.2, (strideVal / 25) * 1.25);

  for (const det of detections) {
    const ts = det.timestamp_sec != null ? Math.round(det.timestamp_sec * 100) / 100 : 0;
    if (!groups.has(ts)) {
      groups.set(ts, {
        source_timestamp: det.timestamp_sec || 0,
        generation: state.overlayRenderer.generation,
        detections: [],
      });
    }
    groups.get(ts).detections.push({
      label: det.label,
      class_id: det.class_id,
      position: det.position,
      confidence: det.confidence,
      bbox: det.bbox || [det.x1, det.y1, det.x2, det.y2],
      track_id: det.track_id,
      state: "confirmed",
      ttl_sec: ttlSec,
    });
  }

  for (const kf of groups.values()) {
    state.overlayRenderer.addKeyframe(kf);
  }
}

/* ------------------------------------------------------------------ luồng legacy polling */

async function pollLegacyJob(jobId) {
  if (state.jobId !== jobId || state.pollInFlight) return;
  state.pollInFlight = true;

  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    state.pollInFlight = false;
    if (state.jobId !== jobId) return;
    const stageText = $("stage-text");
    if (stageText) stageText.textContent = err.message;
    const progFill = $("progress-fill");
    if (progFill) progFill.className = "track-fill failed";
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
    state.pollInFlight = false;
    stopPolling();
    teardownLiveStream();
    stopLiveAudio();
    if ($("live-bar")) $("live-bar").hidden = true;
    await renderResult(job);
    return;
  }

  if (job.status === "failed") {
    state.pollInFlight = false;
    stopPolling();
    renderFailed(job);
    return;
  }

  if (isLiveProcessing) {
    await renderLegacyLiveProgress(job);
  } else {
    renderLegacyProgress(job);
  }

  state.pollInFlight = false;
  if (state.jobId === jobId) {
    const nextInterval = isLiveProcessing ? POLL_MS_LEGACY_LIVE : POLL_MS_QUEUE;
    state.pollTimer = setTimeout(() => pollLegacyJob(jobId), nextInterval);
  }
}

function renderLegacyProgress(job) {
  const p = job.progress || {};
  const jobTitle = $("job-title");
  if (jobTitle) jobTitle.textContent = displayName(job);
  renderJobMeta(job);
  const stageText = $("stage-text");
  if (stageText) stageText.textContent = STAGE_LABEL[p.stage] || STATUS_LABEL[job.status] || job.status;

  const fill = $("progress-fill");
  const progPct = $("progress-pct");
  if (fill) {
    if (p.percent == null && job.status !== "failed") {
      fill.className = "track-fill indeterminate";
      if (progPct) progPct.textContent = "";
    } else {
      fill.className = `track-fill ${job.status === "done" ? "done" : job.status === "failed" ? "failed" : ""}`;
      fill.style.width = `${p.percent ?? 100}%`;
      if (progPct) progPct.textContent = p.percent != null ? `${Math.floor(p.percent)}%` : "";
    }
  }

  const parts = [];
  if (p.frame_count && job.input_type === "video") parts.push(`Khung ${p.frames_done}/${p.frame_count}`);
  if (p.fps) parts.push(`${p.fps.toFixed(1)} khung/giây`);
  if (p.eta_sec != null) parts.push(`còn khoảng ${Math.ceil(p.eta_sec)} giây`);
  const progDetail = $("progress-detail");
  if (progDetail) progDetail.textContent = parts.join(", ");
}

async function renderLegacyLiveProgress(job) {
  state.job = job;
  const p = job.progress || {};
  const stage = p.stage || job.status;

  const jobTitle = $("job-title");
  if (jobTitle) jobTitle.textContent = displayName(job);
  renderJobMeta(job);

  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = true;
  const result = $("result");
  if (result) result.hidden = false;
  const insights = $("insights");
  if (insights) insights.hidden = false;
  const eventsCard = $("events-card");
  if (eventsCard) eventsCard.hidden = false;
  const downloadLink = $("download-link");
  if (downloadLink) downloadLink.hidden = true;

  const video = $("result-video");
  const image = $("result-image");
  const liveImg = $("live-stream");

  if (video) {
    video.hidden = true;
    video.pause();
  }
  if (image) image.hidden = true;
  syncVideoSpeedBarVisibility();

  if (state.activeStreamJobId !== job.id) {
    state.activeStreamJobId = job.id;
    if (liveImg) {
      liveImg.hidden = false;
      liveImg.src = job.stream_url || `/api/jobs/${job.id}/stream`;
    }
  } else if (liveImg && liveImg.getAttribute("src")) {
    liveImg.hidden = false;
  }

  const liveBar = $("live-bar");
  if (liveBar) {
    liveBar.hidden = false;
    const statusText = $("live-status-text");
    if (statusText) {
      statusText.textContent = stage === "finalizing"
        ? "Đang hoàn tất xử lý video…"
        : "Đang nhận diện & phát trực tiếp";
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
      liveDetail.textContent = parts.join(" · ");
    }
  }

  const [events, detections] = await Promise.all([
    api(`/api/jobs/${job.id}/voice-events`).catch(() => []),
    api(`/api/jobs/${job.id}/detections`).catch(() => []),
  ]);
  if (state.jobId !== job.id) return;

  state.events = events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  state.detections = detections;

  renderHud(job);
  renderEvents(true);
  renderInsights();
  drawMarkers();
}

function renderFailed(job) {
  state.pollInFlight = false;
  stopPolling();
  setBufferingState(false);
  teardownEventSource();
  teardownOverlayRenderer();
  teardownLiveStream();
  stopLiveAudio();
  if ($("live-bar")) $("live-bar").hidden = true;
  syncVideoSpeedBarVisibility();
  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = true;
  const errMsg = $("error-message");
  if (errMsg) errMsg.textContent = job.error_message || "Không rõ nguyên nhân.";
  const errPanel = $("error-panel");
  if (errPanel) errPanel.hidden = false;
}

async function renderResult(job) {
  clearTimeout(state.captionTimer);
  state.captionTimer = null;
  state.liveCaptionUntil = 0;
  setBufferingState(false);
  teardownLiveStream();
  stopLiveAudio();
  teardownOverlayRenderer();

  if ($("live-bar")) $("live-bar").hidden = true;
  if ($("cancel-job-btn")) $("cancel-job-btn").hidden = true;
  if ($("live-mode-toggle-btn")) $("live-mode-toggle-btn").hidden = true;

  const [events, detections] = await Promise.all([
    api(`/api/jobs/${job.id}/voice-events`).catch(() => []),
    api(`/api/jobs/${job.id}/detections`).catch(() => []),
  ]);
  if (state.jobId !== job.id) return;

  state.job = job;
  state.events = events.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  state.detections = detections;

  const progPanel = $("progress-panel");
  if (progPanel) progPanel.hidden = true;
  const result = $("result");
  if (result) result.hidden = false;
  const insights = $("insights");
  if (insights) insights.hidden = false;
  const eventsCard = $("events-card");
  if (eventsCard) eventsCard.hidden = false;
  const downloadLink = $("download-link");
  if (downloadLink) {
    downloadLink.hidden = !job.download_url;
    downloadLink.href = job.download_url || "#";
  }

  const video = $("result-video");
  const image = $("result-image");
  const isVideo = job.input_type === "video";
  if (video) video.hidden = !isVideo;
  if (image) image.hidden = isVideo;
  const roadWrap = $("road-wrap");
  if (roadWrap) roadWrap.hidden = !isVideo;
  const caption = $("caption");
  if (caption) caption.hidden = true;

  if (isVideo && video) {
    video.src = job.result_url || job.source_url || `/api/jobs/${job.id}/source`;
    const targetRate = state.videoPlaybackRate || Number(job.playback_rate) || 1.0;
    applyVideoPlaybackRate(video, targetRate);
    updateVideoSpeedButtons(targetRate);
    setVideoMuted(state.videoMuted);
  } else if (image) {
    if (video) video.removeAttribute("src");
    image.src = `${job.result_url}?t=${Date.now()}`;
  }
  syncVideoSpeedBarVisibility();

  renderHud(job);
  renderEvents(isVideo);
  renderInsights();
}

/* ------------------------------------------------------------------ render chi tiết */

function renderJobMeta(job) {
  const p = job.progress || {};
  const metaItems = [
    el("li", {}, MODEL_LABEL[job.model_name] || job.model_name),
    el("li", {}, job.input_type === "image" ? "Ảnh" : "Video"),
    el("li", {}, `Gửi lúc ${formatClock(job.created_at)}`),
  ];
  const strideVal = job.frame_stride || p.frame_stride;
  if (strideVal) {
    metaItems.push(el("li", { class: "num" }, `Mỗi ${strideVal} frame`));
  }
  const jobMeta = $("job-meta");
  if (jobMeta) jobMeta.replaceChildren(...metaItems);
}

function renderHud(job) {
  const p = (job && job.progress) || {};
  const items = [el("li", { class: "live" }, MODEL_LABEL[job.model_name] || job.model_name)];
  if (p.fps) items.push(el("li", {}, el("b", {}, p.fps.toFixed(1)), " khung/giây"));
  if (p.lag_ms != null) items.push(el("li", {}, el("b", {}, Math.round(p.lag_ms).toString()), " ms lag"));
  const signCount = job.unique_sign_count ?? p.unique_sign_count ?? state.events.length;
  const voiceCount = job.voice_play_count ?? p.voice_play_count ?? state.events.length;
  items.push(el("li", {}, el("b", {}, String(signCount)), " biển báo"));
  items.push(el("li", {}, el("b", {}, String(voiceCount)), " lần đọc cảnh báo"));
  const hud = $("hud");
  if (hud) hud.replaceChildren(...items);
}

function updateScrubberUI(curSec) {
  const scrubber = $("video-scrubber");
  const curText = $("current-time-text");
  const durText = $("duration-time-text");
  const duration = state.duration || 0;

  if (curText) curText.textContent = formatTime(curSec);
  if (durText && duration > 0) durText.textContent = formatTime(duration);

  if (scrubber && !state.isScrubbing && duration > 0) {
    const pct = Math.min(100, Math.max(0, (curSec / duration) * 100));
    scrubber.value = pct.toFixed(1);
  }
}

function seekTo(sec) {
  const video = $("result-video");
  if (!video) return;

  const duration = (video && Number.isFinite(video.duration) && video.duration > 0)
    ? video.duration
    : (state.duration || 999999);
  const targetSec = Math.max(0, Math.min(sec, duration));
  interruptCurrentVideoAudio();

  video.currentTime = targetSec;
  state.targetPlayTime = targetSec;

  if (state.overlayRenderer) {
    state.overlayRenderer.setGeneration(state.overlayRenderer.generation + 1);
  }
  resetVideoTrackState();

  // Khôi phục lại danh sách pendingAlerts từ thời điểm seek
  state.pendingAlerts = [];
  for (const ev of state.events) {
    if (ev.timestamp_sec >= targetSec - 0.05) {
      ev.played = false;
      state.pendingAlerts.push(ev);
    } else {
      ev.played = true;
    }
  }

  // Mốc đệm cần thiết: không vượt quá độ dài video, đệm vừa đủ ~2.0s để phát mượt ngay
  const isJobDone = (state.job && state.job.status === "done") || !state.eventsSource;
  const requiredBuffer = (duration > 0 && Number.isFinite(duration))
    ? Math.min(duration, targetSec + Math.min(state.bufferAheadSec, 2.0))
    : (targetSec + Math.min(state.bufferAheadSec, 2.0));

  // Nếu đoạn này đã được xử lý rồi (bufferMaxTimestamp đã vượt qua) hoặc job đã xong:
  const alreadyBuffered = isJobDone || (state.bufferMaxTimestamp >= requiredBuffer - 0.25);

  if (!alreadyBuffered) {
    video.pause();
    setBufferingState(true, `Đang đệm dữ liệu từ ${formatTime(targetSec)}…`);
    fetch(`/api/jobs/${state.jobId}/playback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        timestamp_sec: targetSec,
        seeked: true,
        buffer_ahead_sec: state.bufferAheadSec,
        mode: "buffered",
        playback_rate: video.playbackRate || 1.0,
      }),
    }).catch(() => {});
  } else {
    setBufferingState(false);
    video.play().catch(() => {});
  }

  updateScrubberUI(targetSec);
}

function renderEvents(isVideo) {
  const list = $("events");
  if (!list) return;
  list.replaceChildren();
  const eventsEmpty = $("events-empty");
  if (eventsEmpty) eventsEmpty.hidden = state.events.length > 0;
  list.hidden = state.events.length === 0;

  state.events.forEach((ev, i) => {
    const time = isVideo
      ? el("button", {
        class: "tl-time",
        type: "button",
        onclick: () => seekTo(ev.timestamp_sec),
        "aria-label": `Tua video tới ${formatTime(ev.timestamp_sec)}`,
      }, formatTime(ev.timestamp_sec))
      : el("span", { class: "tl-time" }, "–");

    const play = el("button", {
      class: "tl-play",
      type: "button",
      title: "Nghe câu cảnh báo",
      "aria-label": `Nghe: ${ev.spoken_text}`,
      html: PLAY_ICON,
      onclick: () => {
        if (state.manualAudio) {
          try {
            state.manualAudio.pause();
            state.manualAudio.removeAttribute("src");
          } catch (_) { }
          state.manualAudio = null;
        }
        const a = new Audio(ev.audio_url);
        applyAudioPlaybackRate(a);
        state.manualAudio = a;
        a.addEventListener("ended", () => {
          if (state.manualAudio === a) state.manualAudio = null;
        }, { once: true });
        a.addEventListener("error", () => {
          if (state.manualAudio === a) state.manualAudio = null;
        }, { once: true });
        a.play().catch(() => { });
      },
    });

    list.append(el("li", { "data-index": i },
      time,
      el("span", { class: "tl-dot", "aria-hidden": "true" }),
      el("span", { class: "tl-text" }, ev.spoken_text),
      el("span", { class: "tl-lane" }, POSITION_LABEL[laneOf(ev)] || ev.position),
      play));
  });
  if (isVideo) drawMarkers();
}

function laneOf(ev) {
  if (POSITION_LABEL[ev.position]) return ev.position;
  const text = (ev.spoken_text || "").toLowerCase();
  if (text.includes("bên trái")) return "left";
  if (text.includes("bên phải")) return "right";
  return "front";
}

function drawMarkers() {
  const video = $("result-video");
  const road = $("road");
  if (!road) return;
  road.querySelectorAll(".marker").forEach((m) => m.remove());

  const p = (state.job || {}).progress || {};
  const duration = (video && Number.isFinite(video.duration) && video.duration > 0)
    ? video.duration
    : (p.duration_sec || state.duration);

  if (!Number.isFinite(duration) || duration <= 0) return;
  state.duration = duration;
  const roadEnd = $("road-end");
  if (roadEnd) roadEnd.textContent = formatTime(duration);
  const durText = $("duration-time-text");
  if (durText) durText.textContent = formatTime(duration);

  state.events.forEach((ev, i) => {
    const lane = road.querySelector(`.lane[data-lane="${laneOf(ev)}"]`);
    if (!lane) return;
    const marker = el("button", {
      class: "marker",
      type: "button",
      "data-index": i,
      title: `${formatTime(ev.timestamp_sec)}: ${ev.spoken_text}`,
      "aria-label": `${formatTime(ev.timestamp_sec)}, ${ev.spoken_text}`,
      onclick: (e) => { e.stopPropagation(); seekTo(ev.timestamp_sec); },
    });
    marker.style.left = `${Math.min(100, (100 * ev.timestamp_sec) / duration)}%`;
    lane.append(marker);
  });
}

function renderInsights() {
  const job = state.job;
  if (!job) return;
  renderSpeed(job);
  renderPositions();
  renderTiming(job);
  renderDensity(job);
  renderHud(job);
}

function renderSpeed(job) {
  const box = $("ins-speed");
  if (!box) return;
  box.replaceChildren();
  const p = job.progress || {};
  const videoFps = p.video_fps || (job.input_type === "video" && state.duration && p.frame_count ? p.frame_count / state.duration : null);
  const ratio = p.fps && videoFps ? p.fps / videoFps : null;

  box.append(el("div", {},
    el("div", { class: "big" }, ratio != null ? `${ratio.toFixed(1)}×` : "–"),
    el("p", { class: "caption-sm" }, "so với thời gian thực")));

  const g = svg("svg", {
    viewBox: "0 0 120 64",
    role: "img",
    "aria-label": ratio ? `Xử lý nhanh gấp ${ratio.toFixed(1)} lần thời gian thực` : "Chưa đo được tỉ lệ thời gian thực",
  });
  const arc = "M10 58 A50 50 0 0 1 110 58";
  g.append(svg("path", { d: arc, class: "gauge-track", pathLength: 100 }));
  if (ratio != null) {
    const fill = Math.min(100, (ratio / 2) * 100);
    g.append(svg("path", {
      d: arc,
      class: `gauge-value${ratio < 1 ? " slow" : ""}`,
      pathLength: 100,
      "stroke-dasharray": `${fill} 100`,
    }));
  }
  g.append(svg("line", { x1: 60, y1: 2, x2: 60, y2: 14, class: "gauge-tick" }));
  box.append(g);

  const note = ratio == null ? (job.input_type === "image" ? "Ảnh tĩnh, không tính thời gian thực." : "Đang tính toán…")
    : ratio >= 1 ? `Kịp thời gian thực, ${p.fps.toFixed(1)} khung/giây.` : `Chậm hơn thời gian thực, ${p.fps.toFixed(1)} khung/giây.`;
  box.append(el("p", { class: "caption-sm" }, note));
}

function renderPositions() {
  const box = $("ins-position");
  if (!box) return;
  box.replaceChildren();
  const counts = { left: 0, front: 0, right: 0 };
  state.events.forEach((ev) => {
    const lane = laneOf(ev);
    if (counts[lane] != null) counts[lane] += 1;
  });
  const max = Math.max(1, ...Object.values(counts));

  const cx = 60, cy = 70, r = 58;
  const sectors = [["left", 160, 112], ["front", 112, 68], ["right", 68, 20]];
  const g = svg("svg", {
    viewBox: "0 0 120 80",
    role: "img",
    "aria-label": `Bên trái ${counts.left}, phía trước ${counts.front}, bên phải ${counts.right} biển báo`,
  });
  const pt = (deg, rad) => [cx + rad * Math.cos((deg * Math.PI) / 180), cy - rad * Math.sin((deg * Math.PI) / 180)];
  for (const [key, a1, a2] of sectors) {
    const [x1, y1] = pt(a1, r);
    const [x2, y2] = pt(a2, r);
    g.append(svg("path", {
      d: `M${cx} ${cy} L${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2} Z`,
      class: "fan-sector",
      "fill-opacity": (0.1 + 0.8 * (counts[key] / max)).toFixed(2),
    }));
  }
  g.append(svg("rect", { x: cx - 6, y: cy - 6, width: 12, height: 12, rx: 3, class: "fan-car" }));
  box.append(g);

  box.append(el("div", { class: "figure-row" },
    ...Object.keys(counts).map((k) => el("div", {},
      el("div", { class: "big" }, String(counts[k])),
      el("p", { class: "caption-sm" }, POSITION_LABEL[k])))));
}

function renderTiming(job) {
  const box = $("ins-timing");
  if (!box) return;
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
  const split = el("div", {
    class: "split",
    role: "img",
    "aria-label": parts.map(([, label, ms]) => `${label} ${formatMs(ms)}`).join(", "),
  });
  for (const [cls, , ms] of parts) {
    const seg = el("span", { class: cls });
    seg.style.width = `${total ? (100 * ms) / total : 0}%`;
    split.append(seg);
  }
  box.append(split);
  box.append(el("ul", { class: "legend" },
    ...parts.map(([cls, label, ms]) => el("li", {}, el("i", { class: cls }), label, el("b", {}, formatMs(ms))))));
}

function renderDensity(job) {
  const box = $("ins-density");
  if (!box) return;
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
  for (const d of dets) {
    bins[Math.min(DENSITY_BINS - 1, Math.floor(((d.timestamp_sec || 0) / duration) * DENSITY_BINS))] += 1;
  }
  const max = Math.max(1, ...bins);
  const framesWithSign = new Set(dets.map((d) => d.frame_id || d.frame_index)).size;
  const frameCount = p.frame_count;

  box.append(el("div", {},
    el("div", { class: "big" }, frameCount ? `${Math.round((100 * framesWithSign) / frameCount)}%` : String(framesWithSign)),
    el("p", { class: "caption-sm" }, `khung hình có biển, ${dets.length} lượt phát hiện`)));
  const chart = el("div", { class: "density" });
  bins.forEach((n, i) => {
    const start = (i / DENSITY_BINS) * duration;
    const bar = el("button", {
      type: "button",
      class: n ? "" : "empty",
      title: `${formatTime(start)}: ${n} lượt phát hiện`,
      "aria-label": `Tua tới ${formatTime(start)}, ${n} lượt phát hiện`,
      onclick: () => seekTo(start + 0.3),
    });
    bar.style.height = `${Math.max(4, (100 * n) / max)}%`;
    chart.append(bar);
  });
  box.append(chart);
  box.append(el("div", { class: "density-axis" }, el("span", {}, "0:00"), el("span", {}, formatTime(duration))));
}

function setupPlayer() {
  const video = $("result-video");
  const road = $("road");
  const liveImg = $("live-stream");

  if (liveImg) {
    liveImg.addEventListener("error", () => {
      if (state.activeStreamJobId === state.jobId) {
        liveImg.hidden = true;
        liveImg.removeAttribute("src");
      }
    });
  }

  if (video) {
    video.addEventListener("loadedmetadata", () => {
      drawMarkers();
      renderInsights();
      applyVideoPlaybackRate(video, state.videoPlaybackRate);
      if (state.overlayRenderer) {
        state.overlayRenderer.updateDimensions();
      }
      updateScrubberUI(video.currentTime || 0);
      setVideoMuted(state.videoMuted);
    });

    video.addEventListener("volumechange", () => {
      if (video.muted !== state.videoMuted) {
        setVideoMuted(video.muted);
      }
    });

    video.addEventListener("ratechange", () => {
      const curRate = video.playbackRate || 1.0;
      if (Math.abs(curRate - state.videoPlaybackRate) > 0.04) {
        state.videoPlaybackRate = curRate;
        state.userCustomVideoSpeed = true;
        updateVideoSpeedButtons(curRate);
      }
    });

    video.addEventListener("timeupdate", () => {
      const t = video.currentTime;
      if (state.duration && road) {
        const playhead = road.querySelector(".playhead");
        if (playhead) playhead.style.setProperty("--p", Math.min(1, t / state.duration));
      }

      updateScrubberUI(t);
      pruneAndInterruptVideoAudio(t);

      let current = -1;
      state.events.forEach((ev, i) => {
        if (ev.timestamp_sec <= t && t - ev.timestamp_sec < CAPTION_SECONDS) current = i;
      });

      const caption = $("caption");
      if (caption) {
        const now = Date.now();
        const isLiveCaptionActive = now < (state.liveCaptionUntil || 0);

        if (!isLiveCaptionActive) {
          if (current < 0) {
            caption.hidden = true;
            caption.textContent = "";
          } else {
            const activeEv = state.events[current];
            const isLiveJob = state.job && (state.job.status === "processing" || state.job.status === "queued");
            const shouldSuppress = isLiveJob && state.cutoffOnPass && (
              state.interruptedAlertIds.has(activeEv.id) ||
              isVideoTrackPassed(activeEv.track_id, activeEv.timestamp_sec, t)
            );
            if (shouldSuppress) {
              caption.hidden = true;
              caption.textContent = "";
            } else {
              caption.hidden = false;
              caption.textContent = activeEv.spoken_text;
            }
          }
        }
      }
      if (road) {
        road.querySelectorAll(".marker").forEach((m) => m.classList.toggle("active", Number(m.dataset.index) === current));
      }
      const eventsEl = $("events");
      if (eventsEl) {
        eventsEl.querySelectorAll("li").forEach((r) => r.classList.toggle("active", Number(r.dataset.index) === current));
      }
    });
  }

  // Thanh kéo Scrubber
  const scrubber = $("video-scrubber");
  if (scrubber) {
    scrubber.addEventListener("input", () => {
      state.isScrubbing = true;
      const duration = state.duration || 0;
      const curSec = (parseFloat(scrubber.value) / 100) * duration;
      const curText = $("current-time-text");
      if (curText) curText.textContent = formatTime(curSec);
    });

    scrubber.addEventListener("change", () => {
      state.isScrubbing = false;
      const duration = state.duration || 0;
      const targetSec = (parseFloat(scrubber.value) / 100) * duration;
      seekTo(targetSec);
    });
  }

  if (road) {
    road.addEventListener("click", (e) => {
      const lane = road.querySelector(".lane");
      if (!lane) return;
      const rect = lane.getBoundingClientRect();
      if (!state.duration || e.clientX < rect.left) return;
      seekTo(((e.clientX - rect.left) / rect.width) * state.duration + 0.3);
    });
  }

  const liveAudioBtn = $("live-audio-btn");
  if (liveAudioBtn) {
    liveAudioBtn.addEventListener("click", toggleLiveAudio);
  }

  const videoMuteBtn = $("video-mute-btn");
  if (videoMuteBtn) {
    videoMuteBtn.addEventListener("click", toggleVideoMuted);
  }

  const liveVideoMuteBtn = $("live-video-mute-btn");
  if (liveVideoMuteBtn) {
    liveVideoMuteBtn.addEventListener("click", toggleVideoMuted);
  }

  const liveCutoffBtn = $("live-cutoff-btn");
  if (liveCutoffBtn) {
    liveCutoffBtn.addEventListener("click", toggleCutoffOnPass);
  }

  const liveSpeedSelect = $("live-speed-select");
  if (liveSpeedSelect) {
    liveSpeedSelect.addEventListener("change", (e) => {
      setVoicePlaybackRate(e.target.value);
    });
  }

  // Khả năng tương thích cho stub camera nếu có
  const camCutoffBtn = $("camera-cutoff-btn");
  if (camCutoffBtn) {
    camCutoffBtn.addEventListener("click", toggleCutoffOnPass);
  }
  const camSpeedSelect = $("camera-speed-select");
  if (camSpeedSelect) {
    camSpeedSelect.addEventListener("change", (e) => {
      setVoicePlaybackRate(e.target.value);
    });
  }

  const speedGroup = $("video-speed-group");
  if (speedGroup) {
    speedGroup.addEventListener("click", (e) => {
      const btn = e.target.closest(".video-speed-btn");
      if (!btn) return;
      const rate = parseFloat(btn.dataset.rate || "1.0");
      setVideoPlaybackRate(rate, true);
    });
  }
}

/* ------------------------------------------------------------------ điều hướng */

function route() {
  const match = location.hash.match(/^#\/job\/(\d+)$/);
  if (match) showJobView(Number(match[1]));
  else showUploadView();
}

const newBtn = $("new-btn");
if (newBtn) {
  newBtn.addEventListener("click", () => { location.hash = "#/"; });
}
window.addEventListener("hashchange", route);

// Khởi tạo ứng dụng
setupUpload();
setupPlayer();
loadHealth();
route();
