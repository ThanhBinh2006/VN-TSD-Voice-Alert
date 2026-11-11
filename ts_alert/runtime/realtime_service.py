"""Điều phối video, camera, theo dõi biển báo và TTS bất đồng bộ."""
from __future__ import annotations

import inspect
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ts_alert.config.resolve import resolve_path
from ts_alert.postprocess import position, tts
from ts_alert.postprocess.tts_worker import AsyncTTSWorker
from ts_alert.runtime.cpu_opt import (
    prepare_runtime_detector,
    restore_cpu_tuning,
)
from ts_alert.runtime.scheduler import (
    CancellationToken,
    FrameEnvelope,
    FrameRingBuffer,
    LatestFrameSlot,
    OverlayTimelineStore,
    VideoPlaybackClock,
    decide_video_frame_action,
)
from ts_alert.runtime.tracker import (
    TrackerConfig,
    TrafficSignTracker,
    compute_flow_gray,
)
from ts_alert.types import MODEL_NAMES, POSITIONS, RuntimeAlert
from ts_alert.utils.io import load_classes

_LOG = logging.getLogger(__name__)
_TERMINAL = frozenset({"done", "failed", "cancelled", "stopped"})


def _finite(value: Any, name: str, minimum: float = 0.0) -> float:
    """Kiểm tra số hữu hạn và giới hạn dưới."""
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} phải hữu hạn và >= {minimum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    """Kiểm tra số nguyên dương, không chấp nhận bool."""
    if isinstance(value, bool):
        raise ValueError(f"{name} phải là số nguyên dương")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{name} phải là số nguyên dương")
    return result


def _build_speech_text_compat(
    label: str,
    side: str,
    classes: dict,
    cfg: dict[str, Any],
) -> str:
    """Gọi tts.build_speech_text tương thích cả chữ ký 3 và 4 tham số."""
    try:
        parameters = tuple(
            inspect.signature(tts.build_speech_text).parameters.values()
        )
    except (ValueError, TypeError):
        return tts.build_speech_text(label, side, classes)

    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    for parameter in parameters:
        if parameter.name == "cfg":
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                return tts.build_speech_text(label, side, classes, cfg=cfg)
            if parameter.kind in positional_kinds:
                return tts.build_speech_text(label, side, classes, cfg)

    if len(parameters) >= 4 and parameters[3].kind in positional_kinds:
        return tts.build_speech_text(label, side, classes, cfg)

    return tts.build_speech_text(label, side, classes)


def _optional_call(
    target: Any,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Gọi thành phần tùy chọn và cô lập lỗi ghi nhật ký."""
    if target is None:
        return None
    try:
        return getattr(target, method)(*args, **kwargs)
    except Exception:
        _LOG.exception("Không thể thực hiện %s", method)
        return None


class _SessionEventBus:
    """Phát sự kiện SSE độc lập cho nhiều reader trong cùng phiên."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: deque[tuple[int, str, dict[str, Any]]] = deque(
            maxlen=256
        )
        self._seq = 0
        self._closed = False
        self._status: dict[str, Any] = {"status": "queued"}
        self.latest_overlay: dict[str, Any] | None = None
        self.terminal_status: str | None = None

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        """Lưu bản sao sự kiện và đánh thức toàn bộ reader."""
        if event_name not in {
            "overlay",
            "alert",
            "telemetry",
            "status",
            "track_lost",
        }:
            raise ValueError(f"Sự kiện không được hỗ trợ: {event_name}")
        owned = deepcopy(payload)
        with self._condition:
            if self._closed:
                return
            self._seq += 1
            self._events.append((self._seq, event_name, owned))
            if event_name == "overlay":
                self.latest_overlay = owned
            elif event_name == "status":
                self._status = owned
            self._condition.notify_all()

    def close(self, terminal_status: str = "done") -> None:
        """Đóng bus sau khi bảo đảm có trạng thái kết thúc."""
        with self._condition:
            if self._closed:
                return
            if self._status.get("status") != terminal_status:
                self.publish(
                    "status",
                    {**self._status, "status": terminal_status},
                )
            self.terminal_status = terminal_status
            self._closed = True
            self._condition.notify_all()

    @staticmethod
    def _format(seq: int, name: str, payload: dict[str, Any]) -> str:
        """Mã hóa một bản tin SSE UTF-8."""
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return f"id: {seq}\nevent: {name}\ndata: {data}\n\n"

    def iter_sse(self, heartbeat_sec: float = 0.5) -> Iterator[str]:
        """Gửi snapshot, sự kiện mới và heartbeat tới khi bus đóng."""
        heartbeat = _finite(heartbeat_sec, "heartbeat_sec", 0.001)
        with self._condition:
            cursor = self._seq
            status = deepcopy(self._status)
            overlay = deepcopy(self.latest_overlay)

        yield self._format(cursor, "status", status)
        if overlay is not None:
            yield self._format(cursor, "overlay", overlay)

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._seq > cursor or self._closed,
                    timeout=heartbeat,
                )
                batch = [
                    (seq, name, deepcopy(payload))
                    for seq, name, payload in self._events
                    if seq > cursor
                ]
                closed = self._closed
                if batch:
                    cursor = batch[-1][0]

            for seq, name, payload in batch:
                yield self._format(seq, name, payload)
            if closed:
                return
            if not batch:
                yield ": keepalive\n\n"


@dataclass
class _Session:
    """Trạng thái dùng chung; lock bảo vệ tracker và dữ liệu phiên."""

    key: int | str
    kind: str
    model_name: str
    cfg: dict[str, Any]
    cpu_preset: str
    backend: str
    frame_stride: int
    tracker: TrafficSignTracker
    classes: dict
    lock: Any = field(default_factory=threading.RLock)
    bus: _SessionEventBus = field(default_factory=_SessionEventBus)
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    infer_slot: LatestFrameSlot = field(default_factory=LatestFrameSlot)
    ring: FrameRingBuffer = field(
        default_factory=lambda: FrameRingBuffer(maxlen=48)
    )
    timeline: OverlayTimelineStore = field(
        default_factory=lambda: OverlayTimelineStore(max_keyframes=2400)
    )
    wake: threading.Event = field(default_factory=threading.Event)
    infer_done: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    clock: VideoPlaybackClock | None = None
    tts_worker: AsyncTTSWorker | None = None
    infer_thread: threading.Thread | None = None
    decode_thread: threading.Thread | None = None
    status: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    step_fid: int = -1
    step_ts: float = 0.0
    step_gen: int = 0
    width: int = 0
    height: int = 0
    infer_busy: bool = False
    infer_count: int = 0
    infer_fps: float = 0.0
    infer_latency_ms: float = 0.0
    decoded_frames: int = 0
    skipped_frames: int = 0
    lag_ms: float = 0.0
    profile: dict[str, Any] = field(default_factory=dict)
    voice_play_count: int = 0
    unique_classes: set[int] = field(default_factory=set)
    pending_alerts: dict[str, int] = field(default_factory=dict)
    lost_track_timestamps: dict[int, float] = field(default_factory=dict)
    log_detections: bool = True
    next_frame_id: int = 0


@dataclass
class _VideoSession(_Session):
    """Trạng thái riêng của video tải lên."""

    input_path: Path = field(default_factory=Path)
    fps: float = 25.0
    frame_count: int | None = None


@dataclass
class _CameraSession(_Session):
    """Trạng thái riêng của camera trực tiếp."""


class RealtimeSessionManager:
    """Quản lý vòng đời phiên realtime và khóa suy luận theo mô hình."""

    def __init__(
        self,
        detector_provider: Callable[[str], Any],
        cfg_for: Callable[[str], dict[str, Any]],
        outputs_dir: Path | str,
        store: Any = None,
        progress_tracker: Any = None,
        synthesizer: Callable[
            [str, Path, dict[str, Any]], Path
        ] | None = None,
        max_retained_sessions: int = 16,
    ) -> None:
        """Khởi tạo bộ quản lý và bộ đo tài nguyên tùy chọn."""
        self._detector_provider = detector_provider
        self._cfg_for = cfg_for
        self._outputs_dir = Path(outputs_dir).expanduser().resolve()
        self._store = store
        self._progress_tracker = progress_tracker
        self._synthesizer = synthesizer
        self._max_retained_sessions = _positive_int(
            max_retained_sessions, "max_retained_sessions"
        )
        self._lock = threading.RLock()
        self._video_sessions: OrderedDict[int, _VideoSession] = OrderedDict()
        self._camera_sessions: OrderedDict[str, _CameraSession] = OrderedDict()
        self._model_locks: dict[str, Any] = {}
        self._closed = False
        self._process: Any = None
        self._resource_lock = threading.Lock()

        try:
            import psutil

            self._process = psutil.Process()
            self._process.cpu_percent(interval=None)
        except Exception:
            _LOG.debug("Không sử dụng được psutil", exc_info=True)

    def _configuration(
        self,
        model_name: str,
        cfg: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict, TrafficSignTracker]:
        """Sao chép cấu hình và tạo tracker phù hợp mô hình."""
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")
        config = deepcopy(self._cfg_for(model_name) if cfg is None else cfg)
        post = config.setdefault("postprocess", {})
        thresholds = post.setdefault("position", {})
        thresholds.setdefault("left_max", 0.33)
        thresholds.setdefault("right_min", 0.66)
        realtime = config.setdefault("realtime", {})
        tts_cfg = post.setdefault("tts", {})
        if (
            "tts_cache_enabled" in realtime
            and realtime["tts_cache_enabled"] is not None
        ):
            tts_cfg["cache_enabled"] = bool(realtime["tts_cache_enabled"])
        if tts_cfg.get("cache_enabled") is True and not tts_cfg.get("cache_dir"):
            tts_cfg["cache_dir"] = str(self._outputs_dir / "tts_cache")

        classes: dict = {}
        try:
            csv_path = config.get("paths", {}).get("classes_csv")
            if csv_path:
                classes = load_classes(resolve_path(csv_path))
        except Exception:
            _LOG.warning("Không đọc được danh sách biển báo", exc_info=True)

        if model_name == "faster_rcnn":
            tracker_config = TrackerConfig(
                ttl_sec=1.5,
                reid_ttl_sec=2.5,
                max_extrapolate_sec=0.8,
                max_staleness_sec=1.6,
                use_optical_flow=True,
            )
        else:
            tracker_config = TrackerConfig(
                ttl_sec=1.0,
                reid_ttl_sec=2.0,
                max_extrapolate_sec=0.65,
                max_staleness_sec=0.95,
                use_optical_flow=True,
            )
        tracker = TrafficSignTracker(
            cfg=config, config=tracker_config, classes=classes
        )
        return config, classes, tracker

    def _model_lock(self, model_name: str) -> Any:
        """Lấy khóa ổn định cho một tên mô hình."""
        with self._lock:
            return self._model_locks.setdefault(
                model_name, threading.RLock()
            )

    def _prune_locked(self) -> None:
        """Loại phiên terminal cũ; giữ nguyên toàn bộ phiên đang chạy."""
        while (
            len(self._video_sessions) + len(self._camera_sessions)
            > self._max_retained_sessions
        ):
            candidates = [
                session
                for session in (
                    *self._video_sessions.values(),
                    *self._camera_sessions.values(),
                )
                if session.finished.is_set()
            ]
            if not candidates:
                return
            oldest = min(
                candidates,
                key=lambda session: session.ended_at or session.created_at,
            )
            if oldest.kind == "video":
                self._video_sessions.pop(int(oldest.key), None)
            else:
                self._camera_sessions.pop(str(oldest.key), None)

    def _worker(
        self,
        session: _Session,
        synthesizer: Callable[..., Any] | None = None,
    ) -> AsyncTTSWorker:
        """Tạo worker với callback được ràng buộc theo phiên."""
        cfg = session.cfg.get("realtime", {})
        folder = (
            f"job_{session.key}"
            if session.kind == "video"
            else f"camera_{session.key}"
        )
        synth = synthesizer or self._synthesizer
        if synth is None:
            synth = lambda text, target, config: tts.synthesize(
                text, target, config
            )

        return AsyncTTSWorker(
            cfg=session.cfg,
            out_dir=self._outputs_dir / "runtime_audio" / folder,
            synthesizer=synth,
            max_queue_size=_positive_int(
                cfg.get("tts_max_queue_size", 6), "tts_max_queue_size"
            ),
            cooldown_sec=_finite(
                cfg.get("tts_cooldown_sec", 4.0), "tts_cooldown_sec"
            ),
            timeout_sec=_finite(
                cfg.get("tts_timeout_sec", 3.0),
                "tts_timeout_sec",
                0.001,
            ),
            max_alert_age_sec=_finite(
                cfg.get("tts_max_alert_age_sec", 2.5),
                "tts_max_alert_age_sec",
            ),
            media_clock_fn=(
                session.clock.estimated_playhead_sec
                if session.clock is not None
                else None
            ),
            max_media_lag_sec=3.0 if session.clock is not None else None,
            on_ready=lambda alert: self._on_alert(session, alert, True),
            on_drop=lambda alert: self._on_alert(session, alert, False),
        )

    def _on_alert(
        self,
        session: _Session,
        alert: RuntimeAlert,
        ready: bool,
    ) -> None:
        """Công bố audio hợp lệ và bỏ kết quả từ thế hệ playback cũ."""
        with session.lock:
            alert_id = str(alert.get("alert_id", ""))
            generation = session.pending_alerts.pop(alert_id, None)
            if (
                session.status in _TERMINAL
                or session.status == "finishing"
                or session.cancel_token.is_cancelled()
            ):
                session.wake.set()
                return

            current_generation = (
                session.clock.snapshot()["generation"]
                if session.clock is not None
                else session.step_gen
            )
            if ready and (
                generation is None
                or generation != current_generation
            ):
                alert["status"] = "dropped_stale"
                alert.pop("audio_path", None)
                ready = False

            if ready:
                if session.kind == "video":
                    alert["audio_url"] = (
                        f"/api/jobs/{session.key}/runtime-audio/{alert_id}"
                    )
                else:
                    alert["audio_url"] = (
                        f"/api/camera/sessions/{session.key}/audio/{alert_id}"
                    )
                session.voice_play_count += 1
                if session.kind == "video":
                    job_id = int(session.key)
                    _optional_call(
                        self._progress_tracker,
                        "add_voice_play",
                        job_id,
                        1,
                    )
                    _optional_call(
                        self._store,
                        "add_voice_event",
                        job_id,
                        None,
                        alert.get("spoken_text", ""),
                        max(0.0, float(alert.get("source_timestamp", 0.0))),
                        alert.get("audio_path", ""),
                    )
            payload = dict(alert)
            payload["generation"] = current_generation
            track_id = payload.get("track_id")
            if (
                isinstance(track_id, int)
                and not isinstance(track_id, bool)
                and track_id in session.lost_track_timestamps
            ):
                payload["track_lost"] = True
                payload["lost_timestamp"] = session.lost_track_timestamps[
                    track_id
                ]
            else:
                payload["track_lost"] = False
                payload.pop("lost_timestamp", None)
            session.bus.publish("alert", payload)
        session.wake.set()

    def _dispatch_confirmed_alerts(self, session: _Session) -> None:
        """Gửi mỗi track đã xác nhận tới hàng đợi TTS."""
        worker = session.tts_worker
        if worker is None or session.cancel_token.is_cancelled():
            return

        speech_cfg = session.cfg
        realtime = session.cfg.get("realtime", {})
        compact_template = realtime.get(
            "tts_compact_template", "{position_text}, {name}"
        )
        if realtime.get("tts_compact", False) and compact_template:
            speech_cfg = dict(session.cfg)
            post = dict(session.cfg.get("postprocess") or {})
            tts_cfg = dict(post.get("tts") or {})
            tts_cfg["template"] = compact_template
            post["tts"] = tts_cfg
            speech_cfg["postprocess"] = post

        for detection in session.tracker.get_confirmed_alerts():
            if detection.get("alert_emitted"):
                continue
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            class_id = int(detection.get("class_id", -1))
            label = str(detection.get("label", ""))
            side = str(detection.get("position", "front"))
            alert_id = uuid.uuid4().hex
            session.pending_alerts[alert_id] = session.step_gen

            try:
                result = worker.submit(
                    _build_speech_text_compat(
                        label, side, session.classes, speech_cfg
                    ),
                    class_id=class_id,
                    label=label,
                    position=side,
                    track_id=int(track_id),
                    frame_id=int(detection.get("frame_id", session.step_fid)),
                    source_timestamp=float(
                        detection.get("source_timestamp", session.step_ts)
                    ),
                    alert_id=alert_id,
                )
            except Exception:
                session.pending_alerts.pop(alert_id, None)
                _LOG.exception("Không thể gửi cảnh báo TTS")
                continue

            if class_id >= 0:
                session.unique_classes.add(class_id)
            if result.get("status") in {"queued", "ready", "deduped"}:
                session.tracker.mark_alert_emitted(int(track_id))
            if result.get("status") != "queued":
                session.pending_alerts.pop(alert_id, None)
                if result.get("status") == "deduped":
                    session.bus.publish("alert", dict(result))

    def _publish_lost_tracks(self, session: _Session) -> None:
        """Phát sự kiện track mất dấu khi đang giữ lock phiên."""
        lost_events = session.tracker.pop_lost_tracks()
        for item in lost_events:
            if not (item.get("confirmed") or item.get("alert_emitted")):
                continue
            tid = int(item["track_id"])
            lost_timestamp = float(item["lost_timestamp"])
            session.lost_track_timestamps[tid] = lost_timestamp
            session.bus.publish(
                "track_lost",
                {
                    "track_id": tid,
                    "class_id": int(item["class_id"]),
                    "label": str(item["label"]),
                    "lost_timestamp": lost_timestamp,
                    "reason": str(item.get("reason", "lost")),
                    "generation": session.step_gen,
                },
            )

    def _overlay(
        self,
        session: _Session,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ghi keyframe và phát overlay khi đang giữ lock phiên."""
        for detection in detections:
            track_id = detection.get("track_id")
            if (
                isinstance(track_id, int)
                and not isinstance(track_id, bool)
                and detection.get("state") != "lost"
            ):
                session.lost_track_timestamps.pop(track_id, None)
        payload = {
            "frame_id": session.step_fid,
            "source_timestamp": session.step_ts,
            "timestamp_sec": session.step_ts,
            "generation": session.step_gen,
            "width": session.width,
            "height": session.height,
            "detections": detections,
        }
        session.timeline.append(
            session.step_gen,
            session.step_fid,
            session.step_ts,
            detections,
        )
        session.bus.publish("overlay", payload)
        return payload

    @staticmethod
    def _sanitize(
        raw: Any,
        width: int,
        height: int,
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Chuẩn hóa detection trước khi định vị, lưu DB và tracking."""
        clean: list[dict[str, Any]] = []
        if raw is None:
            return clean
        for item in raw:
            try:
                box = [float(value) for value in item["bbox"]]
                score = float(item.get("confidence", 0.0))
                if len(box) != 4 or not all(map(math.isfinite, box)):
                    continue
                if not math.isfinite(score):
                    continue
                x1, x2 = sorted((box[0], box[2]))
                y1, y2 = sorted((box[1], box[3]))
                clean.append(
                    {
                        "bbox": [
                            min(float(width), max(0.0, x1)),
                            min(float(height), max(0.0, y1)),
                            min(float(width), max(0.0, x2)),
                            min(float(height), max(0.0, y2)),
                        ],
                        "label": str(item.get("label", "")),
                        "class_id": int(item["class_id"]),
                        "confidence": min(1.0, max(0.0, score)),
                    }
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue

        positioned = position.add_positions(clean, width, cfg)
        for detection in positioned:
            if detection.get("position") not in POSITIONS:
                detection["position"] = "front"
        return positioned

    def _inference(self, session: _Session) -> None:
        """Chạy detector ngoài lock phiên; chỉ hợp nhất kết quả còn hợp lệ."""
        raw_detector: Any = None
        model_lock = self._model_lock(session.model_name)
        try:
            with session.lock:
                if (
                    session.status in _TERMINAL
                    or session.status == "finishing"
                    or session.cancel_token.is_cancelled()
                ):
                    return

            raw_detector = self._detector_provider(session.model_name)
            with model_lock:
                if session.cancel_token.is_cancelled():
                    return
                runtime_detector, profile = prepare_runtime_detector(
                    session.model_name,
                    raw_detector,
                    session.cfg,
                    backend=session.backend,
                    preset=session.cpu_preset,
                )
            with session.lock:
                if (
                    session.status in _TERMINAL
                    or session.status == "finishing"
                    or session.cancel_token.is_cancelled()
                ):
                    return
                session.profile = dict(profile)

            while not session.cancel_token.is_cancelled():
                envelope = session.infer_slot.pop_latest(
                    timeout=0.1,
                    cancel_token=session.cancel_token,
                )
                if envelope is None:
                    if session.infer_done.is_set():
                        break
                    continue

                with session.lock:
                    if (
                        session.status in _TERMINAL
                        or session.status == "finishing"
                        or session.cancel_token.is_cancelled()
                    ):
                        break
                    generation = (
                        session.clock.snapshot()["generation"]
                        if session.clock is not None
                        else session.step_gen
                    )
                    if envelope.generation != generation:
                        continue
                    session.infer_busy = True

                try:
                    with model_lock:
                        if session.cancel_token.is_cancelled():
                            break
                        started = time.monotonic()
                        raw_detections = runtime_detector.predict(
                            envelope.frame
                        )
                        inference_ms = (
                            time.monotonic() - started
                        ) * 1000.0

                    with session.lock:
                        if (
                            session.status in _TERMINAL
                            or session.status == "finishing"
                            or session.cancel_token.is_cancelled()
                        ):
                            break

                    detections = self._sanitize(
                        raw_detections,
                        envelope.width,
                        envelope.height,
                        session.cfg,
                    )
                    with session.lock:
                        generation = (
                            session.clock.snapshot()["generation"]
                            if session.clock is not None
                            else session.step_gen
                        )
                        if (
                            session.cancel_token.is_cancelled()
                            or envelope.generation != generation
                            or envelope.generation != session.step_gen
                            or session.status in _TERMINAL
                            or session.status == "finishing"
                        ):
                            continue

                        alpha = 1.0 if session.infer_count == 0 else 0.2
                        session.infer_latency_ms += alpha * (
                            inference_ms - session.infer_latency_ms
                        )
                        measured_fps = 1000.0 / max(inference_ms, 0.001)
                        session.infer_fps += alpha * (
                            measured_fps - session.infer_fps
                        )
                        session.infer_count += 1
                        wall_lag_ms = max(
                            0.0,
                            (time.monotonic() - envelope.captured_wall_time)
                            * 1000.0,
                        )
                        session.lag_ms = wall_lag_ms

                        # Không giữ lock model khi ghi DB; lock phiên
                        # ngăn ghi detection sau khi phiên bắt đầu kết thúc.
                        if (
                            session.kind == "video"
                            and session.log_detections
                            and envelope.frame_id >= 0
                            and envelope.source_timestamp >= 0
                        ):
                            _optional_call(
                                self._store,
                                "add_detections",
                                int(session.key),
                                envelope.frame_id,
                                envelope.source_timestamp,
                                detections,
                            )

                    with session.lock:
                        generation = (
                            session.clock.snapshot()["generation"]
                            if session.clock is not None
                            else session.step_gen
                        )
                        if (
                            envelope.generation != generation
                            or envelope.generation != session.step_gen
                            or session.cancel_token.is_cancelled()
                            or session.status in _TERMINAL
                            or session.status == "finishing"
                        ):
                            continue
                        tracked = session.tracker.update_from_detection(
                            detections,
                            detection_frame_id=envelope.frame_id,
                            detection_timestamp=envelope.source_timestamp,
                            current_frame_id=max(
                                session.step_fid, envelope.frame_id
                            ),
                            current_timestamp=max(
                                session.step_ts, envelope.source_timestamp
                            ),
                            source_gray=envelope.gray_pair,
                            frame_width=envelope.width,
                            frame_height=envelope.height,
                            wall_lag_ms=max(
                                0.0,
                                (
                                    time.monotonic()
                                    - envelope.captured_wall_time
                                )
                                * 1000.0,
                            ),
                            position_cfg=session.cfg,
                            include_tentative=True,
                        )
                        self._overlay(session, tracked)
                        self._dispatch_confirmed_alerts(session)
                        self._publish_lost_tracks(session)
                finally:
                    with session.lock:
                        session.infer_busy = False
                    session.wake.set()
        except Exception as exc:
            _LOG.exception("Inference thất bại: %s", session.key)
            with session.lock:
                if (
                    session.status not in _TERMINAL
                    and session.status != "finishing"
                    and not session.cancel_token.is_cancelled()
                ):
                    session.error = str(exc)
                    session.cancel_token.cancel("inference_failed")
            session.wake.set()
        finally:
            if raw_detector is not None:
                try:
                    with model_lock:
                        restore_cpu_tuning(raw_detector)
                except Exception:
                    _LOG.exception("Không khôi phục được CPU tuning")
            session.infer_done.set()
            session.wake.set()

            if session.kind == "camera" and session.error is not None:
                self._finish(session)

    def start_video_session(
        self,
        job_id: int,
        input_path: Path,
        model_name: str,
        *,
        cfg: dict[str, Any] | None = None,
        cpu_preset: str = "fast_cpu",
        backend: str = "pytorch",
        mode: str = "realtime",
        buffer_ahead_sec: float = 2.0,
        frame_stride: int | None = None,
        log_detections: bool = True,
        playback_rate: float | None = None,
    ) -> dict[str, Any]:
        """Tạo phiên video, kiểm tra nguồn và khởi chạy hai luồng xử lý."""
        stride = _positive_int(
            frame_stride
            if frame_stride is not None
            else (12 if model_name == "yolo" else 20),
            "frame_stride",
        )
        config, classes, tracker = self._configuration(model_name, cfg)
        _finite(
            config["realtime"].get("step_fps_cap", 15.0),
            "step_fps_cap",
        )
        raw_rate = (
            config["realtime"].get("video_playback_rate", 1.0)
            if playback_rate is None
            else playback_rate
        )
        if isinstance(raw_rate, bool):
            raise ValueError("playback_rate không hợp lệ")
        rate = _finite(raw_rate, "playback_rate", minimum=0.0)
        if rate <= 0.0:
            raise ValueError("playback_rate phải > 0.0")
        clock = VideoPlaybackClock()
        clock.update(
            0.0,
            playing=True,
            mode=mode,
            buffer_ahead_sec=buffer_ahead_sec,
            playback_rate=rate,
        )
        session = _VideoSession(
            key=job_id,
            kind="video",
            model_name=model_name,
            cfg=config,
            cpu_preset=cpu_preset,
            backend=backend,
            frame_stride=stride,
            tracker=tracker,
            classes=classes,
            input_path=Path(input_path),
            clock=clock,
            log_detections=log_detections,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Bộ quản lý đã đóng")
            existing = self._video_sessions.get(job_id)
            if existing is not None:
                return self._summary(existing)
            self._video_sessions[job_id] = session
            self._prune_locked()

        cap: Any = None
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError("Không mở được video")

            width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or width < 1
                or height < 1
            ):
                raise ValueError("Kích thước video không hợp lệ")

            session.width, session.height = int(width), int(height)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            session.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            session.frame_count = (
                int(count) if math.isfinite(count) and count >= 1 else None
            )
            ok, first_frame = cap.read()
            if not ok or first_frame is None or first_frame.size == 0:
                raise ValueError("Không đọc được frame đầu tiên")

            # Reopen nếu backend không hỗ trợ quay về frame đầu.
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                cap.release()
                cap = cv2.VideoCapture(str(input_path))
                if not cap.isOpened():
                    raise ValueError("Không mở lại được video")

            with self._lock:
                if self._closed or session.cancel_token.is_cancelled():
                    raise RuntimeError("Phiên đã bị hủy khi khởi tạo")
                session.tts_worker = self._worker(session)
                session.status = "processing"
                session.infer_thread = threading.Thread(
                    target=self._inference,
                    args=(session,),
                    name=f"infer-job-{job_id}",
                    daemon=True,
                )
                session.decode_thread = threading.Thread(
                    target=self._decode_video,
                    args=(session, cap),
                    name=f"decode-job-{job_id}",
                    daemon=True,
                )
                _optional_call(self._store, "set_status", job_id, "processing")
                _optional_call(self._progress_tracker, "queued", job_id)
                _optional_call(
                    self._progress_tracker,
                    "start_processing",
                    job_id,
                    session.frame_count or 0,
                    video_fps=session.fps,
                    frame_stride=stride,
                )
                session.bus.publish("status", self._summary(session))
                session.infer_thread.start()
                session.decode_thread.start()
                cap = None
        except Exception as exc:
            if cap is not None:
                cap.release()
            with session.lock:
                session.error = str(exc)
                session.cancel_token.cancel("startup_failed")
            self._finish(session)

        return self._summary(session)

    def _wait(self, session: _Session, timeout: float) -> None:
        """Chờ ngắn, được đánh thức bởi playback, inference hoặc hủy."""
        session.wake.wait(timeout)
        session.wake.clear()

    @staticmethod
    def _seek_capture(cap: Any, target_sec: float, fps: float) -> int:
        """Tua VideoCapture tới mốc thời gian mong muốn với fallback theo frame."""
        target = max(0.0, float(target_sec))
        target_frame = max(0, int(round(target * fps)))
        if cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
            return target_frame
        if cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0):
            return target_frame
        return target_frame

    def _decode_video(self, session: _VideoSession, cap: Any) -> None:
        """Decode và optical flow độc lập với suy luận và TTS."""
        assert session.clock is not None
        interval = 1.0 / max(
            1.0,
            float(session.cfg["realtime"].get("step_fps_cap", 15.0)),
        )
        active_generation = 0
        frame_idx = 0
        last_step_ts = -math.inf
        last_infer_fid = -1
        last_telemetry = 0.0
        reached_eof = False

        try:
            while not session.cancel_token.is_cancelled():
                snap = session.clock.snapshot()
                if snap["generation"] != active_generation:
                    with session.lock:
                        if (
                            session.cancel_token.is_cancelled()
                            or session.status in _TERMINAL
                            or session.status == "finishing"
                        ):
                            break
                        active_generation = int(snap["generation"])
                        session.tracker.reset()
                        session.lost_track_timestamps.clear()
                        session.ring.clear()
                        session.infer_slot.clear()
                        session.timeline.clear()
                        if session.tts_worker is not None:
                            session.tts_worker.drain_pending()
                            session.tts_worker.reset_dedup()
                        session.step_gen = active_generation
                        session.step_ts = float(snap["playhead_sec"])
                        session.step_fid = int(
                            round(session.step_ts * session.fps)
                        )
                        self._overlay(session, [])
                    frame_idx = self._seek_capture(
                        cap, float(snap["playhead_sec"]), session.fps
                    )
                    last_step_ts = -math.inf
                    last_infer_fid = -1

                frame_ts = frame_idx / session.fps
                playhead = float(snap["estimated_playhead_sec"])
                action = decide_video_frame_action(
                    mode=str(snap["mode"]),
                    frame_timestamp_sec=frame_ts,
                    playhead_sec=playhead,
                    buffer_ahead_sec=float(snap["buffer_ahead_sec"]),
                    playing=bool(snap["playing"]),
                )
                if action == "wait_ahead":
                    self._wait(session, 0.03)
                elif action == "skip_behind":
                    if playhead - frame_ts > 1.5:
                        target_idx = max(
                            frame_idx + 1,
                            self._seek_capture(cap, playhead, session.fps),
                        )
                        with session.lock:
                            session.skipped_frames += max(
                                1, target_idx - frame_idx
                            )
                            session.lag_ms = max(
                                session.lag_ms,
                                max(0.0, (playhead - frame_ts) * 1000.0),
                            )
                        frame_idx = target_idx
                    elif cap.grab():
                        with session.lock:
                            session.skipped_frames += 1
                            session.lag_ms = max(
                                0.0, (playhead - frame_ts) * 1000.0
                            )
                        frame_idx += 1
                    else:
                        reached_eof = True
                        break
                else:
                    due_infer = (
                        last_infer_fid < 0
                        or frame_idx % session.frame_stride == 0
                        or frame_idx - last_infer_fid >= session.frame_stride
                    )
                    due_step = due_infer or (frame_ts - last_step_ts >= interval)
                    if not due_step:
                        if cap.grab():
                            with session.lock:
                                session.decoded_frames += 1
                            frame_idx += 1
                            _optional_call(
                                self._progress_tracker,
                                "frame_done",
                                int(session.key),
                                frame_idx - 1,
                                timestamp_sec=frame_ts,
                            )
                            continue
                        reached_eof = True
                        break

                    ok, frame = cap.read()
                    captured_at = time.monotonic()
                    if not ok or frame is None or frame.size == 0:
                        reached_eof = True
                        break

                    gray_pair = compute_flow_gray(
                        frame, session.tracker.config.flow_max_width
                    )
                    with session.lock:
                        if (
                            session.cancel_token.is_cancelled()
                            or session.status in _TERMINAL
                            or session.status == "finishing"
                        ):
                            break
                        if (
                            session.clock.snapshot()["generation"]
                            != active_generation
                        ):
                            continue
                        height, width = frame.shape[:2]
                        session.width, session.height = width, height
                        session.step_fid = frame_idx
                        session.step_ts = frame_ts
                        session.step_gen = active_generation
                        session.decoded_frames += 1
                        session.ring.put(
                            active_generation,
                            frame_idx,
                            frame_ts,
                            gray_pair[0],
                            gray_pair[1],
                        )
                        before = len(session.unique_classes)
                        tracked = session.tracker.step_frame(
                            frame,
                            frame_id=frame_idx,
                            timestamp_sec=frame_ts,
                            frame_width=width,
                            frame_height=height,
                            include_tentative=True,
                            position_cfg=session.cfg,
                        )
                        self._overlay(session, tracked)
                        self._dispatch_confirmed_alerts(session)
                        self._publish_lost_tracks(session)
                        new_signs = len(session.unique_classes) - before
                        if due_infer:
                            session.infer_slot.push(
                                FrameEnvelope(
                                    frame_id=frame_idx,
                                    source_timestamp=frame_ts,
                                    captured_wall_time=captured_at,
                                    frame=frame,
                                    width=width,
                                    height=height,
                                    generation=active_generation,
                                    gray_pair=gray_pair,
                                )
                            )
                            last_infer_fid = frame_idx
                    _optional_call(
                        self._progress_tracker,
                        "frame_done",
                        int(session.key),
                        frame_idx,
                        timestamp_sec=frame_ts,
                        new_sign_count=new_signs,
                    )
                    last_step_ts = frame_ts
                    frame_idx += 1

                now = time.monotonic()
                if now - last_telemetry >= 0.5:
                    session.bus.publish("telemetry", self._summary(session))
                    last_telemetry = now

            # Cho phép frame cuối được suy luận trước khi đóng slot.
            if reached_eof:
                while not session.cancel_token.is_cancelled():
                    with session.lock:
                        idle = (
                            not session.infer_busy
                            and not session.infer_slot.get_stats()["pending"]
                        )
                    if idle or session.infer_done.is_set():
                        break
                    self._wait(session, 0.03)
        except Exception as exc:
            _LOG.exception("Decode thất bại: %s", session.key)
            with session.lock:
                if (
                    session.status not in _TERMINAL
                    and session.status != "finishing"
                    and not session.cancel_token.is_cancelled()
                ):
                    session.error = str(exc)
                    session.cancel_token.cancel("decode_failed")
        finally:
            session.infer_slot.close()
            session.infer_done.set()
            thread = session.infer_thread
            if (
                thread is not None
                and thread.ident is not None
                and thread is not threading.current_thread()
            ):
                if session.cancel_token.is_cancelled():
                    with session.lock:
                        infer_busy = session.infer_busy
                    if not infer_busy:
                        thread.join(timeout=0.05)
                else:
                    deadline = time.monotonic() + 2.0
                    while (
                        thread.is_alive()
                        and not session.cancel_token.is_cancelled()
                    ):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        thread.join(timeout=min(0.05, remaining))

                    if (
                        thread.is_alive()
                        and not session.cancel_token.is_cancelled()
                    ):
                        with session.lock:
                            session.cancel_token.cancel(
                                "inference_shutdown"
                            )

            try:
                cap.release()
            finally:
                self._finish(session)

    def create_camera_session(
        self,
        model_name: str,
        *,
        cfg: dict[str, Any] | None = None,
        cpu_preset: str = "fast_cpu",
        backend: str = "pytorch",
        frame_stride: int = 1,
        synthesizer: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Tạo phiên camera và chạy inference nền."""
        stride = _positive_int(frame_stride, "frame_stride")
        config, classes, tracker = self._configuration(model_name, cfg)
        session_id = uuid.uuid4().hex
        session = _CameraSession(
            key=session_id,
            kind="camera",
            model_name=model_name,
            cfg=config,
            cpu_preset=cpu_preset,
            backend=backend,
            frame_stride=stride,
            tracker=tracker,
            classes=classes,
            log_detections=False,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Bộ quản lý đã đóng")
            session.tts_worker = self._worker(session, synthesizer)
            self._camera_sessions[session_id] = session
            self._prune_locked()
            session.status = "processing"
            session.infer_thread = threading.Thread(
                target=self._inference,
                args=(session,),
                name=f"infer-camera-{session_id}",
                daemon=True,
            )
            session.bus.publish("status", self._summary(session))
            try:
                session.infer_thread.start()
            except Exception as exc:
                session.error = str(exc)
                session.cancel_token.cancel("startup_failed")
                self._finish(session)
        return self._summary(session)

    def push_camera_frame(
        self,
        session_id: str,
        frame_or_bytes: bytes | np.ndarray,
        *,
        timestamp_sec: float | None = None,
        frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Giải mã frame, cập nhật tracker và gửi frame mới nhất cho model."""
        session = self._get_session("camera", session_id)
        if session is None:
            raise KeyError(session_id)

        if isinstance(frame_or_bytes, bytes):
            if not frame_or_bytes:
                raise ValueError("Dữ liệu ảnh rỗng")
            try:
                frame = cv2.imdecode(
                    np.frombuffer(frame_or_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            except cv2.error as exc:
                raise ValueError("Không giải mã được ảnh") from exc
        elif isinstance(frame_or_bytes, np.ndarray):
            frame = frame_or_bytes
        else:
            raise ValueError("Frame phải là bytes hoặc numpy.ndarray")

        if (
            frame is None
            or frame.size == 0
            or frame.dtype != np.uint8
            or frame.ndim not in {2, 3}
            or (frame.ndim == 3 and frame.shape[2] not in {3, 4})
        ):
            raise ValueError("Frame không hợp lệ")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = np.ascontiguousarray(frame).copy()
        captured_at = time.monotonic()
        timestamp = (
            max(0.0, captured_at - session.created_at)
            if timestamp_sec is None
            else _finite(timestamp_sec, "timestamp_sec")
        )
        gray_pair = compute_flow_gray(
            frame, session.tracker.config.flow_max_width
        )

        with session.lock:
            if (
                session.status in _TERMINAL
                or session.cancel_token.is_cancelled()
            ):
                raise KeyError(session_id)

            if frame_id is None:
                fid = session.next_frame_id
            else:
                if isinstance(frame_id, bool) or int(frame_id) != frame_id:
                    raise ValueError("frame_id phải là số nguyên không âm")
                fid = int(frame_id)
            if fid < session.next_frame_id:
                raise ValueError("frame_id phải tăng dần")
            if timestamp < session.step_ts:
                raise ValueError("timestamp_sec không được giảm")

            session.next_frame_id = fid + 1
            height, width = frame.shape[:2]
            session.width, session.height = width, height
            session.step_fid, session.step_ts = fid, timestamp
            session.decoded_frames += 1
            session.ring.put(
                session.step_gen, fid, timestamp, gray_pair[0], gray_pair[1]
            )
            tracked = session.tracker.step_frame(
                frame,
                frame_id=fid,
                timestamp_sec=timestamp,
                frame_width=width,
                frame_height=height,
                include_tentative=True,
                position_cfg=session.cfg,
            )
            overlay = self._overlay(session, tracked)
            self._dispatch_confirmed_alerts(session)
            self._publish_lost_tracks(session)
            dropped_before = 0
            if fid % session.frame_stride == 0 or session.infer_count == 0:
                dropped_before = session.infer_slot.push(
                    FrameEnvelope(
                        frame_id=fid,
                        source_timestamp=timestamp,
                        captured_wall_time=captured_at,
                        frame=frame,
                        width=width,
                        height=height,
                        generation=session.step_gen,
                        gray_pair=gray_pair,
                    )
                )
            result = {
                "session_id": session_id,
                "status": session.status,
                "dropped_before": dropped_before,
                **overlay,
            }
        telemetry = self._summary(session)
        result["telemetry"] = telemetry
        session.bus.publish("telemetry", telemetry)
        return result

    def stop_camera_session(self, session_id: str) -> dict[str, Any]:
        """Dừng nhận frame và đóng tài nguyên camera."""
        session = self._get_session("camera", session_id)
        if session is None:
            raise KeyError(session_id)
        with session.lock:
            session.cancel_token.cancel("stopped")
            session.infer_slot.close()
            session.infer_done.set()
            session.wake.set()

        thread = session.infer_thread
        if (
            thread is not None
            and thread.ident is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.1)
        self._finish(session)
        return self._summary(session)

    def _finish(self, session: _Session) -> None:
        """Hoàn tất TTS, cập nhật persistence và đóng SSE đúng một lần."""
        with session.lock:
            if session.status in _TERMINAL or session.status == "finishing":
                return
            session.status = "finishing"

        cancelled = session.cancel_token.is_cancelled()
        worker = session.tts_worker
        if worker is not None:
            if not cancelled and session.error is None:
                cfg = session.cfg.get("realtime", {})
                deadline = time.monotonic() + (
                    float(cfg.get("tts_max_alert_age_sec", 2.5))
                    + float(cfg.get("tts_timeout_sec", 3.0))
                    + 0.5
                )
                while time.monotonic() < deadline:
                    with session.lock:
                        pending = bool(session.pending_alerts)
                    if not pending or session.cancel_token.is_cancelled():
                        break
                    self._wait(session, 0.03)
            cancelled = session.cancel_token.is_cancelled()
            worker.shutdown(wait=not cancelled, cancel_pending=True)

        with session.lock:
            cancelled = session.cancel_token.is_cancelled()
            if session.error is not None:
                terminal = "failed"
            elif cancelled:
                terminal = (
                    "stopped" if session.kind == "camera" else "cancelled"
                )
            else:
                terminal = "done"

            if session.kind == "video":
                job_id = int(session.key)
                if terminal == "done":
                    try:
                        if self._store is not None:
                            self._store.finish_job(
                                job_id,
                                "",
                                len(session.unique_classes),
                                session.voice_play_count,
                                0.0,
                                0.0,
                                (time.monotonic() - session.created_at)
                                * 1000.0,
                            )
                            self._store.set_status(job_id, "done")
                    except Exception as exc:
                        _LOG.exception("Không hoàn tất được job %s", job_id)
                        session.error = str(exc)
                        terminal = "failed"

                if terminal != "done":
                    _optional_call(
                        self._store,
                        "fail_job",
                        job_id,
                        session.error
                        or session.cancel_token.reason
                        or "Phiên đã dừng",
                    )
                _optional_call(
                    self._progress_tracker,
                    "set_stage",
                    job_id,
                    "done" if terminal == "done" else "failed",
                )

            session.status = terminal
            session.ended_at = time.monotonic()
            session.pending_alerts.clear()
            session.infer_slot.close()
            session.ring.clear()
            session.bus.publish("status", self._summary(session))
            session.bus.close(terminal)
            session.finished.set()
            session.wake.set()

        with self._lock:
            self._prune_locked()

    def update_video_playback(
        self,
        job_id: int,
        *,
        timestamp_sec: float | None = None,
        playing: bool | None = None,
        playback_rate: float | None = None,
        seeked: bool = False,
        mode: str | None = None,
        buffer_ahead_sec: float | None = None,
    ) -> dict[str, Any]:
        """Cập nhật playback và đánh thức decode mà không hủy phiên."""
        session = self._get_session("video", job_id)
        if session is None:
            raise KeyError(job_id)
        assert session.clock is not None

        with session.lock:
            if (
                session.status in _TERMINAL
                or session.status == "finishing"
                or session.cancel_token.is_cancelled()
            ):
                return self._summary(session)
            previous_generation = int(
                session.clock.snapshot()["generation"]
            )
            timestamp = (
                session.clock.estimated_playhead_sec()
                if timestamp_sec is None
                else _finite(timestamp_sec, "timestamp_sec")
            )
            generation = session.clock.update(
                timestamp,
                playing=playing,
                playback_rate=playback_rate,
                seeked=seeked,
                mode=mode,
                buffer_ahead_sec=buffer_ahead_sec,
            )
            if seeked or generation > previous_generation:
                session.infer_slot.clear()
            result = self._summary(session)
        session.wake.set()
        return result

    def cancel_video_session(self, job_id: int) -> bool:
        """Yêu cầu hủy video và đánh thức mọi bộ chờ."""
        session = self._get_session("video", job_id)
        if session is None:
            return False
        with session.lock:
            if session.status in _TERMINAL:
                return False
            session.cancel_token.cancel("cancelled")
            session.infer_slot.close()
            session.infer_done.set()
            session.wake.set()
        return True

    def _get_session(
        self,
        kind: str,
        session_key: int | str,
    ) -> _Session | None:
        """Tra phiên bằng loại và khóa đã chuẩn hóa."""
        with self._lock:
            if kind == "video":
                return self._video_sessions.get(int(session_key))
            if kind == "camera":
                return self._camera_sessions.get(str(session_key))
        raise ValueError("kind phải là 'video' hoặc 'camera'")

    def _summary(self, session: _Session) -> dict[str, Any]:
        """Tạo snapshot trạng thái, hiệu năng và tài nguyên."""
        with session.lock:
            playback: dict[str, Any] = {}
            generation = session.step_gen
            if isinstance(session, _VideoSession) and session.clock is not None:
                playback = session.clock.snapshot()
                generation = int(playback["generation"])

            elapsed = (
                (session.ended_at or time.monotonic()) - session.created_at
            )
            result: dict[str, Any] = {
                "kind": session.kind,
                "model_name": session.model_name,
                "status": session.status,
                "error": session.error,
                "width": session.width,
                "height": session.height,
                "frame_id": session.step_fid,
                "source_timestamp": session.step_ts,
                "generation": generation,
                "frame_stride": session.frame_stride,
                "infer_count": session.infer_count,
                "infer_fps": session.infer_fps,
                "inference_fps": session.infer_fps,
                "decode_fps": (
                    session.decoded_frames / max(elapsed, 0.001)
                    if session.decoded_frames
                    else 0.0
                ),
                "infer_latency_ms": session.infer_latency_ms,
                "lag_ms": round(session.lag_ms, 2),
                "skipped_frames": session.skipped_frames,
                "decoded_frames": session.decoded_frames,
                "unique_sign_count": len(session.unique_classes),
                "voice_play_count": session.voice_play_count,
                "elapsed_sec": max(0.0, elapsed),
                "cpu_preset": session.cpu_preset,
                "backend": session.backend,
                "profile": deepcopy(session.profile),
                "scheduler": session.infer_slot.get_stats(),
                "tracker": session.tracker.get_stats(),
                "tts": (
                    session.tts_worker.get_stats()
                    if session.tts_worker is not None
                    else {}
                ),
                "rss_mb": None,
                "cpu_percent": None,
            }
            result["dropped_frames"] = result["scheduler"].get("dropped", 0)
            result["dropped_stale_detections"] = result["tracker"].get(
                "dropped_stale_detections", 0
            )
            result["compensated_detections"] = result["tracker"].get(
                "compensated_detections", 0
            )
            result["active_tracks"] = result["tracker"].get("active_count", 0)
            result["late_alerts"] = result["tts"].get("late_alerts", 0)
            result["late_ratio"] = result["tts"].get("late_ratio", 0.0)
            if isinstance(session, _VideoSession):
                result.update(
                    job_id=session.key,
                    fps=session.fps,
                    frame_count=session.frame_count,
                    duration_sec=(
                        session.frame_count / session.fps
                        if session.frame_count is not None
                        else None
                    ),
                    playback=playback,
                    overlay_generation=session.step_gen,
                )
            else:
                result["session_id"] = str(session.key)

        if self._process is not None:
            try:
                with self._resource_lock:
                    result["rss_mb"] = (
                        self._process.memory_info().rss / (1024.0 * 1024.0)
                    )
                    result["cpu_percent"] = self._process.cpu_percent(
                        interval=None
                    )
            except Exception:
                _LOG.debug("Không đọc được tài nguyên", exc_info=True)
        return result

    def get_video_status(self, job_id: int) -> dict[str, Any] | None:
        """Lấy snapshot video hoặc None nếu không còn giữ phiên."""
        session = self._get_session("video", job_id)
        return self._summary(session) if session is not None else None

    def get_camera_status(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Lấy snapshot camera hoặc None nếu không còn giữ phiên."""
        session = self._get_session("camera", session_id)
        return self._summary(session) if session is not None else None

    def get_runtime_audio_path(
        self,
        kind: str,
        session_key: int | str,
        alert_id: str,
    ) -> Path | None:
        """Tra audio qua worker, không ghép đường dẫn từ alert_id."""
        session = self._get_session(kind, session_key)
        if session is None or session.tts_worker is None:
            return None
        return session.tts_worker.get_audio_path(alert_id)

    def iter_sse_events(
        self,
        kind: str,
        session_key: int | str,
        heartbeat_sec: float = 0.5,
    ) -> Iterator[str]:
        """Lấy iterator SSE riêng cho một client."""
        session = self._get_session(kind, session_key)
        if session is None:
            raise KeyError(session_key)
        return session.bus.iter_sse(heartbeat_sec)

    def has_video_session(self, job_id: int) -> bool:
        """Kiểm tra video đang được giữ trong bộ quản lý."""
        return self._get_session("video", job_id) is not None

    def has_camera_session(self, session_id: str) -> bool:
        """Kiểm tra camera đang được giữ trong bộ quản lý."""
        return self._get_session("camera", session_id) is not None

    def shutdown(self, wait: bool = False) -> None:
        """Ngừng nhận phiên mới và yêu cầu dừng mọi tác vụ hiện tại."""
        with self._lock:
            self._closed = True
            sessions = [
                *self._video_sessions.values(),
                *self._camera_sessions.values(),
            ]

        for session in sessions:
            with session.lock:
                session.cancel_token.cancel("shutdown")
                session.infer_slot.close()
                session.infer_done.set()
                session.wake.set()
            if session.kind == "camera":
                self._finish(session)

        if wait:
            # Giới hạn tổng thời gian join cho toàn bộ các phiên.
            # Predict bị kẹt không được ngăn bộ quản lý hoàn tất shutdown.
            deadline = time.monotonic() + 2.0
            for session in sessions:
                for thread in (session.decode_thread, session.infer_thread):
                    if (
                        thread is None
                        or thread.ident is None
                        or thread is threading.current_thread()
                    ):
                        continue
                    while thread.is_alive():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        thread.join(timeout=min(0.05, remaining))

                if not session.finished.is_set():
                    self._finish(session)
