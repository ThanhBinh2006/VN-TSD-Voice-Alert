"""Chạy job nền, theo dõi tiến độ từng frame và cung cấp luồng MJPEG.

Detector được cache theo model. Mỗi job có cấu hình overrides riêng.
Bộ đệm live chỉ giữ một JPEG mới nhất cho mỗi slot.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ts_alert.config.resolve import deep_merge
from ts_alert.pipeline import _resolve_frame_stride, process_media
from ts_alert.runtime import video_io
from ts_alert.runtime.scheduler import CancellationToken

log = logging.getLogger(__name__)

STAGES = (
    "queued",
    "loading_model",
    "processing",
    "finalizing",
    "done",
    "failed",
)


class ProgressTracker:
    """Lưu tiến độ trong RAM, đồng bộ giữa worker và các request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _initial_state() -> dict[str, Any]:
        """Tạo state độc lập cho một job."""
        return {
            "stage": "queued",
            "frames_done": 0,
            "frame_count": None,
            "video_fps": None,
            "frame_stride": 1,
            "current_timestamp_sec": None,
            "layer_time_ms": 0.0,
            "tts_time_ms": 0.0,
            "unique_sign_count": 0,
            "voice_play_count": 0,
            "started_at": None,
            "ended_at": None,
            "updated_at": time.time(),
        }

    def queued(self, job_id: int) -> None:
        """Khởi tạo hoặc reset tiến độ của job đang chờ."""
        with self._lock:
            self._jobs[job_id] = self._initial_state()

    def set_stage(self, job_id: int, stage: str) -> None:
        """Cập nhật stage và chốt đồng hồ khi kết thúc xử lý frame."""
        with self._lock:
            state = self._jobs.setdefault(job_id, self._initial_state())
            state["stage"] = stage
            state["updated_at"] = time.time()
            if stage in ("finalizing", "done", "failed"):
                self._stop_clock(state)

    @staticmethod
    def _stop_clock(state: dict[str, Any]) -> None:
        """Chốt thời gian một lần để FPS không giảm sau khi xử lý xong."""
        if state.get("started_at") is not None and state.get("ended_at") is None:
            state["ended_at"] = time.perf_counter()

    def start_processing(
        self,
        job_id: int,
        frame_count: int | None,
        video_fps: float | None = None,
        frame_stride: int = 1,
    ) -> None:
        """Bắt đầu tính tiến độ; giữ tương thích lời gọi chỉ truyền tổng frame."""
        with self._lock:
            state = self._jobs.setdefault(job_id, self._initial_state())
            state.update(
                stage="processing",
                frames_done=0,
                frame_count=frame_count or None,
                video_fps=video_fps,
                frame_stride=max(1, int(frame_stride)),
                current_timestamp_sec=None,
                layer_time_ms=0.0,
                tts_time_ms=0.0,
                unique_sign_count=0,
                voice_play_count=0,
                started_at=time.perf_counter(),
                ended_at=None,
                updated_at=time.time(),
            )

    def frame_done(
        self,
        job_id: int,
        frame_index: int,
        timestamp_sec: float | None = None,
        layer_time_ms: float | None = None,
        tts_time_ms: float | None = None,
        new_sign_count: int = 0,
    ) -> None:
        """Cập nhật frame đã xử lý và các số liệu tích lũy từ pipeline."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return

            state["frames_done"] = max(state["frames_done"], frame_index + 1)
            if timestamp_sec is not None:
                state["current_timestamp_sec"] = timestamp_sec
            if layer_time_ms is not None:
                state["layer_time_ms"] = layer_time_ms
            if tts_time_ms is not None:
                state["tts_time_ms"] = tts_time_ms
            if new_sign_count:
                state["unique_sign_count"] += new_sign_count

            state["updated_at"] = time.time()
            if (
                state["frame_count"]
                and state["frames_done"] >= state["frame_count"]
            ):
                state["stage"] = "finalizing"
                self._stop_clock(state)

    def add_voice_play(self, job_id: int, count: int = 1) -> None:
        """Cộng số sự kiện giọng nói đã được lưu thành công."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is not None:
                state["voice_play_count"] += count
                state["updated_at"] = time.time()

    def get(self, job_id: int) -> dict[str, Any] | None:
        """Trả snapshot tiến độ, FPS, ETA và thống kê xử lý của job."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            state = dict(state)

        stage = state["stage"]
        done = state["frames_done"]
        total = state["frame_count"]
        video_fps = state["video_fps"]
        started_at = state["started_at"]
        ended_at = state["ended_at"]

        elapsed = (
            (ended_at if ended_at is not None else time.perf_counter())
            - started_at
            if started_at is not None
            else None
        )
        fps = done / elapsed if elapsed and done else None

        if stage == "done":
            percent = 100.0
        elif total:
            percent = min(99.0, round(100.0 * done / total, 1))
        else:
            percent = None

        eta = (
            max(0.0, (total - done) / fps)
            if fps and total and stage == "processing"
            else None
        )

        return {
            "stage": stage,
            "percent": percent,
            "frames_done": done,
            "frame_count": total,
            "fps": round(fps, 2) if fps else None,
            "eta_sec": round(eta, 1) if eta is not None else None,
            "video_fps": video_fps,
            "duration_sec": (
                round(total / video_fps, 2)
                if total and video_fps and video_fps > 0
                else None
            ),
            "current_timestamp_sec": state["current_timestamp_sec"],
            "frame_stride": state["frame_stride"],
            "elapsed_ms": round(elapsed * 1000.0, 1) if elapsed is not None else None,
            "layer_time_ms": state["layer_time_ms"],
            "tts_time_ms": state["tts_time_ms"],
            "unique_sign_count": state["unique_sign_count"],
            "voice_play_count": state["voice_play_count"],
        }


class _TrackingStore:
    """Chuyển tiếp thao tác store và cập nhật thống kê tương ứng."""

    def __init__(self, store: Any, tracker: ProgressTracker) -> None:
        self._store = store
        self._tracker = tracker

    def add_detections(
        self,
        job_id: int,
        frame_index: int,
        timestamp_sec: float,
        detections: list[Any],
    ) -> Any:
        """Lưu detections và giữ hành vi cập nhật tiến độ của phiên bản cũ."""
        result = self._store.add_detections(
            job_id, frame_index, timestamp_sec, detections
        )
        self._tracker.frame_done(job_id, frame_index)
        return result

    def add_voice_event(
        self,
        job_id: int,
        detection_id: int | None,
        spoken_text: str,
        timestamp_sec: float,
        audio_path: str,
    ) -> Any:
        """Lưu sự kiện giọng nói trước khi cộng bộ đếm."""
        result = self._store.add_voice_event(
            job_id, detection_id, spoken_text, timestamp_sec, audio_path
        )
        self._tracker.add_voice_play(job_id, 1)
        return result

    def __getattr__(self, name: str) -> Any:
        """Chuyển tiếp các thuộc tính còn lại tới store thật."""
        return getattr(self._store, name)


@dataclass
class _LiveSlot:
    """Một JPEG mới nhất và condition đánh thức các client đang đọc."""

    cond: threading.Condition = field(default_factory=threading.Condition)
    seq: int = 0
    jpeg: bytes | None = None
    frame_index: int = -1
    timestamp_sec: float = 0.0
    finished: bool = False
    status: str = "queued"


class LiveFrameBuffer:
    """Bộ đệm JPEG có giới hạn slot, không tạo thread hoặc hàng đợi frame.

    Các condition dùng chung khóa để việc thay slot, publish và shutdown
    luôn được quan sát nhất quán.
    """

    def __init__(self, max_jobs: int = 6) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs phải >= 1")

        self._max_jobs = max_jobs
        self._lock = threading.RLock()
        self._slots: OrderedDict[int, _LiveSlot] = OrderedDict()
        self._shutdown = threading.Event()

    def _remove_slot(self, job_id: int) -> None:
        """Xóa slot và đánh thức client; caller phải giữ self._lock."""
        slot = self._slots.pop(job_id)
        slot.finished = True
        slot.cond.notify_all()

    def _prune(self) -> None:
        """Ưu tiên xóa slot finished cũ nhất để giữ giới hạn bộ nhớ."""
        while len(self._slots) > self._max_jobs:
            victim = next(
                (
                    job_id
                    for job_id, slot in self._slots.items()
                    if slot.finished
                ),
                None,
            )
            if victim is None:
                # Vẫn giữ giới hạn cứng nếu buffer được dùng với nhiều producer.
                victim = next(iter(self._slots))
            self._remove_slot(victim)

    def init_job(self, job_id: int) -> None:
        """Tạo hoặc reset slot và đưa job về cuối thứ tự lưu."""
        with self._lock:
            if self._shutdown.is_set():
                return
            if job_id in self._slots:
                self._remove_slot(job_id)
            self._slots[job_id] = _LiveSlot(
                cond=threading.Condition(self._lock)
            )
            self._prune()

    def publish(
        self,
        job_id: int,
        frame: np.ndarray | None,
        frame_index: int,
        timestamp_sec: float,
    ) -> None:
        """Encode ngoài khóa và thay JPEG hiện tại bằng frame mới nhất."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        with self._lock:
            slot = self._slots.get(job_id)
            if slot is None or slot.finished or self._shutdown.is_set():
                return

        try:
            if frame.ndim < 2:
                return
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                return

            if width > 1280:
                frame = cv2.resize(
                    frame,
                    (1280, max(1, round(height * 1280 / width))),
                    interpolation=cv2.INTER_AREA,
                )

            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not encoded_ok:
                return
            jpeg = encoded.tobytes()
        except Exception:
            log.debug("Không encode được frame live của job %s", job_id, exc_info=True)
            return

        with self._lock:
            # Không publish vào slot đã bị reset trong lúc encode.
            if (
                self._shutdown.is_set()
                or self._slots.get(job_id) is not slot
                or slot.finished
            ):
                return
            slot.jpeg = jpeg
            slot.frame_index = frame_index
            slot.timestamp_sec = timestamp_sec
            slot.status = "processing"
            slot.seq += 1
            slot.cond.notify_all()

    def mark_finished(self, job_id: int, status: str = "done") -> None:
        """Đóng luồng của job, giữ JPEG cuối và đánh thức các client."""
        with self._lock:
            slot = self._slots.get(job_id)
            if slot is None:
                return
            slot.finished = True
            slot.status = status
            slot.cond.notify_all()

    def has_slot(self, job_id: int) -> bool:
        """Cho biết job còn slot trong buffer hay không."""
        with self._lock:
            return job_id in self._slots

    def latest_frame(self, job_id: int) -> bytes | None:
        """Trả JPEG mới nhất; bytes là bất biến nên không cần sao chép."""
        with self._lock:
            slot = self._slots.get(job_id)
            return slot.jpeg if slot is not None else None

    def iter_mjpeg(
        self,
        job_id: int,
        heartbeat_sec: float = 0.5,
    ) -> Iterator[bytes]:
        """Yield multipart MJPEG; kết thúc khi slot đóng, bị xóa hoặc shutdown."""
        if not 0 < heartbeat_sec < float("inf"):
            raise ValueError("heartbeat_sec phải là số hữu hạn > 0")

        with self._lock:
            slot = self._slots.get(job_id)
            if slot is None:
                return

        last_seq = 0
        while True:
            with slot.cond:
                if (
                    self._shutdown.is_set()
                    or self._slots.get(job_id) is not slot
                ):
                    break

                if slot.seq > last_seq and slot.jpeg is not None:
                    jpeg = slot.jpeg
                    last_seq = slot.seq
                elif slot.finished:
                    break
                else:
                    slot.cond.wait(timeout=heartbeat_sec)
                    continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            # Không giữ thêm JPEG cũ trong generator khi chờ frame tiếp theo.
            del jpeg

    def shutdown(self) -> None:
        """Đánh thức toàn bộ client để các generator thoát."""
        self._shutdown.set()
        with self._lock:
            for slot in self._slots.values():
                slot.cond.notify_all()


class JobRunner:
    """Chạy tuần tự các job và tái sử dụng detector theo tên model."""

    def __init__(
        self,
        store: Any,
        cfg_for: Callable[[str], dict[str, Any]],
        detector_factory: Callable[[str, dict[str, Any]], Any],
        out_dir: Path,
        sync: bool = False,
    ) -> None:
        """Nhận factory cấu hình, detector và thư mục kết quả."""
        self.store = store
        self.cfg_for = cfg_for
        self.detector_factory = detector_factory
        self.out_dir = Path(out_dir)
        self.sync = sync
        self.progress = ProgressTracker()
        self.live = LiveFrameBuffer()
        self._detectors: dict[str, Any] = {}
        self._detector_lock = threading.Lock()
        self._jobs_lock = threading.RLock()
        self._cancel_tokens: dict[int, CancellationToken] = {}
        self._futures: dict[int, Future[None]] = {}
        self._shutting_down = False
        self._pool = (
            None
            if sync
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsd-job")
        )

    def get_detector(self, model_name: str) -> Any:
        """Nạp detector một lần bằng cấu hình model và giữ trong cache."""
        with self._detector_lock:
            if model_name not in self._detectors:
                log.info("Nạp detector %s ...", model_name)
                self._detectors[model_name] = self.detector_factory(
                    model_name, self.cfg_for(model_name)
                )
            return self._detectors[model_name]

    def loaded_models(self) -> list[str]:
        """Trả danh sách model đã nạp dưới khóa bảo vệ cache."""
        with self._detector_lock:
            return sorted(self._detectors)

    def submit(
        self,
        job_id: int,
        input_path: Path,
        model_name: str,
        overrides: dict[str, Any] | None = None,
        *,
        render_video: bool = True,
    ) -> Future[None] | None:
        """Xếp job vào worker với overrides riêng và token hủy độc lập."""
        job_overrides = deepcopy(overrides) if overrides is not None else None

        with self._jobs_lock:
            if self._shutting_down:
                raise RuntimeError("Worker đang shutdown")

            token = CancellationToken()
            self._cancel_tokens[job_id] = token
            self.progress.queued(job_id)
            try:
                if video_io.media_type(input_path) == "video":
                    self.live.init_job(job_id)
            except Exception:
                pass

            if not self.sync:
                try:
                    if self._pool is None:
                        raise RuntimeError("Worker chưa được khởi tạo")

                    future = self._pool.submit(
                        self._run,
                        job_id,
                        Path(input_path),
                        model_name,
                        job_overrides,
                        render_video=render_video,
                    )
                    self._futures[job_id] = future
                    return future
                except BaseException:
                    if self._cancel_tokens.get(job_id) is token:
                        self._cancel_tokens.pop(job_id, None)
                        self._futures.pop(job_id, None)
                    self.progress.set_stage(job_id, "failed")
                    self.live.mark_finished(job_id, "failed")
                    raise

        self._run(
            job_id,
            Path(input_path),
            model_name,
            job_overrides,
            render_video=render_video,
        )
        return None

    def get_cancel_token(self, job_id: int) -> CancellationToken | None:
        """Trả token của job còn được quản lý, hoặc None khi đã kết thúc."""
        with self._jobs_lock:
            return self._cancel_tokens.get(job_id)

    def is_cancelled(self, job_id: int) -> bool:
        """Cho biết token hiện tại đã bị hủy; job đã dọn token trả False."""
        token = self.get_cancel_token(job_id)
        return token is not None and token.is_cancelled()

    def cancel(self, job_id: int) -> bool:
        """Yêu cầu hủy job; chốt trạng thái ngay nếu future chưa chạy."""
        with self._jobs_lock:
            token = self._cancel_tokens.get(job_id)
            future = self._futures.get(job_id)

        if token is None:
            return False

        # Listener có thể lấy khóa riêng; không gọi khi giữ khóa quản lý job.
        token.cancel("Job cancelled")

        if future is not None and future.cancel():
            with self._jobs_lock:
                if self._cancel_tokens.get(job_id) is not token:
                    return True
                try:
                    self.store.fail_job(job_id, "Job đã bị hủy")
                finally:
                    self.progress.set_stage(job_id, "failed")
                    self.live.mark_finished(job_id, "failed")
                    self._cancel_tokens.pop(job_id, None)
                    if self._futures.get(job_id) is future:
                        self._futures.pop(job_id, None)

        return True

    def _run(
        self,
        job_id: int,
        input_path: Path,
        model_name: str,
        overrides: dict[str, Any] | None = None,
        *,
        render_video: bool = True,
    ) -> None:
        """Xử lý job, hỗ trợ hủy và luôn giải phóng token cùng slot live."""
        with self._jobs_lock:
            token = self._cancel_tokens.get(job_id)
            if token is None:
                token = CancellationToken()
                self._cancel_tokens[job_id] = token
            if self._shutting_down:
                cancel_before_start = True
            else:
                cancel_before_start = False

        if cancel_before_start:
            token.cancel("Job cancelled")

        final_status = "failed"
        try:
            token.raise_if_cancelled()
            base_cfg = self.cfg_for(model_name)
            cfg = deep_merge(base_cfg, overrides) if overrides else base_cfg
            is_video, frame_count, video_fps = _media_info(input_path)
            token.raise_if_cancelled()

            if is_video and not self.live.has_slot(job_id):
                self.live.init_job(job_id)

            frame_stride = _resolve_frame_stride(cfg)
            with self._detector_lock:
                model_loaded = model_name in self._detectors
            if not model_loaded:
                self.progress.set_stage(job_id, "loading_model")

            token.raise_if_cancelled()
            detector = self.get_detector(model_name)
            token.raise_if_cancelled()

            self.progress.start_processing(
                job_id,
                frame_count,
                video_fps=video_fps,
                frame_stride=frame_stride,
            )

            def _on_frame(
                frame_index: int,
                timestamp_sec: float,
                frame: np.ndarray,
                predicted: bool,
                layer_time_ms: float,
                tts_time_ms: float,
                detections: list[Any],
                events: list[Any],
            ) -> None:
                """Ghi nhận mọi frame, kể cả frame không chạy inference."""
                self.progress.frame_done(
                    job_id,
                    frame_index,
                    timestamp_sec=timestamp_sec,
                    layer_time_ms=layer_time_ms,
                    tts_time_ms=tts_time_ms,
                    new_sign_count=len(events),
                )
                if is_video:
                    self.live.publish(
                        job_id, frame, frame_index, timestamp_sec
                    )

            process_media(
                input_path,
                model_name,
                cfg=cfg,
                detector=detector,
                repo=_TrackingStore(self.store, self.progress),
                job_id=job_id,
                out_dir=self.out_dir,
                on_frame=_on_frame,
                render_video=render_video,
                cancel_event=token,
            )
            self.progress.set_stage(job_id, "done")
            final_status = "done"
        except Exception as exc:
            self.progress.set_stage(job_id, "failed")
            if token.is_cancelled():
                log.info("Job %s đã bị hủy", job_id)
                error_message = "Job đã bị hủy"
            else:
                log.exception("Job %s lỗi", job_id)
                error_message = f"{type(exc).__name__}: {exc}"

            job = (
                self.store.get_job(job_id)
                if hasattr(self.store, "get_job")
                else None
            )
            if not job or job.get("status") != "failed":
                self.store.fail_job(job_id, error_message)
        finally:
            with self._jobs_lock:
                if self._cancel_tokens.get(job_id) is token:
                    self._cancel_tokens.pop(job_id, None)
                    self._futures.pop(job_id, None)
            self.live.mark_finished(job_id, final_status)

    def shutdown(self, wait: bool = False) -> None:
        """Hủy mọi job trước khi đóng luồng MJPEG và dừng executor."""
        with self._jobs_lock:
            self._shutting_down = True
            tokens = tuple(self._cancel_tokens.items())

        # Gửi tín hiệu tới tất cả worker trước các thao tác ghi trạng thái.
        for job_id, token in tokens:
            try:
                token.cancel("Job cancelled")
            except Exception:
                log.exception("Lỗi callback khi hủy job %s", job_id)

        try:
            for job_id, _ in tokens:
                try:
                    self.cancel(job_id)
                except Exception:
                    log.exception("Không thể chốt trạng thái hủy job %s", job_id)
        finally:
            self.live.shutdown()
            if self._pool is not None:
                self._pool.shutdown(wait=wait, cancel_futures=not wait)


def _media_info(input_path: Path) -> tuple[bool, int | None, float | None]:
    """Trả loại media, tổng frame và FPS; metadata lỗi được biểu diễn bằng None."""
    try:
        is_video = video_io.media_type(input_path) == "video"
    except Exception:
        return False, None, None

    if not is_video:
        return False, 1, None

    try:
        info = video_io.get_video_info(input_path)
        frame_count = int(info["frame_count"]) if info.get("frame_count") else None
        video_fps = float(info["fps"]) if info.get("fps") else None
        return (
            True,
            frame_count if frame_count is not None and frame_count > 0 else None,
            video_fps if video_fps is not None and video_fps > 0 else None,
        )
    except Exception:
        return True, None, None


def _count_frames(input_path: Path) -> int | None:
    """Trả tổng frame; ảnh bằng 1, metadata không đọc được trả None."""
    return _media_info(input_path)[1]
