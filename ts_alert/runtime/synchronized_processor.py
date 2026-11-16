"""Xử lý đồng bộ ảnh/video, trả frame JPEG và cảnh báo âm thanh, không xuất MP4."""

from __future__ import annotations

import base64
import copy
import json
import logging
import math
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from ts_alert.db.interface import ResultStore
from ts_alert.postprocess import draw_layer, position, tts
from ts_alert.postprocess.dedup import Deduplicator
from ts_alert.runtime import video_io
from ts_alert.types import Detection

logger = logging.getLogger(__name__)

Frame = NDArray[np.uint8]


class Detector(Protocol):
    """Giao diện tương thích với detector hiện có của dự án."""

    def predict(self, frame: Frame) -> list[Detection]:
        """Phát hiện biển báo trên frame BGR ở kích thước gốc."""
        ...


class CancelToken(Protocol):
    """Giao diện hủy tương thích với threading.Event."""

    def is_set(self) -> bool:
        """Trả True khi người gọi yêu cầu hủy."""
        ...


class ProcessingCancelledError(RuntimeError):
    """Tác vụ đã bị hủy trước khi hoàn tất."""


class VoiceAlert(TypedDict):
    """Cảnh báo có âm thanh sẵn sàng phát cùng frame tương ứng."""

    alert_id: str
    text: str
    audio_url: str
    position: str
    class_name: str


@dataclass(slots=True)
class SynchronizedFrame:
    """Frame đã xử lý và âm thanh đã tổng hợp xong.

    fps là FPS nguồn, timestamp_sec thuộc dòng thời gian nguồn.
    jpeg_bytes và image_base64 biểu diễn cùng một ảnh JPEG.

    Với video, phần tử is_done=True là thông báo kết thúc riêng:
    frame_index là frame cuối đã xử lý, không chứa ảnh hoặc cảnh báo.
    progress_pct=0 khi chưa biết tổng số frame; kết thúc luôn bằng 100.

    Bên tiêu thụ chịu trách nhiệm hiển thị ảnh và phát audio_url.
    """

    frame_index: int
    timestamp_sec: float
    image_base64: str
    detections: list[Detection]
    voice_alerts: list[VoiceAlert]
    progress_pct: float
    fps: float
    is_done: bool
    jpeg_bytes: bytes = field(default=b"", repr=False)


def _validate_stride(stride: int) -> int:
    """Kiểm tra stride là số nguyên dương, không chấp nhận bool."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("frame_stride phải là số nguyên >= 1.")
    return stride


def _check_cancelled(token: CancelToken | None) -> None:
    """Phát sinh ngoại lệ nếu nhận yêu cầu hủy."""
    if token is not None and token.is_set():
        raise ProcessingCancelledError("Tác vụ đã bị hủy.")


class VideoFrameSource:
    """Quản lý VideoCapture; đọc tuần tự và chỉ trả frame theo stride."""

    def __init__(self, path: str | Path) -> None:
        """Lưu đường dẫn; tài nguyên được mở khi vào context manager."""
        self.path = Path(path)
        self.total_frames: int = 0
        self.fps: float = video_io.DEFAULT_FPS
        self.width: int = 0
        self.height: int = 0
        self.duration_sec: float = 0.0
        self._capture: cv2.VideoCapture | None = None
        self._first_frame: Frame | None = None
        self._started: bool = False

    def __enter__(self) -> VideoFrameSource:
        """Mở video và lấy kích thước từ frame thực tế đã giải mã."""
        if self._capture is not None:
            raise RuntimeError("VideoFrameSource đang mở.")
        if not self.path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Không mở được video: {self.path}")

            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.fps = (
                raw_fps
                if math.isfinite(raw_fps)
                and 0 < raw_fps <= video_io.MAX_REASONABLE_FPS
                else video_io.DEFAULT_FPS
            )

            raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = (
                max(0, int(raw_count)) if math.isfinite(raw_count) else 0
            )
            self.duration_sec = self.total_frames / self.fps

            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(f"Video không có frame đọc được: {self.path}")

            self.height, self.width = map(int, frame.shape[:2])
            self._first_frame = frame
            self._started = False
            self._capture = capture
            return self
        except BaseException:
            capture.release()
            raise

    def iter_frames(
        self,
        stride: int,
        cancel_token: CancelToken | None = None,
    ) -> Iterator[tuple[int, Frame, float]]:
        """Trả frame 0, stride, 2*stride; timestamp = frame_index / fps.

        Mỗi lần mở nguồn chỉ được duyệt một lượt. grab() bỏ qua frame
        không cần xử lý; retrieve() giải mã frame được chọn.
        """
        stride = _validate_stride(stride)
        capture = self._capture
        if capture is None:
            raise RuntimeError("Phải dùng VideoFrameSource trong khối with.")
        if self._started:
            raise RuntimeError("Nguồn video đã được duyệt.")
        self._started = True

        first_frame = self._first_frame
        self._first_frame = None
        if first_frame is None:
            raise RuntimeError("Không có frame đầu tiên.")

        _check_cancelled(cancel_token)
        yield 0, first_frame, 0.0

        frame_index = 1
        while True:
            _check_cancelled(cancel_token)
            if not capture.grab():
                break

            if frame_index % stride == 0:
                success, frame = capture.retrieve()
                if not success or frame is None:
                    raise RuntimeError(
                        f"Không giải mã được frame {frame_index}: {self.path}"
                    )
                yield frame_index, frame, frame_index / self.fps

            frame_index += 1

    def close(self) -> None:
        """Giải phóng capture; gọi nhiều lần vẫn an toàn."""
        capture, self._capture = self._capture, None
        self._first_frame = None
        if capture is not None:
            capture.release()

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Luôn giải phóng capture khi rời context manager."""
        self.close()


class TrafficSignAnnotator:
    """Vẽ biển báo bằng draw_layer và mã hóa frame JPEG."""

    def annotate(
        self,
        frame: Frame,
        detections: list[Detection],
    ) -> Frame:
        """Vẽ bounding box và nhãn tiếng Việt, giữ nguyên frame đầu vào."""
        return draw_layer.draw_detections(frame, detections)

    @staticmethod
    def to_jpeg(frame: Frame, quality: int = 80) -> bytes:
        """Mã hóa frame BGR uint8 thành JPEG bytes."""
        if (
            isinstance(quality, bool)
            or not isinstance(quality, int)
            or not 1 <= quality <= 100
        ):
            raise ValueError("quality phải là số nguyên trong khoảng 1..100.")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise ValueError("frame phải là ảnh BGR uint8 có dạng (H, W, 3).")

        success, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(frame),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not success:
            raise RuntimeError("Không mã hóa được frame JPEG.")
        return encoded.tobytes()

    @staticmethod
    def jpeg_to_base64(jpeg_bytes: bytes) -> str:
        """Tạo Data URI từ JPEG bytes đã mã hóa."""
        payload = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def to_base64(self, frame: Frame) -> str:
        """Mã hóa frame thành Data URI JPEG với chất lượng mặc định."""
        return self.jpeg_to_base64(self.to_jpeg(frame))


class VoiceAlertManager:
    """Chống lặp, tổng hợp MP3 đồng bộ và ghi voice event vào ResultStore."""

    def __init__(
        self,
        cfg: dict[str, Any],
        classes: dict[int, dict[str, Any]],
        store: ResultStore,
        audio_dir: str | Path,
        job_id: int,
        frame_stride: int = 1,
    ) -> None:
        """Tạo trạng thái độc lập cho một job.

        Chỉ chấp nhận gtts vì engine pyttsx3 hiện tại không bảo đảm xuất
        MP3 thực sự. Cache âm thanh vẫn do tts.synthesize quản lý.
        """
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("inference", {})["frame_stride"] = (
            _validate_stride(frame_stride)
        )
        post = self.cfg.setdefault("postprocess", {})
        dedup = post.setdefault("dedup", {})
        dedup.setdefault("iou_match", 0.3)
        dedup.setdefault("lost_frames", 30)
        pos = post.setdefault("position", {})
        pos.setdefault("left_max", 0.35)
        pos.setdefault("right_min", 0.65)
        if "tts" in self.cfg and "tts" not in post:
            post["tts"] = self.cfg["tts"]
        post.setdefault("tts", {"engine": "mock", "voice": "vi-VN"})
        normalized_classes = {}
        for cid, info in (classes or {}).items():
            if isinstance(info, str):
                normalized_classes[cid] = {"name": info, "vietnamese": info, "tts_text": info}
            elif isinstance(info, dict):
                normalized_classes[cid] = dict(info)
            else:
                normalized_classes[cid] = {"name": str(cid), "vietnamese": str(cid)}
        self.classes = normalized_classes
        self.store = store
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.deduplicator = Deduplicator(self.cfg)
        self.unique_sign_count: int = 0
        self.voice_play_count: int = 0
        self.tts_time_ms: float = 0.0
        self._audio_index: int = 0

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp_sec: float,
        cancel_token: CancelToken | None = None,
    ) -> list[VoiceAlert]:
        """Trả cảnh báo sau khi MP3 đã tồn tại và voice event đã được ghi.

        audio_url là Data URI MP3, có thể phát trực tiếp trên trình duyệt.
        Lỗi TTS được truyền lên để job không báo hoàn tất sai.
        """
        _check_cancelled(cancel_token)
        for det in detections:
            if "label" not in det:
                cls_info = self.classes.get(det.get("class_id", -1))
                if isinstance(cls_info, dict):
                    det["label"] = cls_info.get("name") or cls_info.get("vietnamese") or str(det.get("class_id"))
                elif isinstance(cls_info, str):
                    det["label"] = cls_info
                else:
                    det["label"] = str(det.get("class_id", "unknown"))
            det.setdefault("position", "front")

        events = self.deduplicator.update(
            detections, frame_index, timestamp_sec
        )
        self.unique_sign_count += len(events)
        alerts: list[VoiceAlert] = []

        for event in events:
            _check_cancelled(cancel_token)
            started = time.perf_counter()
            audio_path = self.audio_dir / (
                f"job{self.job_id}_{self._audio_index}.mp3"
            )
            self._audio_index += 1

            try:
                text = tts.build_speech_text(
                    event["label"],
                    event["position"],
                    self.classes,
                    self.cfg,
                )
                tts.synthesize(text, audio_path, self.cfg)
                audio_bytes = audio_path.read_bytes()
                if not audio_bytes:
                    raise RuntimeError(f"File MP3 rỗng: {audio_path}")

                _check_cancelled(cancel_token)
                self.store.add_voice_event(
                    self.job_id,
                    event["detection"].get("db_id"),
                    text,
                    timestamp_sec,
                    str(audio_path),
                )
                self.voice_play_count += 1

                mime = (
                    "audio/wav"
                    if audio_bytes.startswith(b"RIFF")
                    or audio_path.suffix.lower() == ".wav"
                    else "audio/mpeg"
                )
                alerts.append(
                    VoiceAlert(
                        alert_id=uuid.uuid4().hex,
                        text=text,
                        audio_url=(
                            f"data:{mime};base64,"
                            + base64.b64encode(audio_bytes).decode("ascii")
                        ),
                        position=event["position"],
                        class_name=event["label"],
                    )
                )
            finally:
                self.tts_time_ms += (
                    time.perf_counter() - started
                ) * 1000.0

        return alerts


class SynchronizedPipeline:
    """Điều phối detector, vị trí, TTS, vẽ frame và lưu kết quả.

    Không tạo video hoặc ảnh kết quả trên đĩa; output_path khi hoàn tất
    là chuỗi rỗng. Chỉ các file âm thanh được ghi vào audio_dir.

    Một instance xử lý tối đa một job tại một thời điểm để bảo vệ detector.
    Khi dừng duyệt process_video sớm, người gọi phải đóng generator bằng
    close() hoặc contextlib.closing() để giải phóng tài nguyên ngay.
    """

    def __init__(
        self,
        detector: Detector,
        cfg: dict[str, Any],
        classes: dict[int, dict[str, Any]],
        store: ResultStore,
        audio_dir: str | Path,
        annotator: TrafficSignAnnotator | None = None,
    ) -> None:
        """Nhận các phụ thuộc đã khởi tạo từ tầng ứng dụng."""
        self.detector = detector
        self.cfg = copy.deepcopy(cfg)
        post = self.cfg.setdefault("postprocess", {})
        dedup = post.setdefault("dedup", {})
        dedup.setdefault("iou_match", 0.3)
        dedup.setdefault("lost_frames", 30)
        pos = post.setdefault("position", {})
        pos.setdefault("left_max", 0.35)
        pos.setdefault("right_min", 0.65)
        if "tts" in self.cfg and "tts" not in post:
            post["tts"] = self.cfg["tts"]
        post.setdefault("tts", {"engine": "mock", "voice": "vi-VN"})
        normalized_classes = {}
        for cid, info in (classes or {}).items():
            if isinstance(info, str):
                normalized_classes[cid] = {"name": info, "vietnamese": info, "tts_text": info}
            elif isinstance(info, dict):
                normalized_classes[cid] = dict(info)
            else:
                normalized_classes[cid] = {"name": str(cid), "vietnamese": str(cid)}
        self.classes = normalized_classes
        self.store = store
        self.audio_dir = Path(audio_dir)
        self.annotator = annotator or TrafficSignAnnotator()
        self._processing_lock = threading.Lock()

    def _process_frame(
        self,
        frame: Frame,
        frame_index: int,
        timestamp_sec: float,
        fps: float,
        progress_pct: float,
        job_id: int,
        voice_manager: VoiceAlertManager,
        cancel_token: CancelToken | None = None,
    ) -> tuple[SynchronizedFrame, float]:
        """Xử lý một frame; chỉ trả về khi mọi cảnh báo đã có âm thanh."""
        _check_cancelled(cancel_token)
        detections = [
            copy.deepcopy(detection)
            for detection in self.detector.predict(frame)
        ]
        for det in detections:
            if "label" not in det:
                cls_info = self.classes.get(det.get("class_id", -1))
                if isinstance(cls_info, dict):
                    det["label"] = cls_info.get("name") or cls_info.get("vietnamese") or str(det.get("class_id"))
                elif isinstance(cls_info, str):
                    det["label"] = cls_info
                else:
                    det["label"] = str(det.get("class_id", "unknown"))
            det.setdefault("position", "front")
        position.add_positions(detections, int(frame.shape[1]), self.cfg)
        _check_cancelled(cancel_token)

        self.store.add_detections(
            job_id, frame_index, timestamp_sec, detections
        )
        voice_alerts = voice_manager.update(
            detections, frame_index, timestamp_sec, cancel_token
        )

        _check_cancelled(cancel_token)
        started = time.perf_counter()
        annotated = self.annotator.annotate(frame, detections)
        jpeg_bytes = self.annotator.to_jpeg(annotated)
        image_base64 = self.annotator.jpeg_to_base64(jpeg_bytes)
        layer_time_ms = (time.perf_counter() - started) * 1000.0
        _check_cancelled(cancel_token)

        return (
            SynchronizedFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                image_base64=image_base64,
                detections=detections,
                voice_alerts=voice_alerts,
                progress_pct=progress_pct,
                fps=fps,
                is_done=False,
                jpeg_bytes=jpeg_bytes,
            ),
            layer_time_ms,
        )

    def _finish(
        self,
        job_id: int,
        manager: VoiceAlertManager,
        layer_time_ms: float,
        started: float,
    ) -> None:
        """Lưu tổng kết; số voice event là số âm thanh đã chuẩn bị thành công."""
        self.store.finish_job(
            job_id=job_id,
            output_path="",
            unique_sign_count=manager.unique_sign_count,
            voice_play_count=manager.voice_play_count,
            layer_time_ms=layer_time_ms,
            tts_time_ms=manager.tts_time_ms,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        self.store.set_status(job_id, "done")

    def _record_failure(self, job_id: int, message: str) -> None:
        """Ghi thất bại mà không che khuất ngoại lệ xử lý ban đầu."""
        try:
            self.store.fail_job(job_id, message)
        except Exception:
            logger.exception("Không ghi được trạng thái lỗi cho job %s", job_id)

    def process_video(
        self,
        video_path: str | Path,
        frame_stride: int,
        job_id: int,
        cancel_token: CancelToken | None = None,
    ) -> Iterator[SynchronizedFrame]:
        """Yield các frame đã xử lý, sau đó một thông báo is_done=True.

        TTS được tổng hợp trước khi yield frame tương ứng. Thời gian tổng
        của job bao gồm thời gian bên tiêu thụ tạm dừng giữa các lần next().
        Hủy tác vụ được ghi nhận là failed theo hợp đồng ResultStore.
        """
        if not self._processing_lock.acquire(blocking=False):
            raise RuntimeError("Pipeline đang xử lý một job khác.")

        started = time.perf_counter()
        completed = False
        layer_time_ms = 0.0
        last_index = -1
        last_timestamp = 0.0

        try:
            frame_stride = _validate_stride(frame_stride)
            _check_cancelled(cancel_token)
            self.store.set_status(job_id, "processing")
            manager = VoiceAlertManager(
                self.cfg,
                self.classes,
                self.store,
                self.audio_dir,
                job_id,
                frame_stride,
            )

            with VideoFrameSource(video_path) as source:
                fps = source.fps
                for frame_index, frame, timestamp in source.iter_frames(
                    frame_stride, cancel_token
                ):
                    progress = (
                        min(
                            99.9,
                            100.0 * (frame_index + 1) / source.total_frames,
                        )
                        if source.total_frames > 0
                        else 0.0
                    )
                    result, elapsed_ms = self._process_frame(
                        frame,
                        frame_index,
                        timestamp,
                        fps,
                        progress,
                        job_id,
                        manager,
                        cancel_token,
                    )
                    layer_time_ms += elapsed_ms
                    last_index = frame_index
                    last_timestamp = timestamp
                    yield result

            _check_cancelled(cancel_token)
            self._finish(job_id, manager, layer_time_ms, started)
            completed = True

            yield SynchronizedFrame(
                frame_index=last_index,
                timestamp_sec=last_timestamp,
                image_base64="",
                detections=[],
                voice_alerts=[],
                progress_pct=100.0,
                fps=fps,
                is_done=True,
            )
        except GeneratorExit:
            if not completed:
                self._record_failure(job_id, "Luồng xử lý bị đóng trước khi xong.")
            raise
        except Exception as exc:
            self._record_failure(job_id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._processing_lock.release()

    def process_image(
        self,
        image_path: str | Path,
        job_id: int,
    ) -> SynchronizedFrame:
        """Xử lý một ảnh, trả JPEG và cảnh báo sẵn sàng với is_done=True."""
        if not self._processing_lock.acquire(blocking=False):
            raise RuntimeError("Pipeline đang xử lý một job khác.")

        started = time.perf_counter()
        try:
            self.store.set_status(job_id, "processing")
            manager = VoiceAlertManager(
                self.cfg,
                self.classes,
                self.store,
                self.audio_dir,
                job_id,
            )
            frame = video_io.read_image(image_path)
            result, layer_time_ms = self._process_frame(
                frame=frame,
                frame_index=0,
                timestamp_sec=0.0,
                fps=0.0,
                progress_pct=100.0,
                job_id=job_id,
                voice_manager=manager,
            )
            self._finish(job_id, manager, layer_time_ms, started)
            result.is_done = True
            return result
        except Exception as exc:
            self._record_failure(job_id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._processing_lock.release()


class SynchronizedSession:
    """Quản lý một phiên xử lý video đồng bộ chạy ngầm và phát SSE."""

    def __init__(
        self,
        job_id: int,
        video_path: str | Path,
        frame_stride: int,
        pipeline: SynchronizedPipeline,
    ) -> None:
        self.job_id = job_id
        self.video_path = Path(video_path)
        self.frame_stride = frame_stride
        self.pipeline = pipeline
        self.cancel_event = threading.Event()
        self._subscribers_lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self.latest_frame: SynchronizedFrame | None = None
        self.is_done: bool = False
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Khởi động luồng xử lý video."""
        self.thread = threading.Thread(
            target=self._run,
            name=f"sync-job-{self.job_id}",
            daemon=True,
        )
        self.thread.start()

    def _broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Phát sự kiện tới tất cả subscriber."""
        msg = {"event": event_type, "data": data}
        with self._subscribers_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def _run(self) -> None:
        """Thực thi pipeline theo stride và broadcast kết quả."""
        try:
            self._broadcast(
                "status",
                {
                    "stage": "processing",
                    "job_id": self.job_id,
                    "frame_stride": self.frame_stride,
                },
            )
            for frame in self.pipeline.process_video(
                self.video_path,
                self.frame_stride,
                self.job_id,
                cancel_token=self.cancel_event,
            ):
                self.latest_frame = frame
                if frame.is_done:
                    self.is_done = True
                    self._broadcast(
                        "done",
                        {
                            "job_id": self.job_id,
                            "frame_index": frame.frame_index,
                            "timestamp_sec": frame.timestamp_sec,
                            "progress_pct": 100.0,
                            "is_done": True,
                        },
                    )
                    self._broadcast(
                        "status",
                        {
                            "job_id": self.job_id,
                            "status": "done",
                            "stage": "done",
                            "percent": 100.0,
                        },
                    )
                    break

                frame_payload = {
                    "job_id": self.job_id,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "image": frame.image_base64,
                    "detections": frame.detections,
                    "voice_alerts": frame.voice_alerts,
                    "progress_pct": frame.progress_pct,
                    "fps": frame.fps,
                    "is_done": False,
                }
                self._broadcast("frame", frame_payload)
                for alert in frame.voice_alerts:
                    self._broadcast(
                        "alert",
                        {
                            "job_id": self.job_id,
                            "alert_id": alert["alert_id"],
                            "label": alert["class_name"],
                            "class_name": alert["class_name"],
                            "position": alert["position"],
                            "text": alert["text"],
                            "audio_url": alert["audio_url"],
                            "source_timestamp": frame.timestamp_sec,
                            "timestamp_sec": frame.timestamp_sec,
                            "status": "ready",
                        },
                    )
        except Exception as exc:
            self.error = str(exc)
            self._broadcast(
                "error",
                {
                    "job_id": self.job_id,
                    "message": str(exc),
                },
            )
        finally:
            self.is_done = True
            with self._subscribers_lock:
                for q in self._subscribers:
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        """Đăng ký nhận luồng sự kiện SSE."""
        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=120)
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any] | None]) -> None:
        """Hủy đăng ký nhận sự kiện."""
        with self._subscribers_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def cancel(self) -> None:
        """Hủy phiên làm việc."""
        self.cancel_event.set()

    def iter_sse(self, heartbeat_sec: float = 0.5) -> Iterator[str]:
        """Tạo generator SSE phục vụ HTTP response."""
        q = self.subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=heartbeat_sec)
                    if msg is None:
                        break
                    yield (
                        f"event: {msg['event']}\n"
                        f"data: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"
                    )
                    if msg["event"] in ("done", "error"):
                        break
                except queue.Empty:
                    if self.is_done:
                        break
                    yield ": heartbeat\n\n"
        finally:
            self.unsubscribe(q)


class SynchronizedJobManager:
    """Quản lý các phiên xử lý đồng bộ và định tuyến SSE."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[int, SynchronizedSession] = {}

    def create_session(
        self,
        job_id: int,
        video_path: str | Path,
        frame_stride: int,
        pipeline: SynchronizedPipeline,
    ) -> SynchronizedSession:
        """Tạo và kích hoạt một phiên xử lý video đồng bộ."""
        with self._lock:
            session = SynchronizedSession(
                job_id=job_id,
                video_path=video_path,
                frame_stride=frame_stride,
                pipeline=pipeline,
            )
            self._sessions[job_id] = session
            session.start()
            return session

    def get_session(self, job_id: int) -> SynchronizedSession | None:
        """Lấy phiên đang quản lý theo job_id."""
        with self._lock:
            return self._sessions.get(job_id)

    def cancel_session(self, job_id: int) -> bool:
        """Yêu cầu hủy phiên đang chạy."""
        with self._lock:
            session = self._sessions.get(job_id)
        if session is not None:
            session.cancel()
            return True
        return False

