"""Unit tests for ts_alert.runtime.synchronized_processor.

Kiểm thử toàn diện các lớp OOP:
- VideoFrameSource (đọc theo stride chuẩn)
- TrafficSignAnnotator (vẽ bounding box và nhãn tiếng Việt, xuất Base64)
- VoiceAlertManager (dedup, tổng hợp giọng nói, xuất Base64 data URI)
- SynchronizedPipeline (chuỗi xử lý đồng bộ)
- SynchronizedSession & SynchronizedJobManager (SSE streaming đồng bộ)
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.store_memory import InMemoryStore
from ts_alert.runtime import video_io
from ts_alert.runtime.synchronized_processor import (
    ProcessingCancelledError,
    SynchronizedFrame,
    SynchronizedJobManager,
    SynchronizedPipeline,
    SynchronizedSession,
    TrafficSignAnnotator,
    VideoFrameSource,
    VoiceAlertManager,
    _validate_stride,
)
from ts_alert.types import Detection


class DummyDetector:
    """Detector giả lập trả về biển báo ở frame đầu tiên."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.call_count = 0
        self.detections = detections or [
            {
                "class_id": 0,
                "confidence": 0.95,
                "bbox": [10.0, 10.0, 50.0, 50.0],
            }
        ]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        self.call_count += 1
        # Trả về detection ở lần gọi đầu tiên
        if self.call_count == 1:
            return list(self.detections)
        return []


@pytest.fixture(autouse=True)
def mock_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Giả lập hàm tổng hợp giọng nói để test chạy nhanh offline."""
    def _fake_synth(text: str, out_path: Path, cfg: dict) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00dummy-mp3-audio-bytes")
        return p

    monkeypatch.setattr("ts_alert.postprocess.tts.synthesize", _fake_synth)


def _make_dummy_video(path: Path, num_frames: int = 30) -> None:
    """Tạo video giả lập có num_frames khung hình."""
    with video_io.VideoWriter(path, 10.0, (100, 100)) as writer:
        for i in range(num_frames):
            frame = np.full((100, 100, 3), i * 5 % 256, dtype=np.uint8)
            writer.write(frame)


def test_validate_stride() -> None:
    """Kiểm tra validation frame_stride."""
    assert _validate_stride(1) == 1
    assert _validate_stride(15) == 15
    with pytest.raises(ValueError):
        _validate_stride(0)
    with pytest.raises(ValueError):
        _validate_stride(-5)
    with pytest.raises(ValueError):
        _validate_stride(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _validate_stride(1.5)  # type: ignore[arg-type]


def test_video_frame_source(tmp_path: Path) -> None:
    """Kiểm tra VideoFrameSource đọc chính xác theo stride."""
    video_path = tmp_path / "test.mp4"
    _make_dummy_video(video_path, num_frames=25)

    with VideoFrameSource(video_path) as source:
        assert source.fps == 10.0
        assert source.width == 100
        assert source.height == 100
        assert source.total_frames == 25

        frames = list(source.iter_frames(stride=10))
        assert len(frames) == 3
        # Các frame đọc được: 0, 10, 20
        indices = [idx for idx, _, _ in frames]
        assert indices == [0, 10, 20]
        for _, frame, ts in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (100, 100, 3)
            assert ts >= 0.0


def test_traffic_sign_annotator() -> None:
    """Kiểm tra TrafficSignAnnotator vẽ nhãn và xuất Base64 Data URI."""
    annotator = TrafficSignAnnotator()
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections: list[Detection] = [
        {"class_id": 0, "confidence": 0.9, "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]

    annotated = annotator.annotate(dummy_frame, detections)
    assert isinstance(annotated, np.ndarray)
    b64_uri = annotator.to_base64(annotated)
    assert b64_uri.startswith("data:image/jpeg;base64,")
    jpeg_bytes = annotator.to_jpeg(annotated)
    assert len(jpeg_bytes) > 0


def test_voice_alert_manager(tmp_path: Path) -> None:
    """Kiểm tra VoiceAlertManager dedup, phát âm thanh và xuất Base64 Data URI."""
    classes = {0: "Cấm đi ngược chiều"}
    cfg = {
        "tts": {"engine": "mock", "voice": "vi-VN"},
        "postprocess": {"dedup_window_sec": 2.0, "iou_threshold": 0.5},
    }
    store = InMemoryStore()
    job_id = store.create_job("video", "test.mp4", "yolo")
    manager = VoiceAlertManager(
        cfg=cfg,
        classes=classes,
        store=store,
        audio_dir=tmp_path / "audio",
        job_id=job_id,
        frame_stride=1,
    )

    detections: list[Detection] = [
        {"class_id": 0, "confidence": 0.9, "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]

    # Lần 1: Tạo alert mới
    alerts1 = manager.update(detections=detections, frame_index=0, timestamp_sec=0.5)
    assert len(alerts1) == 1
    assert alerts1[0]["class_name"] == "Cấm đi ngược chiều"
    assert alerts1[0]["audio_url"].startswith("data:audio/")
    assert "text" in alerts1[0]

    # Lần 2: Cùng biển trong cửa sổ dedup -> không tạo alert lặp lại
    alerts2 = manager.update(detections=detections, frame_index=1, timestamp_sec=1.0)
    assert len(alerts2) == 0


def test_synchronized_pipeline_end_to_end(tmp_path: Path) -> None:
    """Kiểm tra SynchronizedPipeline xử lý toàn bộ video theo stride."""
    video_path = tmp_path / "test_pipeline.mp4"
    _make_dummy_video(video_path, num_frames=20)

    classes = {0: "Biển báo nguy hiểm"}
    cfg = {
        "tts": {"engine": "mock"},
        "postprocess": {"dedup_window_sec": 2.0, "iou_threshold": 0.5},
    }
    store = InMemoryStore()
    job_id = store.create_job("video", "test.mp4", "yolo")
    pipeline = SynchronizedPipeline(
        detector=DummyDetector(),
        cfg=cfg,
        classes=classes,
        store=store,
        audio_dir=tmp_path / "audio",
    )

    frames = list(pipeline.process_video(job_id=job_id, video_path=video_path, frame_stride=5))
    # 20 frames với stride=5 -> frame 0, 5, 10, 15 và 1 frame sentinel kết thúc -> 5 frames
    assert len(frames) == 5
    assert frames[-1].is_done is True
    assert frames[-1].progress_pct == 100.0

    for frame in frames[:-1]:
        assert isinstance(frame, SynchronizedFrame)
        assert frame.image_base64.startswith("data:image/jpeg;base64,")
        assert 0.0 <= frame.progress_pct <= 100.0

    # Frame đầu tiên có voice_alert
    assert len(frames[0].voice_alerts) == 1
    assert frames[0].voice_alerts[0]["audio_url"].startswith("data:audio/")


def test_synchronized_session_and_manager(tmp_path: Path) -> None:
    """Kiểm tra SynchronizedSession truyền tải sự kiện qua SSE."""
    video_path = tmp_path / "test_session.mp4"
    _make_dummy_video(video_path, num_frames=15)

    classes = {0: "Cấm rẽ trái"}
    cfg = {
        "tts": {"engine": "mock"},
        "postprocess": {"dedup_window_sec": 2.0, "iou_threshold": 0.5},
    }
    store = InMemoryStore()
    job_id = store.create_job("video", "test.mp4", "yolo")
    pipeline = SynchronizedPipeline(
        detector=DummyDetector(),
        cfg=cfg,
        classes=classes,
        store=store,
        audio_dir=tmp_path / "audio",
    )

    manager = SynchronizedJobManager()
    session = manager.create_session(
        job_id=job_id,
        video_path=video_path,
        frame_stride=5,
        pipeline=pipeline,
    )

    assert manager.get_session(job_id) is session

    events = []
    for line in session.iter_sse(heartbeat_sec=0.2):
        for subline in line.splitlines():
            if subline.startswith("event: "):
                events.append(subline.split("event: ")[1].strip())

    # Kiểm tra đã nhận các event frame, alert và kết thúc bằng done
    assert "frame" in events
    assert "alert" in events
    assert "done" in events
    assert session.is_done
