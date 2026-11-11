"""Kiểm tra LUỒNG tổng thể bằng cách giả lập (mock) các hàm TODO và nơi lưu log."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ts_alert.pipeline as flow
from ts_alert.config.resolve import PROJECT_ROOT, resolve_config
from ts_alert.postprocess import draw_layer, merge_av, position, tts
from ts_alert.runtime import video_io


class FakeRepo:
    def __init__(self):
        self.detections, self.voices, self.statuses = [], [], []
        self.finished = self.failed = None

    def create_job(self, input_type, input_path, model_name):
        return 1

    def set_status(self, job_id, status):
        self.statuses.append(status)

    def add_detections(self, job_id, frame_index, timestamp_sec, detections):
        for d in detections:
            self.detections.append(d)
            d["db_id"] = len(self.detections)

    def add_voice_event(self, job_id, detection_id, spoken_text, timestamp_sec, audio_path):
        self.voices.append((detection_id, spoken_text, timestamp_sec))

    def finish_job(self, *args):
        self.finished = args

    def fail_job(self, job_id, error_message):
        self.failed = error_message


class FakeDetector:
    def predict(self, frame):
        return [{"bbox": [0, 0, 10, 10], "label": "Cấm rẽ trái", "class_id": 0, "confidence": 0.9}]


class FakeDedup:
    def __init__(self, cfg):
        self.seen = set()

    def update(self, detections, frame_index, timestamp_sec):
        events = []
        for d in detections:
            if d["label"] not in self.seen:
                self.seen.add(d["label"])
                events.append({"detection": d, "label": d["label"], "position": d["position"],
                               "frame_index": frame_index, "timestamp_sec": timestamp_sec})
        return events


class FakeWriter:
    def __init__(self, path, fps, size):
        self.path = path

    def write(self, frame):
        pass

    def close(self):
        self.path.write_bytes(b"video")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = resolve_config("yolo")
    cfg["inference"] = {**cfg.get("inference", {}), "frame_stride": 1}
    cfg["paths"] = {**cfg["paths"], "classes_csv": str(PROJECT_ROOT / "configs/classes.csv"),
                    "outputs_dir": str(tmp_path / "out")}
    monkeypatch.setattr(video_io, "get_video_info", lambda p: {"fps": 10.0, "width": 30, "height": 20, "frame_count": 3})
    monkeypatch.setattr(video_io, "iter_frames", lambda p: ((i, np.zeros((20, 30, 3), np.uint8), i / 10) for i in range(3)))
    monkeypatch.setattr(video_io, "VideoWriter", FakeWriter)
    monkeypatch.setattr(position, "get_position", lambda bbox, w, cfg: "front")
    monkeypatch.setattr(draw_layer, "draw_detections", lambda frame, dets: frame)
    monkeypatch.setattr(tts, "build_speech_text", lambda label, pos, classes: f"{pos} {label}")
    monkeypatch.setattr(tts, "synthesize", lambda text, out, cfg: out.write_bytes(b"audio") or out)
    monkeypatch.setattr(merge_av, "merge_audio_video", lambda v, ev, out: out.write_bytes(v.read_bytes() + b"+audio"))
    monkeypatch.setattr(flow, "Deduplicator", FakeDedup)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    return cfg, tmp_path, src, FakeRepo()


def test_video_flow(setup):
    cfg, tmp, src, repo = setup
    result = flow.process_media(src, "yolo", cfg=cfg, detector=FakeDetector(), repo=repo)

    assert result["status"] == "done"
    assert result["unique_sign_count"] == 1       # chống lặp: 3 khung hình nhưng chỉ 1 biển
    assert result["voice_play_count"] == 1
    assert repo.statuses == ["processing", "done"]
    assert len(repo.detections) == 3
    assert repo.voices[0][0] == 1                 # voice event tham chiếu detection kích hoạt
    assert repo.finished[2:4] == (1, 1)           # (unique_sign_count, voice_play_count)
    assert (tmp / "out" / "videos" / "job1_clip_result.mp4").read_bytes() == b"video+audio"


def test_failure_marks_job_failed(setup):
    cfg, tmp, src, repo = setup

    class Boom:
        def predict(self, frame):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        flow.process_media(src, "yolo", cfg=cfg, detector=Boom(), repo=repo)
    assert "boom" in repo.failed


def test_video_flow_frame_stride_reuses_boxes_and_emits_on_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chỉ suy luận mỗi 30 khung nhưng vẫn vẽ, ghi và phát callback đủ 35 khung."""
    cfg = resolve_config("yolo")
    cfg["inference"] = {**cfg.get("inference", {}), "frame_stride": 30}
    cfg["paths"] = {
        **cfg["paths"],
        "classes_csv": str(PROJECT_ROOT / "configs/classes.csv"),
        "outputs_dir": str(tmp_path / "out"),
    }

    frames = [
        np.full((20, 30, 3), index, dtype=np.uint8)
        for index in range(35)
    ]
    draw_calls: list[tuple[int, list[dict[str, Any]]]] = []
    written_frames: list[np.ndarray] = []
    callbacks: list[dict[str, Any]] = []

    class SpyDetector(FakeDetector):
        """Ghi nhận khung suy luận và trả bounding box thay đổi theo khung."""

        def __init__(self) -> None:
            self.predict_frames: list[int] = []

        def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
            """Trả một detection có tọa độ phân biệt hai lần suy luận."""
            frame_index = int(frame[0, 0, 0])
            self.predict_frames.append(frame_index)
            detections = super().predict(frame)
            offset = frame_index // 30
            detections[0]["bbox"] = [offset, 0, 10 + offset, 10]
            return detections

    class SpyWriter(FakeWriter):
        """Lưu các khung thực sự được truyền vào writer."""

        def write(self, frame: np.ndarray) -> None:
            """Ghi nhận từng khung đầu ra theo đúng thứ tự."""
            written_frames.append(frame)

    def record_draw(
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """Chụp lại detections tại thời điểm vẽ từng khung."""
        draw_calls.append(
            (
                int(frame[0, 0, 0]),
                [
                    {**detection, "bbox": list(detection["bbox"])}
                    for detection in detections
                ],
            )
        )
        return frame.copy()

    def record_on_frame(**kwargs: Any) -> None:
        """Ghi nhận dữ liệu callback của mỗi khung đầu ra."""
        callbacks.append(kwargs)

    monkeypatch.setattr(
        video_io,
        "get_video_info",
        lambda path: {
            "fps": 10.0,
            "width": 30,
            "height": 20,
            "frame_count": 35,
        },
    )
    monkeypatch.setattr(
        video_io,
        "iter_frames",
        lambda path: (
            (index, frame, index / 10)
            for index, frame in enumerate(frames)
        ),
    )
    monkeypatch.setattr(video_io, "VideoWriter", SpyWriter)
    monkeypatch.setattr(position, "get_position", lambda bbox, width, cfg: "front")
    monkeypatch.setattr(draw_layer, "draw_detections", record_draw)
    monkeypatch.setattr(
        tts,
        "build_speech_text",
        lambda label, pos, classes: f"{pos} {label}",
    )
    monkeypatch.setattr(
        tts,
        "synthesize",
        lambda text, out, cfg: out.write_bytes(b"audio") or out,
    )
    monkeypatch.setattr(
        merge_av,
        "merge_audio_video",
        lambda video, events, out: out.write_bytes(
            video.read_bytes() + b"+audio"
        ),
    )
    monkeypatch.setattr(flow, "Deduplicator", FakeDedup)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    repo = FakeRepo()
    detector = SpyDetector()

    result = flow.process_media(
        src,
        "yolo",
        cfg=cfg,
        detector=detector,
        repo=repo,
        on_frame=record_on_frame,
    )

    assert result["status"] == "done"
    assert detector.predict_frames == [0, 30]
    assert len(repo.detections) == 2
    assert [detection["bbox"] for detection in repo.detections] == [
        [0, 0, 10, 10],
        [1, 0, 11, 10],
    ]

    assert len(draw_calls) == 35
    assert [index for index, _ in draw_calls] == list(range(35))
    assert len(draw_calls[0][1]) == 1
    for frame_index in range(1, 30):
        assert draw_calls[frame_index][1] == draw_calls[0][1]
    assert draw_calls[30][1] != draw_calls[0][1]
    for frame_index in range(31, 35):
        assert draw_calls[frame_index][1] == draw_calls[30][1]

    assert len(written_frames) == 35
    assert [int(frame[0, 0, 0]) for frame in written_frames] == list(range(35))
    assert len(callbacks) == 35
    assert [callback["frame_index"] for callback in callbacks] == list(range(35))
    assert [callback["timestamp_sec"] for callback in callbacks] == [
        index / 10 for index in range(35)
    ]
    for frame_index, callback in enumerate(callbacks):
        assert callback["predicted"] is (frame_index in (0, 30))
        assert callback["frame"] is written_frames[frame_index]
        assert callback["detections"] == draw_calls[frame_index][1]


def test_invalid_frame_stride_raises(setup: tuple[Any, ...]) -> None:
    """Từ chối stride không hợp lệ và đánh dấu job thất bại mỗi lần."""
    cfg, _, src, repo = setup

    for frame_stride in (0, -1, True, 1.5):
        cfg["inference"]["frame_stride"] = frame_stride
        repo.failed = None

        with pytest.raises(ValueError, match=r"inference\.frame_stride"):
            flow.process_media(
                src,
                "yolo",
                cfg=cfg,
                detector=FakeDetector(),
                repo=repo,
            )

        assert repo.failed is not None
        assert repo.failed.startswith("ValueError:")
        assert "inference.frame_stride" in repo.failed
        assert repo.finished is None
