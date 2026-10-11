"""Kiểm tra LUỒNG tổng thể bằng cách giả lập (mock) các hàm TODO và nơi lưu log."""
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
