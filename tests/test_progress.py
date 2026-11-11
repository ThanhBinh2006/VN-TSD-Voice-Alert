"""Test tiến độ % của job (Thành viên 3): ProgressTracker, _TrackingStore và trường "progress" trong API."""
import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.fake import FakeDetector
from app.jobs import LiveFrameBuffer, ProgressTracker, _TrackingStore
from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.runtime import video_io

N_FRAMES = 30


def _video_bytes(tmp_path, n=N_FRAMES, fps=10.0, w=96, h=64):
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, fps, (w, h)) as wr:
        for i in range(n):
            wr.write(np.full((h, w, 3), i * 5 % 256, np.uint8))
    return p.read_bytes()


@pytest.fixture
def app_and_log(tmp_path):
    """App chạy job đồng bộ; detector ghi lại tiến độ mà API trả về TẠI MỖI khung hình."""
    out = tmp_path / "outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        },
        "inference": {"frame_stride": 1},
    }
    app = create_app(fake=True, store=InMemoryStore(), sync=True, overrides=overrides)
    runner = app.extensions["tsd"]["runner"]
    seen = []

    class SpyDetector(FakeDetector):
        def predict(self, frame):
            seen.append(runner.progress.get(1))
            return super().predict(frame)

    runner.detector_factory = lambda name, cfg: SpyDetector()
    yield app, seen
    app.extensions["tsd"]["restore_fakes"]()


def _upload(client, data, filename):
    return client.post("/api/jobs", data={"file": (io.BytesIO(data), filename), "model": "yolo"},
                       content_type="multipart/form-data")


def test_progress_increases_during_job_and_ends_at_100(app_and_log, tmp_path):
    app, seen = app_and_log
    client = app.test_client()
    job_id = _upload(client, _video_bytes(tmp_path), "clip.mp4").get_json()["job_id"]

    assert len(seen) == N_FRAMES
    assert all(p["stage"] == "processing" and p["frame_count"] == N_FRAMES for p in seen)
    percents = [p["percent"] for p in seen]
    assert percents[0] == 0.0
    assert percents == sorted(percents)                 # tăng dần
    assert max(percents) < 100                          # đang chạy thì chưa bao giờ báo 100%

    progress = client.get(f"/api/jobs/{job_id}").get_json()["progress"]
    assert progress["stage"] == "done" and progress["percent"] == 100.0
    assert progress["frames_done"] == N_FRAMES and progress["eta_sec"] is None
    assert progress["fps"] and progress["fps"] > 0     # tốc độ xử lý (khung/giây) -> dùng cho báo cáo
    assert client.get("/api/jobs").get_json()[0]["progress"]["stage"] == "done"


def test_progress_image_and_failed(app_and_log, tmp_path):
    app, _ = app_and_log
    client = app.test_client()
    img = tmp_path / "a.png"
    video_io.write_image(img, np.zeros((32, 32, 3), np.uint8))
    job_id = _upload(client, img.read_bytes(), "a.png").get_json()["job_id"]
    p = client.get(f"/api/jobs/{job_id}").get_json()["progress"]
    assert p["stage"] == "done" and p["percent"] == 100.0 and p["frame_count"] == 1

    job_id = _upload(client, b"not a video", "hong.mp4").get_json()["job_id"]
    p = client.get(f"/api/jobs/{job_id}").get_json()["progress"]
    assert p["stage"] == "failed" and p["percent"] is None   # không biết tổng số khung


def test_tracker_unknown_total_and_finalizing():
    t = ProgressTracker()
    assert t.get(1) is None
    t.queued(1)
    assert t.get(1)["stage"] == "queued" and t.get(1)["percent"] is None
    t.start_processing(1, None)                          # metadata không có frame_count
    t.frame_done(1, 4)
    assert t.get(1)["percent"] is None and t.get(1)["frames_done"] == 5

    t.start_processing(2, 10)
    for i in range(10):
        t.frame_done(2, i)
        time.sleep(0.001)
    p = t.get(2)
    assert p["stage"] == "finalizing" and p["percent"] == 99.0   # đủ khung nhưng chưa ghi xong kết quả
    fps_at_finalizing = p["fps"]
    time.sleep(0.05)
    assert t.get(2)["fps"] == fps_at_finalizing          # đã chốt giờ: fps không giảm dần sau khi xong khung
    t.set_stage(2, "done")
    assert t.get(2)["percent"] == 100.0 and t.get(2)["fps"] == fps_at_finalizing


def test_tracking_store_forwards_everything():
    store, tracker = InMemoryStore(), ProgressTracker()
    wrapped = _TrackingStore(store, tracker)
    jid = wrapped.create_job("video", "x.mp4", "yolo")
    tracker.start_processing(jid, 2)
    dets = [{"bbox": [0, 0, 1, 1], "label": "A", "class_id": 0, "confidence": 0.9, "position": "left"}]
    wrapped.add_detections(jid, 0, 0.0, dets)
    assert dets[0]["db_id"] == 1                         # store thật vẫn nhận và gán db_id
    assert store.get_detections(jid)[0]["label"] == "A"
    assert tracker.get(jid)["frames_done"] == 1 and tracker.get(jid)["percent"] == 50.0


def test_loading_model_stage_only_on_first_job(app_and_log, tmp_path):
    """Job đầu tiên của một model đi qua stage loading_model; job sau dùng lại detector nên không nạp lại."""
    app, _ = app_and_log
    runner = app.extensions["tsd"]["runner"]
    stages_when_loading = []

    def factory(name, cfg):
        stages_when_loading.append(runner.progress.get(len(stages_when_loading) + 1)["stage"])
        return FakeDetector()

    runner.detector_factory = factory
    client = app.test_client()
    video = _video_bytes(tmp_path, n=5)
    _upload(client, video, "a.mp4")
    _upload(client, video, "b.mp4")
    assert stages_when_loading == ["loading_model"]        # chỉ nạp 1 lần cho 2 job
    assert runner.loaded_models() == ["yolo"]


def test_progress_with_frame_stride(tmp_path: Path) -> None:
    """Stride 10 chỉ inference ba lần nhưng tiến độ vẫn tính đủ 30 frame."""
    out = tmp_path / "outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        },
        "inference": {"frame_stride": 10},
    }
    app = create_app(
        fake=True,
        store=InMemoryStore(),
        sync=True,
        overrides=overrides,
    )
    runner = app.extensions["tsd"]["runner"]
    predicted_frames: list[int] = []

    class SpyDetector(FakeDetector):
        """Ghi lại vị trí inference thông qua số frame đã xử lý."""

        def predict(self, frame: np.ndarray) -> list:
            """Ghi nhận lần inference rồi gọi detector giả."""
            progress = runner.progress.get(1)
            assert progress is not None
            predicted_frames.append(progress["frames_done"])
            return super().predict(frame)

    runner.detector_factory = lambda name, cfg: SpyDetector()

    try:
        client = app.test_client()
        response = _upload(client, _video_bytes(tmp_path), "stride.mp4")
        job_id = response.get_json()["job_id"]

        assert len(predicted_frames) == 3
        assert predicted_frames == [0, 10, 20]

        progress = client.get(f"/api/jobs/{job_id}").get_json()["progress"]
        assert progress["stage"] == "done"
        assert progress["percent"] == 100.0
        assert progress["frames_done"] == N_FRAMES
        assert progress["frame_stride"] == 10
        assert progress["video_fps"] is not None and progress["video_fps"] > 0
        assert progress["duration_sec"] is not None and progress["duration_sec"] > 0
    finally:
        runner.shutdown()
        app.extensions["tsd"]["restore_fakes"]()


def test_live_frame_buffer_streams_and_unblocks_on_shutdown() -> None:
    """Kiểm tra JPEG, kết thúc MJPEG, giới hạn slot và đánh thức khi shutdown."""
    buffer = LiveFrameBuffer(max_jobs=2)
    threads: list[threading.Thread] = []

    def collect_stream(job_id: int, chunks: list[bytes]) -> None:
        """Đọc hết stream; heartbeat dài giúp phát hiện thiếu notify."""
        chunks.extend(buffer.iter_mjpeg(job_id, heartbeat_sec=30.0))

    try:
        buffer.init_job(1)
        assert buffer.has_slot(1)
        assert buffer.latest_frame(1) is None

        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        buffer.publish(1, frame, frame_index=0, timestamp_sec=0.0)

        jpeg = buffer.latest_frame(1)
        assert jpeg is not None
        assert jpeg.startswith(b"\xff\xd8")

        buffer.mark_finished(1)
        chunks: list[bytes] = []
        finished_reader = threading.Thread(
            target=collect_stream,
            args=(1, chunks),
            daemon=True,
        )
        threads.append(finished_reader)
        finished_reader.start()
        finished_reader.join(timeout=2.0)

        assert not finished_reader.is_alive()
        assert len(chunks) == 1
        assert chunks[0].startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg\r\n" in chunks[0]
        assert jpeg in chunks[0]

        buffer.init_job(2)
        buffer.init_job(3)
        assert not buffer.has_slot(1)
        assert buffer.latest_frame(1) is None
        assert buffer.has_slot(2)
        assert buffer.has_slot(3)

        waiting_chunks: list[bytes] = []
        waiting_reader = threading.Thread(
            target=collect_stream,
            args=(2, waiting_chunks),
            daemon=True,
        )
        threads.append(waiting_reader)
        waiting_reader.start()
        waiting_reader.join(timeout=0.05)
        assert waiting_reader.is_alive()

        buffer.shutdown()
        waiting_reader.join(timeout=2.0)
        assert not waiting_reader.is_alive()
        assert waiting_chunks == []
    finally:
        buffer.shutdown()
        for thread in threads:
            thread.join(timeout=2.0)
