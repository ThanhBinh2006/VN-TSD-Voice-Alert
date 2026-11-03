"""Web chạy với Repository SQLite thật của Thành viên 5 (Thành viên 3): API trả cùng dạng như khi lưu RAM."""
import io

import numpy as np
import pytest

from app.fake import is_stub
from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.db.repository import Repository
from ts_alert.runtime import video_io


def _overrides(tmp_path):
    out = tmp_path / "outputs"
    return {"paths": {"outputs_dir": str(out), "uploads_dir": str(out / "uploads"),
                      "db_path": str(out / "test.db")}}


@pytest.fixture
def app(tmp_path):
    app = create_app(fake=True, store_kind="auto", sync=True, overrides=_overrides(tmp_path))
    yield app
    app.extensions["tsd"]["restore_fakes"]()


def _video_bytes(tmp_path, n=40):
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, 10.0, (96, 64)) as wr:
        for i in range(n):
            wr.write(np.full((64, 96, 3), i * 5 % 256, np.uint8))
    return p.read_bytes()


def test_auto_store_uses_sqlite_once_repository_is_done(app):
    assert isinstance(app.extensions["tsd"]["store"], Repository)
    assert app.test_client().get("/api/health").get_json()["store"] == "Repository"


def test_video_job_with_sqlite(app, tmp_path):
    client = app.test_client()
    resp = client.post("/api/jobs", data={"file": (io.BytesIO(_video_bytes(tmp_path)), "clip.mp4"), "model": "yolo"},
                       content_type="multipart/form-data")
    job_id = resp.get_json()["job_id"]

    history = client.get("/api/jobs").get_json()             # cột lịch sử (Repository.get_jobs)
    assert [j["id"] for j in history] == [job_id]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done" and job["progress"]["percent"] == 100.0
    assert isinstance(job["created_at"], float)              # giao diện đọc giờ dạng số giây
    assert client.get(job["result_url"]).status_code == 200

    dets = client.get(f"/api/jobs/{job_id}/detections").get_json()
    assert dets and len(dets[0]["bbox"]) == 4
    events = client.get(f"/api/jobs/{job_id}/voice-events").get_json()
    assert events and events[0]["position"] in ("left", "front", "right")
    assert client.get(events[0]["audio_url"]).status_code == 200


def test_unfinished_jobs_marked_failed_on_restart(tmp_path):
    overrides = _overrides(tmp_path)
    repo = Repository(overrides["paths"]["db_path"])
    stuck = repo.create_job("video", "x.mp4", "yolo")
    repo.set_status(stuck, "processing")                     # giả lập server tắt giữa chừng

    app = create_app(fake=True, store_kind="sqlite", sync=True, overrides=overrides)
    try:
        job = app.test_client().get(f"/api/jobs/{stuck}").get_json()
        assert job["status"] == "failed" and "gửi lại" in job["error_message"]
    finally:
        app.extensions["tsd"]["restore_fakes"]()


def test_is_stub_ignores_commented_marker():
    def stub():
        raise NotImplementedError("TODO: Thành viên 4")

    def done():
        return 1
        # raise NotImplementedError("TODO: Thành viên 5")

    assert is_stub(stub) and not is_stub(done)


def test_memory_store_get_jobs_matches_repository():
    s = InMemoryStore()
    for model in ("yolo", "faster_rcnn", "yolo"):
        s.create_job("video", "x.mp4", model)
    s.set_status(3, "done")
    assert [j["id"] for j in s.get_jobs()] == [3, 2, 1]
    assert [j["id"] for j in s.get_jobs(limit=1, offset=1)] == [2]
    assert [j["id"] for j in s.get_jobs(model_name="yolo", status="done")] == [3]
