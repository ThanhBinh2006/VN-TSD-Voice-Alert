"""Test web backend (Thành viên 3): chế độ fake + lưu RAM + chạy job đồng bộ, không cần model thật."""
import io

import numpy as np
import pytest

from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.runtime import video_io


def _video_bytes(tmp_path, n=40, fps=10.0, w=96, h=64):
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, fps, (w, h)) as wr:
        for i in range(n):
            wr.write(np.full((h, w, 3), i * 5 % 256, np.uint8))
    return p.read_bytes()


def _image_bytes(tmp_path):
    p = tmp_path / "src.png"
    video_io.write_image(p, np.zeros((64, 96, 3), np.uint8))
    return p.read_bytes()


@pytest.fixture
def client(tmp_path):
    out = tmp_path / "outputs"
    overrides = {"paths": {"outputs_dir": str(out), "uploads_dir": str(out / "uploads"),
                           "db_path": str(out / "test.db")}}
    app = create_app(fake=True, store=InMemoryStore(), sync=True, overrides=overrides)
    app.config["TESTING"] = True
    yield app.test_client()
    app.extensions["tsd"]["restore_fakes"]()


def _upload(client, data, filename, model="yolo"):
    return client.post("/api/jobs", data={"file": (io.BytesIO(data), filename), "model": model},
                       content_type="multipart/form-data")


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["fake"] is True
    assert set(body["models"]) == {"yolo", "faster_rcnn"}


def test_video_job_end_to_end(client, tmp_path):
    resp = _upload(client, _video_bytes(tmp_path), "đoạn đường.mp4")   # tên file có dấu
    assert resp.status_code == 202
    job_id = resp.get_json()["job_id"]

    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error_message")
    assert job["input_type"] == "video" and job["model_name"] == "yolo"
    assert job["unique_sign_count"] >= 1 and job["voice_play_count"] == job["unique_sign_count"]
    assert job["result_url"] and job["download_url"]

    result = client.get(job["result_url"])
    assert result.status_code == 200 and result.mimetype == "video/mp4" and len(result.data) > 0
    dl = client.get(job["download_url"])
    assert "attachment" in dl.headers["Content-Disposition"]

    dets = client.get(f"/api/jobs/{job_id}/detections").get_json()
    assert dets and {"bbox", "label", "confidence", "position", "frame_index"} <= set(dets[0])
    voices = client.get(f"/api/jobs/{job_id}/voice-events").get_json()
    assert len(voices) == job["voice_play_count"]
    assert client.get(voices[0]["audio_url"]).status_code == 200

    assert [j["id"] for j in client.get("/api/jobs").get_json()] == [job_id]


def test_image_job(client, tmp_path):
    job_id = _upload(client, _image_bytes(tmp_path), "anh.png", model="faster_rcnn").get_json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done" and job["input_type"] == "image"
    assert client.get(job["result_url"]).mimetype == "image/jpeg"


def test_bad_requests(client, tmp_path):
    assert _upload(client, b"x", "a.mp4", model="resnet").status_code == 400
    assert _upload(client, b"x", "a.txt").status_code == 400
    assert client.post("/api/jobs", data={"model": "yolo"}).status_code == 400
    assert client.get("/api/jobs/999").status_code == 404
    assert client.get("/api/jobs/999").get_json()["error"]


def test_broken_video_marks_failed(client):
    job_id = _upload(client, b"not a video", "hong.mp4").get_json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "failed" and job["error_message"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_store_memory_contract():
    s = InMemoryStore()
    jid = s.create_job("video", "x.mp4", "yolo")
    dets = [{"bbox": [1, 2, 3, 4], "label": "A", "class_id": 0, "confidence": 0.5, "position": "left"}]
    s.add_detections(jid, 0, 0.0, dets)
    assert dets[0]["db_id"] == 1
    s.add_voice_event(jid, 1, "Chú ý", 0.0, "a.mp3")
    s.finish_job(jid, "out.mp4", 1, 1, 1.0, 2.0, 3.0)
    s.set_status(jid, "done")
    job = s.get_job(jid)
    assert job["status"] == "done" and job["total_time_ms"] == 3.0
    job["status"] = "hacked"                      # trả bản sao, không sửa được dữ liệu bên trong
    assert s.get_job(jid)["status"] == "done"
    with pytest.raises(KeyError):
        s.set_status(99, "done")
