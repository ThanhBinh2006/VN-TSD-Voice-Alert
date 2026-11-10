"""Test giao diện web (Thành viên 3): trang chính, file tĩnh và các API mà giao diện cần."""
import io

import cv2
import numpy as np
import pytest

from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.runtime import video_io


@pytest.fixture
def client(tmp_path):
    """Tạo client đồng bộ với detector giả và dữ liệu tạm."""
    out = tmp_path / "outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        }
    }
    app = create_app(fake=True, store=InMemoryStore(), sync=True, overrides=overrides)
    try:
        yield app.test_client()
    finally:
        app.extensions["tsd"]["runner"].shutdown(wait=False)
        app.extensions["tsd"]["restore_fakes"]()


def test_index_page_and_static_files(client) -> None:
    """Kiểm tra DOM, tài nguyên offline và các tham chiếu API của giao diện."""
    page = client.get("/")
    assert page.status_code == 200 and page.mimetype == "text/html"
    html = page.get_data(as_text=True)
    assert "Cảnh báo biển báo giao thông" in html
    for el_id in (
        "upload-form",
        "file-input",
        "job-view",
        "road",
        "insights",
        "events-card",
        "history-list",
        "frame-stride-input",
        "live-stream",
        "live-bar",
        "live-audio-btn",
    ):
        assert f'id="{el_id}"' in html
    assert client.get("/static/app.css").status_code == 200
    for font in ("Inter-Regular", "Inter-Medium", "Inter-SemiBold", "InterDisplay-Medium"):
        assert client.get(f"/static/fonts/{font}.woff").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "/api/jobs" in js.get_data(as_text=True)
    source = js.get_data(as_text=True)
    assert "/stream" in source
    assert "frame_stride" in source


def test_health_lists_accepted_extensions(client) -> None:
    """Kiểm tra định dạng được hỗ trợ và frame_stride mặc định."""
    health = client.get("/api/health").get_json()
    accept = health["accept"]
    assert ".mp4" in accept and ".jpg" in accept
    assert type(health["frame_stride"]) is int
    assert health["frame_stride"] > 0
    assert health["frame_stride"] == 30


def test_voice_events_include_position(client, tmp_path) -> None:
    """Kiểm tra upload video với stride tùy chọn, URL live và vị trí cảnh báo."""
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, 10.0, (96, 64)) as wr:
        for i in range(40):
            wr.write(np.full((64, 96, 3), i * 5 % 256, np.uint8))
    resp = client.post(
        "/api/jobs",
        data={
            "file": (io.BytesIO(p.read_bytes()), "a.mp4"),
            "model": "yolo",
            "frame_stride": "10",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["frame_stride"] == 10
    assert body["stream_url"] == f"/api/jobs/{body['job_id']}/stream"
    events = client.get(f"/api/jobs/{body['job_id']}/voice-events").get_json()
    assert events and all(ev["position"] in ("left", "front", "right") for ev in events)
    assert events[0]["position"] == "left"  # FakeDetector: biển đầu tiên nằm bên trái


def test_image_upload_without_stream_url(client) -> None:
    """Kiểm tra upload ảnh vẫn hoàn tất và không cung cấp URL live."""
    encoded, image = cv2.imencode(".png", np.zeros((64, 96, 3), dtype=np.uint8))
    assert encoded
    resp = client.post(
        "/api/jobs",
        data={
            "file": (io.BytesIO(image.tobytes()), "image.png"),
            "model": "yolo",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["status"] == "done"
    assert "stream_url" not in body

    detail = client.get(f"/api/jobs/{body['job_id']}")
    assert detail.status_code == 200
    job = detail.get_json()
    assert job["input_type"] == "image"
    assert job["status"] == "done"
    assert "stream_url" not in job
    assert client.get(job["result_url"]).status_code == 200
