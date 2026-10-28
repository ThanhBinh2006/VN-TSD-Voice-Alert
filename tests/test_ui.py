"""Test giao diện web (Thành viên 3): trang chính, file tĩnh và các API mà giao diện cần."""
import io

import numpy as np
import pytest

from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.runtime import video_io


@pytest.fixture
def client(tmp_path):
    out = tmp_path / "outputs"
    overrides = {"paths": {"outputs_dir": str(out), "uploads_dir": str(out / "uploads"),
                           "db_path": str(out / "test.db")}}
    app = create_app(fake=True, store=InMemoryStore(), sync=True, overrides=overrides)
    yield app.test_client()
    app.extensions["tsd"]["restore_fakes"]()


def test_index_page_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200 and page.mimetype == "text/html"
    html = page.get_data(as_text=True)
    assert "Cảnh báo biển báo giao thông" in html
    for el_id in ("upload-form", "file-input", "job-view", "road", "insights", "events-card", "history-list"):   # app.js cần các id này
        assert f'id="{el_id}"' in html
    assert client.get("/static/app.css").status_code == 200
    for font in ("Inter-Regular", "Inter-Medium", "Inter-SemiBold", "InterDisplay-Medium"):   # font đóng gói, chạy offline
        assert client.get(f"/static/fonts/{font}.woff").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "/api/jobs" in js.get_data(as_text=True)


def test_health_lists_accepted_extensions(client):
    accept = client.get("/api/health").get_json()["accept"]
    assert ".mp4" in accept and ".jpg" in accept


def test_voice_events_include_position(client, tmp_path):
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, 10.0, (96, 64)) as wr:
        for i in range(40):
            wr.write(np.full((64, 96, 3), i * 5 % 256, np.uint8))
    resp = client.post("/api/jobs", data={"file": (io.BytesIO(p.read_bytes()), "a.mp4"), "model": "yolo"},
                       content_type="multipart/form-data")
    events = client.get(f"/api/jobs/{resp.get_json()['job_id']}/voice-events").get_json()
    assert events and all(ev["position"] in ("left", "front", "right") for ev in events)
    assert events[0]["position"] == "left"        # FakeDetector: biển đầu tiên nằm bên trái
