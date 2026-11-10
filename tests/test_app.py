"""Test web backend với detector giả, lưu RAM và xử lý đồng bộ/bất đồng bộ."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from flask import Response
from flask.testing import FlaskClient
from numpy.typing import NDArray

from app.fake import FakeDetector, _fake_merge_audio_video, _fake_synthesize
from app.main import create_app
from app.store_memory import InMemoryStore
from ts_alert.postprocess import merge_av, tts
from ts_alert.runtime import video_io


def _video_bytes(
    tmp_path: Path,
    n: int = 40,
    fps: float = 10.0,
    w: int = 96,
    h: int = 64,
) -> bytes:
    """Tạo video MP4 thử nghiệm và trả nội dung file."""
    p = tmp_path / "src.mp4"
    with video_io.VideoWriter(p, fps, (w, h)) as wr:
        for i in range(n):
            wr.write(np.full((h, w, 3), i * 5 % 256, np.uint8))
    return p.read_bytes()


def _image_bytes(tmp_path: Path) -> bytes:
    """Tạo ảnh PNG thử nghiệm và trả nội dung file."""
    p = tmp_path / "src.png"
    video_io.write_image(p, np.zeros((64, 96, 3), np.uint8))
    return p.read_bytes()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[FlaskClient]:
    """Cấp client chạy đồng bộ và giải phóng tài nguyên sau mỗi test."""
    out = tmp_path / "outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        }
    }
    app = create_app(
        fake=True, store=InMemoryStore(), sync=True, overrides=overrides
    )
    app.config["TESTING"] = True
    try:
        yield app.test_client()
    finally:
        try:
            app.extensions["tsd"]["runner"].shutdown(wait=True)
        finally:
            app.extensions["tsd"]["restore_fakes"]()


def _upload(
    client: FlaskClient,
    data: bytes,
    filename: str,
    model: str = "yolo",
    frame_stride: str | None = None,
) -> Response:
    """Upload media, kèm frame_stride nếu được chỉ định."""
    form: dict[str, Any] = {
        "file": (io.BytesIO(data), filename),
        "model": model,
    }
    if frame_stride is not None:
        form["frame_stride"] = frame_stride
    return client.post(
        "/api/jobs",
        data=form,
        content_type="multipart/form-data",
    )


def test_health(client: FlaskClient) -> None:
    """Health trả trạng thái, hai model và stride mặc định từ shared.yaml."""
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["fake"] is True
    assert set(body["models"]) == {"yolo", "faster_rcnn"}
    assert body["frame_stride"] == 30


def test_video_job_end_to_end(client: FlaskClient, tmp_path: Path) -> None:
    """Video được xử lý, lưu detection, giọng nói và cung cấp file kết quả."""
    resp = _upload(client, _video_bytes(tmp_path), "đoạn đường.mp4")
    assert resp.status_code == 202
    job_id = resp.get_json()["job_id"]

    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error_message")
    assert job["input_type"] == "video" and job["model_name"] == "yolo"
    assert (
        job["unique_sign_count"] >= 1
        and job["voice_play_count"] == job["unique_sign_count"]
    )
    assert job["result_url"] and job["download_url"]

    result = client.get(job["result_url"])
    assert (
        result.status_code == 200
        and result.mimetype == "video/mp4"
        and len(result.data) > 0
    )
    dl = client.get(job["download_url"])
    assert "attachment" in dl.headers["Content-Disposition"]

    dets = client.get(f"/api/jobs/{job_id}/detections").get_json()
    assert dets and {
        "bbox", "label", "confidence", "position", "frame_index"
    } <= set(dets[0])
    voices = client.get(f"/api/jobs/{job_id}/voice-events").get_json()
    assert len(voices) == job["voice_play_count"]
    assert client.get(voices[0]["audio_url"]).status_code == 200

    assert [j["id"] for j in client.get("/api/jobs").get_json()] == [job_id]


def test_image_job(client: FlaskClient, tmp_path: Path) -> None:
    """Ảnh được xử lý bằng Faster R-CNN giả và trả kết quả JPEG."""
    job_id = _upload(
        client, _image_bytes(tmp_path), "anh.png", model="faster_rcnn"
    ).get_json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done" and job["input_type"] == "image"
    assert client.get(job["result_url"]).mimetype == "image/jpeg"


def test_bad_requests(client: FlaskClient, tmp_path: Path) -> None:
    """Từ chối model, định dạng, upload hoặc job không hợp lệ."""
    assert _upload(client, b"x", "a.mp4", model="resnet").status_code == 400
    assert _upload(client, b"x", "a.txt").status_code == 400
    assert client.post("/api/jobs", data={"model": "yolo"}).status_code == 400
    assert client.get("/api/jobs/999").status_code == 404
    assert client.get("/api/jobs/999").get_json()["error"]


def test_broken_video_marks_failed(client: FlaskClient) -> None:
    """Video hỏng chuyển sang failed và chưa cho phép lấy kết quả."""
    job_id = _upload(client, b"not a video", "hong.mp4").get_json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "failed" and job["error_message"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


def test_store_memory_contract() -> None:
    """Store lưu thống kê, cấp detection ID và trả bản sao job."""
    s = InMemoryStore()
    jid = s.create_job("video", "x.mp4", "yolo")
    dets = [{
        "bbox": [1, 2, 3, 4],
        "label": "A",
        "class_id": 0,
        "confidence": 0.5,
        "position": "left",
    }]
    s.add_detections(jid, 0, 0.0, dets)
    assert dets[0]["db_id"] == 1
    s.add_voice_event(jid, 1, "Chú ý", 0.0, "a.mp3")
    s.finish_job(jid, "out.mp4", 1, 1, 1.0, 2.0, 3.0)
    s.set_status(jid, "done")
    job = s.get_job(jid)
    assert job["status"] == "done" and job["total_time_ms"] == 3.0
    job["status"] = "hacked"
    assert s.get_job(jid)["status"] == "done"
    with pytest.raises(KeyError):
        s.set_status(99, "done")


def test_frame_stride_validation_and_override(
    client: FlaskClient,
    tmp_path: Path,
) -> None:
    """Stride sai không tạo file; stride hợp lệ được ghi vào tiến độ job."""
    uploads_dir = Path(client.application.extensions["tsd"]["uploads_dir"])
    data = _video_bytes(tmp_path, n=20)

    for invalid_stride in ("0", "-5", "abc", "1.5", "301"):
        before = set(uploads_dir.rglob("*"))
        response = _upload(
            client,
            data,
            "invalid_stride.mp4",
            frame_stride=invalid_stride,
        )
        assert response.status_code == 400, invalid_stride
        assert set(uploads_dir.rglob("*")) == before, invalid_stride

    response = _upload(
        client,
        data,
        "valid_stride.mp4",
        frame_stride="5",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["frame_stride"] == 5
    assert body["stream_url"] == f"/api/jobs/{body['job_id']}/stream"

    job = client.get(body["status_url"]).get_json()
    assert job["status"] == "done", job.get("error_message")
    assert job["progress"]["frame_stride"] == 5


def test_live_stream_and_frame_endpoints(
    client: FlaskClient,
    tmp_path: Path,
) -> None:
    """Video đã xong còn JPEG/MJPEG; ảnh không cung cấp luồng live."""
    response = _upload(client, _video_bytes(tmp_path), "live.mp4")
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    base_url = f"/api/jobs/{job_id}"

    job = client.get(base_url).get_json()
    assert job["status"] == "done", job.get("error_message")
    assert client.application.extensions["tsd"]["runner"].live.has_slot(job_id)

    with client.get(f"{base_url}/frame") as frame:
        assert frame.status_code == 200
        assert frame.mimetype == "image/jpeg"
        assert frame.data.startswith(b"\xff\xd8")

    with client.get(f"{base_url}/stream", buffered=False) as stream:
        assert stream.status_code == 200
        assert "multipart/x-mixed-replace" in stream.mimetype

        chunks = iter(stream.response)
        first_chunk = next(chunks)
        assert b"--frame" in first_chunk
        assert b"\xff\xd8" in first_chunk
        assert first_chunk.endswith(b"\r\n")
        with pytest.raises(StopIteration):
            next(chunks)

    response = _upload(client, _image_bytes(tmp_path), "live_image.png")
    assert response.status_code == 202
    image_id = response.get_json()["job_id"]

    image_job = client.get(f"/api/jobs/{image_id}").get_json()
    assert image_job["status"] == "done", image_job.get("error_message")
    assert client.get(f"/api/jobs/{image_id}/stream").status_code == 400
    assert client.get(f"/api/jobs/{image_id}/frame").status_code == 404


def test_streaming_during_processing_sync_false(tmp_path: Path) -> None:
    """Chặn frame 1 để kiểm tra dữ liệu frame 0 trong lúc job processing."""
    second_frame_entered = threading.Event()
    release_second_frame = threading.Event()

    class BlockingDetector(FakeDetector):
        """Detector giả dừng trước khi dự đoán frame thứ hai."""

        def predict(self, frame: NDArray[np.uint8]) -> list[dict[str, Any]]:
            """Chờ chốt tại lần gọi thứ hai, sau khi frame 0 đã publish."""
            if self._calls == 1:
                second_frame_entered.set()
                if not release_second_frame.wait(timeout=30.0):
                    raise TimeoutError("Hết thời gian chờ mở chốt frame thứ hai")
            return super().predict(frame)

    detector = BlockingDetector()

    def detector_factory(
        model_name: str,
        cfg: dict[str, Any],
    ) -> BlockingDetector:
        """Cấp detector có chốt đồng bộ cho worker."""
        return detector

    out = tmp_path / "async_outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        },
        "inference": {"frame_stride": 1},
    }
    app = create_app(
        fake=True,
        store=InMemoryStore(),
        sync=False,
        overrides=overrides,
    )
    app.config["TESTING"] = True
    runner = app.extensions["tsd"]["runner"]

    try:
        # Giữ test độc lập với mạng và dịch vụ TTS thực tế.
        with (
            patch.object(tts, "synthesize", _fake_synthesize),
            patch.object(merge_av, "merge_audio_video", _fake_merge_audio_video),
        ):
            try:
                runner.detector_factory = detector_factory

                with app.test_client() as async_client:
                    response = _upload(
                        async_client,
                        _video_bytes(tmp_path, n=20),
                        "processing.mp4",
                        frame_stride="1",
                    )
                    assert response.status_code == 202
                    job_id = response.get_json()["job_id"]
                    base_url = f"/api/jobs/{job_id}"

                    assert second_frame_entered.wait(timeout=15.0), (
                        "Worker chưa đến frame thứ hai",
                        async_client.get(base_url).get_json(),
                    )

                    job = async_client.get(base_url).get_json()
                    assert job["status"] == "processing", job
                    assert job["progress"]["frame_stride"] == 1

                    response = async_client.get(f"{base_url}/detections")
                    assert response.status_code == 200
                    detections = response.get_json()
                    frame_zero_detections = [
                        det for det in detections if det["frame_index"] == 0
                    ]
                    assert frame_zero_detections

                    response = async_client.get(f"{base_url}/voice-events")
                    assert response.status_code == 200
                    events = response.get_json()
                    detection_ids = {
                        det["id"] for det in frame_zero_detections
                    }
                    assert any(
                        event["detection_id"] in detection_ids
                        and event["timestamp_sec"] == 0.0
                        for event in events
                    )

                    with async_client.get(f"{base_url}/frame") as frame:
                        assert frame.status_code == 200
                        assert frame.mimetype == "image/jpeg"
                        assert frame.data.startswith(b"\xff\xd8")

                    assert (
                        async_client.get(f"{base_url}/result").status_code
                        == 409
                    )
                    assert (
                        async_client.get(base_url).get_json()["status"]
                        == "processing"
                    )

                    release_second_frame.set()
                    deadline = time.monotonic() + 30.0
                    polling_interval = threading.Event()

                    while True:
                        job = async_client.get(base_url).get_json()
                        if job["status"] in {"done", "failed"}:
                            break
                        remaining = deadline - time.monotonic()
                        assert remaining > 0, (
                            "Job không hoàn tất trong thời gian cho phép",
                            job,
                        )
                        polling_interval.wait(timeout=min(0.02, remaining))

                    assert job["status"] == "done", job.get("error_message")
                    with async_client.get(f"{base_url}/result") as result:
                        assert result.status_code == 200
                        assert result.mimetype == "video/mp4"
                        assert result.data
            finally:
                release_second_frame.set()
                runner.shutdown(wait=True)
    finally:
        app.extensions["tsd"]["restore_fakes"]()


def test_live_http_mjpeg_stream_and_client_disconnect(tmp_path: Path) -> None:
    """Đọc JPEG qua HTTP thật, đóng client sớm và kiểm tra EOF khi job xong."""
    import json
    from contextlib import ExitStack
    from http.client import HTTPConnection, HTTPResponse

    from werkzeug.serving import make_server

    deadline = time.monotonic() + 15.0
    second_frame_entered = threading.Event()
    release_second_frame = threading.Event()
    polling_interval = threading.Event()

    def remaining_time() -> float:
        """Giới hạn tổng thời gian chờ của test."""
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Hết thời gian chờ HTTP streaming"
        return remaining

    class BlockingDetector(FakeDetector):
        """Dừng trước inference frame 1, sau khi JPEG frame 0 đã sẵn sàng."""

        def predict(self, frame: NDArray[np.uint8]) -> list[dict[str, Any]]:
            """Chờ test đọc JPEG và đóng một kết nối trước khi chạy tiếp."""
            if self._calls == 1:
                second_frame_entered.set()
                if not release_second_frame.wait(timeout=15.0):
                    raise TimeoutError("Hết thời gian chờ mở chốt frame 1")
            return super().predict(frame)

    detector = BlockingDetector()

    def detector_factory(
        model_name: str,
        cfg: dict[str, Any],
    ) -> BlockingDetector:
        """Cấp cùng detector cho worker của test."""
        return detector

    def read_jpeg(response: HTTPResponse) -> bytes:
        """Đọc đúng một phần multipart theo Content-Length."""
        assert response.readline(128) == b"--frame\r\n"
        headers: dict[bytes, bytes] = {}
        while True:
            line = response.readline(1024)
            assert line, "Luồng kết thúc trước phần thân JPEG"
            if line == b"\r\n":
                break
            name, value = line.split(b":", 1)
            headers[name.strip().lower()] = value.strip()

        assert headers[b"content-type"] == b"image/jpeg"
        length = int(headers[b"content-length"])
        assert 0 < length <= 1_000_000
        jpeg = response.read(length)
        assert len(jpeg) == length
        assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
        assert response.read(2) == b"\r\n"
        return jpeg

    out = tmp_path / "http_outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(out / "test.db"),
        },
        "inference": {"frame_stride": 1},
    }
    app = create_app(
        fake=True,
        store=InMemoryStore(),
        sync=False,
        overrides=overrides,
    )
    app.config["TESTING"] = True
    runner = app.extensions["tsd"]["runner"]

    try:
        with (
            patch.object(tts, "synthesize", _fake_synthesize),
            patch.object(merge_av, "merge_audio_video", _fake_merge_audio_video),
        ):
            try:
                runner.detector_factory = detector_factory
                server = make_server("127.0.0.1", 0, app, threaded=True)
                server_thread = threading.Thread(
                    target=server.serve_forever,
                    kwargs={"poll_interval": 0.01},
                    daemon=True,
                )
                try:
                    server_thread.start()
                    with ExitStack() as connections:
                        control = HTTPConnection(
                            "127.0.0.1",
                            server.server_port,
                            timeout=remaining_time(),
                        )
                        connections.callback(control.close)

                        def get_job_status(path: str) -> dict[str, Any]:
                            """Lấy trạng thái qua kết nối HTTP độc lập với stream."""
                            control.timeout = remaining_time()
                            control.request("GET", path)
                            with control.getresponse() as response:
                                assert response.status == 200
                                return json.loads(response.read())

                        boundary = "tsd-live-test-boundary"
                        payload = (
                            f"--{boundary}\r\n"
                            'Content-Disposition: form-data; name="model"\r\n'
                            "\r\nyolo\r\n"
                            f"--{boundary}\r\n"
                            'Content-Disposition: form-data; name="frame_stride"\r\n'
                            "\r\n1\r\n"
                            f"--{boundary}\r\n"
                            'Content-Disposition: form-data; name="file"; '
                            'filename="live.mp4"\r\n'
                            "Content-Type: video/mp4\r\n\r\n"
                        ).encode("ascii")
                        payload += _video_bytes(tmp_path, n=3)
                        payload += f"\r\n--{boundary}--\r\n".encode("ascii")
                        control.request(
                            "POST",
                            "/api/jobs",
                            body=payload,
                            headers={
                                "Content-Type": (
                                    f"multipart/form-data; boundary={boundary}"
                                ),
                            },
                        )
                        with control.getresponse() as response:
                            assert response.status == 202
                            body = json.loads(response.read())

                        base_url = body["status_url"]
                        assert body["frame_stride"] == 1
                        assert second_frame_entered.wait(
                            timeout=remaining_time()
                        ), "Worker chưa đến frame 1"

                        while True:
                            job = get_job_status(base_url)
                            if job["status"] == "processing":
                                break
                            assert job["status"] == "queued", job
                            polling_interval.wait(
                                timeout=min(0.01, remaining_time())
                            )

                        stream_connection = HTTPConnection(
                            "127.0.0.1",
                            server.server_port,
                            timeout=remaining_time(),
                        )
                        connections.callback(stream_connection.close)
                        stream_connection.request("GET", body["stream_url"])
                        stream = connections.enter_context(
                            stream_connection.getresponse()
                        )
                        assert stream.status == 200
                        assert "multipart/x-mixed-replace" in (
                            stream.getheader("Content-Type") or ""
                        )
                        assert "boundary=frame" in (
                            stream.getheader("Content-Type") or ""
                        )
                        assert "no-store" in (
                            stream.getheader("Cache-Control") or ""
                        )
                        assert stream.getheader("X-Accel-Buffering") == "no"
                        first_jpeg = read_jpeg(stream)
                        assert not release_second_frame.is_set()
                        assert get_job_status(base_url)["status"] == "processing"

                        # Client thứ hai ngắt kết nối khi worker vẫn đang chờ.
                        disconnect_connection = HTTPConnection(
                            "127.0.0.1",
                            server.server_port,
                            timeout=remaining_time(),
                        )
                        connections.callback(disconnect_connection.close)
                        disconnect_connection.request("GET", body["stream_url"])
                        with disconnect_connection.getresponse() as disconnected:
                            assert disconnected.status == 200
                            assert read_jpeg(disconnected) == first_jpeg
                        disconnect_connection.close()

                        release_second_frame.set()
                        while True:
                            job = get_job_status(base_url)
                            if job["status"] in {"done", "failed"}:
                                break
                            polling_interval.wait(
                                timeout=min(0.01, remaining_time())
                            )

                        assert job["status"] == "done", job.get("error_message")
                        # read() phải kết thúc bình thường sau chunk cuối.
                        stream.read()
                        assert stream.read(1) == b""
                        assert stream.isclosed()
                finally:
                    release_second_frame.set()
                    try:
                        if server_thread.ident is not None:
                            server.shutdown()
                            server_thread.join(timeout=2.0)
                            assert not server_thread.is_alive()
                    finally:
                        server.server_close()
            finally:
                release_second_frame.set()
                runner.shutdown(wait=True)
    finally:
        app.extensions["tsd"]["restore_fakes"]()


def test_async_streaming_with_sqlite_repository(tmp_path: Path) -> None:
    """SQLite phục vụ dữ liệu live khi dừng ở frame 30 và lưu đủ thống kê."""
    from ts_alert.db.repository import Repository

    deadline = time.monotonic() + 15.0
    frame_thirty_entered = threading.Event()
    release_frame_thirty = threading.Event()
    polling_interval = threading.Event()

    def remaining_time() -> float:
        """Giới hạn tổng thời gian chờ của test."""
        remaining = deadline - time.monotonic()
        assert remaining > 0, "Hết thời gian chờ SQLite streaming"
        return remaining

    class BlockingDetector(FakeDetector):
        """Lần inference thứ hai tương ứng frame 30 khi stride bằng 30."""

        def predict(self, frame: NDArray[np.uint8]) -> list[dict[str, Any]]:
            """Chặn trước khi ghi detection của frame 30."""
            if self._calls == 1:
                frame_thirty_entered.set()
                if not release_frame_thirty.wait(timeout=15.0):
                    raise TimeoutError("Hết thời gian chờ mở chốt frame 30")
            return super().predict(frame)

    detector = BlockingDetector()

    def detector_factory(
        model_name: str,
        cfg: dict[str, Any],
    ) -> BlockingDetector:
        """Cấp detector có chốt cho worker SQLite."""
        return detector

    repository = Repository(tmp_path / "live.db")
    out = tmp_path / "sqlite_outputs"
    overrides = {
        "paths": {
            "outputs_dir": str(out),
            "uploads_dir": str(out / "uploads"),
            "db_path": str(tmp_path / "live.db"),
        },
        "inference": {"frame_stride": 30},
    }
    app = create_app(
        fake=True,
        store=repository,
        sync=False,
        overrides=overrides,
    )
    app.config["TESTING"] = True
    runner = app.extensions["tsd"]["runner"]

    try:
        with (
            patch.object(tts, "synthesize", _fake_synthesize),
            patch.object(merge_av, "merge_audio_video", _fake_merge_audio_video),
        ):
            try:
                runner.detector_factory = detector_factory
                with app.test_client() as async_client:
                    response = _upload(
                        async_client,
                        _video_bytes(tmp_path, n=35),
                        "sqlite_live.mp4",
                        frame_stride="30",
                    )
                    assert response.status_code == 202
                    body = response.get_json()
                    assert body["frame_stride"] == 30
                    job_id = body["job_id"]
                    base_url = body["status_url"]

                    assert frame_thirty_entered.wait(
                        timeout=remaining_time()
                    ), "Worker chưa đến frame 30"

                    # Worker vẫn sống và giữ chốt trong suốt các request này.
                    for _ in range(2):
                        response = async_client.get(base_url)
                        assert response.status_code == 200
                        job = response.get_json()
                        assert job["status"] == "processing", job
                        assert job["progress"]["stage"] == "processing"
                        assert job["progress"]["frame_stride"] == 30
                        assert job["progress"]["frames_done"] == 30
                        assert job["progress"]["current_timestamp_sec"] == (
                            pytest.approx(2.9)
                        )

                        response = async_client.get(f"{base_url}/detections")
                        assert response.status_code == 200
                        detections = response.get_json()
                        assert detections
                        assert sorted({
                            det["frame_index"] for det in detections
                        }) == [0]
                        assert all(len(det["bbox"]) == 4 for det in detections)

                        response = async_client.get(f"{base_url}/voice-events")
                        assert response.status_code == 200
                        events = response.get_json()
                        assert events
                        detection_ids = {det["id"] for det in detections}
                        assert all(
                            event["detection_id"] in detection_ids
                            and event["timestamp_sec"] == 0.0
                            for event in events
                        )

                        with async_client.get(f"{base_url}/frame") as frame:
                            assert frame.status_code == 200
                            assert frame.mimetype == "image/jpeg"
                            assert frame.data.startswith(b"\xff\xd8")

                        assert not release_frame_thirty.is_set()
                        assert repository.get_job(job_id)["status"] == (
                            "processing"
                        )

                    release_frame_thirty.set()
                    while True:
                        response = async_client.get(base_url)
                        assert response.status_code == 200
                        job = response.get_json()
                        if job["status"] in {"done", "failed"}:
                            break
                        polling_interval.wait(
                            timeout=min(0.01, remaining_time())
                        )

                    assert job["status"] == "done", job.get("error_message")
                    assert job["progress"]["frame_stride"] == 30
                    assert job["progress"]["frames_done"] == 35
                    assert detector._calls == 2

                    response = async_client.get(f"{base_url}/detections")
                    assert response.status_code == 200
                    final_detections = response.get_json()
                    assert sorted({
                        det["frame_index"] for det in final_detections
                    }) == [0, 30]
                    assert sorted({
                        det["frame_index"]
                        for det in repository.get_detections(job_id)
                    }) == [0, 30]

                    stored_job = repository.get_job(job_id)
                    assert stored_job is not None
                    assert stored_job["status"] == "done"
                    assert stored_job["started_at"] is not None
                    assert stored_job["finished_at"] is not None
                    assert stored_job["error_message"] is None
                    assert Path(stored_job["output_path"]).is_file()
                    assert stored_job["unique_sign_count"] >= 1
                    assert stored_job["voice_play_count"] == (
                        stored_job["unique_sign_count"]
                    )
                    assert stored_job["voice_play_count"] == len(
                        repository.get_voice_events(job_id)
                    )
                    for key in (
                        "unique_sign_count",
                        "voice_play_count",
                        "layer_time_ms",
                        "tts_time_ms",
                        "total_time_ms",
                    ):
                        assert stored_job[key] is not None
                        assert stored_job[key] >= 0
                        assert stored_job[key] == job[key]
                    assert stored_job["total_time_ms"] > 0
            finally:
                release_frame_thirty.set()
                runner.shutdown(wait=True)
    finally:
        app.extensions["tsd"]["restore_fakes"]()
