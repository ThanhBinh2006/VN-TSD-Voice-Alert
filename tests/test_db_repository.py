from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Cho phép chạy test độc lập khi giải nén bundle; trong repo chính, pyproject.toml đã cấu hình pythonpath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_alert.db.repository import Repository


def _sample_detection(label="Stop", class_id=3, confidence=0.96, position="right"):
    """Tạo detection mẫu theo format đầu vào của Repository.add_detections()."""
    return {
        "bbox": [10, 20, 110, 120],
        "label": label,
        "class_id": class_id,
        "confidence": confidence,
        "position": position,
    }


def test_resultstore_happy_path(tmp_path):
    db_path = tmp_path / "traffic_sign.db"
    repo = Repository(db_path)

    job_id = repo.create_job("video", "data/input/demo.mp4", "yolo")
    repo.set_status(job_id, "processing")
    detections = [_sample_detection()]
    repo.add_detections(job_id, 12, 0.48, detections)

    assert detections[0]["db_id"] > 0
    repo.add_voice_event(
        job_id,
        detections[0]["db_id"],
        "Chú ý, bên phải có biển Stop",
        0.48,
        "outputs/audio/stop.mp3",
    )
    # finish_job phải tự chuyển trạng thái sang done; test không gọi set_status('done').
    repo.finish_job(job_id, "outputs/videos/result.mp4", 1, 1, 20.5, 100.0, 500.2)

    job = repo.get_job(job_id)
    assert job is not None
    assert job["status"] == "done"
    assert job["output_path"] == "outputs/videos/result.mp4"
    assert job["unique_sign_count"] == 1
    assert job["voice_play_count"] == 1
    assert job["layer_time_ms"] == 20.5
    assert job["tts_time_ms"] == 100.0
    assert job["total_time_ms"] == 500.2

    saved_detections = repo.get_detections(job_id)
    saved_voice_events = repo.get_voice_events(job_id)
    assert len(saved_detections) == 1
    assert len(saved_voice_events) == 1
    assert saved_voice_events[0]["detection_id"] == saved_detections[0]["id"]
    assert saved_voice_events[0]["spoken_text"] == "Chú ý, bên phải có biển Stop"
    assert saved_voice_events[0]["audio_path"] == "outputs/audio/stop.mp3"


def test_fail_job(tmp_path):
    repo = Repository(tmp_path / "traffic_sign.db")
    job_id = repo.create_job("image", "data/input/demo.jpg", "faster_rcnn")

    repo.fail_job(job_id, "test failure")

    job = repo.get_job(job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_message"] == "test failure"


def test_foreign_keys_and_constraints(tmp_path):
    db_path = tmp_path / "traffic_sign.db"
    repo = Repository(db_path)
    job_id = repo.create_job("image", "input.jpg", "yolo")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert fk_errors == []

        try:
            conn.execute(
                """INSERT INTO detection
                   (job_id, frame_index, timestamp_sec, label, class_id, confidence,
                    x1, y1, x2, y2, position)
                   VALUES (?, 0, 0, 'bad', 0, 0.5, 0, 0, 1, 1, 'unknown')""",
                (job_id,),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("CHECK constraint phải từ chối position='unknown'")


def test_add_detections_assigns_each_inserted_id_to_correct_detection(tmp_path):
    repo = Repository(tmp_path / "traffic_sign.db")
    job_id = repo.create_job("video", "input.mp4", "yolo")
    detections = [
        _sample_detection("Stop", 3, 0.96, "right"),
        _sample_detection("Speed limit", 5, 0.88, "left"),
    ]

    repo.add_detections(job_id, frame_index=7, timestamp_sec=1.25, detections=detections)

    saved = repo.get_detections(job_id)
    assert len(saved) == 2
    assert detections[0]["db_id"] == saved[0]["id"]
    assert detections[1]["db_id"] == saved[1]["id"]
    assert detections[0]["db_id"] != detections[1]["db_id"]
    assert [row["label"] for row in saved] == ["Stop", "Speed limit"]


def test_model_stats_count_only_completed_jobs_and_their_detections(tmp_path):
    repo = Repository(tmp_path / "traffic_sign.db")

    # Một job YOLO hoàn tất có 2 detection và 1 lần phát giọng nói.
    done_job_id = repo.create_job("video", "completed.mp4", "yolo")
    done_detections = [
        _sample_detection("Stop", 3, 0.96, "right"),
        _sample_detection("Speed limit", 5, 0.88, "left"),
    ]
    repo.add_detections(done_job_id, 1, 0.5, done_detections)
    repo.finish_job(done_job_id, "outputs/completed.mp4", 2, 1, 10.0, 30.0, 100.0)

    # Detection thuộc job lỗi không được tính vào thống kê kết quả hoàn tất.
    failed_job_id = repo.create_job("video", "failed.mp4", "yolo")
    repo.add_detections(failed_job_id, 2, 1.0, [_sample_detection("No entry", 7, 0.91, "front")])
    repo.fail_job(failed_job_id, "mock failure")

    stats = repo.get_model_stats()
    yolo = stats["yolo"]

    assert yolo["total_jobs"] == 2
    assert yolo["completed_jobs"] == 1
    assert yolo["failed_jobs"] == 1
    assert yolo["total_detections"] == 2
    assert yolo["total_voice_plays"] == 1
    assert yolo["avg_total_time_ms"] == 100.0
    assert yolo["avg_layer_time_ms"] == 10.0
    assert yolo["avg_tts_time_ms"] == 30.0

    # Model chưa có job nào vẫn phải có thống kê hợp lệ, không bị chia cho 0.
    faster_rcnn = stats["faster_rcnn"]
    assert faster_rcnn["total_jobs"] == 0
    assert faster_rcnn["completed_jobs"] == 0
    assert faster_rcnn["total_detections"] == 0
    assert faster_rcnn["avg_total_time_ms"] is None


def test_get_job_returns_none_for_unknown_id(tmp_path):
    repo = Repository(tmp_path / "traffic_sign.db")
    assert repo.get_job(999999) is None