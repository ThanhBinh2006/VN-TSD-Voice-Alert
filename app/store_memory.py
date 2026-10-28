"""Nơi lưu log tạm trong RAM cho web khi Repository (SQLite, Thành viên 5) chưa xong.   Phụ trách: Thành viên 3

- Thoả giao diện ResultStore (ts_alert/db/interface.py) -> truyền thẳng vào process_media(repo=...).
- Có thêm các hàm ĐỌC mà web cần. Đây là ĐỀ XUẤT cho Thành viên 5: Repository nên có đúng các hàm này
  (cùng tên, cùng kiểu trả về) để web chỉ cần đổi một dòng khi chuyển sang SQLite:

    get_job(job_id) -> dict | None
        {"id", "input_type", "input_path", "model_name", "status", "created_at", "finished_at",
         "output_path", "unique_sign_count", "voice_play_count",
         "layer_time_ms", "tts_time_ms", "total_time_ms", "error_message"}
    list_jobs(limit=50) -> list[dict]          # mới nhất trước, cùng khoá như get_job
    get_detections(job_id) -> list[dict]       # {"id", "frame_index", "timestamp_sec", "bbox", "label",
                                               #  "class_id", "confidence", "position"}
    get_voice_events(job_id) -> list[dict]     # {"id", "detection_id", "spoken_text", "timestamp_sec", "audio_path"}

- Dữ liệu mất khi tắt server. Mọi hàm có khoá (lock) vì web ghi từ thread nền và đọc từ thread request.
"""
from __future__ import annotations

import copy
import threading
import time

JOB_FIELDS_DEFAULT = {
    "finished_at": None, "output_path": None, "unique_sign_count": None, "voice_play_count": None,
    "layer_time_ms": None, "tts_time_ms": None, "total_time_ms": None, "error_message": None,
}


class InMemoryStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[int, dict] = {}
        self._detections: list[dict] = []
        self._voices: list[dict] = []

    # ------------------------------------------------------------------ ghi (ResultStore)

    def create_job(self, input_type: str, input_path: str, model_name: str) -> int:
        with self._lock:
            job_id = len(self._jobs) + 1
            self._jobs[job_id] = {"id": job_id, "input_type": input_type, "input_path": input_path,
                                  "model_name": model_name, "status": "queued", "created_at": time.time(),
                                  **JOB_FIELDS_DEFAULT}
            return job_id

    def set_status(self, job_id: int, status: str) -> None:
        with self._lock:
            self._job(job_id)["status"] = status

    def add_detections(self, job_id: int, frame_index: int, timestamp_sec: float, detections: list) -> None:
        with self._lock:
            self._job(job_id)
            for det in detections:
                row_id = len(self._detections) + 1
                self._detections.append({
                    "id": row_id, "job_id": job_id, "frame_index": frame_index, "timestamp_sec": timestamp_sec,
                    "bbox": [float(v) for v in det.get("bbox", [])], "label": det.get("label"),
                    "class_id": det.get("class_id"), "confidence": float(det.get("confidence", 0.0)),
                    "position": det.get("position"),
                })
                det["db_id"] = row_id

    def add_voice_event(self, job_id: int, detection_id, spoken_text: str, timestamp_sec: float,
                        audio_path: str) -> None:
        with self._lock:
            self._job(job_id)
            self._voices.append({"id": len(self._voices) + 1, "job_id": job_id, "detection_id": detection_id,
                                 "spoken_text": spoken_text, "timestamp_sec": timestamp_sec,
                                 "audio_path": audio_path})

    def finish_job(self, job_id: int, output_path: str, unique_sign_count: int, voice_play_count: int,
                   layer_time_ms: float, tts_time_ms: float, total_time_ms: float) -> None:
        with self._lock:
            self._job(job_id).update(
                output_path=output_path, unique_sign_count=unique_sign_count, voice_play_count=voice_play_count,
                layer_time_ms=layer_time_ms, tts_time_ms=tts_time_ms, total_time_ms=total_time_ms,
                finished_at=time.time())

    def fail_job(self, job_id: int, error_message: str) -> None:
        with self._lock:
            self._job(job_id).update(status="failed", error_message=error_message, finished_at=time.time())

    # ------------------------------------------------------------------ đọc (đề xuất cho Thành viên 5)

    def get_job(self, job_id: int) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["id"], reverse=True)[:limit]
            return copy.deepcopy(jobs)

    def get_detections(self, job_id: int) -> list[dict]:
        with self._lock:
            return [_without_job_id(d) for d in self._detections if d["job_id"] == job_id]

    def get_voice_events(self, job_id: int) -> list[dict]:
        with self._lock:
            return [_without_job_id(v) for v in self._voices if v["job_id"] == job_id]

    # ------------------------------------------------------------------

    def _job(self, job_id: int) -> dict:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"Không có job {job_id}") from None


def _without_job_id(row: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in row.items() if k != "job_id"}
