"""Lưu log xử lý vào SQLite.   Phụ trách: Thành viên 5

Cần thoả giao diện ResultStore trong ts_alert/db/interface.py (cấu trúc bảng tuỳ nhóm thiết kế).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_VALID_STATUSES = {"queued", "processing", "done", "failed"}
_VALID_MODELS = {"yolo", "faster_rcnn"}
_VALID_INPUT_TYPES = {"image", "video"}
_VALID_POSITIONS = {"left", "front", "right"}

class Repository:
    def __init__(self, db_path):
        """Mở/tạo file SQLite tại db_path và tạo bảng nếu chưa có (đọc ts_alert/db/schema.sql)."""
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self):
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        with self._connection() as conn:
            conn.executescript(schema)

    @staticmethod
    def _require_nonnegative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} không được âm")

    def create_job(self, input_type: str, input_path: str, model_name: str) -> int:
        if input_type not in _VALID_INPUT_TYPES:
            raise ValueError(f"input_type không hợp lệ: {input_type}")
        if model_name not in _VALID_MODELS:
            raise ValueError(f"model_name không hợp lệ: {model_name}")
        with self._connection() as conn:
            cur = conn.execute(
                """INSERT INTO processing_job (input_type, input_path, model_name, status)
                    VALUES (?, ?, ?, 'queued')""",
                (input_type, str(input_path), model_name),
            )
            return int(cur.lastrowid)
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def set_status(self, job_id: int, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"status không hợp lệ: {status}")
        with self._connection() as conn:
            if status == "processing":
                cur = conn.execute(
                    """UPDATE processing_job SET status=?, started_at=COALESCE(started_at, CURRENT_TIMESTAMP),
                       error_message=NULL WHERE id=?""", (status, job_id)
                )
            elif status in {"done", "failed"}:
                cur = conn.execute(
                    """UPDATE processing_job SET status=?, finished_at=COALESCE(finished_at, CURRENT_TIMESTAMP)
                       WHERE id=?""", (status, job_id)
                )
            else:
                cur = conn.execute("UPDATE processing_job SET status=? WHERE id=?", (status, job_id))
            if cur.rowcount == 0:
                raise KeyError(f"Không tìm thấy processing_job id={job_id}")
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def add_detections(self, job_id: int, frame_index: int, timestamp_sec: float, detections: list) -> None:
        if frame_index < 0 or timestamp_sec < 0:
            raise ValueError("frame_index và timestamp_sec phải không âm")
        if not detections:
            return
        rows: list[tuple[Any, ...]] = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("Mỗi detection phải có bbox=[x1,y1,x2,y2]")
            missing = [key for key in ("label", "class_id", "confidence", "position") if key not in det]
            if missing:
                raise ValueError(f"Detection thiếu trường: {', '.join(missing)}")
            if det["position"] not in _VALID_POSITIONS:
                raise ValueError(f"position không hợp lệ: {det['position']}")
            x1, y1, x2, y2 = (float(v) for v in bbox)
            confidence = float(det["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("confidence phải nằm trong [0, 1]")
            if x2 < x1 or y2 < y1:
                raise ValueError("bbox không hợp lệ: x2>=x1 và y2>=y1")
            rows.append((job_id, frame_index, float(timestamp_sec), str(det["label"]),
                         int(det["class_id"]), confidence, x1, y1, x2, y2, det["position"]))

        with self._connection() as conn:
            for det, row in zip(detections, rows):
                cur = conn.execute(
                    """INSERT INTO detection
                    (job_id, frame_index, timestamp_sec, label, class_id, confidence,
                        x1, y1, x2, y2, position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
                det["db_id"] = int(cur.lastrowid)
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def add_voice_event(self, job_id: int, detection_id, spoken_text: str, timestamp_sec: float, audio_path: str) -> None:
        if timestamp_sec < 0:
            raise ValueError("timestamp_sec không được âm")
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO voice_event (job_id, detection_id, spoken_text, timestamp_sec, audio_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (job_id, int(detection_id) if detection_id is not None else None,
                 spoken_text, float(timestamp_sec), str(audio_path)),
            )
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def finish_job(self, job_id: int, output_path: str, unique_sign_count: int, voice_play_count: int,
                   layer_time_ms: float, tts_time_ms: float, total_time_ms: float) -> None:
        counts = (unique_sign_count, voice_play_count)
        if any(int(value) < 0 for value in counts):
            raise ValueError("Các bộ đếm không được âm")
        for name, value in (("layer_time_ms", layer_time_ms), ("tts_time_ms", tts_time_ms),
                            ("total_time_ms", total_time_ms)):
            self._require_nonnegative(name, float(value))
        with self._connection() as conn:
            cur = conn.execute(
                """UPDATE processing_job
                    SET status='done',
                        output_path=?, unique_sign_count=?, voice_play_count=?,
                        layer_time_ms=?, tts_time_ms=?, total_time_ms=?,
                        finished_at=COALESCE(finished_at, CURRENT_TIMESTAMP)
                    WHERE id=?""",
                (str(output_path), int(unique_sign_count), int(voice_play_count),
                 float(layer_time_ms), float(tts_time_ms), float(total_time_ms), job_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Không tìm thấy processing_job id={job_id}")
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    def fail_job(self, job_id: int, error_message: str) -> None:
        with self._connection() as conn:
            cur = conn.execute(
                """UPDATE processing_job
                   SET status='failed', error_message=?, finished_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (str(error_message), job_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Không tìm thấy processing_job id={job_id}")
        # # TODO: Thành viên 5
        # raise NotImplementedError("TODO: Thành viên 5")

    # TODO: Thành viên 5 — thêm các hàm đọc cho web (lịch sử, chi tiết job) và thống kê so sánh YOLO vs Faster R-CNN.
    def get_jobs(self, limit: int = 20, offset: int = 0, model_name: str | None = None, status: str | None = None,
                 ) -> list[dict]:
        """Lấy lịch sử job mới nhất, có phân trang và lọc tuỳ chọn."""
        if limit < 1:
            raise ValueError("limit phải lớn hơn hoặc bằng 1")
        if offset < 0:
            raise ValueError("offset không được âm")
        if model_name is not None and model_name not in _VALID_MODELS:
            raise ValueError(f"model_name không hợp lệ: {model_name}")
        if status is not None and status not in _VALID_STATUSES:
            raise ValueError(f"status không hợp lệ: {status}")

        conditions: list[str] = []
        params: list[Any] = []
        if model_name is not None:
            conditions.append("model_name = ?")
            params.append(model_name)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([int(limit), int(offset)])

        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM processing_job
                    {where_clause}
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_job(self, job_id: int) -> dict | None:
        """Lấy thông tin tổng quan của một job theo ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM processing_job WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_detections(self, job_id: int) -> list[dict]:
        """Lấy các detection của job, theo thứ tự khung hình."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM detection WHERE job_id=? ORDER BY frame_index, id", (job_id,)
            ).fetchall()]

    def get_voice_events(self, job_id: int) -> list[dict]:
        """Lấy các sự kiện phát giọng nói của job theo thời điểm."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM voice_event WHERE job_id=? ORDER BY timestamp_sec, id", (job_id,)
            ).fetchall()]

    def get_model_stats(self) -> dict[str, dict[str, int | float | None]]:
        """Tổng hợp số liệu theo model để hỗ trợ so sánh YOLO và Faster R-CNN.

        Các chỉ số thời gian trung bình chỉ tính trên job có status='done'.
        Detection và số lần phát giọng nói cũng chỉ tính từ các job đã hoàn tất,
        nhằm tránh đưa kết quả xử lý dở dang của job lỗi vào phần so sánh.
        Giá trị trung bình là None nếu model chưa có job hoàn tất."""
        stats: dict[str, dict[str, int | float | None]] = {}

        with self._connection() as conn:
            for model_name in sorted(_VALID_MODELS):
                row = conn.execute(
                    f"""SELECT
                           COUNT(*) AS total_jobs,
                           SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS completed_jobs,
                           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_jobs,
                           SUM(CASE WHEN status='done' THEN COALESCE(voice_play_count, 0) ELSE 0 END)
                               AS total_voice_plays,
                           AVG(CASE WHEN status='done' THEN total_time_ms END) AS avg_total_time_ms,
                           AVG(CASE WHEN status='done' THEN layer_time_ms END) AS avg_layer_time_ms,
                           AVG(CASE WHEN status='done' THEN tts_time_ms END) AS avg_tts_time_ms
                       FROM processing_job
                       WHERE model_name=?""",
                    (model_name,),
                ).fetchone()

                detection_row = conn.execute(
                    f"""SELECT COUNT(*) AS total_detections
                       FROM detection AS d
                       INNER JOIN processing_job AS j ON j.id = d.job_id
                       WHERE j.model_name=? AND j.status='done'""",
                    (model_name,),
                ).fetchone()

                stats[model_name] = {
                    "total_jobs": int(row["total_jobs"] or 0),
                    "completed_jobs": int(row["completed_jobs"] or 0),
                    "failed_jobs": int(row["failed_jobs"] or 0),
                    "total_detections": int(detection_row["total_detections"] or 0),
                    "total_voice_plays": int(row["total_voice_plays"] or 0),
                    "avg_total_time_ms": (
                        float(row["avg_total_time_ms"])
                        if row["avg_total_time_ms"] is not None else None
                    ),
                    "avg_layer_time_ms": (
                        float(row["avg_layer_time_ms"])
                        if row["avg_layer_time_ms"] is not None else None
                    ),
                    "avg_tts_time_ms": (
                        float(row["avg_tts_time_ms"])
                        if row["avg_tts_time_ms"] is not None else None
                    ),
                }

        return stats