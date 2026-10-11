"""Lưu log xử lý vào SQLite.   Phụ trách: Thành viên 5

Cần thoả giao diện ResultStore trong ts_alert/db/interface.py (cấu trúc bảng tuỳ nhóm thiết kế).
"""
from __future__ import annotations


class Repository:
    def __init__(self, db_path):
        """Mở/tạo file SQLite tại db_path và tạo bảng nếu chưa có (đọc ts_alert/db/schema.sql)."""
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def create_job(self, input_type: str, input_path: str, model_name: str) -> int:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def set_status(self, job_id: int, status: str) -> None:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def add_detections(self, job_id: int, frame_index: int, timestamp_sec: float, detections: list) -> None:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def add_voice_event(self, job_id: int, detection_id, spoken_text: str, timestamp_sec: float, audio_path: str) -> None:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def finish_job(self, job_id: int, output_path: str, unique_sign_count: int, voice_play_count: int,
                   layer_time_ms: float, tts_time_ms: float, total_time_ms: float) -> None:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    def fail_job(self, job_id: int, error_message: str) -> None:
        # TODO: Thành viên 5
        raise NotImplementedError("TODO: Thành viên 5")

    # TODO: Thành viên 5 — thêm các hàm đọc cho web (lịch sử, chi tiết job) và thống kê so sánh YOLO vs Faster R-CNN.
