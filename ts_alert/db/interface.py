"""HỢP ĐỒNG giữa luồng xử lý (ts_alert/pipeline.py) và nơi lưu log.

Nhóm được TỰ DO thiết kế cấu trúc DB (SQLite, bảng, cột...) miễn là lớp Repository vẫn có đủ các hàm dưới đây
với đúng ý nghĩa tham số, và lưu được đủ DỮ LIỆU CỐT LÕI:

  - biển báo nào được phát hiện, ở khung hình / thời điểm nào, vị trí trái-phải-trước mặt
  - các lần phát giọng nói: câu đọc, biển báo kích hoạt, thời điểm
  - đường dẫn file âm thanh, đường dẫn video/ảnh kết quả
  - số biển báo duy nhất, số lần giọng nói thực sự được phát
  - thời gian xử lý riêng của phần vẽ layer, phần tạo giọng nói và tổng thời gian
"""
from __future__ import annotations

from typing import Protocol


class ResultStore(Protocol):
    def create_job(self, input_type: str, input_path: str, model_name: str) -> int:
        """input_type: "image" | "video"; model_name: "yolo" | "faster_rcnn". Trả job_id."""
        ...

    def set_status(self, job_id: int, status: str) -> None:
        """status: "queued" | "processing" | "done" | "failed"."""
        ...

    def add_detections(self, job_id: int, frame_index: int, timestamp_sec: float, detections: list) -> None:
        """Ghi các Detection của một khung hình (có đủ bbox, label, class_id, confidence, position).
        NÊN gán det["db_id"] = id dòng vừa ghi để voice event tham chiếu được (không bắt buộc)."""
        ...

    def add_voice_event(self, job_id: int, detection_id, spoken_text: str, timestamp_sec: float, audio_path: str) -> None:
        """Ghi một lần phát giọng nói. detection_id có thể là None."""
        ...

    def finish_job(self, job_id: int, output_path: str, unique_sign_count: int, voice_play_count: int,
                   layer_time_ms: float, tts_time_ms: float, total_time_ms: float) -> None:
        """Đánh dấu hoàn tất và lưu số liệu tổng kết (thời gian tính bằng mili-giây)."""
        ...

    def fail_job(self, job_id: int, error_message: str) -> None:
        ...
