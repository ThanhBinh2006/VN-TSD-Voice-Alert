"""Chống cảnh báo lặp: một biển báo trong video chỉ kích hoạt đọc đúng một lần.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from ts_alert.types import Detection, SpeechEvent


class Deduplicator:
    def __init__(self, cfg: dict):
        self.iou_match = cfg["postprocess"]["dedup"]["iou_match"]
        self.lost_frames = cfg["postprocess"]["dedup"]["lost_frames"]
        self.reset()

    def reset(self) -> None:
        """Xoá toàn bộ trạng thái theo dõi (gọi khi bắt đầu video mới)."""
        # TODO: Thành viên 4 (khởi tạo cấu trúc lưu các biển đang được theo dõi)
        raise NotImplementedError("TODO: Thành viên 4")

    def update(self, detections: list[Detection], frame_index: int, timestamp_sec: float) -> list[SpeechEvent]:
        """Nhận detection của khung hình hiện tại, trả danh sách SpeechEvent CẦN ĐỌC (chỉ biển mới xuất hiện).

        Gợi ý: một detection là "cùng biển" với track cũ nếu cùng label và IoU > iou_match;
        track bị xoá khi vắng quá lost_frames khung hình (biển xuất hiện lại sau đó = biển mới).
        """
        # TODO: Thành viên 4
        raise NotImplementedError("TODO: Thành viên 4")
