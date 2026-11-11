"""Chống cảnh báo lặp: một biển báo trong video chỉ kích hoạt đọc đúng một lần.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from ts_alert.types import Detection, SpeechEvent


class Deduplicator:
    def __init__(self, cfg: dict) -> None:
        """Khởi tạo bộ chống cảnh báo lặp theo cấu hình."""
        self.iou_match = cfg["postprocess"]["dedup"]["iou_match"]
        self.lost_frames = cfg["postprocess"]["dedup"]["lost_frames"]

        frame_stride = (cfg.get("inference") or {}).get("frame_stride", 1)
        if frame_stride is None:
            frame_stride = 1
        if isinstance(frame_stride, bool):
            raise ValueError("inference.frame_stride không được là bool.")

        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError("inference.frame_stride phải >= 1.")

        self.reset()

    def reset(self) -> None:
        """Xoá toàn bộ trạng thái theo dõi (gọi khi bắt đầu video mới)."""
        # TODO: Thành viên 4 (khởi tạo cấu trúc lưu các biển đang được theo dõi)
        self.tracks = []

    @staticmethod
    def _iou(box_a: list, box_b: list) -> float:
        """Tính IoU giữa hai bounding box [x1, y1, x2, y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp_sec: float,
    ) -> list[SpeechEvent]:
        """Nhận detection của khung hình hiện tại, trả danh sách SpeechEvent CẦN ĐỌC (chỉ biển mới xuất hiện).

        Gợi ý: một detection là "cùng biển" với track cũ nếu cùng label và IoU > iou_match;
        track bị xoá khi vắng quá lost_frames khung hình (biển xuất hiện lại sau đó = biển mới).
        """
        effective_lost_frames = max(self.lost_frames, self.frame_stride)

        # 1. Xoá các track đã biến mất quá lâu.
        self.tracks = [
            track
            for track in self.tracks
            if frame_index - track["last_seen"] <= effective_lost_frames
        ]

        events: list[SpeechEvent] = []
        matched_track_indexes: set[int] = set()

        # 2. Xử lý từng detection của frame hiện tại.
        for detection in detections:
            best_track_index: int | None = None
            best_iou = 0.0

            for track_index, track in enumerate(self.tracks):
                if track_index in matched_track_indexes:
                    continue

                if track["label"] != detection["label"]:
                    continue

                iou = self._iou(track["bbox"], detection["bbox"])

                if iou > self.iou_match and iou > best_iou:
                    best_iou = iou
                    best_track_index = track_index

            # 3. Nếu match được track cũ → cập nhật track, không đọc lại.
            if best_track_index is not None:
                track = self.tracks[best_track_index]
                track["bbox"] = detection["bbox"]
                track["last_seen"] = frame_index
                matched_track_indexes.add(best_track_index)
                continue

            # 4. Không match được → đây là biển mới.
            self.tracks.append(
                {
                    "label": detection["label"],
                    "bbox": detection["bbox"],
                    "last_seen": frame_index,
                }
            )

            events.append(
                {
                    "detection": detection,
                    "label": detection["label"],
                    "position": detection["position"],
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                }
            )

        return events
