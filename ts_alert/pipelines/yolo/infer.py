"""YoloDetector: suy luận trên 1 khung hình.   Phụ trách: Thành viên 1"""
from __future__ import annotations

from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.types import Detection


class YoloDetector(BaseDetector):
    name = "yolo"

    def __init__(self, weights, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["inference"]["conf_threshold"]
        self.model = load_model(weights)

    def predict(self, frame) -> list[Detection]:
        """Chạy YOLO trên frame (BGR) và trả danh sách Detection theo docs/DATA_CONTRACT.md:
        [{"bbox": [x1, y1, x2, y2], "label": str, "class_id": int, "confidence": float}]

        Lưu ý: bbox theo pixel của ẢNH GỐC; label lấy từ configs/classes.csv.
        """
        # TODO: Thành viên 1
        raise NotImplementedError("TODO: Thành viên 1")
