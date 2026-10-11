"""FasterRCNNDetector: suy luận trên 1 khung hình.   Phụ trách: Thành viên 2"""
from __future__ import annotations

from ts_alert.config.resolve import get_num_classes, resolve_path
from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.faster_rcnn.model import load_checkpoint
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes


class FasterRCNNDetector(BaseDetector):
    name = "faster_rcnn"

    def __init__(self, weights, cfg: dict):
        self.cfg = cfg
        self.conf = cfg["inference"]["conf_threshold"]
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
        self.model = load_checkpoint(weights, num_classes=get_num_classes(cfg) + 1)

    def predict(self, frame) -> list[Detection]:
        """Chạy Faster R-CNN trên frame (BGR) và trả danh sách Detection theo docs/DATA_CONTRACT.md.

        Lưu ý: BGR -> RGB -> tensor [0,1]; lọc score < self.conf; đổi nhãn torchvision (1..N) -> class_id (0..N-1).
        """
        # TODO: Thành viên 2
        raise NotImplementedError("TODO: Thành viên 2")
