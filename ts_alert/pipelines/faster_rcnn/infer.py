"""FasterRCNNDetector: suy luận trên 1 khung hình.   Phụ trách: Thành viên 2"""
from __future__ import annotations

import torch
import numpy as np

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(weights, num_classes=get_num_classes(cfg) + 1, device=str(self.device))

    def predict(self, frame) -> list[Detection]:
        """Chạy Faster R-CNN trên frame (BGR) và trả danh sách Detection theo docs/DATA_CONTRACT.md.

        Lưu ý: BGR -> RGB -> tensor [0,1]; lọc score < self.conf; đổi nhãn torchvision (1..N) -> class_id (0..N-1).
        """
        if frame is None or frame.size == 0:
            return []

        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        self.model.eval()
        
        with torch.inference_mode():
            outputs = self.model([tensor])[0]

        detections: list[Detection] = []
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        
        for box, score, label in zip(boxes, scores, labels):
            confidence = float(score.item())

            if confidence < self.conf:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            class_id = int(label.item()) - 1
            if class_id < 0:
                continue

            name = self.classes.get(class_id, {}).get("name", "")
            if not name:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": name,
                    "class_id": class_id,
                    "confidence": confidence,
                }
            )

        return detections
