"""Suy luận biển báo bằng Faster R-CNN trên ảnh BGR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ts_alert.config.resolve import get_num_classes, resolve_path
from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.faster_rcnn.model import load_checkpoint
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes


class FasterRCNNDetector(BaseDetector):
    """Phát hiện biển báo và trả tọa độ hộp theo ảnh gốc."""

    name = "faster_rcnn"

    def __init__(self, weights: str | Path, cfg: dict) -> None:
        """Nạp cấu hình, danh sách lớp và mô hình trên thiết bị khả dụng."""
        self.conf = float(cfg["inference"]["conf_threshold"])
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_checkpoint(
            weights,
            num_classes=get_num_classes(cfg) + 1,
            device=str(self.device),
        )

    def predict(self, frame: NDArray[np.uint8] | None) -> list[Detection]:
        """Chuyển ảnh BGR sang RGB và lọc phát hiện theo ngưỡng tin cậy."""
        if frame is None or frame.size == 0:
            return []

        rgb = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.float32)
        rgb /= np.float32(255.0)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        with torch.inference_mode():
            prediction = self.model([tensor])[0]
            keep = (
                (prediction["scores"] >= self.conf)
                & (prediction["labels"] >= 1)
            )
            boxes = prediction["boxes"][keep].cpu().tolist()
            scores = prediction["scores"][keep].cpu().tolist()
            labels = prediction["labels"][keep].cpu().tolist()

        detections: list[Detection] = []
        for box, confidence, label in zip(boxes, scores, labels):
            class_id = int(label) - 1
            detections.append(
                {
                    "bbox": [float(coordinate) for coordinate in box],
                    "label": str(
                        self.classes.get(class_id, {}).get(
                            "name", f"Class_{class_id}"
                        )
                    ),
                    "class_id": class_id,
                    "confidence": float(confidence),
                }
            )

        return detections
