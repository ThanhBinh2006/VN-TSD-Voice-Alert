"""Bộ phát hiện biển báo sử dụng mô hình YOLO."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ts_alert.config.resolve import resolve_path
from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes


class YoloDetector(BaseDetector):
    """Phát hiện biển báo và trả về tọa độ theo ảnh gốc."""

    name: str = "yolo"

    def __init__(self, weights: str | Path, cfg: dict) -> None:
        """Nạp trọng số, ngưỡng suy luận và danh mục lớp biển báo."""
        inference = cfg.get("inference", {})
        data = cfg.get("data", {})

        self.conf: float = float(inference.get("conf_threshold", 0.25))
        self.iou: float = float(inference.get("iou_threshold", 0.5))
        self.img_size: int = int(data.get("image_size", 640))
        self.model = load_model(weights)
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))

    def predict(
        self,
        frame: NDArray[np.uint8] | None,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Trả về các phát hiện; ảnh không tồn tại hoặc rỗng trả về [].

        Dùng self.conf khi conf_threshold là None.
        Raise ValueError nếu ngưỡng truyền vào không phải float trong [0.0, 1.0].
        """
        threshold = self.conf
        if conf_threshold is not None:
            if isinstance(conf_threshold, (bool, np.bool_)):
                raise ValueError("conf_threshold không được là bool.")

            try:
                threshold = float(conf_threshold)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "conf_threshold phải là số hữu hạn trong khoảng [0, 1]."
                ) from exc

            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "conf_threshold phải là số hữu hạn trong khoảng [0, 1]."
                )

        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=threshold,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                bbox = [float(value) for value in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "bbox": bbox,
                        "label": str(
                            self.classes.get(class_id, {}).get(
                                "name", f"Class_{class_id}"
                            )
                        ),
                        "class_id": class_id,
                        "confidence": confidence,
                    }
                )

        return detections
