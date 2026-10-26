"""YoloDetector: suy luận trên 1 khung hình.   Phụ trách: Thành viên 1"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ts_alert.config.resolve import resolve_path
from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes


class YoloDetector(BaseDetector):
    name = "yolo"

    def __init__(self, weights, cfg: dict):
        self.cfg = cfg
        self.conf = float(cfg.get("inference", {}).get("conf_threshold", 0.25))
        self.iou = float(cfg.get("inference", {}).get("iou_threshold", 0.5))
        self.model = load_model(weights)
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))

    def predict(self, frame) -> list[Detection]:
        """Chạy YOLO trên frame (BGR) và trả danh sách Detection theo docs/DATA_CONTRACT.md:
        [{"bbox": [x1, y1, x2, y2], "label": str, "class_id": int, "confidence": float}]

        Lưu ý: bbox theo pixel của ẢNH GỐC; label lấy từ configs/classes.csv.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        # Chạy inference qua Ultralytics YOLO
        img_size = self.cfg.get("data", {}).get("image_size", 640)
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=img_size,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())

            class_info = self.classes.get(cls_id, {})
            label_name = class_info.get("name", f"Class_{cls_id}")

            detections.append({
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                "label": label_name,
                "class_id": cls_id,
                "confidence": conf_val,
            })

        return detections
