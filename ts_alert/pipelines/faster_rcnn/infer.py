"""Suy luận biển báo bằng Faster R-CNN trên ảnh BGR."""

from __future__ import annotations

import math
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
    """Phát hiện biển báo và trả về tọa độ theo ảnh gốc."""

    name: str = "faster_rcnn"

    def __init__(self, weights: str | Path, cfg: dict) -> None:
        """Nạp trọng số, ngưỡng suy luận, danh mục lớp và thiết bị."""
        inference = cfg.get("inference", {})
        self.conf: float = float(inference.get("conf_threshold", 0.25))
        self.iou: float = float(inference.get("iou_threshold", 0.5))
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))

        requested = str(cfg.get("device", "cuda")).strip().lower()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and requested != "cpu" else "cpu"
        )
        self.model = load_checkpoint(
            weights,
            num_classes=get_num_classes(cfg) + 1,
            device=str(self.device),
        )

        if hasattr(self.model, "roi_heads") and hasattr(
            self.model.roi_heads, "nms_thresh"
        ):
            self.model.roi_heads.nms_thresh = self.iou

    def predict(
        self,
        frame: NDArray[np.uint8] | None,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Suy luận ảnh BGR và trả về detection theo DATA_CONTRACT.

        Ảnh không tồn tại hoặc rỗng trả về danh sách rỗng. Ngưỡng
        score nội bộ được tạm hạ khi cần và luôn được khôi phục.

        Raises:
            ValueError: Nếu conf_threshold là bool, không chuyển được
                thành float hữu hạn hoặc nằm ngoài khoảng [0, 1].
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

        rgb = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.float32)
        rgb /= np.float32(255.0)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        roi_heads = None
        original_score_thresh: float | None = None
        if hasattr(self.model, "roi_heads") and hasattr(
            self.model.roi_heads, "score_thresh"
        ):
            roi_heads = self.model.roi_heads
            original_score_thresh = roi_heads.score_thresh

        lower_score_thresh = (
            original_score_thresh is not None
            and threshold < original_score_thresh
        )

        with torch.inference_mode():
            try:
                if lower_score_thresh:
                    roi_heads.score_thresh = threshold
                prediction = self.model([tensor])[0]
            finally:
                if lower_score_thresh:
                    roi_heads.score_thresh = original_score_thresh

            keep = (
                (prediction["scores"] >= threshold)
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
