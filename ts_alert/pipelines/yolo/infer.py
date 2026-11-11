"""Bộ phát hiện biển báo sử dụng mô hình YOLO với profile CPU tùy chọn, an toàn luồng."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ts_alert.config.resolve import resolve_path
from ts_alert.pipelines.base import BaseDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes


_LOCK_INIT = threading.Lock()


class YoloDetector(BaseDetector):
    """Phát hiện biển báo và trả về tọa độ theo ảnh gốc."""

    name: str = "yolo"

    def __init__(self, weights: str | Path, cfg: dict) -> None:
        """Nạp trọng số, ngưỡng suy luận và danh mục lớp biển báo; chỉ áp dụng CPU tuning nếu cấu hình bật."""
        inference = cfg.get("inference", {})
        data = cfg.get("data", {})
        cpu_cfg = inference.get("cpu") or {}

        self._predict_lock = threading.RLock()
        self.weights_path = Path(weights)
        self.conf: float = float(inference.get("conf_threshold", 0.25))
        self.iou: float = float(inference.get("iou_threshold", 0.5))
        self.img_size: int = int(data.get("image_size", 640))
        self.model = load_model(weights)
        self.classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))

        if bool(cpu_cfg.get("apply_on_load", False)):
            self.apply_cpu_optimization(cpu_cfg)

    def _guard(self) -> threading.RLock:
        """Trả về khóa suy luận, khởi tạo an toàn nếu chưa chạy __init__."""
        lock = getattr(self, "_predict_lock", None)
        if lock is None:
            with _LOCK_INIT:
                lock = getattr(self, "_predict_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._predict_lock = lock
        return lock

    def apply_cpu_optimization(
        self,
        cpu_cfg: dict | str | None = None,
    ) -> dict[str, Any]:
        """Áp dụng profile CPU dưới cùng khóa dùng cho suy luận."""
        from ts_alert.runtime.cpu_opt import apply_yolo_cpu_tuning

        with self._guard():
            return apply_yolo_cpu_tuning(self, cpu_cfg)

    def restore_cpu_optimization(self) -> bool:
        """Khôi phục cấu hình CPU gốc nếu đã lưu baseline."""
        from ts_alert.runtime.cpu_opt import restore_cpu_tuning

        with self._guard():
            return restore_cpu_tuning(self)

    def get_runtime_profile(self) -> dict[str, Any]:
        """Trả về cấu hình suy luận hiện tại và trạng thái fuse mô hình."""
        with self._guard():
            cpu_profile = getattr(
                self, "_cpu_profile", getattr(self, "cpu_profile", None)
            )
            fused = bool(getattr(self, "_cpu_fused", False))

            model = getattr(self.model, "model", None)
            is_fused = getattr(model, "is_fused", None)
            if callable(is_fused):
                fused = bool(is_fused())

            return {
                "model": self.name,
                "img_size": self.img_size,
                "conf": self.conf,
                "iou": self.iou,
                "cpu_profile": cpu_profile,
                "fused": fused,
            }

    def predict(
        self,
        frame: NDArray[np.uint8] | None,
        conf_threshold: float | None = None,
        *,
        img_size: int | None = None,
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

        if img_size is not None:
            if (
                isinstance(img_size, (bool, np.bool_))
                or not isinstance(img_size, (int, np.integer))
                or img_size <= 0
            ):
                raise ValueError("img_size phải là số nguyên dương.")

        with self._guard():
            results = self.model.predict(
                source=frame,
                conf=threshold,
                iou=self.iou,
                imgsz=self.img_size if img_size is None else int(img_size),
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
