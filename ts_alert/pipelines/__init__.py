"""Registry nạp pipeline và detector khi cần, không import torch ở cấp module."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ts_alert.types import MODEL_NAMES

if TYPE_CHECKING:
    from ts_alert.pipelines.base import BaseDetector, BasePipeline

__all__ = ["get_pipeline", "get_detector"]


def get_pipeline(name: str, cfg: dict) -> BasePipeline:
    """Khởi tạo pipeline theo tên; báo ValueError nếu tên không được hỗ trợ."""
    if name == "yolo":
        from ts_alert.pipelines.yolo.train import YoloPipeline

        return YoloPipeline(cfg)

    if name == "faster_rcnn":
        from ts_alert.pipelines.faster_rcnn.train import FasterRCNNPipeline

        return FasterRCNNPipeline(cfg)

    raise ValueError(
        f"Model không hợp lệ: {name!r}. Chọn một trong {MODEL_NAMES}"
    )


def get_detector(name: str, cfg: dict) -> BaseDetector:
    """Nạp detector bằng cấu hình và trọng số mặc định của pipeline tương ứng."""
    return get_pipeline(name, cfg).load_detector()
