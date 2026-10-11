"""Đăng ký các mô hình. Import muộn để module không cần torch vẫn chạy được (test, web UI khởi động nhanh)."""
from __future__ import annotations

from ts_alert.pipelines.base import BaseDetector, BasePipeline
from ts_alert.types import MODEL_NAMES


def get_pipeline(name: str, cfg: dict) -> BasePipeline:
    if name == "yolo":
        from ts_alert.pipelines.yolo.train import YoloPipeline
        return YoloPipeline(cfg)
    if name == "faster_rcnn":
        from ts_alert.pipelines.faster_rcnn.train import FasterRCNNPipeline
        return FasterRCNNPipeline(cfg)
    raise ValueError(f"Model không hợp lệ: {name!r}. Chọn một trong {MODEL_NAMES}")


def get_detector(name: str, cfg: dict) -> BaseDetector:
    return get_pipeline(name, cfg).load_detector()
