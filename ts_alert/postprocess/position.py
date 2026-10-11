"""Xác định vị trí biển báo so với người lái.   Phụ trách: Thành viên 4"""
from __future__ import annotations


def get_position(bbox: list, frame_width: int, cfg: dict) -> str:
    """Trả "left" | "front" | "right" dựa vào tâm ngang của bbox / frame_width
    và ngưỡng cfg["postprocess"]["position"]["left_max"], ["right_min"]."""
    # TODO: Thành viên 4
    raise NotImplementedError("TODO: Thành viên 4")


def add_positions(detections: list, frame_width: int, cfg: dict) -> list:
    for det in detections:
        det["position"] = get_position(det["bbox"], frame_width, cfg)
    return detections
