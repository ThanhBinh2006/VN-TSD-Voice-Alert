"""Xác định vị trí biển báo so với người lái.
Phụ trách: Thành viên 4
"""
from __future__ import annotations


def get_position(bbox: list, frame_width: int, cfg: dict) -> str:
    """Trả "left" | "front" | "right" dựa vào tâm ngang của bbox / frame_width
    và ngưỡng cfg["postprocess"]["position"]["left_max"], ["right_min"].
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    ratio = center_x / frame_width

    position_cfg = cfg["postprocess"]["position"]
    left_max = position_cfg["left_max"]
    right_min = position_cfg["right_min"]

    if ratio < left_max:
        return "left"
    if ratio < right_min:
        return "front"
    return "right"


def add_positions(detections: list, frame_width: int, cfg: dict) -> list:
    for det in detections:
        det["position"] = get_position(det["bbox"], frame_width, cfg)
    return detections