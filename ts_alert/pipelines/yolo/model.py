"""Nạp mô hình YOLO (Ultralytics 8.4.175, YOLO26m).   Phụ trách: Thành viên 1"""
from __future__ import annotations

from pathlib import Path


def load_model(weights_or_name: str | Path):
    """Trả về đối tượng ultralytics.YOLO.

    - Huấn luyện: weights_or_name = cfg["model"]["pretrained"] (vd. "yolo26m.pt")
    - Suy luận:   weights_or_name = đường dẫn best.pt đã fine-tune
    Nhớ import ultralytics BÊN TRONG hàm (import muộn).
    """
    from ultralytics import YOLO

    return YOLO(str(weights_or_name))
