"""Kiểu dữ liệu và hằng số dùng chung (xem docs/DATA_CONTRACT.md)."""
from __future__ import annotations

from typing import TypedDict

POSITIONS = ("left", "front", "right")
MODEL_NAMES = ("yolo", "faster_rcnn")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Detection(TypedDict, total=False):
    bbox: list          # [x1, y1, x2, y2] pixel, theo ảnh gốc
    label: str
    class_id: int       # id theo configs/classes.csv (KHÔNG tính background)
    confidence: float
    position: str       # left | front | right  (do postprocess/position.py thêm)
    db_id: int          # id dòng trong bảng detection (do repository thêm)


class SpeechEvent(TypedDict):
    """Một lần cần đọc giọng nói, do Deduplicator trả về."""
    detection: Detection
    label: str
    position: str
    frame_index: int
    timestamp_sec: float
