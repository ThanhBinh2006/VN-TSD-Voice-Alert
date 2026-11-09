from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO


def load_model(weights_or_name: str | Path) -> YOLO:
    """Tải mô hình YOLO từ đường dẫn trọng số hoặc tên mô hình."""
    from ultralytics import YOLO

    return YOLO(str(weights_or_name))
