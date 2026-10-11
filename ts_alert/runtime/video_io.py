"""Đọc/ghi ảnh và video (OpenCV 4.12).   Phụ trách: Thành viên 3"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ts_alert.types import IMAGE_EXTS, VIDEO_EXTS


def media_type(path) -> str:
    """"video" hoặc "image" theo đuôi file; ValueError nếu không hỗ trợ."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    raise ValueError(f"Định dạng không hỗ trợ: {ext}")


def get_video_info(path) -> dict:
    """Trả {"fps": float, "width": int, "height": int, "frame_count": int}."""
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


def iter_frames(path) -> Iterator[tuple]:
    """Yield (frame_index, frame_bgr, timestamp_sec) lần lượt từng khung hình của video.
    timestamp_sec = frame_index / fps."""
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


def read_image(path):
    """Đọc ảnh -> ndarray BGR (H, W, 3); raise nếu không đọc được."""
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


def write_image(path, frame) -> None:
    """Ghi ndarray BGR ra file ảnh (tạo thư mục cha nếu thiếu)."""
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


class VideoWriter:
    """Ghi video KHÔNG tiếng (mp4, codec mp4v hoặc h264 nếu có)."""

    def __init__(self, path, fps: float, size: tuple):
        """size = (width, height)."""
        # TODO: Thành viên 3
        raise NotImplementedError("TODO: Thành viên 3")

    def write(self, frame) -> None:
        # TODO: Thành viên 3
        raise NotImplementedError("TODO: Thành viên 3")

    def close(self) -> None:
        # TODO: Thành viên 3
        raise NotImplementedError("TODO: Thành viên 3")
