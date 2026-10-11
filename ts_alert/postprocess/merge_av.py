"""Ghép âm thanh vào video theo timestamp.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from pathlib import Path


def merge_audio_video(video_path: Path, audio_events: list, out_path: Path) -> Path:
    """audio_events: [{"audio_path": Path, "timestamp_sec": float, "text": str}, ...].

    Đặt từng đoạn audio bắt đầu tại timestamp_sec rồi mix thành một track gắn vào video_path -> out_path.
    Gợi ý: ffmpeg (binary lấy từ imageio-ffmpeg 0.6.0) với filter adelay + amix. Trả out_path.
    """
    # TODO: Thành viên 4
    raise NotImplementedError("TODO: Thành viên 4")
