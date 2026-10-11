"""Text-to-Speech.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from pathlib import Path


def build_speech_text(label: str, position: str, classes: dict) -> str:
    """Tạo câu đọc từ nhãn + vị trí, dùng cfg["postprocess"]["tts"] (template, position_text) trong shared.yaml
    và cột tts_text trong configs/classes.csv (nếu có) — vd. "Chú ý, bên trái có biển cấm rẽ trái"."""
    # TODO: Thành viên 4
    raise NotImplementedError("TODO: Thành viên 4")


def synthesize(text: str, out_path: Path, cfg: dict) -> Path:
    """Tạo file âm thanh cho `text` tại out_path (mp3/wav), engine theo cfg["postprocess"]["tts"]["engine"]
    (gTTS 2.5.4 cần internet; pyttsx3 2.99 chạy offline). Trả out_path."""
    # TODO: Thành viên 4
    raise NotImplementedError("TODO: Thành viên 4")
