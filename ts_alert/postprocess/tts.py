"""Text-to-Speech.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from pathlib import Path


def build_speech_text(label: str, position: str, classes: dict) -> str:
    """Tạo câu đọc từ nhãn + vị trí, dùng cfg["postprocess"]["tts"] (template, position_text) trong shared.yaml
    và cột tts_text trong configs/classes.csv (nếu có) — vd. "Chú ý, bên trái có biển cấm rẽ trái"."""
    # Tìm thông tin class tương ứng với label.
    class_info = None

    for info in classes.values():
        if info.get("name") == label:
            class_info = info
            break

    # Nếu không tìm thấy class, dùng chính label làm tên biển báo.
    name = label
    tts_text = ""

    if class_info is not None:
        name = class_info.get("name", label)
        tts_text = class_info.get("tts_text", "")

    # Nếu classes.csv có câu đọc riêng và không phải placeholder,
    # ưu tiên sử dụng câu đó.
    if tts_text and not tts_text.startswith("TODO_"):
        return tts_text

    # Template mặc định theo contract của project.
    position_text = {
        "left": "bên trái",
        "front": "phía trước",
        "right": "bên phải",
    }.get(position, position)

    return f"Chú ý, {position_text} có biển {name}"


def synthesize(text: str, out_path: Path, cfg: dict) -> Path:
    """Tạo file âm thanh cho `text` tại out_path (mp3/wav), engine theo cfg["postprocess"]["tts"]["engine"]
    (gTTS 2.5.4 cần internet; pyttsx3 2.99 chạy offline). Trả out_path."""
    if not text.strip():
        raise ValueError("text must not be empty")

    tts_cfg = cfg["postprocess"]["tts"]
    engine = tts_cfg["engine"]
    lang = tts_cfg.get("lang", "vi")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "gtts":
        from gtts import gTTS

        gTTS(text=text, lang=lang).save(str(out_path))
        return out_path

    if engine == "pyttsx3":
        import pyttsx3

        speaker = pyttsx3.init()
        speaker.save_to_file(text, str(out_path))
        speaker.runAndWait()
        speaker.stop()
        return out_path

    raise ValueError(f"Unsupported TTS engine: {engine}")