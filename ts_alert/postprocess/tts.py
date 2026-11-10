"""Text-to-Speech.   Phụ trách: Thành viên 4"""
from __future__ import annotations

import re
from pathlib import Path


_DEFAULT_TEMPLATE = "Chú ý, {position_text} có biển {name}"
_DEFAULT_POSITION_TEXT = {
    "left": "bên trái",
    "front": "phía trước",
    "right": "bên phải",
}


def build_speech_text(
    label: str,
    position: str,
    classes: dict,
    cfg: dict | None = None,
) -> str:
    """Tạo cảnh báo chứa vị trí và tên đọc của biển báo.

    Ưu tiên khớp nhãn chính xác, sau đó khớp không phân biệt hoa thường
    và bỏ khoảng trắng hai đầu. Dùng tts_text làm tên đọc hoặc template
    riêng nếu chứa {position_text} hoặc {position}. Khi không truyền cfg,
    sử dụng template và bản dịch vị trí mặc định.
    """
    label_raw = "" if label is None else str(label)
    class_entries = (classes or {}).values()
    class_info = None

    for info in class_entries:
        if info.get("name") == label_raw:
            class_info = info
            break

    label_clean = label_raw.strip()
    if class_info is None and label_clean:
        for info in class_entries:
            if (info.get("name") or "").strip().casefold() == label_clean.casefold():
                class_info = info
                break

    raw_name = class_info.get("name", label_raw) if class_info is not None else label_raw
    raw_tts = (
        str(class_info.get("tts_text") or "").strip()
        if class_info is not None
        else ""
    )
    if raw_tts.startswith("TODO_"):
        raw_tts = ""

    tts_cfg = ((cfg or {}).get("postprocess") or {}).get("tts") or {}
    template = str(tts_cfg.get("template") or _DEFAULT_TEMPLATE)
    pos_map = {
        **_DEFAULT_POSITION_TEXT,
        **{
            str(key).strip().lower(): "" if value is None else str(value)
            for key, value in (tts_cfg.get("position_text") or {}).items()
        },
    }

    pos_raw = "" if position is None else str(position).strip()
    position_text = pos_map.get(pos_raw.lower(), pos_raw)

    if raw_tts and (
        "{position_text}" in raw_tts or "{position}" in raw_tts
    ):
        template = raw_tts
        spoken_name = str(raw_name).strip()
    else:
        spoken_name = raw_tts or str(raw_name).strip()

    if re.search(r"\bbiển\s+\{name\}", template, flags=re.IGNORECASE):
        spoken_name = re.sub(
            r"^biển\s+",
            "",
            spoken_name,
            count=1,
            flags=re.IGNORECASE,
        )

    replacements = {
        "position_text": position_text,
        "position": position_text,
        "name": spoken_name,
        "label": label_raw,
    }
    text = re.sub(
        r"\{(position_text|position|name|label)\}",
        lambda match: replacements[match.group(1)],
        template,
    )

    return re.sub(r"\s+", " ", text).replace(" ,", ",").strip()


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