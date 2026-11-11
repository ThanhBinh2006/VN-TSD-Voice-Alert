"""Text-to-Speech.   Phụ trách: Thành viên 4"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path


_DEFAULT_TEMPLATE = "Chú ý, {position_text} có biển {name}"
_DEFAULT_POSITION_TEXT = {
    "left": "bên trái",
    "front": "phía trước",
    "right": "bên phải",
}

_CACHE_GUARD = threading.Lock()
_CACHE_LOCKS: dict[str, threading.Lock] = {}


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


def _get_cache_lock(digest: str) -> threading.Lock:
    """Lấy khóa dùng chung theo digest giữa các luồng trong tiến trình."""
    with _CACHE_GUARD:
        lock = _CACHE_LOCKS.get(digest)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[digest] = lock
        return lock


def _get_cache_dir(tts_cfg: dict, cfg: dict) -> Path | None:
    """Xác định thư mục cache; trả None nếu không có cấu hình phù hợp."""
    cache_dir = tts_cfg.get("cache_dir")
    if cache_dir:
        return Path(cache_dir)

    outputs_dir = ((cfg or {}).get("paths") or {}).get("outputs_dir")
    if outputs_dir:
        return Path(outputs_dir) / "tts_cache"

    return None


def _has_audio(path: Path) -> bool:
    """Kiểm tra file âm thanh tồn tại và có dữ liệu.

    File bị xóa trong lúc kiểm tra được coi là cache miss.
    Các lỗi truy cập khác được chuyển cho bên gọi xử lý.
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _synthesize_direct(
    text: str,
    out_path: Path,
    engine: str,
    lang: str,
) -> Path:
    """Tổng hợp trực tiếp bằng engine được chọn và trả về out_path."""
    if engine == "gtts":
        from gtts import gTTS

        gTTS(text=text, lang=lang).save(str(out_path))
        return out_path

    if engine == "pyttsx3":
        import pyttsx3

        speaker = pyttsx3.init()
        try:
            speaker.save_to_file(text, str(out_path))
            speaker.runAndWait()
        finally:
            speaker.stop()
        return out_path

    raise ValueError(f"Unsupported TTS engine: {engine}")


def synthesize(text: str, out_path: Path, cfg: dict) -> Path:
    """Tạo file âm thanh cho text và trả về out_path.

    Engine lấy từ cfg["postprocess"]["tts"]["engine"]:
    gTTS cần internet; pyttsx3 chạy offline.

    Cache mặc định tắt. Khi bật và xác định được thư mục cache, dùng
    SHA-256 của engine, lang, phần mở rộng và nguyên văn text làm khóa.
    Khóa theo digest ngăn tổng hợp trùng giữa các luồng trong tiến trình;
    file tạm được thay thế nguyên tử trước khi sao chép đến out_path.
    """
    if not text.strip():
        raise ValueError("text must not be empty")

    tts_cfg = cfg["postprocess"]["tts"]
    engine = tts_cfg["engine"]
    lang = tts_cfg.get("lang", "vi")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine not in ("gtts", "pyttsx3"):
        raise ValueError(f"Unsupported TTS engine: {engine}")

    cache_enabled = bool(tts_cfg.get("cache_enabled", False))
    if not cache_enabled:
        return _synthesize_direct(text, out_path, engine, lang)

    cache_dir = _get_cache_dir(tts_cfg, cfg)
    if cache_dir is None:
        return _synthesize_direct(text, out_path, engine, lang)

    ext = ".wav" if engine == "pyttsx3" else ".mp3"
    cache_key = json.dumps(
        [engine, lang, ext, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{digest}{ext}"

    with _get_cache_lock(digest):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)

            if not _has_audio(cache_file):
                tmp_path = cache_dir / f".tmp_{digest}_{uuid.uuid4().hex}{ext}"
                try:
                    _synthesize_direct(text, tmp_path, engine, lang)
                    if not _has_audio(tmp_path):
                        raise RuntimeError(
                            f"Engine {engine!r} không tạo được file âm thanh "
                            f"hợp lệ tại {tmp_path}"
                        )

                    # File tạm nằm cùng thư mục để thay thế nguyên tử.
                    os.replace(tmp_path, cache_file)
                finally:
                    tmp_path.unlink(missing_ok=True)

            # Caller luôn nhận bản sao riêng để có thể tự dọn out_path.
            shutil.copy2(cache_file, out_path)
            return out_path
        except OSError as exc:
            raise OSError(
                f"Không thể đọc/ghi cache TTS tại {cache_file} "
                f"hoặc sao chép đến {out_path}: {exc}"
            ) from exc
