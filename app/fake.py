"""Chế độ --fake: chạy thử web end-to-end khi model / hậu xử lý của thành viên khác chưa xong.
Phụ trách: Thành viên 3. CHỈ dùng khi phát triển, KHÔNG dùng để báo cáo kết quả.

- FakeDetector: sinh biển báo giả xen kẽ trái / phải theo thời gian (không cần trọng số).
- install_fakes(): chỉ thay những hàm hậu xử lý CÒN là vỏ (`raise NotImplementedError("TODO`).
  Hàm nào Thành viên 4 đã hiện thực thì giữ nguyên bản thật -> càng về sau càng ít phần giả.
  Trả về hàm restore() để khôi phục (test dùng để không ảnh hưởng test khác).
"""
from __future__ import annotations

import inspect
import logging
import shutil
import wave
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

STUB_MARKER = 'raise NotImplementedError("TODO'

_FAKE_SIGNS = [
    # (label, class_id, tâm x tương đối, tâm y tương đối)
    ("Cấm rẽ trái (giả)", 0, 0.15, 0.35),
    None,
    ("Giới hạn tốc độ 50 (giả)", 0, 0.85, 0.35),
    None,
    ("Đường người đi bộ (giả)", 0, 0.50, 0.30),
    None,
]
_PHASE_FRAMES = 15                   # mỗi pha kéo dài 15 khung hình


class FakeDetector:
    """Trả detection theo đúng DATA_CONTRACT, lần lượt các pha: biển trái, trống, biển phải, trống, ..."""

    name = "fake"

    def __init__(self, *_args, **_kwargs):
        self._calls = 0

    def predict(self, frame) -> list[dict]:
        sign = _FAKE_SIGNS[(self._calls // _PHASE_FRAMES) % len(_FAKE_SIGNS)]
        self._calls += 1
        if sign is None:
            return []
        label, class_id, cx, cy = sign
        h, w = frame.shape[:2]
        half = max(4, int(min(w, h) * 0.08))
        x, y = int(cx * w), int(cy * h)
        return [{"bbox": [max(0, x - half), max(0, y - half), min(w - 1, x + half), min(h - 1, y + half)],
                 "label": label, "class_id": class_id, "confidence": 0.9}]


def fake_detector_factory(model_name: str, cfg: dict) -> FakeDetector:
    return FakeDetector()


# ----------------------------------------------------------------------------- hậu xử lý giả

def _fake_get_position(bbox, frame_width, cfg):
    p = cfg["postprocess"]["position"]
    cx = (bbox[0] + bbox[2]) / 2 / max(frame_width, 1)
    return "left" if cx < p["left_max"] else "right" if cx > p["right_min"] else "front"


def _fake_draw_detections(frame, detections):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"FAKE {det.get('confidence', 0):.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out


_POSITION_TEXT = {"left": "bên trái", "front": "phía trước", "right": "bên phải"}


def _fake_build_speech_text(label, position, classes):
    return f"Chú ý, {_POSITION_TEXT.get(position, position)} có biển {label}"


def _fake_synthesize(text, out_path, cfg):
    """Ghi 0,5 giây im lặng (WAV) thay cho giọng đọc thật — không cần internet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 8000)
    return out_path


def _fake_merge_audio_video(video_path, audio_events, out_path):
    """Không ghép tiếng, chỉ chép video sang out_path."""
    shutil.copyfile(video_path, out_path)
    return Path(out_path)


class _FakeDeduplicator:
    """Một nhãn chỉ đọc lại khi đã vắng quá lost_frames khung hình."""

    def __init__(self, cfg):
        self.lost_frames = cfg["postprocess"]["dedup"]["lost_frames"]
        self.reset()

    def reset(self):
        self.last_seen: dict[str, int] = {}

    def update(self, detections, frame_index, timestamp_sec):
        events = []
        for det in detections:
            last = self.last_seen.get(det["label"])
            if last is None or frame_index - last > self.lost_frames:
                events.append({"detection": det, "label": det["label"], "position": det.get("position", "front"),
                               "frame_index": frame_index, "timestamp_sec": timestamp_sec})
            self.last_seen[det["label"]] = frame_index
        return events


# ----------------------------------------------------------------------------- cài / gỡ

def is_stub(obj) -> bool:
    try:
        return STUB_MARKER in inspect.getsource(obj)
    except (OSError, TypeError):
        return False


def install_fakes():
    """Thay các hàm hậu xử lý còn là vỏ bằng bản giả. Trả (restore, danh sách tên đã thay)."""
    import ts_alert.pipeline as flow
    from ts_alert.postprocess import dedup, draw_layer, merge_av, position, tts

    candidates = [
        (position, "get_position", position.get_position, _fake_get_position),
        (draw_layer, "draw_detections", draw_layer.draw_detections, _fake_draw_detections),
        (tts, "build_speech_text", tts.build_speech_text, _fake_build_speech_text),
        (tts, "synthesize", tts.synthesize, _fake_synthesize),
        (merge_av, "merge_audio_video", merge_av.merge_audio_video, _fake_merge_audio_video),
        (flow, "Deduplicator", dedup.Deduplicator.update, _FakeDeduplicator),
    ]
    patched = []
    for module, attr, real_impl, fake_impl in candidates:
        if is_stub(real_impl):
            patched.append((module, attr, getattr(module, attr)))
            setattr(module, attr, fake_impl)
    names = [f"{m.__name__.split('.')[-1]}.{a}" for m, a, _ in patched]
    if names:
        log.warning("--fake: đang dùng bản GIẢ cho: %s", ", ".join(names))

    def restore():
        for module, attr, original in patched:
            setattr(module, attr, original)

    return restore, names
