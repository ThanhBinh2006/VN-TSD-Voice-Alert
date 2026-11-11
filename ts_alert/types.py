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


class TrackedDetection(TypedDict, total=False):
    """Kết quả theo dõi đối tượng trên khung hình gốc trong pipeline realtime."""

    frame_id: int
    source_timestamp: float  # Thời điểm frame gốc, tính bằng giây.
    timestamp_sec: float     # Alias tương thích của source_timestamp.
    class_id: int
    confidence: float
    bbox: list[float]        # [x1, y1, x2, y2] pixel, theo khung hình gốc.
    track_id: int | None
    label: str
    position: str            # left | front | right
    vx: float                # Vận tốc theo trục x, px/giây.
    vy: float                # Vận tốc theo trục y, px/giây.
    state: str               # tentative | active | lost
    confirmed: bool
    hits: int
    missed_detections: int
    age_sec: float
    ttl_sec: float
    lag_ms: float
    source: str              # detector | optical_flow | motion_pred
    last_detect_frame_id: int
    last_detect_timestamp: float
    alert_emitted: bool
    class_votes: dict[int, float]


class RuntimeAlert(TypedDict, total=False):
    """Dữ liệu cảnh báo và trạng thái xử lý của worker TTS bất đồng bộ."""

    alert_id: str
    track_id: int | None
    frame_id: int
    source_timestamp: float  # Thời điểm frame gốc, tính bằng giây.
    class_id: int
    label: str
    position: str
    spoken_text: str
    audio_path: str
    audio_url: str
    created_wall_time: float   # Thời gian tạo, tính bằng giây theo wall clock.
    deadline_wall_time: float  # Hạn xử lý, cùng hệ thời gian với created_wall_time.
    latency_ms: float
    status: str  # queued | ready | dropped_stale | deduped | timeout | error
