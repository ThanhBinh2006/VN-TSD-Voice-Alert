"""Đo tốc độ suy luận (FPS, độ trễ)."""
from __future__ import annotations

import time


def measure_fps(detector, frames: list, warmup: int = 5) -> dict:
    """Chạy detector.predict trên frames (bỏ qua `warmup` lần đầu), trả {"fps": ..., "latency_ms": ...}."""
    if not frames:
        return {"fps": 0.0, "latency_ms": 0.0}
    for frame in frames[:warmup]:
        detector.predict(frame)
    timed = frames[warmup:] or frames
    start = time.perf_counter()
    for frame in timed:
        detector.predict(frame)
    elapsed = time.perf_counter() - start
    return {"fps": len(timed) / elapsed if elapsed > 0 else 0.0,
            "latency_ms": 1000.0 * elapsed / len(timed)}
