"""Lập lịch frame, đồng bộ playback và lưu overlay với bộ nhớ giới hạn."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def _finite(value: float, name: str) -> float:
    """Chuẩn hóa một giá trị số hữu hạn."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} phải là số hữu hạn")
    return result


def _nonnegative(value: float, name: str) -> float:
    """Kiểm tra một giá trị hữu hạn không âm."""
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} phải >= 0")
    return result


def _validate_mode(mode: str) -> str:
    """Kiểm tra chế độ playback được hỗ trợ."""
    if mode not in ("realtime", "buffered"):
        raise ValueError(f"Chế độ playback không hợp lệ: {mode!r}")
    return mode


class CancelledError(RuntimeError):
    """Tác vụ đã bị hủy thông qua CancellationToken."""


class CancellationToken:
    """Token hủy thread-safe; lần hủy đầu tiên quyết định lý do."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._listeners: dict[object, Callable[[], None]] = {}

    def cancel(self, reason: str = "cancelled") -> None:
        """Hủy token và đánh thức các bộ chờ đang đăng ký."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            listeners = tuple(self._listeners.values())
            self._listeners.clear()

        # Không giữ lock token khi lấy lock của đối tượng đang chờ.
        for listener in listeners:
            listener()

    def is_cancelled(self) -> bool:
        """Trả về trạng thái hủy hiện tại."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Trả về lý do hủy, hoặc None nếu chưa hủy."""
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        """Chờ hủy; trả True nếu token đã bị hủy."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        """Ném CancelledError nếu token đã bị hủy."""
        with self._lock:
            if self._event.is_set():
                raise CancelledError(self._reason or "cancelled")

    def _subscribe(self, listener: Callable[[], None]) -> object:
        """Đăng ký đánh thức mà không bỏ lỡ cancellation đồng thời."""
        key = object()
        with self._lock:
            cancelled = self._event.is_set()
            if not cancelled:
                self._listeners[key] = listener
        if cancelled:
            listener()
        return key

    def _unsubscribe(self, key: object) -> None:
        """Giải phóng callback cùng tham chiếu đối tượng chờ."""
        with self._lock:
            self._listeners.pop(key, None)


@dataclass(slots=True)
class FrameEnvelope:
    """Frame nguồn và dữ liệu liên quan trong một thế hệ playback."""

    frame_id: int
    source_timestamp: float
    captured_wall_time: float
    frame: np.ndarray
    width: int
    height: int
    generation: int = 0
    gray_pair: tuple[np.ndarray, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameRingBuffer:
    """Bộ đệm thumbnail xám có dung lượng cố định và bản sao độc lập."""

    def __init__(self, maxlen: int = 24) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen phải > 0")
        self._lock = threading.Lock()
        self._items: deque[tuple[int, int, float, np.ndarray, float]] = deque(
            maxlen=maxlen
        )

    def put(
        self,
        generation: int,
        frame_id: int,
        source_timestamp: float,
        gray: np.ndarray,
        scale: float,
    ) -> None:
        """Lưu thumbnail uint8; tự giải phóng mục cũ khi đầy."""
        timestamp = _finite(source_timestamp, "source_timestamp")
        scale = _finite(scale, "scale")
        if scale <= 0:
            raise ValueError("scale phải > 0")
        if gray.ndim != 2 or gray.size == 0 or gray.dtype != np.uint8:
            raise ValueError("gray phải là thumbnail grayscale uint8 không rỗng")

        # Copy tránh giữ cả ảnh gốc nếu thumbnail là một numpy view.
        owned_gray = gray.copy(order="C")
        owned_gray.setflags(write=False)
        with self._lock:
            self._items.append(
                (generation, frame_id, timestamp, owned_gray, scale)
            )

    def find_closest(
        self,
        target_timestamp: float,
        generation: int = 0,
        max_diff_sec: float = 0.25,
    ) -> tuple[np.ndarray, float] | None:
        """Trả thumbnail gần nhất trong cùng thế hệ và ngưỡng thời gian."""
        target = _finite(target_timestamp, "target_timestamp")
        limit = _nonnegative(max_diff_sec, "max_diff_sec")
        with self._lock:
            best: tuple[np.ndarray, float] | None = None
            best_diff = math.inf
            for item_generation, _, timestamp, gray, scale in reversed(self._items):
                if item_generation != generation:
                    continue
                difference = abs(timestamp - target)
                if difference <= limit and difference < best_diff:
                    best = (gray, scale)
                    best_diff = difference
            return best

    def clear(self, generation: int | None = None) -> None:
        """Xóa toàn bộ hoặc chỉ các thumbnail thuộc thế hệ chỉ định."""
        with self._lock:
            if generation is None:
                self._items.clear()
            else:
                self._items = deque(
                    (item for item in self._items if item[0] != generation),
                    maxlen=self._items.maxlen,
                )


class LatestFrameSlot:
    """Slot một frame; ghi đè frame chưa xử lý để không tạo backlog."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._item: FrameEnvelope | None = None
        self._closed = False
        self._lifecycle = 0
        self._pushed = 0
        self._consumed = 0
        self._dropped = 0

    def push(self, envelope: FrameEnvelope) -> int:
        """Đặt frame mới nhất; trả 1 nếu ghi đè một frame đang chờ."""
        with self._condition:
            if self._closed:
                return 0
            dropped = int(self._item is not None)
            self._item = envelope
            self._pushed += 1
            self._dropped += dropped
            self._condition.notify()
            return dropped

    def _wake_waiters(self) -> None:
        """Đánh thức bộ chờ khi token bị hủy."""
        with self._condition:
            self._condition.notify_all()

    def pop_latest(
        self,
        timeout: float | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> FrameEnvelope | None:
        """Chờ bằng Condition; trả None khi hết hạn, đóng hoặc bị hủy."""
        deadline = None
        if timeout is not None:
            timeout = _finite(timeout, "timeout")
            deadline = time.monotonic() + max(0.0, timeout)

        subscription: object | None = None
        if cancel_token is not None:
            subscription = cancel_token._subscribe(self._wake_waiters)

        try:
            with self._condition:
                lifecycle = self._lifecycle
                while True:
                    if (
                        self._closed
                        or lifecycle != self._lifecycle
                        or (
                            cancel_token is not None
                            and cancel_token.is_cancelled()
                        )
                    ):
                        return None

                    if self._item is not None:
                        envelope = self._item
                        self._item = None
                        self._consumed += 1
                        return envelope

                    remaining = None
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return None
                    self._condition.wait(remaining)
        finally:
            if cancel_token is not None and subscription is not None:
                cancel_token._unsubscribe(subscription)

    def clear(self) -> int:
        """Bỏ frame đang chờ và cộng vào thống kê dropped."""
        with self._condition:
            dropped = int(self._item is not None)
            self._item = None
            self._dropped += dropped
            return dropped

    def close(self) -> None:
        """Đóng slot, giải phóng frame đang chờ và đánh thức mọi bộ chờ."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._lifecycle += 1
            self._dropped += int(self._item is not None)
            self._item = None
            self._condition.notify_all()

    def reopen(self) -> None:
        """Mở lại slot rỗng; giữ nguyên thống kê tích lũy."""
        with self._condition:
            self._closed = False
            self._condition.notify_all()

    def get_stats(self) -> dict[str, Any]:
        """Trả snapshot thống kê nhất quán dưới cùng một lock."""
        with self._condition:
            return {
                "pushed": self._pushed,
                "consumed": self._consumed,
                "dropped": self._dropped,
                "pending": int(self._item is not None),
                "drop_ratio": (
                    self._dropped / self._pushed if self._pushed else 0.0
                ),
            }


class VideoPlaybackClock:
    """Đồng hồ playback thread-safe, dùng monotonic để nội suy playhead.

    seek() giữ trạng thái playing hiện tại. update() bắt đầu thế hệ mới
    sẽ mặc định tiếp tục phát nếu caller không truyền playing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.playing: bool = False
        self.playhead_sec: float = 0.0
        self.playback_rate: float = 1.0
        self.generation: int = 0
        self.mode: str = "realtime"
        self.buffer_ahead_sec: float = 2.0
        self._reference_wall_time = time.monotonic()

    def update(
        self,
        timestamp_sec: float,
        *,
        playing: bool | None = None,
        playback_rate: float | None = None,
        generation: int | None = None,
        seeked: bool = False,
        mode: str | None = None,
        buffer_ahead_sec: float | None = None,
    ) -> int:
        """Áp dụng báo cáo trình duyệt; bỏ qua báo cáo từ thế hệ cũ."""
        timestamp = max(0.0, _finite(timestamp_sec, "timestamp_sec"))
        rate = (
            None
            if playback_rate is None
            else _nonnegative(playback_rate, "playback_rate")
        )
        selected_mode = None if mode is None else _validate_mode(mode)
        ahead = (
            None
            if buffer_ahead_sec is None
            else _nonnegative(buffer_ahead_sec, "buffer_ahead_sec")
        )

        with self._lock:
            if generation is not None and generation < self.generation:
                return self.generation

            previous_generation = self.generation
            if seeked:
                self.generation = max(
                    self.generation + 1,
                    generation if generation is not None else 0,
                )
            elif generation is not None:
                self.generation = generation

            if playing is not None:
                self.playing = playing
            elif self.generation != previous_generation:
                self.playing = True

            if rate is not None:
                self.playback_rate = rate
            if selected_mode is not None:
                self.mode = selected_mode
            if ahead is not None:
                self.buffer_ahead_sec = ahead

            self.playhead_sec = timestamp
            self._reference_wall_time = time.monotonic()
            return self.generation

    def seek(self, timestamp_sec: float) -> int:
        """Tăng thế hệ và đặt playhead, giữ trạng thái phát/tạm dừng."""
        timestamp = max(0.0, _finite(timestamp_sec, "timestamp_sec"))
        with self._lock:
            self.generation += 1
            self.playhead_sec = timestamp
            self._reference_wall_time = time.monotonic()
            return self.generation

    def _estimate_locked(self, now: float) -> float:
        """Nội suy khi caller đang giữ lock."""
        elapsed = max(0.0, now - self._reference_wall_time)
        return self.playhead_sec + (
            elapsed * self.playback_rate if self.playing else 0.0
        )

    def estimated_playhead_sec(self) -> float:
        """Trả playhead nội suy tại thời điểm gọi."""
        with self._lock:
            return self._estimate_locked(time.monotonic())

    def snapshot(self) -> dict[str, Any]:
        """Trả trạng thái cùng playhead tham chiếu và playhead nội suy."""
        with self._lock:
            return {
                "playing": self.playing,
                "playhead_sec": self.playhead_sec,
                "estimated_playhead_sec": self._estimate_locked(time.monotonic()),
                "playback_rate": self.playback_rate,
                "generation": self.generation,
                "mode": self.mode,
                "buffer_ahead_sec": self.buffer_ahead_sec,
            }


def decide_video_frame_action(
    mode: str,
    frame_timestamp_sec: float,
    playhead_sec: float,
    max_lag_drop_sec: float = 0.6,
    buffer_ahead_sec: float = 2.0,
    playing: bool = True,
) -> str:
    """Quyết định process, skip_behind hoặc wait_ahead.

    Khi tạm dừng, playhead đứng yên nhưng vẫn cho phép xử lý phần buffer
    theo cùng giới hạn thời gian.
    """
    mode = _validate_mode(mode)
    timestamp = _finite(frame_timestamp_sec, "frame_timestamp_sec")
    playhead = _finite(playhead_sec, "playhead_sec")
    lag = _nonnegative(max_lag_drop_sec, "max_lag_drop_sec")
    ahead = _nonnegative(buffer_ahead_sec, "buffer_ahead_sec")

    if mode == "realtime":
        behind_limit = lag
        ahead_limit = max(ahead, 0.5)
    else:
        behind_limit = lag * 1.5
        ahead_limit = ahead

    if timestamp < playhead - behind_limit:
        return "skip_behind"
    if timestamp > playhead + ahead_limit:
        return "wait_ahead"
    return "process"


class OverlayTimelineStore:
    """Lưu keyframe giới hạn dung lượng với sequence tăng liên tục."""

    def __init__(self, max_keyframes: int = 1200) -> None:
        if max_keyframes <= 0:
            raise ValueError("max_keyframes phải > 0")
        self._lock = threading.Lock()
        self._keyframes: deque[dict[str, Any]] = deque(maxlen=max_keyframes)
        self._seq = 0

    def append(
        self,
        generation: int,
        frame_id: int,
        source_timestamp: float,
        detections: list[dict],
    ) -> None:
        """Lưu bản sao detections; tự giải phóng keyframe cũ khi đầy."""
        timestamp = _finite(source_timestamp, "source_timestamp")
        owned_detections = deepcopy(detections)
        with self._lock:
            self._seq += 1
            self._keyframes.append(
                {
                    "seq": self._seq,
                    "generation": generation,
                    "frame_id": frame_id,
                    "source_timestamp": timestamp,
                    "detections": owned_detections,
                }
            )

    def clear(self, generation: int | None = None) -> None:
        """Xóa toàn bộ hoặc thế hệ chỉ định; không đặt lại sequence."""
        with self._lock:
            if generation is None:
                self._keyframes.clear()
            else:
                self._keyframes = deque(
                    (
                        item
                        for item in self._keyframes
                        if item["generation"] != generation
                    ),
                    maxlen=self._keyframes.maxlen,
                )

    def get_since(self, last_seq: int) -> tuple[int, list[dict]]:
        """Trả sequence hiện tại và bản sao các keyframe mới còn lưu."""
        with self._lock:
            items = [
                deepcopy(item)
                for item in self._keyframes
                if item["seq"] > last_seq
            ]
            return self._seq, items

    def lookup_near(
        self,
        timestamp_sec: float,
        generation: int,
        max_age_sec: float = 1.0,
    ) -> list[dict] | None:
        """Lấy detections gần nhất; ưu tiên quá khứ nếu cách đều.

        Trả [] cho keyframe không có detection, None nếu không tìm thấy.
        Kết quả là bản sao độc lập với dữ liệu trong store.
        """
        timestamp = _finite(timestamp_sec, "timestamp_sec")
        max_age = _nonnegative(max_age_sec, "max_age_sec")

        with self._lock:
            best: dict[str, Any] | None = None
            best_key = (math.inf, True)
            for item in reversed(self._keyframes):
                if item["generation"] != generation:
                    continue
                source_timestamp = item["source_timestamp"]
                difference = abs(source_timestamp - timestamp)
                key = (difference, source_timestamp > timestamp)
                if difference <= max_age and key < best_key:
                    best = item
                    best_key = key
            return None if best is None else deepcopy(best["detections"])
