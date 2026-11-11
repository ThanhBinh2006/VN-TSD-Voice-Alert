"""Worker TTS bất đồng bộ với queue giới hạn, dedup và kiểm tra độ trễ.

Timeout giới hạn thời gian worker chờ synthesizer. Python không thể cưỡng
bức dừng một thread đang chạy; tối đa hai tác vụ synth được phép tồn tại.
Khi cả hai bị treo, cảnh báo mới bị loại thay vì tích lũy trong executor.

Các callback cần chạy nhanh và không thực hiện I/O chặn.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ts_alert.postprocess import tts as _tts_module
from ts_alert.types import RuntimeAlert

__all__ = ["AsyncTTSWorker"]

_LOGGER = logging.getLogger(__name__)

Synthesizer = Callable[[str, Path, dict[str, Any]], Path]
AlertCallback = Callable[[RuntimeAlert], None]
DedupKey = tuple[int | None, str]


@dataclass(eq=False)
class _PendingAlert:
    """Trạng thái nội bộ, độc lập với dictionary trả về cho caller."""

    alert: RuntimeAlert
    key: DedupKey
    track_id: int | None
    created: float
    deadline: float
    source_timestamp: float
    epoch: int
    generation: int


@dataclass(eq=False)
class _SynthesisTask:
    """Theo dõi tác vụ để dọn file kể cả khi hoàn thành sau timeout."""

    target: Path
    abandoned: threading.Event = field(default_factory=threading.Event)


class AsyncTTSWorker:
    """Tổng hợp âm thanh tuần tự trên một worker daemon.

    ``created_wall_time`` và ``deadline_wall_time`` dùng cùng hệ đồng hồ
    ``clock_fn``; mặc định đây là đồng hồ monotonic, không phải Unix time.

    ``submitted`` tính mọi lần gọi submit, bao gồm dedup và đầu vào lỗi.
    ``synthesized`` tính các lần synthesizer trả về thành công trong hạn chờ,
    kể cả kết quả sau đó bị loại do cảnh báo lỗi thời.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        out_dir: str | Path,
        synthesizer: Synthesizer | None = None,
        max_queue_size: int = 8,
        cooldown_sec: float = 4.0,
        timeout_sec: float = 3.0,
        max_alert_age_sec: float = 2.5,
        clock_fn: Callable[[], float] | None = None,
        media_clock_fn: Callable[[], float] | None = None,
        max_media_lag_sec: float | None = None,
        on_ready: AlertCallback | None = None,
        on_drop: AlertCallback | None = None,
    ) -> None:
        """Khởi tạo queue, executor hai slot và tự khởi động worker."""
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size < 1
        ):
            raise ValueError("max_queue_size must be a positive integer")

        for name, value in (
            ("cooldown_sec", cooldown_sec),
            ("max_alert_age_sec", max_alert_age_sec),
            ("max_media_lag_sec", max_media_lag_sec),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be finite and positive")

        self.cfg = cfg
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.max_queue_size = max_queue_size
        self.cooldown_sec = float(cooldown_sec)
        self.timeout_sec = float(timeout_sec)
        self.max_alert_age_sec = float(max_alert_age_sec)
        self.max_media_lag_sec = max_media_lag_sec

        self._synthesizer = synthesizer or (
            lambda text, target, cfg: _tts_module.synthesize(text, target, cfg)
        )
        self._clock = clock_fn or time.monotonic
        self._media_clock = media_clock_fn
        self._on_ready = on_ready
        self._on_drop = on_drop

        self._condition = threading.Condition()
        self._queue: deque[_PendingAlert] = deque()
        self._closed = False
        self._epoch = 0
        self._generation = 0

        self._reserved_keys: dict[DedupKey, _PendingAlert] = {}
        self._reserved_tracks: dict[int, _PendingAlert] = {}
        self._last_ready: dict[DedupKey, float] = {}
        self._alerted_track_ids: set[int] = set()

        self.ready_history: deque[RuntimeAlert] = deque(maxlen=100)
        self._ready_count = 0
        self._stats: dict[str, int] = {
            "submitted": 0,
            "synthesized": 0,
            "deduped": 0,
            "dropped_queue_full": 0,
            "dropped_pre_stale": 0,
            "dropped_post_stale": 0,
            "dropped_drained": 0,
            "timeouts": 0,
            "errors": 0,
            "late_alerts": 0,
        }

        # Không submit thêm khi hai tác vụ trước vẫn chưa kết thúc.
        self._slots = threading.BoundedSemaphore(2)
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="tts-synth",
        )
        self._worker = threading.Thread(
            target=self._run,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        spoken_text: str,
        class_id: int | None = None,
        label: str = "",
        position: str = "front",
        track_id: int | None = None,
        frame_id: int = 0,
        source_timestamp: float = 0.0,
        alert_id: str | None = None,
    ) -> RuntimeAlert:
        """Đưa cảnh báo vào queue mà không chờ tổng hợp âm thanh."""
        now = self._clock()
        text = spoken_text.strip()
        alert: RuntimeAlert = {
            "alert_id": alert_id if alert_id is not None else uuid.uuid4().hex,
            "track_id": track_id,
            "frame_id": frame_id,
            "source_timestamp": source_timestamp,
            "label": label,
            "position": position,
            "spoken_text": text,
            "audio_path": "",
            "audio_url": "",
            "created_wall_time": now,
            "deadline_wall_time": now + self.max_alert_age_sec,
            "latency_ms": 0.0,
            "status": "queued",
        }
        # RuntimeAlert.class_id là int; bỏ trường khi không có class_id.
        if class_id is not None:
            alert["class_id"] = class_id

        key = (class_id, position)
        dropped: list[RuntimeAlert] = []

        with self._condition:
            self._stats["submitted"] += 1

            if self._closed or not text:
                alert["status"] = "error"
                self._stats["errors"] += 1
                dropped.append(alert)
            elif (
                (
                    track_id is not None
                    and (
                        track_id in self._alerted_track_ids
                        or track_id in self._reserved_tracks
                    )
                )
                or key in self._reserved_keys
                or (
                    key in self._last_ready
                    and now - self._last_ready[key] < self.cooldown_sec
                )
            ):
                alert["status"] = "deduped"
                self._stats["deduped"] += 1
            else:
                if len(self._queue) >= self.max_queue_size:
                    victim = self._queue.popleft()
                    self._release_locked(victim)
                    victim.alert["status"] = "dropped_stale"
                    victim.alert["latency_ms"] = max(
                        0.0, (now - victim.created) * 1000.0
                    )
                    self._stats["dropped_queue_full"] += 1
                    dropped.append(victim.alert)

                pending = _PendingAlert(
                    alert=alert,
                    key=key,
                    track_id=track_id,
                    created=now,
                    deadline=now + self.max_alert_age_sec,
                    source_timestamp=source_timestamp,
                    epoch=self._epoch,
                    generation=self._generation,
                )
                self._reserved_keys[key] = pending
                if track_id is not None:
                    self._reserved_tracks[track_id] = pending
                self._queue.append(pending)
                self._condition.notify()

        for item in dropped:
            self._notify(self._on_drop, item)
        return alert

    def drain_pending(self, reason: str = "dropped_stale") -> int:
        """Dọn queue và vô hiệu hóa kết quả synth đang chạy trước lần dọn."""
        with self._condition:
            self._generation += 1
            drained = self._drain_locked(reason)
            self._condition.notify_all()

        for alert in drained:
            self._notify(self._on_drop, alert)
        return len(drained)

    def reset_dedup(self) -> None:
        """Xóa lịch sử và reservation; kết quả cũ không tái lập dedup."""
        with self._condition:
            self._epoch += 1
            self._reserved_keys.clear()
            self._reserved_tracks.clear()
            self._last_ready.clear()
            self._alerted_track_ids.clear()

    def get_audio_path(self, alert_id: str) -> Path | None:
        """Tra đường dẫn audio còn tồn tại trong 100 cảnh báo ready gần nhất."""
        with self._condition:
            raw_path = next(
                (
                    alert.get("audio_path", "")
                    for alert in reversed(self.ready_history)
                    if alert.get("alert_id") == alert_id
                    and alert.get("status") == "ready"
                ),
                "",
            )

        if not raw_path:
            return None
        path = Path(raw_path)
        return path if path.is_file() else None

    def poll_ready(
        self,
        since_index: int = 0,
    ) -> tuple[int, list[RuntimeAlert]]:
        """Trả cursor tuyệt đối mới và các kết quả còn giữ sau cursor cũ.

        Khi caller chậm hơn giới hạn 100 item, chỉ trả phần lịch sử còn lại.
        Dictionary trả về là bản sao để caller không sửa lịch sử nội bộ.
        """
        if since_index < 0:
            raise ValueError("since_index must be non-negative")

        with self._condition:
            end = self._ready_count
            start = end - len(self.ready_history)
            offset = min(len(self.ready_history), max(0, since_index - start))
            history = list(self.ready_history)
            return end, [alert.copy() for alert in history[offset:]]

    def get_stats(self) -> dict[str, Any]:
        """Lấy snapshot thống kê; late_ratio dùng tổng submitted trừ dedup."""
        with self._condition:
            stats: dict[str, Any] = dict(self._stats)
            denominator = self._stats["submitted"] - self._stats["deduped"]
            stats.update(
                pending=len(self._queue),
                late_ratio=(
                    self._stats["late_alerts"] / denominator
                    if denominator
                    else 0.0
                ),
                late_ratio_denominator=denominator,
                late_ratio_basis="submitted_minus_deduped",
            )
            return stats

    def shutdown(
        self,
        wait: bool = False,
        cancel_pending: bool = True,
    ) -> None:
        """Đóng worker; không chờ vô hạn các thread synth hoặc callback.

        wait=True chỉ join worker trong timeout_sec + 0.25 giây.
        Executor luôn shutdown(wait=False). Thread synth đã chạy không thể
        bị cưỡng bức dừng và có thể vẫn giữ tiến trình Python khi thoát.
        """
        with self._condition:
            self._closed = True
            self._generation += 1
            drained = (
                self._drain_locked("dropped_stale") if cancel_pending else []
            )
            self._condition.notify_all()

        self._executor.shutdown(wait=False, cancel_futures=True)

        for alert in drained:
            self._notify(self._on_drop, alert)

        if wait and threading.current_thread() is not self._worker:
            self._worker.join(timeout=self.timeout_sec + 0.25)

    def _release_locked(self, pending: _PendingAlert) -> None:
        """Hoàn reservation chỉ khi nó vẫn thuộc đúng cảnh báo này."""
        if self._reserved_keys.get(pending.key) is pending:
            del self._reserved_keys[pending.key]

        track_id = pending.track_id
        if (
            track_id is not None
            and self._reserved_tracks.get(track_id) is pending
        ):
            del self._reserved_tracks[track_id]

    def _drain_locked(self, reason: str) -> list[RuntimeAlert]:
        """Dọn queue khi caller đang giữ Condition."""
        drained: list[RuntimeAlert] = []
        while self._queue:
            pending = self._queue.popleft()
            self._release_locked(pending)
            pending.alert["status"] = reason
            drained.append(pending.alert)

        self._stats["dropped_drained"] += len(drained)
        return drained

    def _invalidated_locked(self, pending: _PendingAlert) -> bool:
        """Kiểm tra shutdown hoặc lần drain sau khi cảnh báo được tạo."""
        return self._closed or pending.generation != self._generation

    def _media_stale(self, pending: _PendingAlert) -> bool:
        """Kiểm tra độ trễ theo timestamp của video hiện tại."""
        return (
            self._media_clock is not None
            and self.max_media_lag_sec is not None
            and self._media_clock() - pending.source_timestamp
            > self.max_media_lag_sec
        )

    def _drop(
        self,
        pending: _PendingAlert,
        status: str,
        counter: str,
        *,
        late: bool = False,
    ) -> None:
        """Kết thúc cảnh báo thất bại và hoàn reservation."""
        with self._condition:
            self._release_locked(pending)
            pending.alert["status"] = status
            self._stats[counter] += 1
            if late:
                self._stats["late_alerts"] += 1

        self._notify(self._on_drop, pending.alert)

    def _run(self) -> None:
        """Lấy từng cảnh báo; lỗi của một cảnh báo không dừng worker."""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or bool(self._queue)
                )
                if not self._queue:
                    return
                pending = self._queue.popleft()

            try:
                self._process(pending)
            except Exception:
                _LOGGER.exception("Unexpected TTS worker failure")
                self._drop(pending, "error", "errors")

    def _process(self, pending: _PendingAlert) -> None:
        """Kiểm tra stale, synth có timeout rồi công bố kết quả."""
        now = self._clock()
        with self._condition:
            invalidated = self._invalidated_locked(pending)

        if (
            invalidated
            or now - pending.created > self.max_alert_age_sec
            or self._media_stale(pending)
        ):
            self._drop(
                pending,
                "dropped_stale",
                "dropped_pre_stale",
                late=True,
            )
            return

        if not self._slots.acquire(blocking=False):
            self._drop(pending, "timeout", "timeouts", late=True)
            return

        engine = (self.cfg.get("postprocess") or {}).get("tts") or {}
        extension = ".wav" if engine.get("engine") == "pyttsx3" else ".mp3"
        task = _SynthesisTask(
            target=self.out_dir / f"tts_{uuid.uuid4().hex}{extension}"
        )

        try:
            # Đồng bộ submit với shutdown để tránh submit vào executor đã đóng.
            with self._condition:
                invalidated = self._invalidated_locked(pending)
                if invalidated:
                    future = None
                else:
                    future = self._executor.submit(
                        self._synthesizer,
                        pending.alert["spoken_text"],
                        task.target,
                        self.cfg,
                    )
        except Exception:
            self._slots.release()
            raise

        if future is None:
            self._slots.release()
            self._drop(
                pending,
                "dropped_stale",
                "dropped_post_stale",
                late=True,
            )
            return

        future.add_done_callback(
            lambda completed: self._synthesis_finished(task, completed)
        )

        try:
            # Dùng monotonic thật cho timeout, độc lập clock giả trong test.
            timeout_deadline = time.monotonic() + self.timeout_sec

            while True:
                with self._condition:
                    invalidated = self._invalidated_locked(pending)

                if invalidated:
                    self._drop(
                        pending,
                        "dropped_stale",
                        "dropped_post_stale",
                        late=True,
                    )
                    return

                remaining = timeout_deadline - time.monotonic()
                if remaining <= 0:
                    self._drop(pending, "timeout", "timeouts", late=True)
                    return

                try:
                    result = future.result(timeout=min(remaining, 0.05))
                    break
                except TimeoutError:
                    # TimeoutError do chính synthesizer ném cũng là timeout.
                    if future.done():
                        self._drop(
                            pending, "timeout", "timeouts", late=True
                        )
                        return
                except Exception:
                    _LOGGER.exception("TTS synthesis failed")
                    self._drop(pending, "error", "errors")
                    return

            with self._condition:
                self._stats["synthesized"] += 1

            audio_path = Path(result).expanduser().resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Synthesizer did not produce an audio file: {audio_path}"
                )

            media_stale = self._media_stale(pending)
            after = self._clock()
            pending.alert["latency_ms"] = max(
                0.0, (after - pending.created) * 1000.0
            )

            with self._condition:
                stale = (
                    self._invalidated_locked(pending)
                    or after > pending.deadline
                    or after - pending.created > self.max_alert_age_sec
                    or media_stale
                )
                if not stale:
                    self._release_locked(pending)
                    pending.alert["audio_path"] = str(audio_path)
                    pending.alert["status"] = "ready"

                    if pending.epoch == self._epoch:
                        self._last_ready[pending.key] = after
                        if pending.track_id is not None:
                            self._alerted_track_ids.add(pending.track_id)

                    self.ready_history.append(pending.alert.copy())
                    self._ready_count += 1

            if stale:
                self._abandon(task, future)
                self._drop(
                    pending,
                    "dropped_stale",
                    "dropped_post_stale",
                    late=True,
                )
                return

            self._notify(self._on_ready, pending.alert)
        finally:
            if pending.alert.get("status") != "ready":
                self._abandon(task, future)

    def _synthesis_finished(
        self,
        task: _SynthesisTask,
        future: Future[Path],
    ) -> None:
        """Trả slot và xóa output nếu tác vụ đã bị bỏ."""
        try:
            if task.abandoned.is_set():
                self._cleanup_task(task, future)
        finally:
            self._slots.release()

    def _abandon(
        self,
        task: _SynthesisTask,
        future: Future[Path],
    ) -> None:
        """Đánh dấu bỏ; xử lý cả race giữa timeout và future hoàn thành."""
        task.abandoned.set()
        future.cancel()
        if future.done():
            self._cleanup_task(task, future)

    def _cleanup_task(
        self,
        task: _SynthesisTask,
        future: Future[Path],
    ) -> None:
        """Xóa target và đường dẫn trả về của tác vụ đã kết thúc."""
        paths = {task.target}
        if not future.cancelled():
            try:
                paths.add(Path(future.result()))
            except Exception:
                pass

        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                _LOGGER.warning(
                    "Cannot remove discarded TTS file: %s",
                    path,
                    exc_info=True,
                )

    @staticmethod
    def _notify(
        callback: AlertCallback | None,
        alert: RuntimeAlert,
    ) -> None:
        """Cô lập lỗi callback và tránh cho callback sửa trạng thái nội bộ."""
        if callback is None:
            return
        try:
            callback(alert.copy())
        except Exception:
            _LOGGER.exception("TTS callback failed")
