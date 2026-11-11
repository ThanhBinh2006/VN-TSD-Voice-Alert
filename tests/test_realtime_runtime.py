"""Regression tests for CPU realtime scheduling, tracking, TTS, and HTTP APIs."""

from __future__ import annotations

import functools
import importlib
import inspect
import io
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import yaml

from app.main import create_app
from app.store_memory import InMemoryStore
import ts_alert.runtime.realtime_service as realtime_service
from ts_alert.config.resolve import resolve_config
from ts_alert.pipelines.faster_rcnn.infer import FasterRCNNDetector
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.postprocess.tts_worker import AsyncTTSWorker
from ts_alert.runtime.cpu_opt import (
    CPU_PRESETS,
    apply_faster_rcnn_cpu_tuning,
    apply_yolo_cpu_tuning,
    prepare_runtime_detector,
    probe_runtime_backends,
    restore_cpu_tuning,
)
from ts_alert.runtime.scheduler import (
    CancellationToken,
    CancelledError,
    FrameEnvelope,
    FrameRingBuffer,
    LatestFrameSlot,
    OverlayTimelineStore,
    VideoPlaybackClock,
    decide_video_frame_action,
)
from ts_alert.runtime.tracker import (
    TrackerConfig,
    TrafficSignTracker,
    compute_flow_gray,
)

WAIT_SEC = 4.0
FPS = 30.0


def eventually(
    predicate: Callable[[], Any],
    *,
    timeout: float = WAIT_SEC,
    message: str = "Condition did not become true",
) -> Any:
    """Poll a condition with a bounded deadline and return its truthy result."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(message)
        threading.Event().wait(min(0.01, remaining))


def join_thread(thread: threading.Thread, timeout: float = WAIT_SEC) -> None:
    """Require a test-owned thread to terminate."""
    thread.join(timeout)
    assert not thread.is_alive(), f"Thread {thread.name!r} did not terminate"


def envelope(frame_id: int, generation: int = 0) -> FrameEnvelope:
    """Create a small scheduler input without model dependencies."""
    return FrameEnvelope(
        frame_id=frame_id,
        source_timestamp=frame_id / FPS,
        captured_wall_time=time.time(),
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
        width=32,
        height=24,
        generation=generation,
    )


def detection(
    bbox: list[float] | None = None,
    *,
    class_id: int = 1,
    confidence: float = 0.99,
) -> dict[str, Any]:
    """Return a detection using the shared data contract."""
    return {
        "bbox": list(bbox or [55.0, 45.0, 105.0, 95.0]),
        "class_id": class_id,
        "label": f"sign-{class_id}",
        "confidence": confidence,
        "position": "front",
    }


def center_x(item: dict[str, Any]) -> float:
    """Return the horizontal center of a detection."""
    x1, _, x2, _ = item["bbox"]
    return float(x1 + x2) / 2.0


def make_scene(shift: float = 0.0) -> np.ndarray:
    """Generate a repeatable textured road scene translated horizontally."""
    rng = np.random.default_rng(731)
    gray = rng.integers(35, 95, size=(160, 240), dtype=np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for y in range(100, 160, 12):
        cv2.line(frame, (0, y), (239, y - 20), (160, 160, 160), 1)

    cv2.circle(frame, (80, 70), 25, (0, 0, 230), -1)
    cv2.circle(frame, (80, 70), 18, (245, 245, 245), -1)
    cv2.putText(
        frame,
        "30",
        (65, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (5, 5, 5),
        2,
        cv2.LINE_AA,
    )

    for x, y in ((68, 58), (90, 60), (72, 82), (91, 81)):
        cv2.rectangle(frame, (x, y), (x + 3, y + 3), (0, 0, 0), -1)

    transform = np.float32([[1, 0, shift], [0, 1, 0]])
    return cv2.warpAffine(
        frame,
        transform,
        (240, 160),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def tracker_config(**overrides: Any) -> TrackerConfig:
    """Build a deterministic tracker configuration for short tests."""
    values: dict[str, Any] = {
        "ttl_sec": 0.4,
        "reid_ttl_sec": 1.5,
        "max_missed_detections": 3,
        "max_extrapolate_sec": 0.65,
        "max_staleness_sec": 0.85,
        "min_hits_to_confirm": 1,
        "single_hit_conf_threshold": 0.6,
        "use_optical_flow": True,
        "flow_max_width": 320,
    }
    values.update(overrides)
    return TrackerConfig(**values)


def start_track(
    tracker: TrafficSignTracker,
    *,
    timestamp: float = 0.0,
    frame_id: int = 0,
) -> dict[str, Any]:
    """Create and return one confirmed track."""
    frame = make_scene()
    tracker.step_frame(frame, frame_id=frame_id, timestamp_sec=timestamp)
    result = tracker.update_from_detection(
        [detection()],
        detection_frame_id=frame_id,
        detection_timestamp=timestamp,
        current_frame_id=frame_id,
        current_timestamp=timestamp,
        current_frame=frame,
        source_gray=compute_flow_gray(frame),
        frame_width=240,
        frame_height=160,
    )
    assert len(result) == 1
    assert result[0]["confirmed"]
    return result[0]


def test_latest_slot_overwrites_pending_without_backlog() -> None:
    """A busy consumer retains exactly the newest pending frame."""
    slot = LatestFrameSlot()
    try:
        assert slot.push(envelope(0)) == 0
        assert slot.pop_latest(timeout=0.1).frame_id == 0

        for frame_id in range(1, 11):
            assert slot.push(envelope(frame_id)) == int(frame_id > 1)

        stats = slot.get_stats()
        assert stats["pushed"] == 11
        assert stats["consumed"] == 1
        assert stats["dropped"] == 9
        assert stats["pending"] == 1
        assert stats["drop_ratio"] == pytest.approx(9 / 11)

        assert slot.pop_latest(timeout=0.1).frame_id == 10
        assert slot.pop_latest(timeout=0.01) is None
        assert slot.get_stats()["pending"] == 0

        slot.push(envelope(11))
        assert slot.clear() == 1
        assert slot.clear() == 0
    finally:
        slot.close()


@pytest.mark.parametrize("wake_action", ["close", "cancel"])
def test_latest_slot_wakes_blocked_consumer(wake_action: str) -> None:
    """Closing or cancelling interrupts an otherwise unbounded wait."""
    slot = LatestFrameSlot()
    token = CancellationToken()
    entered = threading.Event()
    finished = threading.Event()
    outcomes: list[Any] = []

    def consume() -> None:
        entered.set()
        try:
            outcomes.append(slot.pop_latest(timeout=None, cancel_token=token))
        except CancelledError:
            outcomes.append("cancelled")
        finally:
            finished.set()

    thread = threading.Thread(target=consume, name="slot-consumer", daemon=True)
    thread.start()
    try:
        assert entered.wait(1.0)
        if wake_action == "close":
            slot.close()
        else:
            token.cancel("test cancellation")

        assert finished.wait(0.75), "Blocked consumer was not awakened promptly"
        join_thread(thread)
        assert outcomes in ([None], ["cancelled"])

        if wake_action == "cancel":
            assert token.is_cancelled()
            assert token.reason == "test cancellation"
            assert token.wait(0)
            with pytest.raises(CancelledError):
                token.raise_if_cancelled()
    finally:
        token.cancel("cleanup")
        slot.close()
        join_thread(thread)


def test_latest_slot_reopens() -> None:
    """A closed slot can be explicitly reopened for a new session."""
    slot = LatestFrameSlot()
    slot.close()
    assert slot.pop_latest(timeout=0.01) is None
    slot.reopen()
    try:
        slot.push(envelope(9))
        assert slot.pop_latest(timeout=0.1).frame_id == 9
    finally:
        slot.close()


def test_frame_ring_capacity_generation_and_copy_isolation() -> None:
    """Ring lookups respect capacity, generations, age, and ownership."""
    ring = FrameRingBuffer(maxlen=2)
    original = np.full((8, 8), 7, dtype=np.uint8)
    ring.put(1, 1, 1.0, original, 0.5)
    original[:] = 99

    first = ring.find_closest(1.0, generation=1)
    assert first is not None
    np.testing.assert_array_equal(first[0], np.full((8, 8), 7))
    assert first[1] == 0.5

    try:
        first[0][:] = 123
    except ValueError:
        pass

    again = ring.find_closest(1.0, generation=1)
    assert again is not None
    assert np.all(again[0] == 7)

    ring.put(1, 2, 2.0, np.full((8, 8), 2, np.uint8), 1.0)
    ring.put(2, 3, 2.1, np.full((8, 8), 3, np.uint8), 1.0)

    assert ring.find_closest(1.0, generation=1, max_diff_sec=0.01) is None
    assert ring.find_closest(2.1, generation=1, max_diff_sec=0.05) is None
    assert ring.find_closest(2.0, generation=2, max_diff_sec=0.05) is None

    closest = ring.find_closest(2.08, generation=2, max_diff_sec=0.05)
    assert closest is not None
    assert np.all(closest[0] == 3)

    ring.clear(generation=1)
    assert ring.find_closest(2.0, generation=1) is None
    assert ring.find_closest(2.1, generation=2) is not None
    ring.clear()
    assert ring.find_closest(2.1, generation=2) is None


def test_playback_clock_interpolation_pause_seek_and_old_generation() -> None:
    """Playback advances only while playing and rejects stale updates."""
    clock = VideoPlaybackClock()
    clock.update(10.0, playing=True, playback_rate=2.0)
    started = time.monotonic()
    threading.Event().wait(0.08)
    elapsed = time.monotonic() - started
    assert clock.estimated_playhead_sec() == pytest.approx(
        10.0 + elapsed * 2.0, abs=0.12
    )

    clock.update(12.0, playing=False)
    threading.Event().wait(0.05)
    assert clock.estimated_playhead_sec() == pytest.approx(12.0, abs=0.01)

    generation = clock.generation
    clock.seek(2.0)
    assert clock.generation > generation

    generation = clock.generation
    clock.update(
        3.0,
        playing=False,
        seeked=True,
        mode="buffered",
        buffer_ahead_sec=1.5,
    )
    assert clock.generation > generation
    current = clock.generation

    clock.update(
        999.0,
        playing=True,
        generation=current - 1,
        mode="realtime",
    )
    snapshot = clock.snapshot()
    assert snapshot["generation"] == current
    assert snapshot["mode"] == "buffered"
    assert clock.estimated_playhead_sec() == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("mode", ["realtime", "buffered"])
@pytest.mark.parametrize(
    ("frame_timestamp", "expected"),
    [
        (8.0, "skip_behind"),
        (10.0, "process"),
        (20.0, "wait_ahead"),
    ],
)
def test_video_frame_action_branches(
    mode: str, frame_timestamp: float, expected: str
) -> None:
    """Both playback modes enforce lag and lookahead limits."""
    assert decide_video_frame_action(
        mode,
        frame_timestamp,
        10.0,
        max_lag_drop_sec=0.6,
        buffer_ahead_sec=2.0,
        playing=True,
    ) == expected


def test_buffered_mode_processes_within_lookahead() -> None:
    """Buffered playback may analyze frames within its configured window."""
    assert decide_video_frame_action(
        "buffered", 11.0, 10.0, buffer_ahead_sec=2.0, playing=True
    ) == "process"


def test_overlay_lookup_ties_age_and_generation() -> None:
    """Overlay selection prefers the past keyframe on an exact tie."""
    store = OverlayTimelineStore(max_keyframes=8)
    past = detection([10.0, 20.0, 30.0, 40.0])
    future = detection([50.0, 20.0, 70.0, 40.0])
    store.append(1, 10, 1.0, [past])
    store.append(1, 30, 3.0, [future])
    store.append(2, 20, 2.0, [detection(class_id=2)])

    assert store.lookup_near(2.0, generation=1, max_age_sec=1.1) == [past]
    assert store.lookup_near(2.0, generation=2) == [detection(class_id=2)]
    assert store.lookup_near(20.0, generation=1, max_age_sec=0.1) is None
    assert store.lookup_near(2.0, generation=99) is None


def test_overlay_clear_preserves_monotonic_sequence() -> None:
    """Seeking clears obsolete generations without reusing sequence numbers."""
    store = OverlayTimelineStore(max_keyframes=3)
    store.append(1, 1, 0.0, [detection()])
    previous_seq, previous = store.get_since(0)
    assert previous

    store.clear(generation=1)
    assert store.lookup_near(0.0, generation=1) is None

    store.append(2, 2, 0.0, [detection(class_id=2)])
    current_seq, items = store.get_since(previous_seq)
    assert current_seq > previous_seq
    assert len(items) == 1

    store.clear()
    store.append(3, 3, 0.0, [])
    newest_seq, _ = store.get_since(current_seq)
    assert newest_seq > current_seq


def test_overlay_capacity_is_bounded() -> None:
    """Timeline retention is limited independently of its sequence counter."""
    store = OverlayTimelineStore(max_keyframes=3)
    for index in range(10):
        store.append(0, index, float(index), [detection()])
    seq, items = store.get_since(0)
    assert len(items) == 3
    assert seq >= 10
    assert store.lookup_near(0.0, generation=0, max_age_sec=0.1) is None


def test_optical_flow_follows_textured_translation() -> None:
    """Sparse flow reduces position error compared with a stationary box."""
    tracker = TrafficSignTracker(config=tracker_config())
    initial = start_track(tracker)
    track_id = initial["track_id"]
    static_center = center_x(initial)

    result: list[dict[str, Any]] = []
    for frame_id in range(1, 8):
        result = tracker.step_frame(
            make_scene(2.0 * frame_id),
            frame_id=frame_id,
            timestamp_sec=frame_id / FPS,
        )
        assert len(result) == 1
        assert result[0]["track_id"] == track_id

    expected_center = static_center + 14.0
    moving_error = abs(center_x(result[0]) - expected_center)
    static_error = abs(static_center - expected_center)
    assert moving_error < static_error * 0.5
    assert moving_error < 6.0

    x1, y1, x2, y2 = result[0]["bbox"]
    assert 0 <= x1 < x2 <= 240
    assert 0 <= y1 < y2 <= 160


def test_tracker_ttl_hides_lost_track_and_reidentifies() -> None:
    """Expired overlays disappear while Re-ID retains identity and alert state."""
    tracker = TrafficSignTracker(
        config=tracker_config(use_optical_flow=False)
    )
    original = start_track(tracker)
    tracker.mark_alert_emitted(original["track_id"])

    assert tracker.step_frame(
        None,
        frame_id=17,
        timestamp_sec=0.55,
        frame_width=240,
        frame_height=160,
    ) == []
    assert tracker.get_stats()["lost_count"] >= 1

    restored = tracker.update_from_detection(
        [detection()],
        detection_frame_id=18,
        detection_timestamp=0.6,
        current_frame_id=18,
        current_timestamp=0.6,
        frame_width=240,
        frame_height=160,
    )
    assert len(restored) == 1
    assert restored[0]["track_id"] == original["track_id"]
    assert restored[0]["state"] == "active"
    assert restored[0]["alert_emitted"]
    assert tracker.get_stats()["reidentified_tracks"] >= 1
    assert tracker.get_confirmed_alerts() == []


def test_tracker_expires_identity_after_reid_window() -> None:
    """A detection beyond the Re-ID window creates a fresh identity."""
    tracker = TrafficSignTracker(
        config=tracker_config(use_optical_flow=False)
    )
    original = start_track(tracker)
    assert tracker.step_frame(
        None,
        frame_id=60,
        timestamp_sec=2.0,
        frame_width=240,
        frame_height=160,
    ) == []

    restored = tracker.update_from_detection(
        [detection()],
        detection_frame_id=61,
        detection_timestamp=2.01,
        frame_width=240,
        frame_height=160,
    )
    assert restored
    assert restored[0]["track_id"] != original["track_id"]
    assert tracker.get_stats()["expired_tracks"] >= 1


def test_class_voting_resists_one_low_confidence_error() -> None:
    """One noisy class observation does not overturn accumulated evidence."""
    tracker = TrafficSignTracker(
        config=tracker_config(use_optical_flow=False)
    )
    original = start_track(tracker)

    for index in range(1, 5):
        result = tracker.update_from_detection(
            [detection()],
            detection_frame_id=index,
            detection_timestamp=index / FPS,
            frame_width=240,
            frame_height=160,
        )
        assert result[0]["track_id"] == original["track_id"]

    result = tracker.update_from_detection(
        [detection(class_id=2, confidence=0.35)],
        detection_frame_id=5,
        detection_timestamp=5 / FPS,
        frame_width=240,
        frame_height=160,
    )
    current = next(
        item for item in result if item["track_id"] == original["track_id"]
    )
    assert current["class_id"] == 1


def test_delayed_detection_compensates_coordinates_and_overlay_alignment() -> None:
    """Delayed source coordinates are moved into the current frame geometry."""
    tracker = TrafficSignTracker(config=tracker_config())
    source = make_scene(0.0)
    current = make_scene(12.0)
    tracker.step_frame(source, frame_id=0, timestamp_sec=0.0)
    tracker.step_frame(current, frame_id=8, timestamp_sec=0.25)

    result = tracker.update_from_detection(
        [detection()],
        detection_frame_id=0,
        detection_timestamp=0.0,
        current_frame_id=8,
        current_timestamp=0.25,
        current_frame=current,
        source_gray=compute_flow_gray(source),
        frame_width=240,
        frame_height=160,
        wall_lag_ms=250.0,
    )
    assert len(result) == 1
    actual = result[0]
    expected_center = 92.0
    assert abs(center_x(actual) - expected_center) < 6.0
    assert abs(center_x(actual) - expected_center) < abs(80.0 - expected_center)
    assert tracker.get_stats()["compensated_detections"] >= 1

    timeline = OverlayTimelineStore()
    timeline.append(0, 8, 0.25, result)
    aligned = timeline.lookup_near(0.25, generation=0, max_age_sec=0.01)
    assert aligned is not None
    assert aligned[0]["bbox"] == pytest.approx(actual["bbox"])


def test_stale_detection_does_not_overwrite_current_track() -> None:
    """Results older than the staleness limit are discarded before mutation."""
    tracker = TrafficSignTracker(
        config=tracker_config(use_optical_flow=False, ttl_sec=2.0)
    )
    start_track(tracker, timestamp=1.0, frame_id=30)
    before = tracker.step_frame(
        None,
        frame_id=31,
        timestamp_sec=1.01,
        frame_width=240,
        frame_height=160,
    )
    stats = tracker.get_stats()

    tracker.update_from_detection(
        [detection([150.0, 20.0, 200.0, 70.0])],
        detection_frame_id=32,
        detection_timestamp=0.0,
        current_frame_id=33,
        current_timestamp=1.02,
        frame_width=240,
        frame_height=160,
    )
    after = tracker.step_frame(
        None,
        frame_id=34,
        timestamp_sec=1.03,
        frame_width=240,
        frame_height=160,
    )
    assert tracker.get_stats()["dropped_stale_detections"] > stats[
        "dropped_stale_detections"
    ]
    assert len(after) == len(before) == 1
    assert after[0]["track_id"] == before[0]["track_id"]
    assert after[0]["bbox"] == pytest.approx(before[0]["bbox"])


@pytest.mark.parametrize("old_frame_id", [29, 30])
def test_out_of_order_detection_batch_is_discarded(old_frame_id: int) -> None:
    """Duplicate and older detector batches cannot overwrite newer state."""
    tracker = TrafficSignTracker(
        config=tracker_config(use_optical_flow=False)
    )
    original = start_track(tracker, timestamp=1.0, frame_id=30)
    before = tracker.get_stats()["dropped_out_of_order_batches"]

    tracker.update_from_detection(
        [detection([160.0, 20.0, 210.0, 70.0])],
        detection_frame_id=old_frame_id,
        detection_timestamp=1.0,
        current_frame_id=31,
        current_timestamp=1.02,
        frame_width=240,
        frame_height=160,
    )
    result = tracker.step_frame(
        None,
        frame_id=32,
        timestamp_sec=1.03,
        frame_width=240,
        frame_height=160,
    )
    assert tracker.get_stats()["dropped_out_of_order_batches"] > before
    assert len(result) == 1
    assert result[0]["track_id"] == original["track_id"]
    assert result[0]["bbox"] == pytest.approx(original["bbox"])


def test_seek_back_resets_tracker() -> None:
    """A backward jump greater than 100 ms clears existing tracks."""
    tracker = TrafficSignTracker(config=tracker_config())
    start_track(tracker, timestamp=2.0, frame_id=60)
    assert tracker.step_frame(
        make_scene(),
        frame_id=3,
        timestamp_sec=0.1,
    ) == []
    assert tracker.get_stats()["active_count"] == 0
    assert tracker.get_stats()["tentative_count"] == 0
    assert tracker.get_confirmed_alerts() == []


@dataclass
class ManualClock:
    """Provide a thread-safe adjustable clock for TTS age checks."""

    value: float = 100.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def set(self, value: float) -> None:
        with self.lock:
            self.value = value


@dataclass
class Synthesizer:
    """Write deterministic dummy audio, optionally waiting on a test gate."""

    gate: threading.Event | None = None
    delay_sec: float = 0.0
    entered: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    calls: list[str] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, text: str, path: Path, cfg: dict[str, Any]) -> Path:
        del cfg
        try:
            with self.lock:
                self.calls.append(text)
            self.entered.set()
            if self.gate is not None:
                if not self.gate.wait(WAIT_SEC):
                    raise TimeoutError("Test synthesizer gate was not released")
            if self.delay_sec:
                threading.Event().wait(self.delay_sec)
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            with self.lock:
                self.paths.append(path)
            return path
        finally:
            self.finished.set()


def ready_alerts(worker: AsyncTTSWorker) -> list[dict[str, Any]]:
    """Read currently published ready alerts."""
    _, alerts = worker.poll_ready(0)
    return alerts


def assert_late_ratio(worker: AsyncTTSWorker) -> None:
    """Validate the documented denominator, including the empty case."""
    stats = worker.get_stats()
    denominator = stats["submitted"] - stats["deduped"]
    expected = stats["late_alerts"] / denominator if denominator else 0.0
    assert stats["late_ratio"] == pytest.approx(expected)


def test_tts_submit_is_nonblocking(tmp_path: Path) -> None:
    """Submitting does not wait for a 200 ms synthesizer."""
    synth = Synthesizer(delay_sec=0.2)
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        timeout_sec=1.0,
        max_alert_age_sec=5.0,
    )
    try:
        started = time.perf_counter()
        alert = worker.submit("Có biển báo phía trước", class_id=1, track_id=1)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.020
        assert alert["status"] == "queued"

        alerts = eventually(lambda: ready_alerts(worker))
        assert alerts[0]["status"] == "ready"
        assert worker.get_audio_path(alert["alert_id"]).is_file()
        assert len(synth.calls) == 1
        assert_late_ratio(worker)
    finally:
        worker.shutdown(wait=True, cancel_pending=True)


def test_tts_cooldown_and_ready_track_deduplication(tmp_path: Path) -> None:
    """Cooldown and emitted track identities independently suppress repeats."""
    clock = ManualClock()
    synth = Synthesizer()
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        clock_fn=clock,
        cooldown_sec=4.0,
        max_alert_age_sec=10.0,
    )
    try:
        worker.submit("first", class_id=1, position="front", track_id=10)
        eventually(lambda: len(ready_alerts(worker)) == 1)

        repeated_class = worker.submit(
            "same class", class_id=1, position="front", track_id=11
        )
        assert repeated_class["status"] == "deduped"

        clock.set(110.0)
        repeated_track = worker.submit(
            "same track", class_id=2, position="left", track_id=10
        )
        assert repeated_track["status"] == "deduped"

        worker.submit("new track", class_id=1, position="front", track_id=12)
        eventually(lambda: len(ready_alerts(worker)) == 2)
        assert len(synth.calls) == 2
        assert worker.get_stats()["deduped"] == 2
        assert_late_ratio(worker)
    finally:
        worker.shutdown(wait=True, cancel_pending=True)


def test_tts_queue_drops_oldest_pending_item(tmp_path: Path) -> None:
    """Queue pressure retains recent alerts and bounds pending work."""
    gate = threading.Event()
    synth = Synthesizer(gate=gate)
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        max_queue_size=2,
        cooldown_sec=0.0,
        timeout_sec=3.0,
        max_alert_age_sec=10.0,
    )
    try:
        worker.submit("in-flight", class_id=0, track_id=0)
        assert synth.entered.wait(1.0)

        for index in range(1, 9):
            worker.submit(f"pending-{index}", class_id=index, track_id=index)

        stats = worker.get_stats()
        assert stats["dropped_queue_full"] >= 6
        assert stats["pending"] <= 2

        gate.set()
        eventually(lambda: worker.get_stats()["synthesized"] >= 3)
        assert synth.calls == ["in-flight", "pending-7", "pending-8"]
        assert_late_ratio(worker)
    finally:
        gate.set()
        worker.shutdown(wait=True, cancel_pending=True)


def test_tts_timeout_publishes_drop_and_removes_late_audio(tmp_path: Path) -> None:
    """A timed-out synthesis cannot later publish or retain its audio."""
    gate = threading.Event()
    synth = Synthesizer(gate=gate)
    drops: queue.Queue[dict[str, Any]] = queue.Queue()
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        timeout_sec=0.08,
        max_alert_age_sec=10.0,
        on_drop=drops.put,
    )
    try:
        alert = worker.submit("timeout", class_id=1, track_id=1)
        assert synth.entered.wait(1.0)
        eventually(lambda: worker.get_stats()["timeouts"] >= 1)
        dropped = drops.get(timeout=WAIT_SEC)
        assert dropped["status"] == "timeout"
        assert dropped["alert_id"] == alert["alert_id"]
        assert worker.get_audio_path(alert["alert_id"]) is None

        gate.set()
        assert synth.finished.wait(WAIT_SEC)
        eventually(
            lambda: bool(synth.paths) and all(
                not path.exists() for path in synth.paths
            ),
            message="Timed-out synthesis left an audio file behind",
        )
        assert not any(
            item["alert_id"] == alert["alert_id"] for item in ready_alerts(worker)
        )
        assert_late_ratio(worker)
    finally:
        gate.set()
        worker.shutdown(wait=True, cancel_pending=True)


@pytest.mark.parametrize("stale_clock", ["media", "wall"])
def test_tts_pre_stale_alert_never_calls_synthesizer(
    tmp_path: Path, stale_clock: str
) -> None:
    """Queued alerts are checked for both media lag and wall-clock age."""
    gate = threading.Event()
    synth = Synthesizer(gate=gate)
    wall = ManualClock(100.0)
    media = ManualClock(0.0)
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        clock_fn=wall,
        media_clock_fn=media,
        max_media_lag_sec=0.5,
        max_alert_age_sec=0.5,
        timeout_sec=3.0,
        cooldown_sec=0.0,
    )
    try:
        worker.submit("in-flight", class_id=1, source_timestamp=0.0)
        assert synth.entered.wait(1.0)
        stale = worker.submit("queued-stale", class_id=2, source_timestamp=0.0)

        if stale_clock == "media":
            media.set(2.0)
        else:
            wall.set(102.0)

        gate.set()
        eventually(lambda: worker.get_stats()["dropped_pre_stale"] >= 1)
        assert "queued-stale" not in synth.calls
        assert worker.get_audio_path(stale["alert_id"]) is None
        assert_late_ratio(worker)
    finally:
        gate.set()
        worker.shutdown(wait=True, cancel_pending=True)


def test_tts_post_stale_audio_is_deleted(tmp_path: Path) -> None:
    """Media advancing during synthesis invalidates and deletes the result."""
    gate = threading.Event()
    synth = Synthesizer(gate=gate)
    media = ManualClock(0.0)
    drops: queue.Queue[dict[str, Any]] = queue.Queue()
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        media_clock_fn=media,
        max_media_lag_sec=0.25,
        max_alert_age_sec=10.0,
        timeout_sec=3.0,
        on_drop=drops.put,
    )
    try:
        alert = worker.submit("post-stale", class_id=1, source_timestamp=0.0)
        assert synth.entered.wait(1.0)
        media.set(2.0)
        gate.set()

        eventually(lambda: worker.get_stats()["dropped_post_stale"] >= 1)
        dropped = drops.get(timeout=WAIT_SEC)
        assert dropped["status"] == "dropped_stale"
        assert worker.get_audio_path(alert["alert_id"]) is None
        assert synth.paths
        assert all(not path.exists() for path in synth.paths)
        assert not ready_alerts(worker)
        assert_late_ratio(worker)
    finally:
        gate.set()
        worker.shutdown(wait=True, cancel_pending=True)


def test_tts_seek_drains_queue_and_resets_dedup(tmp_path: Path) -> None:
    """Seek cleanup drains queued items, invalidates pre-seek synth, and resets dedup."""
    gate = threading.Event()
    gate.set()
    synth = Synthesizer(gate=gate)
    worker = AsyncTTSWorker(
        {},
        tmp_path,
        synthesizer=synth,
        cooldown_sec=60.0,
        timeout_sec=3.0,
        max_alert_age_sec=10.0,
    )
    try:
        worker.submit("ready-1", class_id=1, track_id=1)
        eventually(lambda: len(ready_alerts(worker)) == 1)
        assert worker.submit("duplicate", class_id=1, track_id=1)[
            "status"
        ] == "deduped"

        gate.clear()
        synth.entered.clear()
        worker.submit("in-flight-blocker", class_id=9, track_id=9)
        assert synth.entered.wait(1.0)
        worker.submit("discard-2", class_id=2, track_id=2)
        worker.submit("discard-3", class_id=3, track_id=3)

        assert worker.drain_pending() == 2
        assert worker.get_stats()["pending"] == 0
        assert worker.get_stats()["dropped_drained"] == 2

        gate.set()
        eventually(lambda: worker.get_stats()["dropped_post_stale"] == 1)
        assert worker.submit("still-deduped", class_id=1, track_id=1)[
            "status"
        ] == "deduped"

        worker.reset_dedup()
        assert worker.submit("after-seek", class_id=1, track_id=1)[
            "status"
        ] == "queued"
        eventually(lambda: len(ready_alerts(worker)) == 2)
        assert synth.calls == ["ready-1", "in-flight-blocker", "after-seek"]
        assert_late_ratio(worker)
    finally:
        gate.set()
        worker.shutdown(wait=True, cancel_pending=True)


class DummyDetector:
    """Expose CPU-tunable attributes without loading torch or model weights."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {
            "img_size": 640,
            "conf": 0.25,
            "iou": 0.45,
            "max_det": 300,
            "device": "cpu",
            "half": False,
        }
        self.device = "cpu"
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 300
        self.half = False
        self.model = SimpleNamespace(
            transform=SimpleNamespace(min_size=(800,), max_size=1333),
            rpn=SimpleNamespace(
                _pre_nms_top_n={"training": 2000, "testing": 1000},
                _post_nms_top_n={"training": 2000, "testing": 1000},
            ),
            roi_heads=SimpleNamespace(detections_per_img=100),
            overrides=dict(self.cfg),
            args=dict(self.cfg),
        )


def faster_rcnn_values(detector: DummyDetector) -> tuple[Any, ...]:
    """Capture actual inference controls, including untouched training values."""
    model = detector.model
    return (
        tuple(model.transform.min_size),
        model.transform.max_size,
        dict(model.rpn._pre_nms_top_n),
        dict(model.rpn._post_nms_top_n),
        model.roi_heads.detections_per_img,
    )


def test_backend_probe_is_safe_without_optional_packages() -> None:
    """Backend discovery returns serializable data on a CPU-only installation."""
    result = probe_runtime_backends()
    assert isinstance(result, dict)
    assert result
    json.dumps(result)


def test_faster_rcnn_tuning_changes_model_and_restores_defaults() -> None:
    """CPU tuning modifies real model controls and restoration reverses it."""
    detector = DummyDetector()
    original = faster_rcnn_values(detector)
    old_threads = torch.get_num_threads()
    cfg = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 120,
        "rpn_post_nms_top_n_test": 60,
        "box_detections_per_img": 12,
        "threads": 1,
    }
    try:
        report = apply_faster_rcnn_cpu_tuning(detector, cfg)
        assert isinstance(report, dict)

        model = detector.model
        assert tuple(model.transform.min_size) == (320,)
        assert model.transform.max_size == 640
        assert model.rpn._pre_nms_top_n["testing"] == 120
        assert model.rpn._post_nms_top_n["testing"] == 60
        assert model.roi_heads.detections_per_img == 12
        assert model.rpn._pre_nms_top_n["training"] == 2000
        assert model.rpn._post_nms_top_n["training"] == 2000
        assert torch.get_num_threads() == 1

        assert restore_cpu_tuning(detector)
        assert faster_rcnn_values(detector) == original
        assert torch.get_num_threads() == old_threads
    finally:
        torch.set_num_threads(old_threads)


def test_yolo_tuning_changes_inference_configuration_and_restores() -> None:
    """YOLO tuning updates inference settings without model weights."""
    detector = DummyDetector()
    original_size = detector.img_size
    old_threads = torch.get_num_threads()
    try:
        report = apply_yolo_cpu_tuning(
            detector, {"img_size": 320, "fuse": False, "threads": 1}
        )
        assert isinstance(report, dict)
        assert detector.img_size == 320
        assert torch.get_num_threads() == 1

        assert restore_cpu_tuning(detector)
        assert detector.img_size == original_size
        assert torch.get_num_threads() == old_threads
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_missing_export_falls_back_to_original_pytorch_detector(
    tmp_path: Path, backend: str
) -> None:
    """Unavailable runtime artifacts produce an explicit PyTorch fallback."""
    detector = DummyDetector()
    old_threads = torch.get_num_threads()
    try:
        prepared, report = prepare_runtime_detector(
            "yolo",
            detector,
            cfg={},
            backend=backend,
            export_dir=tmp_path / "missing-runtime-artifacts",
        )
        assert prepared is detector
        assert report["fallback_used"] is True
        assert isinstance(report["reason"], str)
        assert report["reason"].strip()
        assert "pytorch" in json.dumps(report).lower()
    finally:
        restore_cpu_tuning(detector)
        torch.set_num_threads(old_threads)


def nested_values(value: Any, key: str) -> Iterator[Any]:
    """Yield matching keys from JSON-like nested structures."""
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                yield child
            yield from nested_values(child, key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from nested_values(child, key)


def test_raw_evaluation_cpu_tuning_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shipped default keeps runtime tuning out of raw ML evaluation."""
    path = Path(__file__).resolve().parents[1] / "configs" / "shared.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = list(nested_values(config, "apply_on_load"))
    assert defaults, "shared.yaml must explicitly declare apply_on_load"
    assert all(value is False for value in defaults)

    yolo_cfg = resolve_config("yolo", "local")
    frcnn_cfg = resolve_config("faster_rcnn", "local")
    assert yolo_cfg["inference"]["cpu"]["apply_on_load"] is False
    assert frcnn_cfg["inference"]["cpu"]["apply_on_load"] is False

    dummy_model = SimpleNamespace(
        transform=SimpleNamespace(min_size=(800,), max_size=1333),
        rpn=SimpleNamespace(
            _pre_nms_top_n={"training": 2000, "testing": 1000},
            _post_nms_top_n={"training": 2000, "testing": 1000},
        ),
        roi_heads=SimpleNamespace(detections_per_img=100, nms_thresh=0.5),
    )
    monkeypatch.setattr(
        "ts_alert.pipelines.yolo.infer.load_model", lambda *_a, **_kw: dummy_model
    )
    monkeypatch.setattr(
        "ts_alert.pipelines.faster_rcnn.infer.load_checkpoint",
        lambda *_a, **_kw: dummy_model,
    )

    yolo_det = YoloDetector(tmp_path / "best.pt", yolo_cfg)
    frcnn_det = FasterRCNNDetector(tmp_path / "best.pth", frcnn_cfg)
    assert yolo_det.img_size == int(yolo_cfg["data"]["image_size"])
    assert not hasattr(yolo_det, "_cpu_profile")
    assert frcnn_det.model.transform.min_size == (800,)
    assert frcnn_det.model.transform.max_size == 1333
    assert frcnn_det.model.roi_heads.detections_per_img == 100
    assert not hasattr(frcnn_det, "_cpu_profile")


@dataclass
class DetectorControl:
    """Control fake inference with events for cancellation and overload tests."""

    delay_sec: float = 0.0
    block: bool = False
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    exited: threading.Event = field(default_factory=threading.Event)
    patched_methods: int = 0
    local: threading.local = field(default_factory=threading.local)

    def wrap(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one fake inference method while avoiding nested delays."""

        @functools.wraps(method)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if getattr(self.local, "inside", False):
                return method(*args, **kwargs)
            self.local.inside = True
            try:
                self.entered.set()
                if self.block:
                    if not self.release.wait(WAIT_SEC):
                        raise TimeoutError("Fake detector gate was not released")
                elif self.delay_sec:
                    self.release.wait(self.delay_sec)
                return method(*args, **kwargs)
            finally:
                self.local.inside = False
                self.exited.set()

        return wrapped


@dataclass
class WebRuntime:
    """Keep app resources and bounded SSE readers under one cleanup owner."""

    app: Any
    control: DetectorControl
    streams: list["SSEReader"] = field(default_factory=list)

    @property
    def extension(self) -> dict[str, Any]:
        return self.app.extensions["tsd"]

    def stream(self, url: str) -> "SSEReader":
        reader = SSEReader(self.app, url)
        self.streams.append(reader)
        return reader


class SSEReader:
    """Read SSE on a dedicated client without blocking the pytest thread."""

    def __init__(self, app: Any, url: str) -> None:
        self.app = app
        self.url = url
        self.events: list[tuple[str, Any]] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = threading.Event()
        self.errors: queue.Queue[BaseException] = queue.Queue()
        self.thread = threading.Thread(
            target=self._run, name="test-sse-reader", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        response: Any = None
        try:
            with self.app.test_client() as client:
                response = client.get(self.url, buffered=False)
                assert response.status_code == 200
                assert "text/event-stream" in response.content_type
                self.started.set()
                buffer = ""
                event_name = "message"
                data_lines: list[str] = []

                for chunk in response.response:
                    if self.stop.is_set():
                        break
                    buffer += (
                        chunk.decode("utf-8")
                        if isinstance(chunk, bytes)
                        else str(chunk)
                    )
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.rstrip("\r")
                        if not line:
                            if data_lines:
                                payload = json.loads("\n".join(data_lines))
                                with self.lock:
                                    self.events.append((event_name, payload))
                            event_name = "message"
                            data_lines = []
                        elif line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
        except BaseException as exc:
            self.errors.put(exc)
        finally:
            self.started.set()
            if response is not None:
                response.close()

    def snapshot(self) -> list[tuple[str, Any]]:
        """Return received events and propagate background failures."""
        if not self.errors.empty():
            raise self.errors.get_nowait()
        with self.lock:
            return list(self.events)

    def has_event(self, *names: str) -> bool:
        """Recognize named SSE events or explicit JSON event type fields."""
        for event_name, payload in self.snapshot():
            if event_name in names:
                return True
            if isinstance(payload, dict) and payload.get("type") in names:
                return True
        return False

    def has_status(self, *statuses: str) -> bool:
        """Find a status emitted anywhere inside an SSE payload."""
        return any(
            value in statuses
            for _, payload in self.snapshot()
            for value in nested_values(payload, "status")
            if isinstance(value, str)
        )


def patch_fake_detectors(
    monkeypatch: pytest.MonkeyPatch,
    control: DetectorControl,
    fake_parts: Any = None,
) -> None:
    """Instrument exposed fake detector classes without using real models."""
    module = importlib.import_module("app.fake")
    classes: set[type[Any]] = {
        value
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and ("detector" in name.lower() or "detector" in value.__name__.lower())
    }

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                collect(child)
        elif inspect.isclass(value):
            if "detector" in value.__name__.lower():
                classes.add(value)
        elif value is not None and "detector" in type(value).__name__.lower():
            classes.add(type(value))

    collect(fake_parts)
    for cls in classes:
        for name in ("detect", "predict", "infer", "__call__"):
            method = getattr(cls, name, None)
            if method is None or not callable(method):
                continue
            if getattr(method, "__test_runtime_wrapped__", False):
                continue
            if name == "__call__" and "__call__" not in cls.__dict__:
                continue
            wrapped = control.wrap(method)
            wrapped.__test_runtime_wrapped__ = True
            monkeypatch.setattr(cls, name, wrapped)
            control.patched_methods += 1


@pytest.fixture
def web_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[WebRuntime]:
    """Create an offline fake app and always shut down all owned services."""
    from app.fake import _fake_synthesize

    control = DetectorControl()
    patch_fake_detectors(monkeypatch, control)
    monkeypatch.setattr("ts_alert.postprocess.tts.synthesize", _fake_synthesize)
    app = create_app(
        runtime="local",
        fake=True,
        store=InMemoryStore(),
        sync=False,
        overrides={
            "paths": {
                "outputs_dir": str(tmp_path / "outputs"),
                "uploads_dir": str(tmp_path / "uploads"),
                "db_path": str(tmp_path / "test.db"),
            }
        },
    )
    app.config.update(TESTING=True)
    runtime = WebRuntime(app, control)
    patch_fake_detectors(
        monkeypatch, control, runtime.extension.get("fake_parts")
    )
    cleanup_errors: list[BaseException] = []
    try:
        yield runtime
    finally:
        control.release.set()
        for reader in runtime.streams:
            reader.stop.set()

        for key in ("realtime", "runner"):
            service = runtime.extension.get(key)
            if service is None:
                continue
            try:
                service.shutdown()
            except BaseException as exc:
                cleanup_errors.append(exc)

        restore = runtime.extension.get("restore_fakes")
        if callable(restore):
            try:
                restore()
            except BaseException as exc:
                cleanup_errors.append(exc)

        for reader in runtime.streams:
            reader.thread.join(WAIT_SEC)
            if reader.thread.is_alive():
                cleanup_errors.append(
                    AssertionError("SSE reader survived runtime shutdown")
                )
        if cleanup_errors:
            raise AssertionError(
                "Runtime cleanup failed: "
                + "; ".join(repr(error) for error in cleanup_errors)
            )


@pytest.fixture
def video_bytes(tmp_path: Path) -> bytes:
    """Encode a short 30 FPS video using locally available OpenCV codecs."""
    candidates = (("MJPG", ".avi"), ("mp4v", ".mp4"))
    for codec, suffix in candidates:
        path = tmp_path / f"synthetic{suffix}"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), FPS, (240, 160)
        )
        if not writer.isOpened():
            writer.release()
            continue
        try:
            for index in range(90):
                writer.write(make_scene(float(index % 35)))
        finally:
            writer.release()

        capture = cv2.VideoCapture(str(path))
        try:
            valid, _ = capture.read()
        finally:
            capture.release()
        if valid:
            return path.read_bytes()
    pytest.fail("No local OpenCV video encoder/decoder is available")


def video_filename(content: bytes) -> str:
    """Choose an upload extension matching the generated video container."""
    return "synthetic.avi" if content.startswith(b"RIFF") else "synthetic.mp4"


def upload_video(
    client: Any, content: bytes, **fields: Any
) -> Any:
    """Upload a fresh byte stream for each request."""
    data: dict[str, Any] = {
        "file": (io.BytesIO(content), video_filename(content)),
        "model": "yolo",
        "frame_stride": "1",
        "analysis_mode": "realtime_overlay",
        "playback_mode": "realtime",
        "backend": "pytorch",
        "buffer_ahead_sec": "0.25",
    }
    data.update(fields)
    return client.post(
        "/api/jobs", data=data, content_type="multipart/form-data"
    )


def response_generation(payload: dict[str, Any]) -> int:
    """Require a playback response to expose its resulting generation."""
    generations = list(nested_values(payload, "generation"))
    assert generations, payload
    assert isinstance(generations[0], int)
    return generations[0]


def test_health_and_backend_endpoints(web_runtime: WebRuntime) -> None:
    """Basic runtime discovery remains available without model downloads."""
    with web_runtime.app.test_client() as client:
        for url in ("/api/health", "/api/runtime/backends"):
            response = client.get(url)
            assert response.status_code == 200
            assert isinstance(response.get_json(), dict)


def test_video_overlay_source_sse_playback_and_cancel(
    web_runtime: WebRuntime, video_bytes: bytes
) -> None:
    """Exercise upload, source range, overlay streaming, seeking, and cancel."""
    with web_runtime.app.test_client() as client:
        response = upload_video(client, video_bytes)
        assert response.status_code == 202
        payload = response.get_json()
        assert payload["analysis_mode"] == "realtime_overlay"
        for name in (
            "source_url",
            "events_url",
            "playback_url",
            "cancel_url",
            "export_url",
        ):
            assert isinstance(payload[name], str) and payload[name]

        source = client.get(
            payload["source_url"], headers={"Range": "bytes=0-99"}
        )
        try:
            assert source.status_code in (200, 206)
            assert source.headers["Accept-Ranges"].lower() == "bytes"
            assert source.data
            if source.status_code == 206:
                assert len(source.data) == 100
                assert source.headers["Content-Range"].startswith("bytes 0-99/")
        finally:
            source.close()

        reader = web_runtime.stream(payload["events_url"])
        eventually(lambda: reader.has_event("status"))
        eventually(lambda: reader.has_event("overlay", "alert"))

        first = client.post(
            payload["playback_url"],
            json={
                "timestamp_sec": 0.2,
                "playing": False,
                "playback_rate": 1.0,
                "mode": "buffered",
                "buffer_ahead_sec": 0.5,
            },
        )
        assert first.status_code == 200
        generation = response_generation(first.get_json())

        sought = client.post(
            payload["playback_url"],
            json={
                "timestamp_sec": 0.0,
                "playing": False,
                "playback_rate": 1.0,
                "seeked": True,
                "mode": "realtime",
                "buffer_ahead_sec": 0.25,
            },
        )
        assert sought.status_code == 200
        assert response_generation(sought.get_json()) > generation

        cancelled = client.post(payload["cancel_url"])
        assert cancelled.status_code == 200


def test_cancellation_does_not_wait_for_blocked_detector(
    web_runtime: WebRuntime, video_bytes: bytes
) -> None:
    """Cancellation completes while an inference call is still blocked."""
    control = web_runtime.control
    assert control.patched_methods, "Fake detector exposes no inference hook"
    control.block = True

    with web_runtime.app.test_client() as client:
        response = upload_video(client, video_bytes)
        assert response.status_code == 202
        payload = response.get_json()
        reader = web_runtime.stream(payload["events_url"])
        try:
            assert control.entered.wait(WAIT_SEC)
            assert not control.exited.is_set()

            started = time.monotonic()
            cancelled = client.post(payload["cancel_url"])
            assert cancelled.status_code == 200
            eventually(
                lambda: reader.has_status("failed", "cancelled"),
                timeout=1.8,
                message="Cancellation did not publish a terminal status",
            )
            assert time.monotonic() - started < 2.0
            assert not control.release.is_set()
        finally:
            control.release.set()


def test_cpu_overload_drops_frames_without_deadlock(
    web_runtime: WebRuntime, video_bytes: bytes
) -> None:
    """An 80 ms detector cannot accumulate an unbounded 30 FPS backlog."""
    control = web_runtime.control
    assert control.patched_methods, "Fake detector exposes no inference hook"
    control.delay_sec = 0.08

    with web_runtime.app.test_client() as client:
        response = upload_video(client, video_bytes)
        assert response.status_code == 202
        payload = response.get_json()
        reader = web_runtime.stream(payload["events_url"])
        try:
            assert control.entered.wait(WAIT_SEC)

            def overload_visible() -> bool:
                for _, event in reader.snapshot():
                    for key in ("dropped_frames", "skipped_frames"):
                        if any(
                            isinstance(value, (int, float)) and value > 0
                            for value in nested_values(event, key)
                        ):
                            return True
                return False

            eventually(
                overload_visible,
                message="No dropped/skipped frames reported under CPU overload",
            )
            assert reader.has_event("overlay", "alert")
        finally:
            assert client.post(payload["cancel_url"]).status_code == 200


def jpeg_bytes() -> bytes:
    """Encode a valid synthetic camera frame."""
    success, encoded = cv2.imencode(".jpg", make_scene())
    assert success
    return encoded.tobytes()


def test_camera_upload_backpressure_sse_and_stop(
    web_runtime: WebRuntime,
) -> None:
    """Live camera input validates bytes, replaces pending work, and stops."""
    control = web_runtime.control
    assert control.patched_methods, "Fake detector exposes no inference hook"
    control.block = True
    jpeg = jpeg_bytes()

    with web_runtime.app.test_client() as client:
        created = client.post(
            "/api/camera/sessions",
            json={"model": "yolo", "backend": "pytorch", "frame_stride": 1},
        )
        assert created.status_code == 201
        payload = created.get_json()
        session_id = payload.get("session_id", payload.get("id"))
        assert session_id
        base = f"/api/camera/sessions/{session_id}"

        try:
            invalid = client.post(
                f"{base}/frame",
                data=b"not-an-image",
                content_type="application/octet-stream",
            )
            assert invalid.status_code == 400

            first = client.post(
                f"{base}/frame",
                data={
                    "file": (io.BytesIO(jpeg), "frame.jpg"),
                    "frame_id": "0",
                    "timestamp_sec": "0.0",
                },
                content_type="multipart/form-data",
            )
            assert first.status_code == 200
            assert control.entered.wait(WAIT_SEC)

            dropped: list[int] = []
            for index in range(1, 9):
                pushed = client.post(
                    f"{base}/frame",
                    data={
                        "file": (io.BytesIO(jpeg), "frame.jpg"),
                        "frame_id": str(index),
                        "timestamp_sec": str(index / FPS),
                    },
                    content_type="multipart/form-data",
                )
                assert pushed.status_code == 200
                body = pushed.get_json()
                assert "dropped_before" in body
                dropped.append(int(body["dropped_before"]))
            assert any(value > 0 for value in dropped)

            reader = web_runtime.stream(f"{base}/events")
            control.release.set()
            eventually(lambda: reader.has_event("status"))
            eventually(lambda: reader.has_event("overlay", "alert"))

            stopped = client.post(f"{base}/stop")
            assert stopped.status_code == 200
            after_stop = client.post(
                f"{base}/frame",
                data=jpeg,
                content_type="image/jpeg",
            )
            assert after_stop.status_code == 404
        finally:
            control.release.set()
            client.post(f"{base}/stop")


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("cpu_preset", "does-not-exist"),
        ("backend", "does-not-exist"),
        ("analysis_mode", "does-not-exist"),
        ("playback_mode", "does-not-exist"),
        ("buffer_ahead_sec", "not-a-number"),
        ("buffer_ahead_sec", "-1"),
        ("buffer_ahead_sec", "nan"),
        ("buffer_ahead_sec", "inf"),
    ],
)
def test_job_validation_precedes_upload_persistence(
    web_runtime: WebRuntime,
    field_name: str,
    invalid_value: str,
) -> None:
    """Invalid runtime options fail before any uploaded file is persisted."""
    uploads = Path(web_runtime.extension["uploads_dir"])
    before = {
        path.relative_to(uploads)
        for path in uploads.rglob("*")
        if path.is_file()
    } if uploads.exists() else set()

    with web_runtime.app.test_client() as client:
        response = upload_video(
            client,
            b"validation-must-run-before-video-decoding",
            **{field_name: invalid_value},
        )

    assert response.status_code == 400
    assert response.is_json
    assert isinstance(response.get_json(), dict)
    after = {
        path.relative_to(uploads)
        for path in uploads.rglob("*")
        if path.is_file()
    } if uploads.exists() else set()
    assert after == before


def test_cpu_presets_have_structured_configurations() -> None:
    """Public CPU presets are nonempty mappings suitable for explicit opt-in."""
    assert isinstance(CPU_PRESETS, dict)
    assert CPU_PRESETS
    for name, config in CPU_PRESETS.items():
        assert isinstance(name, str) and name
        assert isinstance(config, dict)
        assert all(isinstance(section, dict) for section in config.values())


def test_realtime_completion_and_export_transition(
    web_runtime: WebRuntime, video_bytes: bytes
) -> None:
    """Completing a realtime job exposes stored detections and allows MP4 export."""
    control = web_runtime.control
    control.block = True

    with web_runtime.app.test_client() as client:
        response = upload_video(
            client,
            video_bytes,
            playback_mode="buffered",
            buffer_ahead_sec="5.0",
            frame_stride="15",
        )
        assert response.status_code == 202
        payload = response.get_json()
        job_id = payload["job_id"]

        assert control.entered.wait(WAIT_SEC)
        conflict = client.post(payload["export_url"])
        assert conflict.status_code == 409

        control.block = False
        control.release.set()

        done_job = eventually(
            lambda: (
                detail.get_json()
                if (detail := client.get(f"/api/jobs/{job_id}")).get_json()["status"] == "done"
                else None
            ),
            timeout=6.0,
            message="Realtime buffered job did not finish in time",
        )
        assert done_job["analysis_mode"] == "realtime_overlay"
        assert "download_url" not in done_job
        dets = client.get(f"/api/jobs/{job_id}/detections").get_json()
        assert dets and all("bbox" in item for item in dets)

        exported = client.post(payload["export_url"])
        assert exported.status_code == 202
        export_body = exported.get_json()
        assert export_body["analysis_mode"] == "legacy"

        final_job = eventually(
            lambda: (
                detail.get_json()
                if (detail := client.get(f"/api/jobs/{job_id}")).get_json()["status"] == "done"
                else None
            ),
            timeout=6.0,
            message="Exported MP4 job did not finish in time",
        )
        assert final_job["download_url"].endswith("download=1")
        assert client.get(final_job["result_url"]).status_code == 200


def test_tracker_pop_lost_tracks_reasons_and_reid_reset() -> None:
    """Lost events record their cause, drain once, and clear on Re-ID."""
    tracker = TrafficSignTracker(
        config=tracker_config(
            use_optical_flow=False,
            ttl_sec=2.0,
            min_hits_to_confirm=1,
            single_hit_conf_threshold=0.6,
            # Default smoothing leaves >25% visible at t=0.35.
            velocity_smoothing=1.0,
        )
    )
    original: dict[str, Any] | None = None
    for fid, timestamp, box in (
        (0, 0.0, [150.0, 40.0, 200.0, 90.0]),
        (1, 0.1, [185.0, 40.0, 235.0, 90.0]),
    ):
        tracked = tracker.update_from_detection(
            [detection(box)],
            detection_frame_id=fid,
            detection_timestamp=timestamp,
            frame_width=240,
            frame_height=160,
        )
        assert len(tracked) == 1
        if original is None:
            original = tracked[0]
        else:
            assert tracked[0]["track_id"] == original["track_id"]
            assert tracked[0]["vx"] > 0.0

    assert original is not None
    assert tracker.step_frame(
        None,
        frame_id=10,
        timestamp_sec=0.35,
        frame_width=240,
        frame_height=160,
    ) == []

    events = tracker.pop_lost_tracks()
    assert len(events) == 1
    assert events[0]["track_id"] == original["track_id"]
    assert events[0]["reason"] == "exited_frame"
    assert events[0]["confirmed"] is True
    assert events[0]["lost_timestamp"] == pytest.approx(0.35)
    assert tracker.pop_lost_tracks() == []

    tracker = TrafficSignTracker(
        config=tracker_config(
            use_optical_flow=False,
            max_missed_detections=2,
            ttl_sec=5.0,
        )
    )
    original = start_track(tracker)
    for fid in (1, 2):
        tracked = tracker.update_from_detection(
            [],
            detection_frame_id=fid,
            detection_timestamp=fid / FPS,
            frame_width=240,
            frame_height=160,
        )
        if fid == 1:
            assert len(tracked) == 1
            assert tracked[0]["missed_detections"] == 1
        else:
            assert tracked == []

    events = tracker.pop_lost_tracks()
    assert len(events) == 1
    assert events[0]["track_id"] == original["track_id"]
    assert events[0]["reason"] == "missed"
    assert events[0]["confirmed"] is True
    assert events[0]["lost_timestamp"] == pytest.approx(2 / FPS)
    assert tracker.pop_lost_tracks() == []

    tracker = TrafficSignTracker(
        config=tracker_config(
            use_optical_flow=False,
            ttl_sec=0.4,
            reid_ttl_sec=1.5,
        )
    )
    original = start_track(tracker)
    tracker.mark_alert_emitted(original["track_id"])

    assert tracker.step_frame(
        None,
        frame_id=15,
        timestamp_sec=0.5,
        frame_width=240,
        frame_height=160,
    ) == []
    assert tracker.get_stats()["lost_count"] == 1
    assert tracker.get_stats()["expired_tracks"] == 1

    # Leave the expiry event unread until Re-ID clears it.
    restored = tracker.update_from_detection(
        [detection()],
        detection_frame_id=17,
        detection_timestamp=0.55,
        frame_width=240,
        frame_height=160,
    )
    assert len(restored) == 1
    assert restored[0]["track_id"] == original["track_id"]
    assert restored[0]["alert_emitted"] is True
    assert tracker.get_stats()["reidentified_tracks"] == 1
    assert tracker.pop_lost_tracks() == []

    assert tracker.step_frame(
        None,
        frame_id=63,
        timestamp_sec=2.1,
        frame_width=240,
        frame_height=160,
    ) == []
    events = tracker.pop_lost_tracks()
    assert len(events) == 1
    assert events[0]["track_id"] == original["track_id"]
    assert events[0]["reason"] == "expired"
    assert events[0]["confirmed"] is True
    assert events[0]["alert_emitted"] is True
    assert events[0]["lost_timestamp"] == pytest.approx(2.1)
    assert tracker.pop_lost_tracks() == []
    assert tracker.get_stats()["lost_count"] == 0


def test_realtime_track_lost_sse_preserves_audio_file_and_logs_voice_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A track lost during synthesis retains ready audio and persistence."""
    permits: queue.Queue[None] = queue.Queue()
    stop_decode = threading.Event()
    synth_gate = threading.Event()
    synth_entered = threading.Event()
    synth_finished = threading.Event()
    captures: list[Any] = []

    class FakeDetector:
        """Emit one sign followed by empty detector cycles."""

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
            """Return the sign only for the first permitted frame."""
            assert frame.shape == (160, 240, 3)
            self.calls += 1
            if self.calls != 1:
                return []
            return [
                {
                    "bbox": [60.0, 40.0, 120.0, 100.0],
                    "class_id": 7,
                    "label": "Cấm rẽ trái",
                    "confidence": 0.95,
                }
            ]

    class ControlledCapture:
        """Deliver exactly one decoded frame per test permit."""

        def __init__(self, source: str) -> None:
            self.source = source
            self.priming = True
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            """Report an open synthetic video source."""
            return not self.released

        def get(self, prop: int) -> float:
            """Expose valid video metadata without codec dependencies."""
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 240.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 160.0,
                cv2.CAP_PROP_FPS: FPS,
                cv2.CAP_PROP_FRAME_COUNT: 90.0,
            }.get(prop, 0.0)

        def set(self, prop: int, value: float) -> bool:
            """Accept the startup rewind and enable gated decoding."""
            if prop == cv2.CAP_PROP_POS_FRAMES and value == 0:
                self.priming = False
                return True
            return False

        def read(self) -> tuple[bool, np.ndarray | None]:
            """Allow startup validation, then wait interruptibly for permits."""
            if self.priming:
                self.priming = False
                return True, make_scene()

            while not stop_decode.is_set():
                try:
                    permits.get(timeout=0.02)
                except queue.Empty:
                    continue
                if stop_decode.is_set():
                    break
                return True, make_scene()
            return False, None

        def release(self) -> None:
            """Record ownership release by the decode worker."""
            self.released = True

    def synthesize(
        text: str,
        target: Path,
        cfg: dict[str, Any],
    ) -> Path:
        """Keep synthesis in flight until the lost event is observed."""
        del cfg
        try:
            assert text.strip()
            synth_entered.set()
            if not synth_gate.wait(WAIT_SEC * 3):
                raise TimeoutError("Synthesis gate was not released")
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
            return target
        finally:
            synth_finished.set()

    detector = FakeDetector()

    # Replace only the service's cv2 reference; leave global OpenCV untouched.
    monkeypatch.setattr(
        realtime_service,
        "cv2",
        SimpleNamespace(
            VideoCapture=ControlledCapture,
            CAP_PROP_FRAME_WIDTH=cv2.CAP_PROP_FRAME_WIDTH,
            CAP_PROP_FRAME_HEIGHT=cv2.CAP_PROP_FRAME_HEIGHT,
            CAP_PROP_FPS=cv2.CAP_PROP_FPS,
            CAP_PROP_FRAME_COUNT=cv2.CAP_PROP_FRAME_COUNT,
            CAP_PROP_POS_FRAMES=cv2.CAP_PROP_POS_FRAMES,
            CAP_PROP_POS_MSEC=cv2.CAP_PROP_POS_MSEC,
        ),
    )
    monkeypatch.setattr(
        realtime_service,
        "prepare_runtime_detector",
        lambda _model, raw, _cfg, **_kwargs: (raw, {}),
    )
    monkeypatch.setattr(
        realtime_service, "restore_cpu_tuning", lambda _detector: False
    )
    monkeypatch.setattr(
        realtime_service, "load_classes", lambda _path: {}
    )
    monkeypatch.setattr(
        "ts_alert.postprocess.tts.synthesize", synthesize
    )

    app = create_app(
        runtime="local",
        fake=False,
        store=InMemoryStore(),
        sync=False,
        overrides={
            "paths": {
                "outputs_dir": str(tmp_path / "outputs"),
                "uploads_dir": str(tmp_path / "uploads"),
                "db_path": str(tmp_path / "test.db"),
            },
            "realtime": {
                "tts_compact": True,
                "tts_compact_template": "{position_text}, {name}",
                "tts_cache_enabled": False,
                "tts_timeout_sec": WAIT_SEC * 3,
                "tts_max_alert_age_sec": WAIT_SEC * 3,
            },
        },
    )
    app.config.update(TESTING=True)
    extension = app.extensions["tsd"]
    manager = extension["realtime"]
    monkeypatch.setattr(
        manager, "_detector_provider", lambda _model: detector
    )

    reader: SSEReader | None = None
    session: Any = None

    try:
        with app.test_client() as client:
            # ControlledCapture supplies frames; uploaded bytes need no codec.
            response = upload_video(
                client,
                b"RIFF\x04\x00\x00\x00AVI ",
                frame_stride="1",
                analysis_mode="realtime_overlay",
                playback_mode="buffered",
                buffer_ahead_sec="5.0",
            )
            assert response.status_code == 202
            payload = response.get_json()
            job_id = payload["job_id"]
            session = manager._get_session("video", job_id)
            assert session is not None
            assert session.tracker.config.max_missed_detections == 2

            # Pin playhead at 0.0 while keeping buffered mode so slow CPU does not skip frames.
            paused = client.post(
                payload["playback_url"],
                json={
                    "timestamp_sec": 0.0,
                    "playing": False,
                    "playback_rate": 1.0,
                    "mode": "buffered",
                    "buffer_ahead_sec": 5.0,
                },
            )
            assert paused.status_code == 200

            reader = SSEReader(app, payload["events_url"])
            eventually(lambda: reader.has_event("status"))

            # Frame 0 confirms the sign and starts blocked synthesis.
            permits.put(None)
            assert synth_entered.wait(WAIT_SEC)
            assert not synth_finished.is_set()
            eventually(
                lambda: manager.get_video_status(job_id)["infer_count"] == 1
            )

            # Admit each miss only after the preceding cycle was merged.
            permits.put(None)
            eventually(
                lambda: manager.get_video_status(job_id)["infer_count"] == 2
            )
            permits.put(None)

            def lost_event() -> dict[str, Any] | None:
                """Find the confirmed sign's lost notification."""
                assert reader is not None
                return next(
                    (
                        event
                        for name, event in reader.snapshot()
                        if name == "track_lost"
                        and event.get("class_id") == 7
                    ),
                    None,
                )

            lost = eventually(lost_event)
            assert lost["reason"] == "missed"
            assert lost["label"] == "Cấm rẽ trái"
            assert lost["lost_timestamp"] == pytest.approx(2 / FPS)
            assert not synth_finished.is_set()

            # Decode remains blocked, keeping the session out of finishing.
            synth_gate.set()

            def ready_event() -> dict[str, Any] | None:
                """Find ready audio belonging to the lost identity."""
                assert reader is not None
                return next(
                    (
                        event
                        for name, event in reader.snapshot()
                        if name == "alert"
                        and event.get("status") == "ready"
                        and event.get("track_id") == lost["track_id"]
                    ),
                    None,
                )

            alert = eventually(ready_event)
            assert synth_finished.wait(WAIT_SEC)
            assert alert["track_lost"] is True
            assert isinstance(alert["lost_timestamp"], float)
            assert alert["lost_timestamp"] == pytest.approx(
                lost["lost_timestamp"]
            )
            assert alert["generation"] == lost["generation"]
            assert alert["spoken_text"].strip()
            assert "Cấm rẽ trái" in alert["spoken_text"]
            assert "Chú ý," not in alert["spoken_text"]
            assert alert["audio_url"] == (
                f"/api/jobs/{job_id}/runtime-audio/{alert['alert_id']}"
            )

            audio = client.get(alert["audio_url"])
            try:
                assert audio.status_code == 200
                assert audio.data
            finally:
                audio.close()

            audio_path = manager.get_runtime_audio_path(
                "video", job_id, alert["alert_id"]
            )
            assert audio_path is not None
            assert audio_path.is_file()
            assert audio_path.read_bytes()

            voices = client.get(f"/api/jobs/{job_id}/voice-events")
            assert voices.status_code == 200
            matching = [
                event
                for event in voices.get_json()
                if event["spoken_text"] == alert["spoken_text"]
                and Path(event["audio_path"]).resolve()
                == audio_path.resolve()
            ]
            assert len(matching) == 1
            assert Path(matching[0]["audio_path"]).is_file()
            voice_audio = client.get(matching[0]["audio_url"])
            try:
                assert voice_audio.status_code == 200
                assert voice_audio.data
            finally:
                voice_audio.close()
            assert detector.calls == 3
    finally:
        # Release every gate before shutdown, including assertion-failure paths.
        synth_gate.set()
        stop_decode.set()
        permits.put(None)
        if reader is not None:
            reader.stop.set()

        try:
            manager.shutdown(wait=True)
        finally:
            try:
                extension["runner"].shutdown(wait=True)
            finally:
                extension["shutdown"]()

        if session is not None:
            for thread in (session.decode_thread, session.infer_thread):
                if thread is not None and thread.ident is not None:
                    join_thread(thread)

        if reader is not None:
            join_thread(reader.thread)
            reader.snapshot()

        if synth_entered.is_set():
            assert synth_finished.wait(WAIT_SEC)
        assert all(capture.released for capture in captures)
