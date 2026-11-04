"""Chạy process_media trong thread nền và giữ detector trong bộ nhớ.   Phụ trách: Thành viên 3

- Một worker duy nhất (max_workers=1): các job xếp hàng lần lượt, tránh hai model cùng chiếm GPU/RAM.
- Detector của mỗi model chỉ nạp MỘT lần (lần đầu có job dùng model đó) rồi tái sử dụng.
- sync=True: chạy ngay trong thread gọi (dùng cho test).
- Tiến độ %: process_media gọi repo.add_detections(...) đúng MỘT lần cho mỗi khung hình (kể cả khung không có
  biển), nên bọc store bằng _TrackingStore để đếm khung đã xử lý / tổng số khung — không cần sửa pipeline.py.
  Tiến độ chỉ giữ trong RAM (mất khi tắt server), không ghi vào DB.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ts_alert.pipeline import process_media
from ts_alert.runtime import video_io

log = logging.getLogger(__name__)

# stage: queued -> loading_model -> processing -> finalizing (ghép âm thanh, ghi kết quả) -> done | failed
STAGES = ("queued", "loading_model", "processing", "finalizing", "done", "failed")


class ProgressTracker:
    """Tiến độ của từng job, an toàn khi thread nền ghi và thread request đọc cùng lúc."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[int, dict] = {}

    def queued(self, job_id: int) -> None:
        with self._lock:
            self._jobs[job_id] = {"stage": "queued", "frames_done": 0, "frame_count": None,
                                  "started_at": None, "updated_at": time.time()}

    def set_stage(self, job_id: int, stage: str) -> None:
        with self._lock:
            p = self._jobs.setdefault(job_id, {"frames_done": 0, "frame_count": None, "started_at": None})
            p["stage"] = stage
            p["updated_at"] = time.time()
            if stage in ("finalizing", "done", "failed"):
                self._stop_clock(p)

    @staticmethod
    def _stop_clock(p: dict) -> None:
        """Chốt thời gian xử lý khung hình (lần đầu rời stage processing) để fps không giảm dần sau khi xong."""
        if p.get("started_at") and not p.get("ended_at"):
            p["ended_at"] = time.perf_counter()

    def start_processing(self, job_id: int, frame_count: int | None) -> None:
        with self._lock:
            p = self._jobs.setdefault(job_id, {"frames_done": 0})
            p.update(stage="processing", frame_count=frame_count or None, frames_done=0,
                     started_at=time.perf_counter(), ended_at=None, updated_at=time.time())

    def frame_done(self, job_id: int, frame_index: int) -> None:
        with self._lock:
            p = self._jobs.get(job_id)
            if p is None:
                return
            p["frames_done"] = max(p["frames_done"], frame_index + 1)
            p["updated_at"] = time.time()
            # frame_count lấy từ metadata nên có thể lệch vài khung: đủ số khung thì coi như đang hoàn tất.
            if p["frame_count"] and p["frames_done"] >= p["frame_count"]:
                p["stage"] = "finalizing"
                self._stop_clock(p)

    def get(self, job_id: int) -> dict | None:
        """{"stage", "percent", "frames_done", "frame_count", "fps", "eta_sec"} hoặc None nếu không theo dõi."""
        with self._lock:
            p = self._jobs.get(job_id)
            if p is None:
                return None
            p = dict(p)
        stage, done, total = p["stage"], p["frames_done"], p["frame_count"]
        elapsed = ((p.get("ended_at") or time.perf_counter()) - p["started_at"]) if p.get("started_at") else None
        fps = done / elapsed if elapsed and done else None
        if stage == "done":
            percent = 100.0
        elif total:
            percent = min(99.0, round(100.0 * done / total, 1))   # chưa ghi xong kết quả thì chưa báo 100%
        else:
            percent = None                                         # không biết tổng số khung
        eta = max(0.0, (total - done) / fps) if fps and total and stage == "processing" else None
        return {"stage": stage, "percent": percent, "frames_done": done, "frame_count": total,
                "fps": round(fps, 2) if fps else None, "eta_sec": round(eta, 1) if eta is not None else None}


class _TrackingStore:
    """Bọc ResultStore thật: chuyển tiếp mọi lời gọi, riêng add_detections thì báo thêm cho ProgressTracker."""

    def __init__(self, store, tracker: ProgressTracker):
        self._store = store
        self._tracker = tracker

    def add_detections(self, job_id, frame_index, timestamp_sec, detections):
        self._store.add_detections(job_id, frame_index, timestamp_sec, detections)
        self._tracker.frame_done(job_id, frame_index)

    def __getattr__(self, name):
        return getattr(self._store, name)


class JobRunner:
    def __init__(self, store, cfg_for: Callable[[str], dict], detector_factory: Callable[[str, dict], object],
                 out_dir: Path, sync: bool = False):
        """cfg_for(model_name) -> cfg đã resolve; detector_factory(model_name, cfg) -> detector."""
        self.store = store
        self.cfg_for = cfg_for
        self.detector_factory = detector_factory
        self.out_dir = Path(out_dir)
        self.sync = sync
        self.progress = ProgressTracker()
        self._detectors: dict[str, object] = {}
        self._detector_lock = threading.Lock()
        self._pool = None if sync else ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsd-job")

    def get_detector(self, model_name: str):
        with self._detector_lock:
            if model_name not in self._detectors:
                log.info("Nạp detector %s ...", model_name)
                self._detectors[model_name] = self.detector_factory(model_name, self.cfg_for(model_name))
            return self._detectors[model_name]

    def loaded_models(self) -> list[str]:
        return sorted(self._detectors)

    def submit(self, job_id: int, input_path: Path, model_name: str) -> Future | None:
        self.progress.queued(job_id)
        if self.sync:
            self._run(job_id, input_path, model_name)
            return None
        return self._pool.submit(self._run, job_id, input_path, model_name)

    def _run(self, job_id: int, input_path: Path, model_name: str) -> None:
        try:
            cfg = self.cfg_for(model_name)
            if model_name not in self._detectors:
                self.progress.set_stage(job_id, "loading_model")
            detector = self.get_detector(model_name)
            self.progress.start_processing(job_id, _count_frames(input_path))
            process_media(input_path, model_name, cfg=cfg, detector=detector,
                          repo=_TrackingStore(self.store, self.progress), job_id=job_id, out_dir=self.out_dir)
            self.progress.set_stage(job_id, "done")
        except Exception as exc:     # lỗi trong process_media đã được fail_job; lỗi trước đó (nạp model...) thì ghi ở đây
            self.progress.set_stage(job_id, "failed")
            log.exception("Job %s lỗi", job_id)
            job = self.store.get_job(job_id) if hasattr(self.store, "get_job") else None
            if not job or job.get("status") != "failed":
                self.store.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    def shutdown(self, wait: bool = False) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)


def _count_frames(input_path: Path) -> int | None:
    """Tổng số khung để tính %. Ảnh = 1. Video lỗi / không đọc được metadata -> None (process_media tự báo lỗi)."""
    try:
        if video_io.media_type(input_path) == "image":
            return 1
        return video_io.get_video_info(input_path)["frame_count"] or None
    except Exception:
        return None
