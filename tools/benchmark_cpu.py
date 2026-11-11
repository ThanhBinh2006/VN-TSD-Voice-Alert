#!/usr/bin/env python3
"""Benchmark CPU, tracking và streaming; chỉ ghi số liệu đo được thực tế."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import platform
import queue
import sys
import tempfile
import threading
import time
import traceback
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_alert.config.resolve import get_num_classes, resolve_config
from ts_alert.pipelines import get_detector
from ts_alert.pipelines.faster_rcnn.infer import FasterRCNNDetector
from ts_alert.pipelines.faster_rcnn.model import build_model
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.postprocess.tts_worker import AsyncTTSWorker
from ts_alert.runtime.cpu_opt import (
    apply_faster_rcnn_cpu_tuning,
    apply_yolo_cpu_tuning,
    configure_cpu_threads,
    logical_core_count,
    physical_core_count,
    prepare_runtime_detector,
    probe_runtime_backends,
    restore_cpu_tuning,
)
from ts_alert.runtime.scheduler import FrameEnvelope, LatestFrameSlot
from ts_alert.runtime.tracker import (
    TrackerConfig,
    TrafficSignTracker,
    compute_flow_gray,
)

PROCESS = psutil.Process()
FPS = 30.0


def error_record(exc: BaseException) -> dict[str, Any]:
    """Chuyển lỗi thành bản ghi có thể lưu JSON."""
    return {
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def json_safe(value: Any) -> Any:
    """Chuẩn hóa dữ liệu và loại NaN/Infinity trước khi ghi JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def rss_mb() -> float:
    """Đọc RSS hiện tại của tiến trình benchmark."""
    return PROCESS.memory_info().rss / (1024**2)


def timing_summary(milliseconds: list[float]) -> dict[str, Any]:
    """Tổng hợp latency; FPS là throughput tuần tự tương đương."""
    if not milliseconds:
        return {
            "samples": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "fps": None,
        }
    values = np.asarray(milliseconds, dtype=np.float64)
    mean = float(values.mean())
    return {
        "samples": len(milliseconds),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": 1000.0 / mean if mean > 0 else None,
    }


def parse_args() -> argparse.Namespace:
    """Đọc và kiểm tra tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--micro-runs", type=int, default=8)
    parser.add_argument("--frcnn-runs", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/cpu_benchmark.json",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    for name in ("num_frames", "micro_runs", "frcnn_runs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} phải lớn hơn 0.")
    if args.quick:
        args.num_frames = min(args.num_frames, 60)
        args.micro_runs = min(args.micro_runs, 2)
        args.frcnn_runs = min(args.frcnn_runs, 1)
    return args


def synthetic_sequence(count: int = 45) -> tuple[list[np.ndarray], list[list[int]]]:
    """Tạo biển báo giàu texture dịch chuyển đúng 4 px/frame ở 30 FPS."""
    rng = np.random.default_rng(42)
    height, width = 360, 640
    background = rng.integers(45, 85, (height, width, 3), dtype=np.uint8)
    cv2.line(background, (80, height), (280, 110), (220, 220, 220), 4)
    cv2.line(background, (560, height), (360, 110), (220, 220, 220), 4)
    patch = rng.integers(80, 255, (80, 80, 3), dtype=np.uint8)
    cv2.rectangle(patch, (2, 2), (77, 77), (0, 0, 255), 5)
    cv2.putText(patch, "30", (13, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    frames, boxes = [], []
    for index in range(count):
        x, y = 80 + 4 * index, 120
        frame = background.copy()
        frame[y:y + 80, x:x + 80] = patch
        frames.append(frame)
        boxes.append([x, y, x + 80, y + 80])
    return frames, boxes


def find_video(explicit: Path | None) -> Path | None:
    """Chọn MP4 đầu tiên, ưu tiên tên kết thúc bằng test_video_1.mp4."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = sorted((ROOT / "outputs/uploads").glob("*.mp4"))
    preferred = [p for p in candidates if p.match("*test_video_1.mp4")]
    return next(iter(preferred or candidates), None)


def read_video(path: Path | None, count: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Đọc frame thực và luôn đóng VideoCapture."""
    if path is None:
        return [], {"status": "missing_video"}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return [], {"status": "unreadable_video", "path": str(path)}
        metadata = {
            "status": "ok",
            "path": str(path),
            "source_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        metadata["frames_loaded"] = len(frames)
        return frames, metadata
    finally:
        cap.release()


def hardware_probe() -> dict[str, Any]:
    """Kiểm tra import và khả năng tính toán CPU, không suy diễn tương thích."""
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "physical_cores": physical_core_count(),
        "logical_cores": logical_core_count(),
        "memory": psutil.virtual_memory()._asdict(),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "runtime_probe": None,
        "packages": {},
    }
    try:
        result["runtime_probe"] = probe_runtime_backends()
    except Exception as exc:
        result["runtime_probe"] = error_record(exc)

    for module_name in ("onnx", "onnxruntime", "openvino"):
        try:
            module = importlib.import_module(module_name)
            package = {
                "import_ok": True,
                "version": getattr(module, "__version__", None),
                "model_compatibility": "not_tested_until_local_export_is_loaded",
            }
            if module_name == "onnxruntime":
                package["available_providers"] = module.get_available_providers()
            if module_name == "openvino":
                package["available_devices"] = module.Core().available_devices
            result["packages"][module_name] = package
        except Exception as exc:
            result["packages"][module_name] = {
                "import_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        x = torch.ones((8, 8), device="cpu")
        result["pytorch_cpu_operation_ok"] = bool((x @ x).sum().item() == 512)
    except Exception as exc:
        result["pytorch_cpu_operation"] = error_record(exc)

    result["mkldnn"] = {
        "available": torch.backends.mkldnn.is_available(),
        "enabled": torch.backends.mkldnn.enabled,
    }
    if result["mkldnn"]["available"]:
        try:
            x = torch.ones((1, 3, 32, 32)).to_mkldnn()
            layer = torch.nn.Conv2d(3, 4, 3).eval()
            with torch.inference_mode():
                layer(x).to_dense()
            result["mkldnn"]["operation_ok"] = True
        except Exception as exc:
            result["mkldnn"]["operation"] = error_record(exc)
    return result


def cpu_config(name: str) -> dict[str, Any]:
    """Tải config và vô hiệu hóa tuning tự động khi nạp detector."""
    cfg = resolve_config(
        pipeline=name,
        runtime="local",
        overrides={
            "device": "cpu",
            "inference": {
                "device": "cpu",
                "cpu": {"apply_on_load": False, "backend": "pytorch"},
            },
        },
    )
    return copy.deepcopy(cfg)


def force_cpu(detector: Any) -> None:
    """Chuyển model PyTorch và thuộc tính device sang CPU."""
    detector.model.to(torch.device("cpu"))
    detector.model.eval()
    detector.device = torch.device("cpu")
    overrides = getattr(detector.model, "overrides", None)
    if isinstance(overrides, dict):
        overrides["device"] = "cpu"


def measure_detector(
    detector: Any,
    frames: list[np.ndarray],
    runs: int,
    warmup: int,
) -> dict[str, Any]:
    """Đo predict đồng bộ, loại warmup khỏi mẫu latency."""
    durations, counts, memory = [], [], []
    with torch.inference_mode():
        for index in range(warmup + runs):
            frame = frames[index % len(frames)]
            started = time.perf_counter()
            detections = detector.predict(frame)
            elapsed = (time.perf_counter() - started) * 1000
            if index >= warmup:
                durations.append(elapsed)
                counts.append(len(detections))
                memory.append(rss_mb())
    return {
        **timing_summary(durations),
        "rss_mb": max(memory) if memory else rss_mb(),
        "rss_measurement": "maximum_post_inference_sample",
        "avg_detections": float(np.mean(counts)) if counts else 0.0,
        "latency_samples_ms": durations,
        "warmup_runs": warmup,
    }


def benchmark_yolo(
    cfg: dict[str, Any],
    frames: list[np.ndarray],
    runs: int,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    """Đo từng cấu hình trên instance mới để tránh ảnh hưởng fuse không đảo được."""
    weights = ROOT / "models/yolo/best.pt"
    if not weights.is_file():
        return {"status": "missing_weights", "path": str(weights)}
    logical, physical = logical_core_count(), physical_core_count()
    profiles = {
        "baseline_640_t4": (640, False, logical),
        "balanced_416_t2": (416, True, physical),
        "fast_cpu_320_t2": (320, True, physical),
        "fast_cpu_320_t4": (320, True, logical),
    }
    result: dict[str, Any] = {
        "status": "ok",
        "weights": str(weights),
        "model_identity": "checkpoint_best.pt; tên YOLO26m chưa được xác minh",
        "profile_suffix_note": "t2/t4 là tên profile; số luồng thực nằm trong requested.",
        "profiles": {},
        "backends": {},
    }
    for name, (size, fuse, threads) in profiles.items():
        detector = None
        record: dict[str, Any] = {}
        try:
            detector = YoloDetector(weights, copy.deepcopy(cfg))
            force_cpu(detector)
            restore_cpu_tuning(detector)
            requested = {
                "preset": "default",
                "img_size": size,
                "fuse": fuse,
                "threads": threads,
                "backend": "pytorch",
            }
            applied = apply_yolo_cpu_tuning(detector, requested)
            record.update({
                "status": "ok",
                "requested": requested,
                "applied": applied,
                "actual_img_size": detector.img_size,
                "actual_torch_threads": torch.get_num_threads(),
                **measure_detector(detector, frames, runs, 2),
            })
        except Exception as exc:
            record.update(error_record(exc))
        finally:
            if detector is not None:
                try:
                    record["restore_result"] = restore_cpu_tuning(detector)
                except Exception as exc:
                    record["restore_error"] = str(exc)
        result["profiles"][name] = record

    export_dir = weights.parent
    exports = {
        "onnx": sorted(export_dir.glob("*.onnx")),
        "openvino": sorted(export_dir.rglob("*.xml")),
    }
    for backend, paths in exports.items():
        package = "onnxruntime" if backend == "onnx" else "openvino"
        available = hardware["packages"].get(package, {}).get("import_ok", False)
        if not available or not paths:
            result["backends"][backend] = {
                "status": "skipped_not_installed_or_no_export",
                "reason": {
                    "runtime": hardware["packages"].get(package),
                    "local_exports": [str(p) for p in paths],
                    "missing_runtime": not available,
                    "missing_local_export": not bool(paths),
                },
                "reason_source": "measured_local_preconditions",
                "prepare_runtime_detector": (
                    "not_called: tránh tự export khi chưa có runtime/export cục bộ"
                ),
            }
            continue

        detector = runtime_detector = None
        record = {}
        try:
            detector = YoloDetector(weights, copy.deepcopy(cfg))
            force_cpu(detector)
            restore_cpu_tuning(detector)
            configure_cpu_threads(physical)
            runtime_detector, preparation = prepare_runtime_detector(
                "yolo",
                detector,
                cfg=copy.deepcopy(cfg),
                backend=backend,
                preset="fast_cpu",
                export_dir=export_dir,
            )
            record["prepare_runtime_detector"] = preparation
            record["local_exports"] = [str(p) for p in paths]
            # Không gán nhãn ONNX/OpenVINO nếu API thực tế trả về fallback.
            if isinstance(runtime_detector, YoloDetector):
                record["status"] = "skipped_not_installed_or_no_export"
                record["reason"] = "prepare_runtime_detector_returned_pytorch_detector"
            else:
                record.update({
                    "status": "ok",
                    "runtime_class": type(runtime_detector).__name__,
                    **measure_detector(runtime_detector, frames, runs, 2),
                })
        except Exception as exc:
            record.update(error_record(exc))
            record["status"] = "skipped_not_installed_or_no_export"
        finally:
            for instance in (runtime_detector, detector):
                if instance is not None:
                    try:
                        restore_cpu_tuning(instance)
                    except Exception as exc:
                        record.setdefault("cleanup_errors", []).append(str(exc))
        result["backends"][backend] = record
    return result


def actual_frcnn_parameters(detector: Any) -> dict[str, Any]:
    """Đọc trực tiếp transform và giới hạn proposal của model đã tuning."""
    model = detector.model
    return {
        "min_size": model.transform.min_size,
        "max_size": model.transform.max_size,
        "rpn_pre_nms_top_n_test": model.rpn.pre_nms_top_n(),
        "rpn_post_nms_top_n_test": model.rpn.post_nms_top_n(),
        "box_detections_per_img": model.roi_heads.detections_per_img,
        "rpn_nms_thresh": model.rpn.nms_thresh,
        "box_nms_thresh": model.roi_heads.nms_thresh,
        "box_score_thresh": model.roi_heads.score_thresh,
        "torch_threads": torch.get_num_threads(),
        "device": str(next(model.parameters()).device),
    }


def benchmark_frcnn(
    cfg: dict[str, Any], frames: list[np.ndarray], runs: int
) -> dict[str, Any]:
    """Đo kiến trúc Faster R-CNN; khai báo rõ checkpoint chưa huấn luyện."""
    weights = ROOT / "models/faster_rcnn/best.pth"
    if weights.is_file():
        detector = get_detector("faster_rcnn", cfg)
        if Path(detector.weights_path).resolve() != weights.resolve():
            detector = FasterRCNNDetector(weights, cfg)
        weights_status = "trained_checkpoint_loaded"
    else:
        weights_status = "missing_best_pth_using_untrained_architecture"
        print(f"[Faster R-CNN] {weights_status}", flush=True)
        detector = object.__new__(FasterRCNNDetector)
        detector._predict_lock = threading.RLock()
        detector.weights_path = weights
        detector.device = torch.device("cpu")
        detector.classes = {
            index: {"name": str(index)} for index in range(get_num_classes(cfg))
        }
        detector.conf, detector.iou = 0.25, 0.5
        # build_model nhận tổng số lớp, gồm một lớp background.
        detector.model = build_model(
            get_num_classes(cfg) + 1, pretrained=False
        ).eval()
    force_cpu(detector)
    logical, physical = logical_core_count(), physical_core_count()
    presets = {
        "default": (800, 1333, 1000, 1000, 100, logical),
        "balanced": (480, 640, 300, 150, 50, physical),
        "fast_cpu": (320, 480, 150, 80, 30, physical),
    }
    result: dict[str, Any] = {
        "status": "ok",
        "weights_status": weights_status,
        "profiles": {},
        "accuracy_claim": "none; benchmark này chỉ đo hiệu năng",
    }
    try:
        for name, values in presets.items():
            record: dict[str, Any] = {}
            try:
                restore_cpu_tuning(detector)
                minimum, maximum, pre, post, detections, threads = values
                requested = {
                    "preset": name,
                    "min_size": minimum,
                    "max_size": maximum,
                    "rpn_pre_nms_top_n_test": pre,
                    "rpn_post_nms_top_n_test": post,
                    "box_detections_per_img": detections,
                    "threads": threads,
                }
                record.update({
                    "status": "ok",
                    "requested": requested,
                    "applied": apply_faster_rcnn_cpu_tuning(detector, requested),
                    "actual": actual_frcnn_parameters(detector),
                    **measure_detector(detector, frames, runs, 1),
                })
            except Exception as exc:
                record.update(error_record(exc))
            finally:
                record["restore_result"] = restore_cpu_tuning(detector)
            result["profiles"][name] = record
        baseline = result["profiles"]["default"].get("mean_ms")
        for record in result["profiles"].values():
            latency = record.get("mean_ms")
            record["speedup_vs_default"] = (
                baseline / latency if baseline and latency else None
            )
        return result
    finally:
        restore_cpu_tuning(detector)


def detection(box: list[float] | np.ndarray, class_id: int = 0) -> dict[str, Any]:
    """Tạo detection có confidence cao cho thí nghiệm ground truth mô phỏng."""
    return {
        "bbox": list(map(float, box)),
        "class_id": class_id,
        "confidence": 0.99,
        "score": 0.99,
        "label": str(class_id),
    }


def bbox(item: dict[str, Any]) -> np.ndarray | None:
    """Đọc bbox xyxy; loại box rỗng hoặc không hữu hạn."""
    value = item.get("bbox", item.get("box"))
    if value is None:
        return None
    box = np.asarray(value, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Tính IoU hai box xyxy."""
    size = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


def accuracy(
    predictions: list[list[dict[str, Any]]],
    references: list[list[dict[str, Any]]],
    frame_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Ghép một-một cùng class; box mất được tính IoU=0."""
    ious, centers = [], []
    retained = total = missing = 0
    indices = frame_ids if frame_ids is not None else list(range(len(references)))
    for index in indices:
        available = [
            (item, bbox(item)) for item in predictions[index]
            if bbox(item) is not None
        ]
        for reference in references[index]:
            target = bbox(reference)
            if target is None:
                continue
            total += 1
            candidates = [
                (position, box_iou(target, box))
                for position, (item, box) in enumerate(available)
                if item.get("class_id", 0) == reference.get("class_id", 0)
            ]
            if not candidates:
                ious.append(0.0)
                missing += 1
                continue
            position, overlap = max(candidates, key=lambda pair: pair[1])
            _, predicted = available.pop(position)
            ious.append(overlap)
            centers.append(float(np.linalg.norm(
                (predicted[:2] + predicted[2:] - target[:2] - target[2:]) / 2
            )))
            retained += int(overlap >= 0.5)
    return {
        "mean_iou": float(np.mean(ious)) if ious else None,
        "center_error_px": float(np.mean(centers)) if centers else None,
        "center_error_scope": "matched_boxes_only",
        "retention_percent": 100.0 * retained / total if total else None,
        "retention_definition": "same-class match with IoU >= 0.5",
        "reference_boxes": total,
        "missing_boxes": missing,
        "evaluated_frames": len(indices),
    }


def make_tracker() -> TrafficSignTracker:
    """Tạo tracker với optical flow và xác nhận ngay detection rõ ràng."""
    return TrafficSignTracker(
        config=TrackerConfig(
            min_hits_to_confirm=1,
            single_hit_conf_threshold=0.60,
            use_optical_flow=True,
        ),
        classes={0: "benchmark_sign"},
    )


def tracker_step(
    tracker: TrafficSignTracker, frame: np.ndarray, index: int
) -> list[dict[str, Any]]:
    """Cập nhật một frame theo đồng hồ media 30 FPS."""
    height, width = frame.shape[:2]
    return tracker.step_frame(
        frame,
        frame_id=index,
        timestamp_sec=index / FPS,
        frame_width=width,
        frame_height=height,
        include_tentative=True,
    )


def tracker_update(
    tracker: TrafficSignTracker,
    detections: list[dict[str, Any]],
    source_index: int,
    current_index: int,
    source_gray: tuple[np.ndarray, float],
    shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Truyền đúng timestamp nguồn và frame hiện tại để bù độ trễ."""
    return tracker.update_from_detection(
        detections,
        detection_frame_id=source_index,
        detection_timestamp=source_index / FPS,
        current_frame_id=current_index,
        current_timestamp=current_index / FPS,
        source_gray=source_gray,
        frame_width=shape[1],
        frame_height=shape[0],
        wall_lag_ms=(current_index - source_index) * 1000 / FPS,
    )


def benchmark_tracking(frames: list[np.ndarray], real_video: bool) -> dict[str, Any]:
    """Đo overhead có track active, accuracy mô phỏng và detection trễ."""
    tracker = make_tracker()
    durations, active = [], []
    for index, frame in enumerate(frames):
        started = time.perf_counter()
        gray = compute_flow_gray(frame)
        tracks = tracker_step(tracker, frame, index)
        durations.append((time.perf_counter() - started) * 1000)
        if index % 15 == 0:
            height, width = frame.shape[:2]
            tracks = tracker_update(
                tracker,
                [detection([width * .35, height * .30, width * .55, height * .60])],
                index, index, gray, frame.shape,
            )
        active.append(bool(tracks))
    active_times = [value for value, present in zip(durations, active) if present]

    sequence, boxes = synthetic_sequence()
    references = [[detection(box)] for box in boxes]
    optical, delayed = make_tracker(), make_tracker()
    predictions: dict[str, list[list[dict[str, Any]]]] = {
        "naive_hold": [],
        "optical_flow_tracker": [],
        "delayed_naive_hold": [],
        "delayed_compensated_tracker": [],
    }
    held: list[dict[str, Any]] = []
    delayed_held: list[dict[str, Any]] = []
    source_grays: dict[int, tuple[np.ndarray, float]] = {}
    delay_frames = math.ceil(.250 * FPS)
    for index, frame in enumerate(sequence):
        optical_tracks = tracker_step(optical, frame, index)
        delayed_tracks = tracker_step(delayed, frame, index)
        if index % 15 == 0:
            source_grays[index] = compute_flow_gray(frame)
            held = references[index]
            optical_tracks = tracker_update(
                optical, held, index, index, source_grays[index], frame.shape
            )
        source_index = index - delay_frames
        if source_index >= 0 and source_index % 15 == 0:
            delayed_held = references[source_index]
            delayed_tracks = tracker_update(
                delayed,
                delayed_held,
                source_index,
                index,
                source_grays[source_index],
                frame.shape,
            )
        for name, values in (
            ("naive_hold", held),
            ("optical_flow_tracker", optical_tracks),
            ("delayed_naive_hold", delayed_held),
            ("delayed_compensated_tracker", delayed_tracks),
        ):
            predictions[name].append(copy.deepcopy(values))

    active_summary = timing_summary(active_times)
    return {
        "status": "ok",
        "overhead": {
            **active_summary,
            "max_fps": active_summary["fps"],
            "max_fps_definition": "1000 / mean_ms; không phải FPS hiển thị",
            "source": "real_video" if real_video else "synthetic_fallback",
            "active_frames": sum(active),
            "total_frames": len(frames),
            "scope": "compute_flow_gray + step_frame; không tính update_from_detection",
            "seed_detection": "synthetic ROI; không khẳng định là biển báo thật",
            "all_frames": timing_summary(durations),
            "tracker_stats": tracker.get_stats(),
        },
        "synthetic_accuracy": {
            "fps": FPS,
            "frames": 45,
            "translation_px_per_frame": 4,
            "detection_stride": 15,
            "requested_delay_ms": 250,
            "actual_delivery_frame": delay_frames,
            "actual_delivery_ms": delay_frames * 1000 / FPS,
            "methods": {
                name: accuracy(values, references)
                for name, values in predictions.items()
            },
            "delay_comparison_after_first_delivery": {
                name: accuracy(values, references, list(range(delay_frames, 45)))
                for name, values in predictions.items()
                if name.startswith("delayed_")
            },
        },
    }


def benchmark_real_accuracy(
    cfg: dict[str, Any], path: Path | None, count: int, quick: bool
) -> dict[str, Any]:
    """Tìm đoạn có detection; dùng YOLO mỗi frame làm pseudo-reference."""
    weights = ROOT / "models/yolo/best.pt"
    if path is None or not weights.is_file():
        return {"status": "skipped_missing_video_or_weights"}
    detector = None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"status": "skipped_unreadable_video"}
        detector = YoloDetector(weights, copy.deepcopy(cfg))
        force_cpu(detector)
        restore_cpu_tuning(detector)
        apply_yolo_cpu_tuning(detector, {
            "preset": "default",
            "img_size": 640,
            "fuse": False,
            "threads": physical_core_count(),
            "backend": "pytorch",
        })
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return {"status": "skipped_unknown_video_length"}
        scan_count = min(total, 8 if quick else 40)
        positions = sorted(set(np.linspace(0, total - 1, scan_count, dtype=int).tolist()))
        start = None
        scanned = []
        with torch.inference_mode():
            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                if not ok:
                    continue
                detections = detector.predict(frame, img_size=640)
                clear = [
                    item for item in detections
                    if float(item.get("confidence", item.get("score", 0))) >= .60
                    and bbox(item) is not None
                ]
                scanned.append({"frame": position, "clear_detections": len(clear)})
                if clear:
                    start = position
                    break
            if start is None:
                return {
                    "status": "skipped_no_clear_detection_in_scanned_frames",
                    "scan": scanned,
                    "search_is_exhaustive": len(positions) == total,
                }
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            sequence, references = [], []
            for _ in range(count):
                ok, frame = cap.read()
                if not ok:
                    break
                sequence.append(frame)
                references.append(detector.predict(frame, img_size=640))
        tracker = make_tracker()
        held: list[dict[str, Any]] = []
        naive, hybrid = [], []
        for index, frame in enumerate(sequence):
            tracks = tracker_step(tracker, frame, index)
            if index % 10 == 0:
                held = references[index]
                tracks = tracker_update(
                    tracker, held, index, index, compute_flow_gray(frame), frame.shape
                )
            naive.append(copy.deepcopy(held))
            hybrid.append(copy.deepcopy(tracks))
        # Bỏ frame refresh để tránh IoU=1 tại frame detect làm đẹp số liệu.
        interpolation_ids = [i for i in range(len(sequence)) if i % 10 != 0]
        return {
            "status": "ok",
            "reference": "YOLO imgsz=640 stride=1; pseudo-reference, không phải nhãn GT",
            "scan": scanned,
            "start_frame": start,
            "frames": len(sequence),
            "detection_stride": 10,
            "naive_hold": accuracy(naive, references),
            "hybrid_optical_flow": accuracy(hybrid, references),
            "between_detections": {
                "naive_hold": accuracy(naive, references, interpolation_ids),
                "hybrid_optical_flow": accuracy(hybrid, references, interpolation_ids),
            },
        }
    finally:
        cap.release()
        if detector is not None:
            restore_cpu_tuning(detector)


class FakeSynthesizer:
    """Callable offline trả WAV thật; benchmark overload cấu hình delay 1.2 s/lượt."""

    def __init__(self, directory: Path, delay_sec: float = 0.005) -> None:
        self.directory = directory
        self.delay_sec = delay_sec
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Hỗ trợ adapter truyền text/config bằng positional hoặc keyword."""
        with self.lock:
            self.calls += 1
            number = self.calls
        time.sleep(self.delay_sec)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"fake_{id(self)}_{number}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 800)
        return str(path)

    def synthesize(self, *args: Any, **kwargs: Any) -> str:
        """Adapter cho worker gọi phương thức synthesize."""
        return self(*args, **kwargs)


def tts_ratio(stats: dict[str, Any], synthesizer: FakeSynthesizer) -> float | None:
    """Chỉ công bố late_ratio khi worker thực sự gọi synthesizer."""
    if not synthesizer.calls:
        return None
    value = stats.get("late_ratio")
    return float(value) if value is not None else None


def benchmark_tts_overload(cfg: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Gây timeout 0.3 s với synthesizer mất 1.2 s để đo tỷ lệ cảnh báo trễ."""
    synthesizer = FakeSynthesizer(directory, delay_sec=1.2)
    worker = AsyncTTSWorker(
        cfg=copy.deepcopy(cfg),
        out_dir=directory,
        synthesizer=synthesizer,
        max_queue_size=6,
        cooldown_sec=0.0,
        timeout_sec=0.3,
        max_alert_age_sec=0.4,
    )
    submissions = []
    try:
        for index in range(12):
            submissions.append(worker.submit(
                f"Kiểm tra nghẽn {index}",
                class_id=index,
                label="benchmark",
                track_id=10000 + index,
                frame_id=index,
                source_timestamp=index / FPS,
            ))
        # Chờ cả tác vụ đang chạy kết thúc trước khi dọn thư mục tạm.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            time.sleep(0.05)
        observed = worker.get_stats()
    finally:
        worker.shutdown(wait=True, cancel_pending=True)
    final = worker.get_stats()
    return {
        "status": "ok" if synthesizer.calls else "synthesizer_not_invoked",
        "simulated_synthesis_ms": 1200,
        "timeout_sec": 0.3,
        "max_alert_age_sec": 0.4,
        "synthesizer_calls": synthesizer.calls,
        "submissions": submissions,
        "stats_before_shutdown": observed,
        "stats": final,
        "late_ratio": tts_ratio(final, synthesizer),
        "note": "Timeout và pre/post stale đều tăng late_alerts khi nghẽn TTS.",
    }


class FallbackDetector:
    """Detector giả có latency thật 50 ms; chỉ dùng khi không có checkpoint."""

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Trả ROI mô phỏng, không dùng để kết luận chất lượng YOLO."""
        time.sleep(0.05)
        height, width = frame.shape[:2]
        return [detection([width * .35, height * .3, width * .55, height * .6])]


def benchmark_streaming(
    cfg: dict[str, Any],
    frames: list[np.ndarray],
    count: int,
    real_video: bool,
    directory: Path,
) -> dict[str, Any]:
    """Chạy display, inference, tracking và TTS bằng các luồng độc lập."""
    weights = ROOT / "models/yolo/best.pt"
    detector: Any = None
    worker: AsyncTTSWorker | None = None
    slot = LatestFrameSlot()
    tracking_frames: queue.Queue[FrameEnvelope | None] = queue.Queue(maxsize=1)
    results: queue.Queue[dict[str, Any]] = queue.Queue()
    lock = threading.Lock()
    overlay: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    inference_times: list[float] = []
    tracking_times: list[float] = []
    display_times: list[float] = []
    latency_ms: list[float] = []
    raw_age_ms: list[float] = []
    compensated_age_ms: list[float] = []
    compensation_horizon_ms: list[float] = []
    overlay_updates = tracking_drops = 0
    resource_samples: list[dict[str, float]] = []
    stop_sampling = threading.Event()
    inference_done = threading.Event()
    clock_state = {"media": 0.0}
    synthesizer = FakeSynthesizer(directory)
    threads: list[threading.Thread] = []

    def media_clock() -> float:
        """Đọc timestamp media mới nhất."""
        return clock_state["media"]

    def infer_loop() -> None:
        """Chỉ lấy frame mới nhất; không khóa luồng display."""
        try:
            while True:
                envelope = slot.pop_latest(timeout=0.1)
                if envelope is None:
                    if capture_done.is_set():
                        break
                    continue
                started = time.perf_counter()
                with torch.inference_mode():
                    detections = detector.predict(envelope.frame)
                finished = time.perf_counter()
                inference_times.append((finished - started) * 1000)
                if envelope.gray_pair is None:
                    envelope.gray_pair = compute_flow_gray(envelope.frame)
                results.put({
                    "envelope": envelope,
                    "detections": detections,
                    "finished": finished,
                })
        except Exception as exc:
            errors.append({"thread": "inference", **error_record(exc)})
        finally:
            inference_done.set()

    def tracking_loop() -> None:
        """Sở hữu tracker duy nhất; áp kết quả inference bất đồng bộ."""
        nonlocal overlay, overlay_updates
        tracker = make_tracker()
        last_detection: dict[str, Any] | None = None
        try:
            while True:
                envelope = tracking_frames.get()
                if envelope is None:
                    break
                started = time.perf_counter()
                tracks = tracker.step_frame(
                    envelope.frame,
                    frame_id=envelope.frame_id,
                    timestamp_sec=envelope.source_timestamp,
                    frame_width=envelope.width,
                    frame_height=envelope.height,
                    include_tentative=True,
                )
                newest = None
                while True:
                    try:
                        candidate = results.get_nowait()
                    except queue.Empty:
                        break
                    if candidate["envelope"].frame_id <= envelope.frame_id:
                        newest = candidate
                    else:
                        results.put(candidate)
                        break
                if newest is not None:
                    source = newest["envelope"]
                    tracks = tracker.update_from_detection(
                        newest["detections"],
                        detection_frame_id=source.frame_id,
                        detection_timestamp=source.source_timestamp,
                        current_frame_id=envelope.frame_id,
                        current_timestamp=envelope.source_timestamp,
                        source_gray=source.gray_pair,
                        frame_width=envelope.width,
                        frame_height=envelope.height,
                        wall_lag_ms=(
                            time.perf_counter() - source.captured_wall_time
                        ) * 1000,
                    )
                    last_detection = newest
                    if newest["detections"]:
                        worker.submit(
                            f"Biển báo tại khung {source.frame_id}",
                            class_id=0,
                            label="benchmark",
                            track_id=source.frame_id,
                            frame_id=source.frame_id,
                            source_timestamp=source.source_timestamp,
                        )
                tracking_times.append((time.perf_counter() - started) * 1000)
                with lock:
                    overlay = {
                        "frame_id": envelope.frame_id,
                        "timestamp": envelope.source_timestamp,
                        "tracks": copy.deepcopy(tracks),
                        "detection": last_detection,
                    }
                    overlay_updates += 1
        except Exception as exc:
            errors.append({"thread": "tracking", **error_record(exc)})

    def sample_resources() -> None:
        """Lấy mẫu RSS và CPU tiến trình, CPU có thể vượt 100% đa lõi."""
        PROCESS.cpu_percent(interval=None)
        while not stop_sampling.wait(0.1):
            resource_samples.append({
                "rss_mb": rss_mb(),
                "cpu_percent": PROCESS.cpu_percent(interval=None),
            })

    capture_done = threading.Event()
    streaming_started = cpu_started = 0.0
    display_finished = streaming_finished = 0.0
    worker_stats: dict[str, Any] = {}
    try:
        configure_cpu_threads(physical_core_count())
        if weights.is_file():
            detector = YoloDetector(weights, copy.deepcopy(cfg))
            force_cpu(detector)
            restore_cpu_tuning(detector)
            tuning = apply_yolo_cpu_tuning(detector, {
                "preset": "fast_cpu",
                "img_size": 320,
                "fuse": True,
                "threads": physical_core_count(),
                "backend": "pytorch",
            })
            # Warmup ngoài khoảng đo streaming.
            with torch.inference_mode():
                detector.predict(frames[0])
            detector_status = "real_yolo_checkpoint"
        else:
            detector = FallbackDetector()
            detector_status = "fake_detector_missing_weights"
            tuning = None

        worker = AsyncTTSWorker(
            cfg=copy.deepcopy(cfg),
            out_dir=directory,
            synthesizer=synthesizer,
            max_queue_size=6,
            cooldown_sec=0.0,
            timeout_sec=3.0,
            max_alert_age_sec=2.5,
            media_clock_fn=media_clock,
            max_media_lag_sec=2.5,
        )
        threads = [
            threading.Thread(target=infer_loop, name="benchmark-inference"),
            threading.Thread(target=tracking_loop, name="benchmark-tracking"),
            threading.Thread(target=sample_resources, name="benchmark-resources"),
        ]
        cpu_times = PROCESS.cpu_times()
        cpu_started = cpu_times.user + cpu_times.system
        streaming_started = time.perf_counter()
        for thread in threads:
            thread.start()

        seen_detection_ids: set[int] = set()
        for index in range(count):
            deadline = streaming_started + index / FPS
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            captured = time.perf_counter()
            frame = frames[index % len(frames)]
            height, width = frame.shape[:2]
            clock_state["media"] = index / FPS
            # Gray do luồng tracking/inference sử dụng; display không đợi model.
            envelope = FrameEnvelope(
                frame_id=index,
                source_timestamp=index / FPS,
                captured_wall_time=captured,
                frame=frame,
                width=width,
                height=height,
            )
            slot.push(envelope)
            try:
                tracking_frames.put_nowait(envelope)
            except queue.Full:
                try:
                    tracking_frames.get_nowait()
                    tracking_drops += 1
                except queue.Empty:
                    pass
                try:
                    tracking_frames.put_nowait(envelope)
                except queue.Full:
                    tracking_drops += 1

            with lock:
                current = overlay.copy()
            canvas = frame.copy()
            for item in current.get("tracks", []):
                box = bbox(item)
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            displayed = time.perf_counter()
            display_times.append(displayed)
            source_result = current.get("detection")
            if source_result is not None:
                source = source_result["envelope"]
                if source.frame_id not in seen_detection_ids:
                    seen_detection_ids.add(source.frame_id)
                    latency_ms.append(
                        (displayed - source.captured_wall_time) * 1000
                    )
                    raw_age_ms.append((index / FPS - source.source_timestamp) * 1000)
                    compensated_age_ms.append(
                        (index / FPS - current["timestamp"]) * 1000
                    )
                    compensation_horizon_ms.append(
                        (current["timestamp"] - source.source_timestamp) * 1000
                    )
        display_finished = time.perf_counter()
    finally:
        capture_done.set()
        slot.close()
        if threads:
            threads[0].join()
            # Đóng tracking sau inference; các kết quả chưa được vẽ không tính latency.
            tracking_frames.put(None)
            threads[1].join()
        if worker is not None:
            try:
                worker.shutdown(wait=True, cancel_pending=True)
            finally:
                worker_stats = worker.get_stats()
        stop_sampling.set()
        if len(threads) > 2:
            threads[2].join()
        streaming_finished = time.perf_counter()
        if detector is not None and not isinstance(detector, FallbackDetector):
            restore_cpu_tuning(detector)

    playback_duration = (
        display_times[-1] - display_times[0] if len(display_times) > 1 else 0
    )
    elapsed = streaming_finished - streaming_started
    final_cpu = PROCESS.cpu_times()
    cpu_seconds = final_cpu.user + final_cpu.system - cpu_started
    displayed_duration = display_finished - streaming_started
    return {
        "status": "ok" if not errors else "partial",
        "source": "real_video" if real_video else "synthetic_fallback",
        "real_video_requirement_met": real_video and count >= 60,
        "display_mode": "headless: copy frame + draw overlay; không gồm GUI/vsync",
        "source_frames_loaded": len(frames),
        "playback_frames": len(display_times),
        "repeated_source_frames": max(0, count - len(frames)),
        "target_fps": FPS,
        "detector": detector_status,
        "tuning": tuning,
        "video_playback_fps": (
            (len(display_times) - 1) / playback_duration if playback_duration > 0 else None
        ),
        "inference_fps": len(inference_times) / elapsed if elapsed > 0 else None,
        "inference_service": timing_summary(inference_times),
        "overlay_step_fps": (
            overlay_updates / displayed_duration if displayed_duration > 0 else None
        ),
        "overlay_service": timing_summary(tracking_times),
        "detection_to_display_latency_ms": timing_summary(latency_ms),
        "timestamp_age_before_compensation_ms": timing_summary(raw_age_ms),
        "timestamp_age_after_compensation_ms": timing_summary(compensated_age_ms),
        "motion_compensation_horizon_ms": timing_summary(compensation_horizon_ms),
        "compensation_note": (
            "Bù box tới timestamp tracking; không làm giảm thời gian inference. "
            "Sai số hình học được đo riêng trong tracking accuracy."
        ),
        "frame_drop_ratio": slot.get_stats().get("drop_ratio"),
        "latest_frame_slot": slot.get_stats(),
        "tracking_frame_drops": tracking_drops,
        "results_never_displayed": len(inference_times) - len(latency_ms),
        "peak_rss_mb": max(
            [sample["rss_mb"] for sample in resource_samples] + [rss_mb()]
        ),
        "peak_rss_definition": "sampled process RSS every 100 ms",
        "process_cpu_percent": 100 * cpu_seconds / elapsed if elapsed > 0 else None,
        "cpu_percent_definition": "process CPU seconds / wall seconds; đa lõi có thể >100%",
        "process_cpu_percent_samples": resource_samples,
        "measurement_duration_sec": elapsed,
        "tts_stats_normal": worker_stats,
        "tts_synthesizer_calls": synthesizer.calls,
        "tts_late_ratio_normal": tts_ratio(worker_stats, synthesizer),
        "errors": errors,
    }


def guarded(function: Callable[..., Any], *args: Any) -> dict[str, Any]:
    """Cô lập lỗi từng phần để vẫn ghi được báo cáo cuối."""
    try:
        return function(*args)
    except Exception as exc:
        print(f"[Lỗi] {function.__name__}: {exc}", flush=True)
        return error_record(exc)


def print_summary(report: dict[str, Any]) -> None:
    """In bảng kết quả ngắn gọn bằng tiếng Việt."""
    def number(value: Any) -> str:
        return f"{value:.2f}" if isinstance(value, (float, int)) else "N/A"

    print("\nKẾT QUẢ BENCHMARK CPU")
    print(f"{'Mô hình / cấu hình':<36} {'ms/lượt':>12} {'FPS':>12} {'RSS MB':>12}")
    print("-" * 76)
    for key, title in (("yolo", "YOLO"), ("faster_rcnn", "FRCNN")):
        section = report.get(key, {})
        if not section.get("profiles"):
            print(f"{title:<36} {section.get('status', 'N/A')}")
        for name, row in section.get("profiles", {}).items():
            print(
                f"{title + '/' + name:<36}"
                f"{number(row.get('mean_ms')):>12}"
                f"{number(row.get('fps')):>12}"
                f"{number(row.get('rss_mb')):>12}"
            )
    streaming = report.get("streaming", {})
    print("\nStreaming:")
    for key in (
        "video_playback_fps", "inference_fps", "overlay_step_fps",
        "frame_drop_ratio", "peak_rss_mb", "process_cpu_percent",
        "tts_late_ratio_normal", "tts_late_ratio_overload",
    ):
        print(f"  {key:<34} {number(streaming.get(key))}")
    methods = report.get("tracking", {}).get("synthetic_accuracy", {}).get("methods", {})
    print("\nTracking mô phỏng:")
    print(f"{'Phương pháp':<36} {'Mean IoU':>12} {'Lỗi tâm px':>12} {'Giữ track %':>12}")
    for name, row in methods.items():
        print(
            f"{name:<36}"
            f"{number(row.get('mean_iou')):>12}"
            f"{number(row.get('center_error_px')):>12}"
            f"{number(row.get('retention_percent')):>12}"
        )


def main() -> int:
    """Chạy các phép đo, phục hồi CPU settings và lưu JSON hợp lệ."""
    args = parse_args()
    old_threads, old_cv_threads = torch.get_num_threads(), cv2.getNumThreads()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": vars(args),
        "limitations": [
            "Pseudo-reference YOLO không thay thế ground truth gán nhãn.",
            "RSS và CPU bao gồm tiến trình benchmark, không chỉ model.",
            "Synthetic fallback không thỏa yêu cầu kiểm chứng bằng video thật.",
            "Smoke test --quick dùng ít mẫu; không dùng để kết luận hiệu năng.",
        ],
    }
    exit_code = 0
    try:
        report["hardware"] = guarded(hardware_probe)
        video = find_video(args.video)
        frames, report["video"] = read_video(video, args.num_frames)
        real_video = bool(frames)
        if not frames:
            frames, _ = synthetic_sequence()
            print("[Dữ liệu] Không đọc được video; dùng ảnh mô phỏng.", flush=True)

        yolo_cfg = cpu_config("yolo")
        frcnn_cfg = cpu_config("faster_rcnn")
        print("[1/6] Benchmark YOLO CPU...", flush=True)
        report["yolo"] = guarded(
            benchmark_yolo, yolo_cfg, frames, args.micro_runs, report["hardware"]
        )
        print("[2/6] Benchmark Faster R-CNN CPU...", flush=True)
        report["faster_rcnn"] = guarded(
            benchmark_frcnn, frcnn_cfg, frames, args.frcnn_runs
        )
        print("[3/6] Đo optical flow, tracking và bù detection trễ...", flush=True)
        report["tracking"] = guarded(benchmark_tracking, frames, real_video)
        print("[4/6] Tìm đoạn video có detection và đo pseudo-accuracy...", flush=True)
        report["real_video_accuracy"] = guarded(
            benchmark_real_accuracy,
            yolo_cfg,
            video,
            min(args.num_frames, 10) if args.quick else args.num_frames,
            args.quick,
        )
        with tempfile.TemporaryDirectory(prefix="ts_cpu_benchmark_") as temporary:
            directory = Path(temporary)
            print("[5/6] Đo playback, inference, tracking và TTS độc lập...", flush=True)
            report["streaming"] = guarded(
                benchmark_streaming,
                yolo_cfg,
                frames,
                max(60, args.num_frames),
                real_video,
                directory,
            )
            print("[6/6] Đo TTS overload: 1200 ms, timeout 0.3 s...", flush=True)
            report["tts_overload"] = guarded(
                benchmark_tts_overload, yolo_cfg, directory
            )
            report["streaming"]["tts_late_ratio_overload"] = (
                report["tts_overload"].get("late_ratio")
            )
    except Exception as exc:
        report["fatal_error"] = error_record(exc)
        exit_code = 1
    finally:
        try:
            torch.set_num_threads(old_threads)
            cv2.setNumThreads(old_cv_threads)
            report["global_thread_settings_restored"] = True
        except Exception as exc:
            report["global_thread_restore_error"] = str(exc)
        report = json_safe(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        print_summary(report)
        print(f"\nĐã lưu báo cáo: {args.output.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
