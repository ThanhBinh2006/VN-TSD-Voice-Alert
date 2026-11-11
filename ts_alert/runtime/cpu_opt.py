"""Tinh chỉnh CPU và adapter suy luận YOLO bằng ONNX Runtime/OpenVINO.

Không tải trọng số hoặc import bắt buộc các backend tăng tốc. Các thay đổi
profile được tuần tự hóa trong module; caller phải dừng suy luận đang chạy
trước khi thay đổi hoặc khôi phục cấu hình của cùng detector.
"""

from __future__ import annotations

import importlib
import math
import os
import threading
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ts_alert.pipelines.base import BaseDetector
from ts_alert.types import Detection

_LOCK = threading.RLock()

CPU_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "yolo": {
        "default": {"img_size": 640, "fuse": False, "threads": None},
        "balanced": {"img_size": 416, "fuse": True, "threads": None},
        "fast_cpu": {"img_size": 320, "fuse": True, "threads": None},
    },
    "faster_rcnn": {
        "default": {
            "min_size": 800,
            "max_size": 1333,
            "rpn_pre_nms_top_n_test": 1000,
            "rpn_post_nms_top_n_test": 1000,
            "box_detections_per_img": 100,
            "threads": None,
        },
        "balanced": {
            "min_size": 480,
            "max_size": 640,
            "rpn_pre_nms_top_n_test": 300,
            "rpn_post_nms_top_n_test": 150,
            "box_detections_per_img": 50,
            "threads": None,
        },
        "fast_cpu": {
            "min_size": 320,
            "max_size": 480,
            "rpn_pre_nms_top_n_test": 150,
            "rpn_post_nms_top_n_test": 80,
            "box_detections_per_img": 30,
            "threads": None,
        },
    },
}
for _presets in CPU_PRESETS.values():
    _presets["realtime"] = _presets["fast_cpu"]

_BACKENDS = {"auto", "pytorch", "onnx", "openvino"}
_OPTION_KEYS = {
    "img_size",
    "min_size",
    "max_size",
    "rpn_pre_nms_top_n_test",
    "rpn_post_nms_top_n_test",
    "box_detections_per_img",
    "threads",
    "fuse",
    "backend",
}


def logical_core_count() -> int:
    """Trả về số CPU logic, luôn ít nhất một."""
    return max(1, os.cpu_count() or 2)


def physical_core_count() -> int:
    """Phát hiện nhân vật lý bằng psutil, fallback một nửa CPU logic."""
    try:
        psutil = importlib.import_module("psutil")
        count = psutil.cpu_count(logical=False)
        if count is not None and int(count) >= 1:
            return int(count)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    """Kiểm tra giá trị nguyên mà không âm thầm cắt phần thập phân."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} phải là số nguyên >= {minimum}.")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} phải là số nguyên >= {minimum}.") from exc
    if not math.isfinite(numeric) or numeric != result or result < minimum:
        raise ValueError(f"{name} phải là số nguyên >= {minimum}.")
    return result


def _threshold(value: Any, name: str) -> float:
    """Kiểm tra ngưỡng hữu hạn trong khoảng đóng [0, 1]."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} không được là bool.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} phải nằm trong [0, 1].") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} phải nằm trong [0, 1].")
    return result


def configure_cpu_threads(
    num_threads: int | None = None,
    *,
    cv_threads: int | None = None,
) -> dict[str, Any]:
    """Cấu hình pool CPU toàn tiến trình và trả về trạng thái thực tế."""
    import cv2
    import torch

    threads = (
        physical_core_count()
        if num_threads is None
        else _integer(num_threads, "num_threads")
    )
    cv_count = (
        None if cv_threads is None else _integer(cv_threads, "cv_threads", 0)
    )

    with _LOCK:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch không cho đổi interop sau khi pool đã được sử dụng.
            pass
        if cv_count is not None:
            cv2.setNumThreads(cv_count)

        return {
            "torch_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
            "cv_threads": int(cv2.getNumThreads()),
            "physical_cores": physical_core_count(),
            "logical_cores": logical_core_count(),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        }


def _probe_package(name: str) -> dict[str, Any]:
    """Import động một package và giữ lại nguyên nhân không khả dụng."""
    try:
        module = importlib.import_module(name)
        return {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "reason": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def probe_runtime_backends() -> dict[str, Any]:
    """Probe package cục bộ; không tải model hoặc truy cập mạng."""
    pytorch = _probe_package("torch")
    ort = _probe_package("onnxruntime")
    onnx = _probe_package("onnx")
    openvino = _probe_package("openvino")
    mkldnn = False

    if pytorch["available"]:
        import torch

        mkldnn = bool(torch.backends.mkldnn.is_available())

    if ort["available"]:
        try:
            ort_module = importlib.import_module("onnxruntime")
            providers = list(ort_module.get_available_providers())
            ort["providers"] = providers
            ort["available"] = "CPUExecutionProvider" in providers
            if not ort["available"]:
                ort["reason"] = "Không có CPUExecutionProvider."
        except Exception as exc:
            ort["available"] = False
            ort["reason"] = f"{type(exc).__name__}: {exc}"

    return {
        "pytorch": pytorch,
        "onnx": {
            "available": ort["available"],
            "version": ort["version"],
            "onnxruntime": ort,
            "onnx": onnx,
            "export_available": bool(ort["available"] and onnx["available"]),
            "reason": ort["reason"],
        },
        "openvino": openvino,
        "physical_cores": physical_core_count(),
        "logical_cores": logical_core_count(),
        "mkldnn": mkldnn,
    }


def resolve_cpu_inference_config(
    model_name: str,
    cfg: dict | None = None,
    preset: str | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Hợp nhất preset < inference.cpu < overrides, chuẩn hóa và validate."""
    name = str(model_name).strip().lower()
    if name not in CPU_PRESETS:
        raise ValueError(f"Model CPU không được hỗ trợ: {model_name!r}.")

    inference = (cfg or {}).get("inference") or {}
    cpu = inference.get("cpu") or {}
    if not isinstance(cpu, Mapping):
        raise ValueError("inference.cpu phải là mapping.")

    selected = str(preset or cpu.get("preset") or "balanced").strip().lower()
    if selected not in CPU_PRESETS[name]:
        raise ValueError(f"Preset CPU không hợp lệ: {selected!r}.")

    result = deepcopy(CPU_PRESETS[name][selected])
    result.update({"model_name": name, "preset": selected, "backend": "pytorch"})

    for source in (cpu, overrides or {}):
        normalized = dict(source)
        if "image_size" in normalized and "img_size" not in normalized:
            normalized["img_size"] = normalized["image_size"]
        for key in _OPTION_KEYS:
            if key in normalized:
                result[key] = normalized[key]

    for key in (
        "img_size",
        "min_size",
        "max_size",
        "rpn_pre_nms_top_n_test",
        "rpn_post_nms_top_n_test",
        "box_detections_per_img",
    ):
        if key in result:
            result[key] = _integer(result[key], key)
    if result.get("threads") is not None:
        result["threads"] = _integer(result["threads"], "threads")
    if "fuse" in result and not isinstance(result["fuse"], bool):
        raise ValueError("fuse phải là bool.")
    if name == "faster_rcnn" and result["min_size"] > result["max_size"]:
        raise ValueError("min_size không được lớn hơn max_size.")

    result["backend"] = str(result["backend"]).strip().lower()
    if result["backend"] not in _BACKENDS:
        raise ValueError(f"Backend không hợp lệ: {result['backend']!r}.")
    return result


def _direct_config(model_name: str, cpu_cfg: dict | str | None) -> dict[str, Any]:
    """Chuẩn hóa đối số tuning trực tiếp."""
    if isinstance(cpu_cfg, str):
        return resolve_cpu_inference_config(model_name, preset=cpu_cfg)
    options = dict(cpu_cfg or {})
    return resolve_cpu_inference_config(
        model_name,
        preset=options.get("preset"),
        overrides=options,
    )


def _is_mock(value: Any) -> bool:
    """Nhận diện unittest.mock, kể cả subclass ở module khác."""
    return any(
        cls.__module__.startswith("unittest.mock")
        for cls in type(value).__mro__
    )


def _capture_baseline(detector: Any, model_name: str) -> None:
    """Lưu baseline một lần, gồm các trường thực sự tồn tại."""
    import torch

    if getattr(detector, "_baseline_cpu_profile", None) is not None:
        return

    baseline: dict[str, Any] = {
        "model_name": model_name,
        "torch_threads": int(torch.get_num_threads()),
    }
    if model_name == "yolo":
        if hasattr(detector, "img_size"):
            baseline["img_size"] = deepcopy(detector.img_size)
    else:
        model = getattr(detector, "model", None)
        transform = getattr(model, "transform", None)
        for key in ("min_size", "max_size"):
            if hasattr(transform, key):
                baseline[key] = deepcopy(getattr(transform, key))

        rpn = getattr(model, "rpn", None)
        for key, attr in (
            ("rpn_pre_nms_top_n_test", "_pre_nms_top_n"),
            ("rpn_post_nms_top_n_test", "_post_nms_top_n"),
        ):
            values = getattr(rpn, attr, None)
            if isinstance(values, dict) and "testing" in values:
                baseline[key] = deepcopy(values["testing"])

        roi = getattr(model, "roi_heads", None)
        if hasattr(roi, "detections_per_img"):
            baseline["box_detections_per_img"] = deepcopy(
                roi.detections_per_img
            )

    detector._baseline_cpu_profile = baseline


def _set_faster_fields(model: Any, profile: dict[str, Any]) -> list[str]:
    """Gán các trường torchvision hiện hữu và trả danh sách đã áp dụng."""
    applied: list[str] = []
    transform = getattr(model, "transform", None)
    for key in ("min_size", "max_size"):
        if key in profile and hasattr(transform, key):
            value = profile[key]
            if key == "min_size":
                value = (value,) if isinstance(value, int) else tuple(value)
            setattr(transform, key, value)
            applied.append(key)

    rpn = getattr(model, "rpn", None)
    for key, attr in (
        ("rpn_pre_nms_top_n_test", "_pre_nms_top_n"),
        ("rpn_post_nms_top_n_test", "_post_nms_top_n"),
    ):
        values = getattr(rpn, attr, None)
        if key in profile and isinstance(values, dict) and "testing" in values:
            values["testing"] = profile[key]
            applied.append(key)

    roi = getattr(model, "roi_heads", None)
    if (
        "box_detections_per_img" in profile
        and hasattr(roi, "detections_per_img")
    ):
        roi.detections_per_img = profile["box_detections_per_img"]
        applied.append("box_detections_per_img")
    return applied


def apply_yolo_cpu_tuning(
    detector: Any,
    cpu_cfg: dict | str | None = None,
) -> dict[str, Any]:
    """Áp dụng kích thước YOLO, CPU threads và fuse tối đa một lần thành công."""
    config = _direct_config("yolo", cpu_cfg)
    with _LOCK:
        _capture_baseline(detector, "yolo")
        thread_info = configure_cpu_threads(config["threads"])
        detector.img_size = config["img_size"]
        fusion_reason: str | None = None

        if config["fuse"] and not getattr(detector, "_cpu_fused", False):
            model = getattr(detector, "model", None)
            fuse_fn = getattr(model, "fuse", None)
            inner_model = getattr(model, "model", None)
            if (
                model is None
                or _is_mock(model)
                or inner_model is None
                or _is_mock(inner_model)
                or not callable(fuse_fn)
            ):
                fusion_reason = "Bỏ qua fuse: model không có fuse thực khả dụng."
            else:
                try:
                    fuse_fn()
                    detector._cpu_fused = True
                except Exception as exc:
                    fusion_reason = f"Fuse thất bại: {type(exc).__name__}: {exc}"

        report = {
            **config,
            **thread_info,
            "backend": "pytorch",
            "fallback_used": False,
            "fused": bool(getattr(detector, "_cpu_fused", False)),
            "fusion_reason": fusion_reason,
        }
        detector._cpu_profile = deepcopy(report)
        return deepcopy(report)


def apply_faster_rcnn_cpu_tuning(
    detector: Any,
    cpu_cfg: dict | str | None = None,
) -> dict[str, Any]:
    """Tinh chỉnh transform, số proposal và số detection của Faster R-CNN."""
    config = _direct_config("faster_rcnn", cpu_cfg)
    with _LOCK:
        _capture_baseline(detector, "faster_rcnn")
        thread_info = configure_cpu_threads(config["threads"])
        applied = _set_faster_fields(getattr(detector, "model", None), config)
        report = {
            **config,
            **thread_info,
            "backend": "pytorch",
            "fallback_used": False,
            "applied_fields": applied,
        }
        detector._cpu_profile = deepcopy(report)
        return deepcopy(report)


def apply_runtime_cpu_tuning(
    detector: Any,
    model_name: str,
    cfg: dict | None = None,
    preset: str | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Áp dụng tuning khi được gọi tường minh; không tự thay đổi evaluation."""
    config = resolve_cpu_inference_config(model_name, cfg, preset, overrides)
    if config["model_name"] == "yolo":
        return apply_yolo_cpu_tuning(detector, config)
    return apply_faster_rcnn_cpu_tuning(detector, config)


def restore_cpu_tuning(detector: Any) -> bool:
    """Khôi phục trường tuning và intra-op threads nếu đã lưu baseline.

    Fusion Conv/BatchNorm không đảo ngược được; cờ `_cpu_fused` được giữ lại.
    Inter-op threads cũng không thể khôi phục an toàn sau khi pool đã chạy.
    """
    import torch

    with _LOCK:
        baseline = getattr(detector, "_baseline_cpu_profile", None)
        if not isinstance(baseline, dict):
            return False

        if baseline["model_name"] == "yolo":
            if "img_size" in baseline:
                detector.img_size = deepcopy(baseline["img_size"])
        else:
            _set_faster_fields(getattr(detector, "model", None), baseline)
        torch.set_num_threads(baseline["torch_threads"])
        detector._cpu_profile = {
            **deepcopy(baseline),
            "restored": True,
            "fused": bool(getattr(detector, "_cpu_fused", False)),
        }
        return True


def _nms(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    iou_threshold: float,
) -> list[int]:
    """NMS theo lớp bằng NumPy, dùng tọa độ xyxy liên tục."""
    keep: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        while order.size:
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if not remaining.size:
                break
            intersection_wh = np.maximum(
                0.0,
                np.minimum(boxes[current, 2:], boxes[remaining, 2:])
                - np.maximum(boxes[current, :2], boxes[remaining, :2]),
            )
            intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
            current_wh = np.maximum(0.0, boxes[current, 2:] - boxes[current, :2])
            remaining_wh = np.maximum(
                0.0, boxes[remaining, 2:] - boxes[remaining, :2]
            )
            union = (
                current_wh.prod()
                + remaining_wh.prod(axis=1)
                - intersection
            )
            overlap = intersection / np.maximum(union, np.finfo(np.float32).eps)
            order = remaining[overlap <= iou_threshold]
    return sorted(keep, key=lambda index: -float(scores[index]))


class _YoloCpuAdapter(BaseDetector):
    """Tiền/hậu xử lý chung cho graph YOLO float32, batch một."""

    name: str = "yolo"

    def _initialize(
        self,
        weights_path: str | Path,
        classes: Mapping[int, Any] | None,
        conf: float,
        iou: float,
        img_size: int,
        output_format: str,
    ) -> None:
        """Lưu hợp đồng graph: auto, raw hoặc end_to_end."""
        if output_format not in {"auto", "raw", "end_to_end"}:
            raise ValueError("output_format phải là auto/raw/end_to_end.")
        self.weights_path = Path(weights_path)
        self.classes: dict[int, Any] = {
            int(key): value for key, value in (classes or {}).items()
        }
        self.conf = _threshold(conf, "conf")
        self.iou = _threshold(iou, "iou")
        self.img_size = _integer(img_size, "img_size")
        self.output_format = output_format
        self._input_hw = (self.img_size, self.img_size)
        self._inference_lock = threading.RLock()

    def _set_input_shape(self, shape: Any) -> None:
        """Dùng kích thước tĩnh của graph hoặc fallback kích thước preset."""
        if len(shape) != 4:
            raise ValueError("Graph phải nhận tensor NCHW bốn chiều.")
        if isinstance(shape[0], (int, np.integer)) and int(shape[0]) != 1:
            raise ValueError("Graph phải hỗ trợ batch một.")
        if isinstance(shape[1], (int, np.integer)) and int(shape[1]) != 3:
            raise ValueError("Graph phải nhận ba kênh RGB.")
        height, width = shape[2], shape[3]
        self._input_hw = (
            int(height)
            if isinstance(height, (int, np.integer)) and height > 0
            else self.img_size,
            int(width)
            if isinstance(width, (int, np.integer)) and width > 0
            else self.img_size,
        )

    def _run(self, tensor: NDArray[np.float32]) -> list[np.ndarray]:
        """Thực thi backend; được lớp con triển khai."""
        raise NotImplementedError

    def predict(
        self,
        frame: NDArray[np.uint8] | None,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Letterbox BGR, suy luận và quy đổi detection về pixel ảnh gốc."""
        import cv2

        threshold = (
            self.conf
            if conf_threshold is None
            else _threshold(conf_threshold, "conf_threshold")
        )
        if frame is None or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame phải là ảnh BGR uint8 với ba kênh.")

        height, width = frame.shape[:2]
        target_h, target_w = self._input_hw
        ratio = min(target_w / width, target_h / height)
        resized_w = max(1, min(target_w, round(width * ratio)))
        resized_h = max(1, min(target_h, round(height * ratio)))
        left = (target_w - resized_w) // 2
        top = (target_h - resized_h) // 2

        resized = cv2.resize(frame, (resized_w, resized_h))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = resized
        tensor = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        tensor /= np.float32(255.0)

        with self._inference_lock:
            outputs = self._run(tensor)
        boxes, scores, labels, already_nms = self._decode(outputs, threshold)
        if boxes.size == 0:
            return []

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        keep = (
            np.argsort(-scores, kind="stable").tolist()
            if already_nms
            else _nms(boxes, scores, labels, self.iou)
        )

        detections: list[Detection] = []
        for index in keep:
            class_id = int(labels[index])
            entry = self.classes.get(class_id)
            label = (
                entry.get("name", f"Class_{class_id}")
                if isinstance(entry, Mapping)
                else entry if entry is not None else f"Class_{class_id}"
            )
            detections.append(
                {
                    "bbox": [float(value) for value in boxes[index]],
                    "label": str(label),
                    "class_id": class_id,
                    "confidence": float(scores[index]),
                }
            )
        return detections

    def _decode(
        self,
        outputs: list[np.ndarray],
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Giải mã YOLO raw xywh hoặc end-to-end xyxy/score/class.

        Với hai lớp, shape [1, N, 6] có thể mang hai ý nghĩa. Caller có
        thể đặt output_format='raw' để loại bỏ sự nhập nhằng này.
        """
        if not outputs:
            raise ValueError("Graph không trả tensor detection.")
        data = np.asarray(outputs[0], dtype=np.float32)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"Shape YOLO không hỗ trợ: {data.shape}.")

        expected_channels = 4 + len(self.classes) if self.classes else None
        end_to_end = self.output_format == "end_to_end"
        if self.output_format == "auto":
            end_to_end = data.shape[1] == 6

        if end_to_end:
            if data.shape[1] != 6:
                raise ValueError("Output end-to-end phải có sáu cột.")
            boxes = data[:, :4].copy()
            scores = data[:, 4]
            raw_labels = data[:, 5]
        else:
            if expected_channels is not None:
                if data.shape[1] == expected_channels:
                    pass
                elif data.shape[0] == expected_channels:
                    data = data.T
                else:
                    raise ValueError("Số lớp của graph khác danh mục classes.")
            elif data.shape[0] < data.shape[1]:
                data = data.T

            if data.shape[1] < 5:
                raise ValueError("Output raw phải có 4 + nc cột.")
            class_scores = data[:, 4:]
            raw_labels = np.argmax(class_scores, axis=1)
            scores = np.max(class_scores, axis=1)
            boxes = data[:, :4].copy()
            boxes[:, :2] = data[:, :2] - data[:, 2:4] / 2.0
            boxes[:, 2:] = data[:, :2] + data[:, 2:4] / 2.0

        valid = (
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(scores)
            & np.isfinite(raw_labels)
            & (scores >= threshold)
            & (scores <= 1.0)
            & (raw_labels >= 0)
            & (raw_labels == np.floor(raw_labels))
            & (raw_labels <= np.iinfo(np.int32).max)
        )
        return (
            boxes[valid],
            scores[valid],
            raw_labels[valid].astype(np.int64),
            end_to_end,
        )


class OnnxDetectorAdapter(_YoloCpuAdapter):
    """Detector YOLO chạy ONNX Runtime CPUExecutionProvider."""

    def __init__(
        self,
        weights_path: str | Path,
        classes: Mapping[int, Any] | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        img_size: int = 640,
        *,
        session: Any | None = None,
        threads: int | None = None,
        output_format: str = "auto",
    ) -> None:
        """Nạp graph cục bộ; cho phép truyền session để kiểm thử."""
        self._initialize(
            weights_path, classes, conf, iou, img_size, output_format
        )
        if session is None:
            ort = importlib.import_module("onnxruntime")
            options = ort.SessionOptions()
            options.intra_op_num_threads = (
                physical_core_count()
                if threads is None
                else _integer(threads, "threads")
            )
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(self.weights_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self.session = session
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Adapter chỉ hỗ trợ graph YOLO có một input.")
        if getattr(inputs[0], "type", "tensor(float)") != "tensor(float)":
            raise ValueError("Graph phải nhận float32.")
        self._input_name = inputs[0].name
        self._set_input_shape(inputs[0].shape)

    def _run(self, tensor: NDArray[np.float32]) -> list[np.ndarray]:
        """Chạy session ONNX."""
        return list(self.session.run(None, {self._input_name: tensor}))


class OpenVinoDetectorAdapter(_YoloCpuAdapter):
    """Detector YOLO chạy compiled model trên OpenVINO CPU."""

    def __init__(
        self,
        weights_path: str | Path,
        classes: Mapping[int, Any] | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        img_size: int = 640,
        *,
        compiled_model: Any | None = None,
        threads: int | None = None,
        output_format: str = "auto",
    ) -> None:
        """Nạp graph ONNX/XML cục bộ, không yêu cầu OpenVINO ở cấp module."""
        self._initialize(
            weights_path, classes, conf, iou, img_size, output_format
        )
        if compiled_model is None:
            ov = importlib.import_module("openvino")
            core = ov.Core()
            graph = core.read_model(str(self.weights_path))
            compiled_model = core.compile_model(
                graph,
                "CPU",
                {
                    "INFERENCE_NUM_THREADS": (
                        physical_core_count()
                        if threads is None
                        else _integer(threads, "threads")
                    ),
                    "PERFORMANCE_HINT": "LATENCY",
                },
            )
        self.compiled_model = compiled_model
        if len(self.compiled_model.inputs) != 1:
            raise ValueError("Adapter chỉ hỗ trợ graph YOLO có một input.")
        input_port = self.compiled_model.input(0)
        if input_port.get_element_type().get_type_name() != "f32":
            raise ValueError("Graph phải nhận float32.")
        shape = [
            int(dim.get_length()) if dim.is_static else None
            for dim in input_port.get_partial_shape()
        ]
        self._set_input_shape(shape)
        self._request = self.compiled_model.create_infer_request()

    def _run(self, tensor: NDArray[np.float32]) -> list[np.ndarray]:
        """Chạy request đồng bộ và sao chép tensor khỏi bộ nhớ request."""
        self._request.infer({self.compiled_model.input(0): tensor})
        return [
            np.array(self._request.get_output_tensor(index).data, copy=True)
            for index in range(len(self.compiled_model.outputs))
        ]


def _find_export(
    detector: Any,
    cpu_options: Mapping[str, Any],
    export_dir: str | Path | None,
    backend: str,
) -> Path | None:
    """Chỉ chọn graph đã có; không tự export hoặc tìm graph không liên quan."""
    suffixes = {".onnx"} if backend == "onnx" else {".onnx", ".xml"}
    explicit = cpu_options.get(f"{backend}_path") or cpu_options.get("export_path")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
        raise FileNotFoundError(f"Graph {backend} không tồn tại: {path}")

    candidates: list[Path] = []
    for attr in ("weights_path", "weights"):
        value = getattr(detector, attr, None)
        if isinstance(value, (str, Path)):
            candidates.append(Path(value))
    model = getattr(detector, "model", None)
    checkpoint = getattr(model, "ckpt_path", None)
    if isinstance(checkpoint, (str, Path)):
        candidates.append(Path(checkpoint))

    for candidate in candidates:
        for suffix in sorted(suffixes):
            path = candidate.with_suffix(suffix)
            if path.is_file():
                return path

    if export_dir is not None:
        directory = Path(export_dir).expanduser()
        if directory.is_file() and directory.suffix.lower() in suffixes:
            return directory
        if directory.is_dir():
            graphs = sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in suffixes
            )
            # Tránh âm thầm chọn nhầm trọng số khi thư mục chứa nhiều graph.
            if len(graphs) == 1:
                return graphs[0]
            if len(graphs) > 1:
                raise ValueError("Có nhiều graph; hãy chỉ định export_path.")
    return None


def prepare_runtime_detector(
    model_name: str,
    detector: BaseDetector,
    cfg: dict | None = None,
    backend: str = "pytorch",
    preset: str | None = None,
    export_dir: str | Path | None = None,
) -> tuple[BaseDetector, dict[str, Any]]:
    """Chuẩn bị detector CPU; backend tăng tốc lỗi sẽ fallback PyTorch.

    Chỉ dùng graph YOLO đã export tại máy. Không tự export Faster R-CNN
    vì hợp đồng input/output và phép resize cần được xác minh riêng.
    """
    requested = str(backend).strip().lower()
    if requested not in _BACKENDS:
        raise ValueError(f"Backend không hợp lệ: {backend!r}.")

    config = resolve_cpu_inference_config(model_name, cfg, preset)
    name = config["model_name"]

    if requested == "pytorch":
        report = apply_runtime_cpu_tuning(detector, name, cfg, preset)
        report["requested_backend"] = requested
        with _LOCK:
            detector._cpu_profile = deepcopy(report)
        return detector, report

    reasons: list[str] = []
    status = probe_runtime_backends()
    inference = (cfg or {}).get("inference") or {}
    options = inference.get("cpu") or {}
    candidates = ["openvino", "onnx"] if requested == "auto" else [requested]

    if name == "faster_rcnn":
        reasons.append(
            "Faster R-CNN chưa có adapter graph tĩnh được xác minh; "
            "sử dụng PyTorch CPU-tuned."
        )
    else:
        for candidate in candidates:
            if not status[candidate]["available"]:
                reasons.append(
                    f"{candidate} không khả dụng: "
                    f"{status[candidate].get('reason') or 'thiếu runtime CPU'}"
                )
                continue

            try:
                path = _find_export(detector, options, export_dir, candidate)
                if path is None:
                    reasons.append(
                        f"Không tìm thấy graph {candidate} đã export cục bộ; "
                        "không tự export hoặc tải trọng số."
                    )
                    continue

                adapter_class = (
                    OnnxDetectorAdapter
                    if candidate == "onnx"
                    else OpenVinoDetectorAdapter
                )
                adapter = adapter_class(
                    weights_path=path,
                    classes=getattr(detector, "classes", {}),
                    conf=getattr(detector, "conf", 0.25),
                    iou=getattr(detector, "iou", 0.5),
                    img_size=config["img_size"],
                    threads=config["threads"],
                    output_format=str(options.get("output_format", "auto")),
                )

                # Xác minh shape và decoding trước khi thay detector.
                target_h, target_w = adapter._input_hw
                adapter.predict(
                    np.zeros((target_h, target_w, 3), dtype=np.uint8)
                )
                thread_info = configure_cpu_threads(config["threads"])
                report = {
                    **config,
                    **thread_info,
                    "backend": candidate,
                    "requested_backend": requested,
                    "weights_path": str(path),
                    "input_hw": adapter._input_hw,
                    "fallback_used": False,
                    "reason": None,
                    "backend_attempts": list(reasons),
                }
                with _LOCK:
                    adapter._cpu_profile = deepcopy(report)
                return adapter, report
            except Exception as exc:
                reasons.append(f"{candidate}: {type(exc).__name__}: {exc}")

    report = apply_runtime_cpu_tuning(detector, name, cfg, preset)
    report.update(
        {
            "requested_backend": requested,
            "fallback_used": True,
            "reason": "; ".join(reasons),
        }
    )
    with _LOCK:
        detector._cpu_profile = deepcopy(report)
    return detector, report

