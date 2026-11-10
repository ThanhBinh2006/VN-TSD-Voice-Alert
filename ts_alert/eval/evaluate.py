"""Đọc tập dữ liệu YOLO, đánh giá detector và lưu kết quả."""

from __future__ import annotations

import inspect
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ts_alert.config.resolve import (
    get_num_classes,
    get_processed_root,
    resolve_path,
)
from ts_alert.eval import benchmark, metrics
from ts_alert.runtime import video_io
from ts_alert.types import IMAGE_EXTS
from ts_alert.utils.io import load_classes, write_json


_DETECTION_METRIC_KEYS = ("mAP50-95", "mAP50", "mAP75", "AR100")
_ALLOWED_EVAL_METRICS = frozenset(
    {
        "mAP50-95",
        "mAP50",
        "mAP75",
        "AR100",
        "precision",
        "recall",
        "f1",
        "fps",
        "latency_ms",
    }
)


def _resolve_requested_metrics(cfg: dict) -> list[str]:
    """Đọc và kiểm tra eval.metrics; thiếu hoặc rỗng nghĩa là giữ tất cả."""
    configured_metrics = cfg.get("eval", {}).get("metrics")
    requested: list[str] = []

    if configured_metrics is not None:
        if isinstance(configured_metrics, (str, bytes)):
            raise ValueError("eval.metrics phải là danh sách tên metric hợp lệ.")

        try:
            metric_items = iter(configured_metrics)
        except TypeError as exc:
            raise ValueError(
                "eval.metrics phải là danh sách tên metric hợp lệ."
            ) from exc

        for item in metric_items:
            if not isinstance(item, str) or item not in _ALLOWED_EVAL_METRICS:
                raise ValueError(f"Metric đánh giá không hợp lệ: {item!r}")
            requested.append(item)

    return requested


def _resolve_class_names(cfg: dict, num_classes: int) -> list[str]:
    """Đọc tên lớp từ CSV, dùng Class_<id> nếu tên thiếu hoặc rỗng."""
    classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
    class_names: list[str] = []

    for cid in range(num_classes):
        name = classes.get(cid, {}).get("name")
        class_names.append(
            name.strip()
            if isinstance(name, str) and name.strip()
            else f"Class_{cid}"
        )

    return class_names


def _serialize_preds(
    preds: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Chuyển các giá trị NumPy/Torch của prediction thành kiểu Python."""
    return [
        [
            {
                "bbox": [float(value) for value in prediction["bbox"]],
                "class_id": int(prediction["class_id"]),
                "confidence": float(prediction.get("confidence", 1.0)),
            }
            for prediction in image_predictions
        ]
        for image_predictions in preds
    ]


def _save_confusion_artifacts(
    preds: list[list[dict[str, Any]]],
    gts: list[list[dict[str, Any]]],
    num_classes: int,
    class_names: list[str],
    iou_threshold: float,
    conf_threshold: float,
    output: Path | str,
    filename_prefix: str = "confusion_matrix",
    split: str | None = None,
) -> None:
    """Lưu ma trận nhầm lẫn PNG và JSON cùng thông tin ngưỡng đánh giá.

    JSON giữ prediction có confidence >= conf_threshold. Prediction thiếu
    confidence luôn được giữ và được tuần tự hóa với confidence bằng 1.0.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    cm, y_true, y_pred = metrics.compute_confusion_matrix(
        preds,
        gts,
        num_classes,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )

    split_label = split.capitalize() if split is not None else "Evaluation"
    metrics.save_confusion_matrix_plot(
        cm,
        class_names,
        output / f"{filename_prefix}.png",
        title=(
            f"Confusion Matrix - {split_label} "
            f"(conf*={conf_threshold:.4f}, IoU={iou_threshold:.4f})"
        ),
    )

    filtered_preds = [
        [
            prediction
            for prediction in image_predictions
            if "confidence" not in prediction
            or float(prediction["confidence"]) >= conf_threshold
        ]
        for image_predictions in preds
    ]

    write_json(
        {
            "labels": class_names + ["background"],
            "class_names": class_names,
            "matrix": cm.tolist(),
            "y_true": [int(label) for label in y_true],
            "y_pred": [int(label) for label in y_pred],
            "preds": _serialize_preds(filtered_preds),
            "split": split,
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
        },
        output / f"{filename_prefix}.json",
    )


def _predict_for_eval(
    detector: Any,
    frames_or_samples: Iterable[np.ndarray | dict[str, Any]],
    raw_conf: float,
) -> list[list[dict[str, Any]]]:
    """Suy luận một lần mỗi ảnh, truyền raw_conf nếu predict hỗ trợ.

    Đầu vào chứa frame hoặc sample có image_path. Chữ ký predict được
    kiểm tra đúng một lần trước vòng lặp; lỗi suy luận được giữ nguyên.
    """
    raw_conf = float(raw_conf)
    if not np.isfinite(raw_conf) or not 0.0 <= raw_conf <= 1.0:
        raise ValueError("raw_conf phải là số hữu hạn thuộc [0, 1].")

    predict = detector.predict
    try:
        signature = inspect.signature(predict)
    except (TypeError, ValueError):
        accepts_conf_threshold = False
    else:
        parameter = signature.parameters.get("conf_threshold")
        accepts_conf_threshold = (
            parameter is not None
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ) or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )

    predictions: list[list[dict[str, Any]]] = []
    for item in frames_or_samples:
        frame = (
            video_io.read_image(item["image_path"])
            if isinstance(item, dict)
            else item
        )
        if accepts_conf_threshold:
            predictions.append(predict(frame, conf_threshold=raw_conf))
        else:
            predictions.append(predict(frame))

    return predictions


def load_split_samples(cfg: dict, split: str = "test") -> list[dict]:
    """Đọc ảnh và nhãn YOLO của một tập dữ liệu.

    Nhãn YOLO có dạng ``class_id cx cy bw bh`` với tọa độ chuẩn hóa.
    Kết quả chứa đường dẫn ảnh và GT có bbox dạng xyxy theo pixel.
    Ảnh không có tệp nhãn được xem là ảnh không có đối tượng.
    """
    root = get_processed_root(cfg)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    samples: list[dict] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        gt: list[dict[str, Any]] = []
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(),
                start=1,
            ):
                fields = line.split()
                if not fields:
                    continue

                try:
                    if len(fields) != 5:
                        raise ValueError("Mỗi dòng phải có đúng 5 giá trị.")

                    class_id = int(fields[0])
                    cx, cy, bw, bh = map(float, fields[1:])
                    if class_id < 0:
                        raise ValueError("class_id phải không âm.")
                    if not np.isfinite([cx, cy, bw, bh]).all():
                        raise ValueError("Tọa độ phải hữu hạn.")
                    if bw < 0 or bh < 0:
                        raise ValueError("Kích thước bbox phải không âm.")
                except ValueError as exc:
                    raise ValueError(
                        f"Nhãn YOLO không hợp lệ tại "
                        f"{label_path}:{line_number}: {exc}"
                    ) from exc

                gt.append(
                    {
                        "class_id": class_id,
                        "bbox": [
                            (cx - bw / 2) * width,
                            (cy - bh / 2) * height,
                            (cx + bw / 2) * width,
                            (cy + bh / 2) * height,
                        ],
                    }
                )

        samples.append({"image_path": image_path, "gt": gt})

    return samples


def evaluate_detector(
    detector: Any,
    cfg: dict,
    split: str = "test",
    out_dir: Path | str | None = None,
    conf_threshold: float | None = None,
    raw_conf_threshold: float | None = None,
) -> dict:
    """Tính mAP, AR, precision, recall, F1 và đo tốc độ detector.

    eval.metrics chọn các chỉ số mAP/AR cần trả về; thiếu hoặc rỗng
    thì giữ toàn bộ. Precision, recall, F1 và benchmark luôn được giữ.
    Khi có out_dir, lưu ma trận nhầm lẫn dạng PNG và JSON kèm các
    dự đoán. Tên lớp thiếu hoặc rỗng trong CSV dùng dạng Class_<id>.

    conf_threshold ghi đè ngưỡng đánh giá trong inference khi được truyền.
    raw_conf_threshold chỉ được truyền tới predict nếu detector hỗ trợ.
    Khi cả hai bằng None, cách gọi suy luận và tính metric giữ nguyên.
    """
    requested = _resolve_requested_metrics(cfg)

    inference = cfg.get("inference", {})
    conf_threshold = float(
        inference.get("conf_threshold", 0.25)
        if conf_threshold is None
        else conf_threshold
    )
    iou_threshold = float(inference.get("iou_threshold", 0.5))
    num_classes = get_num_classes(cfg)

    frames: list[np.ndarray] = []
    preds: list[list[dict[str, Any]]] = []
    gts: list[list[dict[str, Any]]] = []

    for sample in load_split_samples(cfg, split):
        frame = video_io.read_image(sample["image_path"])
        frames.append(frame)
        if raw_conf_threshold is None:
            preds.append(detector.predict(frame))
        gts.append(sample["gt"])

    if raw_conf_threshold is not None:
        preds = _predict_for_eval(
            detector,
            frames,
            raw_conf=raw_conf_threshold,
        )

    result = metrics.compute_metrics(
        preds,
        gts,
        num_classes,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )

    if requested:
        selected = set(requested)
        result = {
            k: v
            for k, v in result.items()
            if k not in _DETECTION_METRIC_KEYS or k in selected
        }

    if out_dir is not None:
        _save_confusion_artifacts(
            preds,
            gts,
            num_classes,
            _resolve_class_names(cfg, num_classes),
            iou_threshold,
            conf_threshold,
            Path(out_dir),
            split=split,
        )

    bench = cfg["eval"]["benchmark"]
    result.update(
        benchmark.measure_fps(
            detector,
            frames[:bench["num_frames"]],
            warmup=bench["warmup"],
        )
    )
    return result


def evaluate_best_model(
    detector: Any,
    cfg: dict,
    out_dir: Path | str,
    splits: tuple[str, ...] = ("test", "val"),
    tune_split: str = "test",
    raw_conf: float = 0.01,
) -> dict[str, Any]:
    """Chọn confidence tối ưu trên tune_split và đánh giá các split.

    Mỗi split được suy luận một lần với raw_conf, ngoài các lượt benchmark.
    tune_split luôn được nạp; split phụ thiếu thư mục ảnh được cảnh báo
    và bỏ qua. Artifact của tune_split có alias tương thích W&B.
    """
    requested = _resolve_requested_metrics(cfg)

    def validate_split(value: str) -> None:
        """Yêu cầu tên split là một thành phần đường dẫn hợp lệ."""
        if (
            not isinstance(value, str)
            or not value.strip()
            or value in {".", ".."}
            or any(character in value for character in '/\\<>:"|?*')
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"Tên split không hợp lệ: {value!r}")

    validate_split(tune_split)
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits phải là một tuple các tên split.")

    active_candidates: list[str] = []
    for split in splits:
        validate_split(split)
        if split not in active_candidates:
            active_candidates.append(split)

    if tune_split not in active_candidates:
        active_candidates.append(tune_split)

    inference = cfg.get("inference", {})
    default_conf = float(inference.get("conf_threshold", 0.25))
    iou_threshold = float(inference.get("iou_threshold", 0.5))
    raw_conf = float(raw_conf)

    for name, value in (
        ("inference.conf_threshold", default_conf),
        ("inference.iou_threshold", iou_threshold),
        ("raw_conf", raw_conf),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} phải là số hữu hạn thuộc [0, 1].")

    num_classes = get_num_classes(cfg)
    root = get_processed_root(cfg)

    # Nạp bắt buộc trước để không bỏ qua lỗi hoặc thiếu tune_split.
    tune_samples = load_split_samples(cfg, tune_split)
    samples_by_split: dict[str, list[dict]] = {}

    for split in active_candidates:
        if split == tune_split:
            samples_by_split[split] = tune_samples
        elif (root / "images" / split).is_dir():
            samples_by_split[split] = load_split_samples(cfg, split)
        else:
            warnings.warn(
                f"Bỏ qua split {split!r}: không tìm thấy thư mục ảnh "
                f"{root / 'images' / split}.",
                UserWarning,
                stacklevel=2,
            )

    frames_by_split: dict[str, list[np.ndarray]] = {}
    preds_by_split: dict[str, list[list[dict[str, Any]]]] = {}
    gts_by_split: dict[str, list[list[dict[str, Any]]]] = {}
    curves: dict[str, dict[str, Any]] = {}

    for split, samples in samples_by_split.items():
        frames = [
            video_io.read_image(sample["image_path"])
            for sample in samples
        ]
        preds = _predict_for_eval(detector, frames, raw_conf=raw_conf)
        gts = [sample["gt"] for sample in samples]

        frames_by_split[split] = frames
        preds_by_split[split] = preds
        gts_by_split[split] = gts
        curves[split] = metrics.compute_pr_curve(
            preds,
            gts,
            num_classes,
            iou_threshold=iou_threshold,
            default_conf=default_conf,
        )

    optimal_conf = float(curves[tune_split]["best_threshold"])
    bench = cfg["eval"]["benchmark"]
    benchmark_result = benchmark.measure_fps(
        detector,
        frames_by_split[tune_split][:bench["num_frames"]],
        warmup=bench["warmup"],
    )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    class_names = _resolve_class_names(cfg, num_classes)
    selected = set(requested)
    split_results: dict[str, dict[str, Any]] = {}

    for split in samples_by_split:
        result = metrics.compute_metrics(
            preds_by_split[split],
            gts_by_split[split],
            num_classes,
            iou_threshold=iou_threshold,
            conf_threshold=optimal_conf,
        )

        if requested:
            result = {
                key: value
                for key, value in result.items()
                if key not in _DETECTION_METRIC_KEYS or key in selected
            }

        result.update(
            {
                "split": split,
                "conf_threshold": round(optimal_conf, 4),
                "tuned_on_split": tune_split,
                "iou_threshold": round(iou_threshold, 4),
                "pr_auc": curves[split]["pr_auc"],
            }
        )
        result.update(benchmark_result)
        split_results[split] = result

        filename_prefix = f"confusion_matrix_{split}"
        _save_confusion_artifacts(
            preds_by_split[split],
            gts_by_split[split],
            num_classes,
            class_names,
            iou_threshold,
            optimal_conf,
            output,
            filename_prefix=filename_prefix,
            split=split,
        )

        if split == tune_split:
            for suffix in (".png", ".json"):
                shutil.copyfile(
                    output / f"{filename_prefix}{suffix}",
                    output / f"confusion_matrix{suffix}",
                )

        metrics_payload = {**result, "pipeline": cfg.get("pipeline")}
        write_json(metrics_payload, output / f"metrics_{split}.json")
        if split == tune_split and tune_split == "test":
            write_json(metrics_payload, output / "metrics.json")

    metrics.save_pr_curve_plot(
        curves,
        output / "pr_curve.png",
        selected_threshold=optimal_conf,
    )
    metrics.save_eval_comparison_plot(
        split_results,
        output / "eval_comparison.png",
    )
    write_json(
        {
            "tuned_on_split": tune_split,
            "selected_conf_threshold": optimal_conf,
            "iou_threshold": iou_threshold,
            "curves": curves,
        },
        output / "pr_curve_data.json",
    )

    return {
        "best_conf_threshold": round(optimal_conf, 4),
        "tuned_on_split": tune_split,
        "test_metrics": split_results[tune_split],
        "val_metrics": split_results.get("val"),
        "splits": split_results,
        "curves": curves,
    }
