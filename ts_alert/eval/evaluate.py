"""Đọc tập dữ liệu YOLO, đánh giá detector và lưu kết quả."""

from __future__ import annotations

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
from ts_alert.types import IMAGE_EXTS
from ts_alert.runtime import video_io
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
) -> dict:
    """Tính mAP, AR, precision, recall, F1 và đo tốc độ detector.

    eval.metrics chọn các chỉ số mAP/AR cần trả về; thiếu hoặc rỗng
    thì giữ toàn bộ. Precision, recall, F1 và benchmark luôn được giữ.
    Khi có out_dir, lưu ma trận nhầm lẫn dạng PNG và JSON kèm các
    dự đoán. Tên lớp thiếu hoặc rỗng trong CSV dùng dạng Class_<id>.
    """
    eval_cfg = cfg.get("eval", {})
    configured_metrics = eval_cfg.get("metrics")
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

    inference = cfg.get("inference", {})
    conf_threshold = float(inference.get("conf_threshold", 0.25))
    iou_threshold = float(inference.get("iou_threshold", 0.5))
    num_classes = get_num_classes(cfg)

    frames: list[np.ndarray] = []
    preds: list[list[dict[str, Any]]] = []
    gts: list[list[dict[str, Any]]] = []

    for sample in load_split_samples(cfg, split):
        frame = video_io.read_image(sample["image_path"])
        frames.append(frame)
        preds.append(detector.predict(frame))
        gts.append(sample["gt"])

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
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)

        cm, y_true, y_pred = metrics.compute_confusion_matrix(
            preds,
            gts,
            num_classes,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
        )

        classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
        class_names: list[str] = []
        for cid in range(num_classes):
            name = classes.get(cid, {}).get("name")
            class_names.append(
                name.strip()
                if isinstance(name, str) and name.strip()
                else f"Class_{cid}"
            )

        metrics.save_confusion_matrix_plot(
            cm,
            class_names,
            output / "confusion_matrix.png",
        )

        # Chuyển dữ liệu số NumPy/Torch thành kiểu Python để ghi JSON.
        serialized_preds: list[list[dict[str, Any]]] = [
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

        write_json(
            {
                "labels": class_names + ["background"],
                "class_names": class_names,
                "matrix": cm.tolist(),
                "y_true": [int(label) for label in y_true],
                "y_pred": [int(label) for label in y_pred],
                "preds": serialized_preds,
            },
            output / "confusion_matrix.json",
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
