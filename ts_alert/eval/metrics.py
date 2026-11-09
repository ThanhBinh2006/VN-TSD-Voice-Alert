"""Chỉ số phát hiện đối tượng và ma trận nhầm lẫn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision


# Mỗi ảnh gồm: boxes (N, 4), labels (N,), scores (N,).
_ImageData = tuple[np.ndarray, np.ndarray, np.ndarray]


def _prepare(
    preds: list,
    gts: list,
    num_classes: int,
    iou_threshold: float,
    conf_threshold: float,
) -> tuple[list[_ImageData], list[_ImageData]]:
    """Kiểm tra đầu vào và chuyển dữ liệu từng ảnh thành mảng NumPy."""
    if num_classes <= 0:
        raise ValueError("num_classes phải lớn hơn 0.")
    if len(preds) != len(gts):
        raise ValueError("preds và gts phải có cùng số ảnh.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold phải thuộc [0, 1].")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold phải thuộc [0, 1].")

    def convert(items: list, prediction: bool) -> _ImageData:
        """Chuyển và kiểm tra các đối tượng trong một ảnh."""
        boxes = np.asarray(
            [item["bbox"] for item in items], dtype=np.float64
        ).reshape(-1, 4)
        raw_labels = np.asarray(
            [item["class_id"] for item in items], dtype=np.float64
        )
        scores = np.asarray(
            [
                item.get("confidence", 1.0) if prediction else 1.0
                for item in items
            ],
            dtype=np.float64,
        )
        if len(boxes) != len(items) or not np.isfinite(boxes).all():
            raise ValueError("Mỗi bbox phải chứa đúng 4 tọa độ hữu hạn.")
        if np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError("bbox phải thỏa x2 >= x1 và y2 >= y1.")
        if (
            not np.isfinite(raw_labels).all()
            or np.any(raw_labels != np.floor(raw_labels))
            or np.any(raw_labels < 0)
            or np.any(raw_labels >= num_classes)
        ):
            raise ValueError("class_id phải là số nguyên thuộc [0, num_classes).")
        if (
            not np.isfinite(scores).all()
            or np.any(scores < 0)
            or np.any(scores > 1)
        ):
            raise ValueError("confidence phải là số hữu hạn thuộc [0, 1].")
        return boxes, raw_labels.astype(np.int64), scores

    return (
        [convert(items, True) for items in preds],
        [convert(items, False) for items in gts],
    )


def _filter_predictions(data: _ImageData, threshold: float) -> _ImageData:
    """Giữ các dự đoán có confidence lớn hơn hoặc bằng ngưỡng."""
    boxes, labels, scores = data
    keep = scores >= threshold
    return boxes[keep], labels[keep], scores[keep]


def _greedy_matches(
    prediction: _ImageData,
    target: _ImageData,
    iou_threshold: float,
    same_class: bool,
) -> list[tuple[int, int]]:
    """Ghép (GT, dự đoán) theo IoU giảm dần, mỗi đối tượng dùng tối đa một lần."""
    pred_boxes, pred_labels, _ = prediction
    gt_boxes, gt_labels, _ = target
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    top_left = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    gt_areas = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)
    pred_areas = (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(axis=1)
    union = gt_areas[:, None] + pred_areas[None, :] - intersection
    ious = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )

    eligible = (union > 0) & (ious >= iou_threshold)
    if same_class:
        eligible &= gt_labels[:, None] == pred_labels[None, :]

    gt_indices, pred_indices = np.nonzero(eligible)
    order = np.argsort(-ious[gt_indices, pred_indices], kind="stable")
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for index in order:
        gt_index = int(gt_indices[index])
        pred_index = int(pred_indices[index])
        if gt_index not in used_gt and pred_index not in used_pred:
            matches.append((gt_index, pred_index))
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    return matches


def compute_metrics(
    preds: list,
    gts: list,
    num_classes: int,
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> dict:
    """Tính mAP/AR và precision, recall, F1 micro trên toàn bộ tập dữ liệu.

    mAP/AR sử dụng toàn bộ dự đoán. Precision/recall/F1 lọc confidence,
    rồi ghép một-một theo IoU giảm dần trong từng ảnh và từng lớp.
    Confidence bị thiếu được xem là 1.0; mẫu số bằng 0 cho kết quả 0.
    """
    predictions, targets = _prepare(
        preds, gts, num_classes, iou_threshold, conf_threshold
    )
    result = {
        "mAP50-95": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "AR100": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if not predictions:
        return result

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        sync_on_compute=False,
    )
    metric.update(
        [
            {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "scores": torch.as_tensor(scores, dtype=torch.float32),
            }
            for boxes, labels, scores in predictions
        ],
        [
            {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }
            for boxes, labels, _ in targets
        ],
    )
    values = metric.compute()
    for output_key, metric_key in (
        ("mAP50-95", "map"),
        ("mAP50", "map_50"),
        ("mAP75", "map_75"),
        ("AR100", "mar_100"),
    ):
        value = float(values[metric_key].item())
        result[output_key] = round(value, 4) if np.isfinite(value) and value >= 0 else 0.0

    tp = fp = fn = 0
    for prediction, target in zip(predictions, targets):
        filtered = _filter_predictions(prediction, conf_threshold)
        matched = len(_greedy_matches(filtered, target, iou_threshold, True))
        tp += matched
        fp += len(filtered[0]) - matched
        fn += len(target[0]) - matched

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.update(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
    return result


def compute_confusion_matrix(
    preds: list,
    gts: list,
    num_classes: int,
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Trả về (cm, y_true, y_pred), hàng là lớp thật, cột là lớp dự đoán.

    Ghép một-một theo IoU giảm dần qua mọi lớp để ghi nhận nhầm lớp.
    Index num_classes biểu diễn background cho GT hoặc dự đoán không ghép.
    """
    predictions, targets = _prepare(
        preds, gts, num_classes, iou_threshold, conf_threshold
    )
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    y_true: list[int] = []
    y_pred: list[int] = []

    def record(true_class: int, pred_class: int) -> None:
        """Ghi một sự kiện vào ma trận và hai danh sách nhãn."""
        cm[true_class, pred_class] += 1
        y_true.append(true_class)
        y_pred.append(pred_class)

    for prediction, target in zip(predictions, targets):
        filtered = _filter_predictions(prediction, conf_threshold)
        matches = _greedy_matches(filtered, target, iou_threshold, False)
        used_gt = {gt_index for gt_index, _ in matches}
        used_pred = {pred_index for _, pred_index in matches}

        for gt_index, pred_index in matches:
            record(int(target[1][gt_index]), int(filtered[1][pred_index]))
        for pred_index, label in enumerate(filtered[1]):
            if pred_index not in used_pred:
                record(num_classes, int(label))
        for gt_index, label in enumerate(target[1]):
            if gt_index not in used_gt:
                record(int(label), num_classes)

    return cm, y_true, y_pred


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path | str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Path:
    """Lưu PNG 200 DPI; class_names không chứa background.

    Chuẩn hóa theo hàng khi normalize=True. Tự tăng kích thước hình và
    bỏ số trong ô khi có nhiều lớp để giữ nhãn của 53 lớp dễ đọc.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from textwrap import fill

    matrix = np.asarray(cm)
    size = len(class_names) + 1
    if matrix.shape != (size, size):
        raise ValueError("cm phải có kích thước (len(class_names) + 1)².")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("cm phải chứa các giá trị hữu hạn, không âm.")

    values = matrix.astype(np.float64, copy=True)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values, totals, out=np.zeros_like(values), where=totals > 0
        )

    path = Path(out_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [fill(name, width=24) for name in class_names] + ["background"]
    side = max(9.0, size * 0.42)
    fig, ax = plt.subplots(figsize=(side + 3, side + 2))
    try:
        image = ax.imshow(
            values,
            cmap="Blues",
            interpolation="nearest",
            vmin=0,
            vmax=1.0 if normalize else max(1.0, float(values.max())),
        )
        ticks = np.arange(size)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Lớp dự đoán")
        ax.set_ylabel("Lớp thực tế")
        ax.set_title(title, fontsize=14, pad=16)
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            label="Tỷ lệ theo hàng" if normalize else "Số lượng",
        )

        if size <= 20:
            midpoint = image.norm.vmax / 2
            for row, col in np.ndindex(values.shape):
                if matrix[row, col] > 0:
                    text = (
                        f"{values[row, col]:.2f}"
                        if normalize
                        else f"{matrix[row, col]:.0f}"
                    )
                    ax.text(
                        col,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if values[row, col] > midpoint else "black",
                    )

        fig.tight_layout()
        fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)

    return path
