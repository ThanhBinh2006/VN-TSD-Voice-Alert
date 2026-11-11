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


def compute_pr_curve(
    preds: list,
    gts: list,
    num_classes: int,
    iou_threshold: float = 0.50,
    thresholds: list[float] | np.ndarray | None = None,
    default_conf: float = 0.25,
) -> dict:
    """Tính PR/F1 micro theo confidence, dùng cùng matching với compute_metrics.

    Chỉ ghép lại khi tập dự đoán được giữ trong một ảnh thay đổi.
    PR-AUC dùng nội suy AP toàn điểm trên đường bao precision.
    """
    predictions, targets = _prepare(
        preds, gts, num_classes, iou_threshold, 0.0
    )
    if not np.isfinite(default_conf) or not 0.0 <= default_conf <= 1.0:
        raise ValueError("default_conf phải là số hữu hạn thuộc [0, 1].")

    threshold_values = np.asarray(
        [round(0.01 * i, 2) for i in range(1, 100)]
        if thresholds is None
        else thresholds,
        dtype=np.float64,
    )
    if (
        threshold_values.ndim != 1
        or threshold_values.size == 0
        or not np.isfinite(threshold_values).all()
        or np.any(threshold_values < 0.0)
        or np.any(threshold_values > 1.0)
    ):
        raise ValueError(
            "thresholds phải là danh sách một chiều, không rỗng, "
            "chứa các số hữu hạn thuộc [0, 1]."
        )

    unique_thresholds, inverse = np.unique(
        threshold_values, return_inverse=True
    )
    totals = np.zeros((len(unique_thresholds), 3), dtype=np.int64)

    for prediction, target in zip(predictions, targets):
        sorted_scores = np.sort(prediction[2])
        retained_counts = len(sorted_scores) - np.searchsorted(
            sorted_scores, unique_thresholds, side="left"
        )
        previous_count = -1
        matched = 0
        target_count = len(target[0])

        for index, count_value in enumerate(retained_counts):
            count = int(count_value)
            # Các tập score >= threshold lồng nhau: cùng số lượng thì cùng tập.
            if count != previous_count:
                filtered = _filter_predictions(
                    prediction, float(unique_thresholds[index])
                )
                matched = len(
                    _greedy_matches(
                        filtered, target, iou_threshold, same_class=True
                    )
                )
                previous_count = count

            totals[index, 0] += matched
            totals[index, 1] += count - matched
            totals[index, 2] += target_count - matched

    counts = totals[inverse]
    raw_precision: list[float] = []
    raw_recall: list[float] = []
    raw_f1: list[float] = []

    for tp_value, fp_value, fn_value in counts:
        tp, fp, fn = int(tp_value), int(fp_value), int(fn_value)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        raw_precision.append(precision)
        raw_recall.append(recall)
        raw_f1.append(f1)

    maximum_f1 = max(raw_f1)
    best_index = max(
        (
            index
            for index, value in enumerate(raw_f1)
            if abs(value - maximum_f1) <= 1e-12
        ),
        key=lambda index: float(threshold_values[index]),
    )
    best_threshold = (
        float(threshold_values[best_index])
        if maximum_f1 > 0.0
        else float(default_conf)
    )

    recall_values = np.asarray(raw_recall, dtype=np.float64)
    precision_values = np.asarray(raw_precision, dtype=np.float64)
    order = np.argsort(recall_values, kind="stable")
    sorted_recall = np.concatenate(
        ([0.0], recall_values[order], [1.0])
    )
    sorted_precision = np.concatenate(
        (
            [1.0 if np.max(recall_values) > 0.0 else 0.0],
            precision_values[order],
            [0.0],
        )
    )
    envelope = np.maximum.accumulate(sorted_precision[::-1])[::-1]
    pr_auc = float(
        np.sum(np.diff(sorted_recall) * envelope[1:])
    )

    return {
        "iou_threshold": float(iou_threshold),
        "thresholds": threshold_values.tolist(),
        "precision": [round(value, 4) for value in raw_precision],
        "recall": [round(value, 4) for value in raw_recall],
        "f1": [round(value, 4) for value in raw_f1],
        "tp": [int(value) for value in counts[:, 0]],
        "fp": [int(value) for value in counts[:, 1]],
        "fn": [int(value) for value in counts[:, 2]],
        "best_threshold": best_threshold,
        "best_precision": round(raw_precision[best_index], 4),
        "best_recall": round(raw_recall[best_index], 4),
        "best_f1": round(raw_f1[best_index], 4),
        "pr_auc": round(float(np.clip(pr_auc, 0.0, 1.0)), 4),
    }


def _validate_pr_plot_data(
    curve: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kiểm tra các dãy thresholds, precision, recall và F1 để vẽ."""
    arrays: list[np.ndarray] = []
    for key in ("thresholds", "precision", "recall", "f1"):
        values = np.asarray(curve[key], dtype=np.float64)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                f"{key} phải là dãy một chiều, không rỗng, "
                "chứa các số hữu hạn thuộc [0, 1]."
            )
        arrays.append(values)

    if any(len(values) != len(arrays[0]) for values in arrays[1:]):
        raise ValueError(
            "thresholds, precision, recall và f1 phải có cùng độ dài."
        )
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _eval_split_style(split: str, index: int) -> tuple[str, str]:
    """Trả màu và kiểu nét nhất quán giữa các biểu đồ evaluation."""
    if split.lower() == "test":
        return "#174A8B", "-"
    if split.lower() == "val":
        return "#E1812C", "--"
    return f"C{index % 10}", "-."


def save_pr_curve_plot(
    curves: dict[str, dict],
    out_path: Path | str,
    selected_threshold: float | None = None,
    title: str = "Precision-Recall & F1-Confidence Curves",
) -> Path:
    """Lưu PNG gồm đường PR và P/R/F1 theo confidence cho các split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("curves không được rỗng.")

    primary_split = "test" if "test" in curves else next(iter(curves))
    if selected_threshold is None:
        selected_threshold = float(
            curves[primary_split]["best_threshold"]
        )
    if (
        not np.isfinite(selected_threshold)
        or not 0.0 <= selected_threshold <= 1.0
    ):
        raise ValueError(
            "selected_threshold phải là số hữu hạn thuộc [0, 1]."
        )
    selected_threshold = float(selected_threshold)

    prepared = {
        split: _validate_pr_plot_data(curve)
        for split, curve in curves.items()
    }
    auc_values: dict[str, float] = {}
    for split, curve in curves.items():
        auc = float(curve["pr_auc"])
        if not np.isfinite(auc) or not 0.0 <= auc <= 1.0:
            raise ValueError("pr_auc phải là số hữu hạn thuộc [0, 1].")
        auc_values[split] = auc

    path = Path(out_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (pr_axis, confidence_axis) = plt.subplots(
        1, 2, figsize=(14, 6)
    )
    try:
        for level in (0.2, 0.4, 0.6, 0.8):
            recall_grid = np.linspace(
                level / (2.0 - level), 1.0, 200
            )
            precision_grid = (
                level * recall_grid / (2.0 * recall_grid - level)
            )
            pr_axis.plot(
                recall_grid,
                precision_grid,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.25,
                zorder=1,
            )
            pr_axis.text(
                0.98,
                level / (2.0 - level),
                f"F1={level:.1f}",
                fontsize=8,
                color="gray",
                alpha=0.7,
                ha="right",
                va="bottom",
            )

        for index, (split, data) in enumerate(prepared.items()):
            thresholds, precision, recall, f1 = data
            color, linestyle = _eval_split_style(split, index)
            label = split.capitalize()
            order = np.argsort(-thresholds, kind="stable")

            pr_axis.plot(
                recall[order],
                precision[order],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label} (PR-AUC={auc_values[split]:.4f})",
                zorder=2,
            )

            selected_index = int(
                np.argmin(np.abs(thresholds - selected_threshold))
            )
            selected_p = float(precision[selected_index])
            selected_r = float(recall[selected_index])
            selected_f1 = float(f1[selected_index])
            pr_axis.scatter(
                selected_r,
                selected_p,
                marker="*",
                s=180,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
            pr_axis.annotate(
                f"{label}: P={selected_p:.4f}, "
                f"R={selected_r:.4f}, F1={selected_f1:.4f}",
                xy=(selected_r, selected_p),
                xytext=(6, -18 - 16 * index),
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="left" if selected_r < 0.5 else "right",
                annotation_clip=False,
            )

            confidence_order = np.argsort(thresholds, kind="stable")
            confidence_axis.plot(
                thresholds[confidence_order],
                f1[confidence_order],
                color=color,
                linestyle=linestyle,
                linewidth=2.8 if split == primary_split else 2,
                label=f"{label} F1",
            )
            if split == primary_split:
                confidence_axis.plot(
                    thresholds[confidence_order],
                    precision[confidence_order],
                    color="#2A9D8F",
                    linestyle=":",
                    linewidth=1.8,
                    label=f"{label} Precision",
                )
                confidence_axis.plot(
                    thresholds[confidence_order],
                    recall[confidence_order],
                    color="#9467BD",
                    linestyle="-.",
                    linewidth=1.8,
                    label=f"{label} Recall",
                )

        confidence_axis.axvline(
            selected_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"conf* = {selected_threshold:.2f}",
        )
        pr_axis.set_title("Precision-Recall Curve")
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        confidence_axis.set_title(
            "F1 / Precision / Recall vs Confidence"
        )
        confidence_axis.set_xlabel("Confidence")
        confidence_axis.set_ylabel("Score")

        for axis in (pr_axis, confidence_axis):
            axis.set_xlim(0.0, 1.02)
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.3)
            axis.legend(loc="best", fontsize=9)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)

    return path


def save_eval_comparison_plot(
    split_metrics: dict[str, dict],
    out_path: Path | str,
    title: str = "Evaluation Metrics Summary (Test vs Val)",
) -> Path:
    """Lưu biểu đồ cột nhóm cho các metric số hữu hạn của từng split."""
    from numbers import Real

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not split_metrics:
        raise ValueError("split_metrics không được rỗng.")

    standard_metrics = (
        "mAP50-95",
        "mAP50",
        "mAP75",
        "AR100",
        "precision",
        "recall",
        "f1",
    )

    def numeric_value(value: object) -> float | None:
        """Lấy số thực hữu hạn, loại boolean và các giá trị thiếu."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return None
        number = float(value)
        return number if np.isfinite(number) else None

    prepared: dict[str, dict[str, float]] = {}
    subtitles: list[str] = []
    for split, values in split_metrics.items():
        prepared[split] = {}
        for key in standard_metrics:
            value = numeric_value(values.get(key))
            if value is not None:
                prepared[split][key] = value

        confidence = numeric_value(values.get("conf_threshold"))
        if confidence is not None:
            subtitles.append(
                f"{split.capitalize()}: conf* = {confidence:.2f}"
            )

    metric_names = [
        key
        for key in standard_metrics
        if any(key in values for values in prepared.values())
    ]
    if not metric_names:
        raise ValueError("Không có metric chuẩn mang giá trị số hữu hạn.")

    path = Path(out_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(14, 6))
    try:
        positions = np.arange(len(metric_names), dtype=np.float64)
        width = 0.8 / len(prepared)
        drawn_values: list[float] = []

        for index, (split, values) in enumerate(prepared.items()):
            available = [
                (position, values[key])
                for position, key in enumerate(metric_names)
                if key in values
            ]
            if not available:
                continue

            offset = (index - (len(prepared) - 1) / 2.0) * width
            color, _ = _eval_split_style(split, index)
            heights = [value for _, value in available]
            bars = axis.bar(
                [positions[position] + offset for position, _ in available],
                heights,
                width=width,
                color=color,
                label=split.capitalize(),
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in heights],
                padding=3,
                fontsize=9,
            )
            drawn_values.extend(heights)

        lower = min(0.0, min(drawn_values))
        upper = max(1.0, max(drawn_values))
        margin = (upper - lower) * 0.10
        axis.set_ylim(
            lower - margin if lower < 0.0 else 0.0,
            upper + margin,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(metric_names)
        axis.set_ylabel("Score")
        axis.set_title(
            "\n".join(subtitles) if subtitles else "",
            fontsize=10,
            pad=12,
        )
        axis.grid(axis="y", alpha=0.3, zorder=0)
        axis.set_axisbelow(True)
        axis.legend(loc="best")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)

    return path
