"""Bộ chỉ số chung cho cả hai mô hình.   Phụ trách: Thành viên 2"""
from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision


def compute_metrics(preds: list, gts: list, num_classes: int) -> dict:
    """preds[i]: list Detection của ảnh i; gts[i]: [{"bbox": [x1,y1,x2,y2], "class_id": int}, ...].

    Trả về đúng các khoá: {"mAP50-95": float, "mAP50": float, "mAP75": float, "AR100": float,
                           "precision": float, "recall": float}
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    preds_fmt = []
    for p_list in preds:
        if not p_list:
            preds_fmt.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })
        else:
            boxes = [item["bbox"] for item in p_list]
            scores = [item["confidence"] for item in p_list]
            labels = [item["class_id"] for item in p_list]
            preds_fmt.append({
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "scores": torch.as_tensor(scores, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            })

    targets_fmt = []
    for g_list in gts:
        if not g_list:
            targets_fmt.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })
        else:
            boxes = [item["bbox"] for item in g_list]
            labels = [item["class_id"] for item in g_list]
            targets_fmt.append({
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            })

    metric.update(preds_fmt, targets_fmt)
    res = metric.compute()

    map_val = float(res["map"].item()) if res["map"].item() >= 0 else 0.0
    map50_val = float(res["map_50"].item()) if res["map_50"].item() >= 0 else 0.0
    map75_val = float(res["map_75"].item()) if res["map_75"].item() >= 0 else 0.0
    mar100_val = float(res["mar_100"].item()) if res["mar_100"].item() >= 0 else 0.0

    return {
        "mAP50-95": round(map_val, 4),
        "mAP50": round(map50_val, 4),
        "mAP75": round(map75_val, 4),
        "AR100": round(mar100_val, 4),
        "precision": round(map50_val, 4),
        "recall": round(mar100_val, 4),
    }
