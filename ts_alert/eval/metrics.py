"""Bộ chỉ số chung cho cả hai mô hình.   Phụ trách: Thành viên 2"""
from __future__ import annotations


def compute_metrics(preds: list, gts: list, num_classes: int) -> dict:
    """preds[i]: list Detection của ảnh i; gts[i]: [{"bbox": [x1,y1,x2,y2], "class_id": int}, ...].

    Trả về đúng các khoá: {"mAP50-95": float, "mAP50": float, "mAP75": float, "AR100": float,
                           "precision": float, "recall": float}
    Gợi ý: torchmetrics.detection.MeanAveragePrecision (backend pycocotools).
    """
    # TODO: Thành viên 2
    raise NotImplementedError("TODO: Thành viên 2")
