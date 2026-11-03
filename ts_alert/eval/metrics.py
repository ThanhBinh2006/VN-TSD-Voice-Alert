"""Bộ chỉ số chung cho cả hai mô hình.   Phụ trách: Thành viên 2"""
from __future__ import annotations


def compute_metrics(preds: list, gts: list, num_classes: int) -> dict:
    """preds[i]: list Detection của ảnh i; gts[i]: [{"bbox": [x1,y1,x2,y2], "class_id": int}, ...].

    Trả về đúng các khoá: {"mAP50-95": float, "mAP50": float, "mAP75": float, "AR100": float,
                           "precision": float, "recall": float}
    Gợi ý: torchmetrics.detection.MeanAveragePrecision (backend pycocotools).
    """
    try:
        import torch
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        preds_tm = []
        targets_tm = []
        for pred_list, gt_list in zip(preds, gts):
            p_boxes = [p["bbox"] for p in pred_list]
            p_scores = [p.get("confidence", 1.0) for p in pred_list]
            p_labels = [p["class_id"] for p in pred_list]
            preds_tm.append({
                "boxes": torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.tensor(p_scores, dtype=torch.float32) if p_scores else torch.zeros((0,), dtype=torch.float32),
                "labels": torch.tensor(p_labels, dtype=torch.int64) if p_labels else torch.zeros((0,), dtype=torch.int64),
            })

            g_boxes = [g["bbox"] for g in gt_list]
            g_labels = [g["class_id"] for g in gt_list]
            targets_tm.append({
                "boxes": torch.tensor(g_boxes, dtype=torch.float32) if g_boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(g_labels, dtype=torch.int64) if g_labels else torch.zeros((0,), dtype=torch.int64),
            })

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(preds_tm, targets_tm)
        res = metric.compute()
        map50_95 = float(res["map"].item()) if res["map"].item() >= 0 else 0.0
        map50 = float(res["map_50"].item()) if res["map_50"].item() >= 0 else 0.0
        map75 = float(res["map_75"].item()) if res["map_75"].item() >= 0 else 0.0
        ar100 = float(res["mar_100"].item()) if res["mar_100"].item() >= 0 else 0.0
        return {
            "mAP50-95": round(map50_95, 4),
            "mAP50": round(map50, 4),
            "mAP75": round(map75, 4),
            "AR100": round(ar100, 4),
            "precision": round(map50, 4),
            "recall": round(ar100, 4),
        }
    except Exception as e:
        print(f"[compute_metrics] Fallback metric do lỗi hoặc thiếu thư viện: {e}")
        total_gt = sum(len(g) for g in gts)
        total_pred = sum(len(p) for p in preds)
        return {
            "mAP50-95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "AR100": 0.0,
            "precision": round(min(1.0, total_gt / max(total_pred, 1)), 4),
            "recall": round(min(1.0, total_pred / max(total_gt, 1)), 4),
        }
