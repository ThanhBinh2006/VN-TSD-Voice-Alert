"""Online preprocessing/augmentation khi huấn luyện (mặc định TẮT: online_preprocess.enabled=false).

Phụ trách: Thành viên 1 (YOLO) + Thành viên 2 (Faster R-CNN)
"""
from __future__ import annotations


def _val(op: dict, key: str, default: float) -> float:
    """Đọc tham số dạng số thực từ op, hỗ trợ boolean (True -> default, False -> 0.0)."""
    v = op.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return default if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _hsv(op: dict, key: str, default: float) -> float:
    """Đọc giá trị HSV, ưu tiên key trực tiếp rồi đến cờ color_jitter."""
    if key in op and op[key] is not None:
        return float(op[key])
    return default if op.get("color_jitter", True) else 0.0


def build_online_transforms(cfg: dict, backend: str):
    """Đọc cfg["online_preprocess"] và sinh tham số augment tương ứng.

    - backend == "ultralytics": trả dict tham số train của Ultralytics (tắt toàn bộ khi enabled=false).
    - backend == "torchvision": trả transform Compose của torchvision cho Faster R-CNN.
    """
    op = cfg.get("online_preprocess", {})
    enabled = bool(op.get("enabled", False))

    if backend == "ultralytics":
        if not enabled:
            out = {k: 0.0 for k in (
                "mosaic", "mixup", "fliplr", "flipud", "hsv_h", "hsv_s",
                "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "erasing"
            )}
            out["auto_augment"] = None
            return out

        jitter = float(op.get("jitter", 0.1))
        return {
            "mosaic": _val(op, "mosaic", 1.0),
            "mixup": _val(op, "mixup", 0.15),
            "fliplr": 0.5 if op.get("hflip") else 0.0,
            "flipud": 0.0,
            "hsv_h": _hsv(op, "hsv_h", 0.015),
            "hsv_s": _hsv(op, "hsv_s", 0.7),
            "hsv_v": _hsv(op, "hsv_v", 0.4),
            "degrees": _val(op, "degrees", 0.0),
            "translate": _val(op, "translate", jitter),
            "scale": _val(op, "scale", float(op.get("jitter", 0.5))),
        }

    if backend == "torchvision":
        from torchvision import transforms

        if not enabled:
            return transforms.Compose([])
        t_list = []
        if op.get("hflip"):
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))

        h, s, v = _hsv(op, "hsv_h", 0.015), _hsv(op, "hsv_s", 0.7), _hsv(op, "hsv_v", 0.4)
        if h > 0 or s > 0 or v > 0 or op.get("color_jitter"):
            t_list.append(transforms.ColorJitter(
                brightness=v or 0.2, contrast=0.2, saturation=s or 0.2, hue=min(h, 0.5)
            ))
        return transforms.Compose(t_list)

    raise ValueError(f"Backend không hợp lệ: {backend!r}. Chọn 'ultralytics' hoặc 'torchvision'.")
