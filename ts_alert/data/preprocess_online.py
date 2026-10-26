"""Online preprocessing/augmentation khi huấn luyện (mặc định TẮT: online_preprocess.enabled=false).

Phụ trách: Thành viên 1 (YOLO) + Thành viên 2 (Faster R-CNN)
"""
from __future__ import annotations


def build_online_transforms(cfg: dict, backend: str):
    """Đọc cfg["online_preprocess"] (letterbox, normalize, hflip, color_jitter, mosaic, mixup, jitter).

    - backend == "ultralytics": trả dict tham số train của Ultralytics (mosaic, mixup, fliplr, hsv_*...).
      Khi enabled=false phải tắt các augmentation mặc định của Ultralytics (mosaic=0, mixup=0, ...).
    - backend == "torchvision": trả transform cho Dataset của Faster R-CNN (hflip, color_jitter...).
      mosaic/mixup không có sẵn trong torchvision -> tự cài hoặc bỏ qua và ghi chú trong báo cáo.
    """
    op = cfg.get("online_preprocess", {})
    enabled = bool(op.get("enabled", False))

    if backend == "ultralytics":
        if not enabled:
            # Tắt toàn bộ augmentation mặc định của Ultralytics để so sánh công bằng
            return {
                "mosaic": 0.0,
                "mixup": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "erasing": 0.0,
                "auto_augment": None,
            }
        else:
            jitter = float(op.get("jitter", 0.1))
            return {
                "mosaic": 1.0 if op.get("mosaic", False) else 0.0,
                "mixup": 0.15 if op.get("mixup", False) else 0.0,
                "fliplr": 0.5 if op.get("hflip", False) else 0.0,
                "flipud": 0.0,
                "hsv_h": 0.015 if op.get("color_jitter", False) else 0.0,
                "hsv_s": 0.7 if op.get("color_jitter", False) else 0.0,
                "hsv_v": 0.4 if op.get("color_jitter", False) else 0.0,
                "degrees": 0.0,
                "translate": jitter,
                "scale": jitter,
            }

    if backend == "torchvision":
        from torchvision import transforms

        if not enabled:
            return transforms.Compose([])
        t_list = []
        if op.get("hflip"):
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if op.get("color_jitter"):
            t_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        return transforms.Compose(t_list)

    raise ValueError(f"Backend không hợp lệ: {backend!r}. Chọn 'ultralytics' hoặc 'torchvision'.")
