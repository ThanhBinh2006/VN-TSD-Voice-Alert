"""Online preprocessing/augmentation khi huấn luyện (mặc định TẮT: online_preprocess.enabled=false).

Phụ trách: Thành viên 1 (YOLO) + Thành viên 2 (Faster R-CNN)
"""
from __future__ import annotations


def build_online_transforms(cfg: dict, backend: str):
    """Đọc cfg["online_preprocess"] (letterbox, normalize, hflip, mosaic, mixup, scale, degrees, translate, hsv_*...).

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
            def _parse_val(key: str, default: float) -> float:
                v = op.get(key)
                if v is None:
                    return default
                if isinstance(v, bool):
                    return default if v else 0.0
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            mosaic = _parse_val("mosaic", 1.0)
            mixup = _parse_val("mixup", 0.15)
            fliplr = 0.5 if op.get("hflip", False) else 0.0

            # Scale & translate (hỗ trợ jitter cũ làm fallback)
            jitter = float(op.get("jitter", 0.1))
            scale = _parse_val("scale", jitter if "jitter" in op else 0.5)
            translate = _parse_val("translate", jitter if "jitter" in op else 0.1)
            degrees = _parse_val("degrees", 0.0)

            # HSV parameters
            if "hsv_h" in op and op["hsv_h"] is not None:
                hsv_h = float(op["hsv_h"])
            elif "color_jitter" in op:
                hsv_h = 0.015 if op["color_jitter"] else 0.0
            else:
                hsv_h = 0.015

            if "hsv_s" in op and op["hsv_s"] is not None:
                hsv_s = float(op["hsv_s"])
            elif "color_jitter" in op:
                hsv_s = 0.7 if op["color_jitter"] else 0.0
            else:
                hsv_s = 0.7

            if "hsv_v" in op and op["hsv_v"] is not None:
                hsv_v = float(op["hsv_v"])
            elif "color_jitter" in op:
                hsv_v = 0.4 if op["color_jitter"] else 0.0
            else:
                hsv_v = 0.4

            return {
                "mosaic": mosaic,
                "mixup": mixup,
                "fliplr": fliplr,
                "flipud": 0.0,
                "hsv_h": hsv_h,
                "hsv_s": hsv_s,
                "hsv_v": hsv_v,
                "degrees": degrees,
                "translate": translate,
                "scale": scale,
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
