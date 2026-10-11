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
    # TODO: Thành viên 1 + 2
    raise NotImplementedError("TODO: Thành viên 1 + 2")
