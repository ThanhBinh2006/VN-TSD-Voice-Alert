"""Tiện ích xử lý ảnh dùng chung cho phân tích EDA và tiền xử lý.

Cung cấp:
  - crop_active_region: Loại bỏ viền đệm màu xám (114, 114, 114) sinh ra do letterbox
  - calculate_dhash: Tính mã Difference Hash (dHash) chuẩn trên vùng ảnh thực tế
  - calculate_brightness_sharpness: Đo độ sáng và độ nét quang học
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def crop_active_region(img: Image.Image | np.ndarray) -> Image.Image:
    """Loại bỏ hoàn toàn dải màu xám đệm (giá trị 114) sinh ra khi letterbox-resize.

    Đảm bảo việc tính mã băm dHash chỉ dựa trên nội dung bối cảnh thực của ảnh,
    tránh nhận nhầm hai ảnh khác nhau là giống nhau chỉ vì có cùng dải đệm màu xám.
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img

    w, h = pil_img.size
    arr = np.array(pil_img.convert("L"))

    # 1. Kiểm tra viền đệm trên và dưới (Landscape letterbox)
    is_top_pad = np.all(np.abs(arr[0:20, :].astype(int) - 114) <= 2)
    is_bot_pad = np.all(np.abs(arr[-20:, :].astype(int) - 114) <= 2)
    if is_top_pad and is_bot_pad:
        non_pad_rows = np.where(np.any(np.abs(arr.astype(int) - 114) > 3, axis=1))[0]
        if len(non_pad_rows) > 0:
            y1, y2 = non_pad_rows[0], non_pad_rows[-1]
            if (y2 - y1) >= 40:
                return pil_img.crop((0, int(y1), w, int(y2 + 1)))

    # 2. Kiểm tra viền đệm trái và phải (Portrait letterbox)
    is_left_pad = np.all(np.abs(arr[:, 0:20].astype(int) - 114) <= 2)
    is_right_pad = np.all(np.abs(arr[:, -20:].astype(int) - 114) <= 2)
    if is_left_pad and is_right_pad:
        non_pad_cols = np.where(np.any(np.abs(arr.astype(int) - 114) > 3, axis=0))[0]
        if len(non_pad_cols) > 0:
            x1, x2 = non_pad_cols[0], non_pad_cols[-1]
            if (x2 - x1) >= 40:
                return pil_img.crop((int(x1), 0, int(x2 + 1), h))

    return pil_img


def calculate_dhash(
    img_or_path: Image.Image | np.ndarray | str | Path,
    hash_size: int = 32,
    strip_padding: bool = True,
) -> np.ndarray | None:
    """Tính mã băm dHash (Difference Hash) dạng boolean vector độ dài hash_size^2.

    Mặc định strip_padding=True sẽ tự động bóc tách vùng màu xám padding
    trước khi resize về (hash_size + 1, hash_size).
    """
    try:
        if isinstance(img_or_path, (str, Path)):
            p = Path(img_or_path)
            if not p.exists():
                return None
            img = Image.open(p)
        elif isinstance(img_or_path, np.ndarray):
            if img_or_path.ndim == 2:
                img = Image.fromarray(img_or_path)
            else:
                img = Image.fromarray(cv2.cvtColor(img_or_path, cv2.COLOR_BGR2RGB))
        else:
            img = img_or_path

        if strip_padding:
            img = crop_active_region(img)

        gray = img.convert("L").resize(
            (hash_size + 1, hash_size),
            Image.Resampling.BILINEAR,
        )
        arr = np.array(gray, dtype=np.int16)
        diff = (arr[:, 1:] > arr[:, :-1]).flatten()
        return diff
    except Exception:
        return None


def calculate_brightness_sharpness(
    img_path: str | Path,
) -> tuple[float, float] | None:
    """Tính độ sáng trung bình [0, 255] và độ nét Laplacian variance."""
    p = Path(img_path)
    if not p.exists():
        return None
    try:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return brightness, sharpness
    except Exception:
        return None
