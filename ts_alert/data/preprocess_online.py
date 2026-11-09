"""Online preprocessing/augmentation khi huấn luyện (mặc định TẮT: online_preprocess.enabled=false).

Phụ trách: Thành viên 1 (YOLO) + Thành viên 2 (Faster R-CNN)
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np

BoxXYXY = tuple[float, float, float, float]

_OCCLUDER_COLORS: tuple[tuple[int, int, int], ...] = (
    (25, 75, 35),
    (35, 50, 80),
    (40, 40, 40),
    (70, 70, 70),
    (210, 210, 215),
)


def _val(op: dict[str, Any], key: str, default: float) -> float:
    """Đọc tham số số thực, hỗ trợ boolean (True -> default, False -> 0.0)."""
    value = op.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return default if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_online_transforms(cfg: dict[str, Any], backend: str) -> Any:
    """Đọc cấu hình online_preprocess và sinh tham số augment theo backend.

    Backend ``ultralytics`` trả dict tham số train; khi tắt, mọi tham số
    augment bằng 0.0 và auto_augment bằng None.
    Backend ``torchvision`` trả transform Compose cho Faster R-CNN.

    Raises:
        ValueError: Backend không được hỗ trợ.
    """
    op = cfg.get("online_preprocess", {})
    enabled = bool(op.get("enabled", False))

    if backend == "ultralytics":
        if not enabled:
            out: dict[str, Any] = {
                key: 0.0
                for key in (
                    "mosaic",
                    "mixup",
                    "fliplr",
                    "flipud",
                    "hsv_h",
                    "hsv_s",
                    "hsv_v",
                    "degrees",
                    "translate",
                    "scale",
                    "shear",
                    "perspective",
                    "erasing",
                )
            }
            out["auto_augment"] = None
            return out

        return {
            "mosaic": _val(op, "mosaic", 1.0),
            "mixup": _val(op, "mixup", 0.15),
            "fliplr": 0.5 if op.get("hflip") else 0.0,
            "flipud": 0.0,
            "hsv_h": _val(op, "hsv_h", 0.015),
            "hsv_s": _val(op, "hsv_s", 0.7),
            "hsv_v": _val(op, "hsv_v", 0.4),
            "degrees": _val(op, "degrees", 0.0),
            "translate": _val(op, "translate", 0.1),
            "scale": _val(op, "scale", 0.5),
        }

    if backend == "torchvision":
        from torchvision import transforms

        if not enabled:
            return transforms.Compose([])

        t_list: list[Any] = []
        if op.get("hflip"):
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))

        h = _val(op, "hsv_h", 0.015)
        s = _val(op, "hsv_s", 0.7)
        v = _val(op, "hsv_v", 0.4)
        if h > 0 or s > 0 or v > 0:
            t_list.append(
                transforms.ColorJitter(
                    brightness=v,
                    contrast=0.2 if (v > 0 or s > 0) else 0.0,
                    saturation=s,
                    hue=min(h, 0.5),
                )
            )
        return transforms.Compose(t_list)

    raise ValueError(
        f"Backend không hợp lệ: {backend!r}. Chọn 'ultralytics' hoặc 'torchvision'."
    )


def _finite_float(value: Any, name: str) -> float:
    """Đọc số thực hữu hạn; báo ValueError khi giá trị không hợp lệ."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} phải là số hữu hạn.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} phải là số hữu hạn.")
    return result


def _validate_occlusion_p(value: Any) -> float:
    """Kiểm tra xác suất che khuất thuộc đoạn [0, 1]."""
    probability = _finite_float(value, "occlusion_p")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("occlusion_p phải nằm trong [0.0, 1.0].")
    return probability


def _validate_occlusion_ratio(value: Any) -> float:
    """Kiểm tra tỷ lệ che khuất hữu hạn và giới hạn vào [0.05, 0.90]."""
    ratio = _finite_float(value, "occlusion_ratio")
    return max(0.05, min(0.90, ratio))


def _validate_image(img: np.ndarray) -> None:
    """Kiểm tra ảnh uint8 không rỗng, có shape (H, W, 3)."""
    if (
        not isinstance(img, np.ndarray)
        or img.dtype != np.uint8
        or img.ndim != 3
        or img.shape[2] != 3
        or min(img.shape[:2]) == 0
    ):
        raise ValueError("Ảnh phải là mảng uint8 không rỗng, shape (H, W, 3).")


def _validate_box_xyxy(
    box: Sequence[float],
    width: int,
    height: int,
) -> BoxXYXY:
    """Kiểm tra box hữu hạn, có diện tích dương và nằm trong ảnh."""
    try:
        if len(box) != 4:
            raise ValueError("Box phải chứa đúng bốn tọa độ.")
        x1, y1, x2, y2 = (
            _finite_float(value, "Tọa độ box") for value in box
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Box phải chứa đúng bốn tọa độ hữu hạn.") from exc

    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("Box phải có diện tích dương và nằm trong ảnh.")
    return x1, y1, x2, y2


def _draw_occluder_inplace(
    out_img: np.ndarray,
    box: BoxXYXY,
    rand: Any,
    ratio: float,
) -> None:
    """Vẽ một vật cản tại góc/cạnh box trực tiếp lên ảnh đã được sao chép."""
    import cv2

    height, width = out_img.shape[:2]
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1

    mode = rand.choice(("corner", "edge"))
    color = rand.choice(_OCCLUDER_COLORS)
    side = rand.randrange(4)

    if mode == "corner":
        low = max(0.05, 0.75 * ratio)
        high = min(0.95, 1.25 * ratio)
        rw = bw * rand.uniform(low, high)
        rh = bh * rand.uniform(low, high)
        x1 = bx1 if side in (0, 2) else bx2 - rw
        y1 = by1 if side in (0, 1) else by2 - rh
        x2, y2 = x1 + rw, y1 + rh
    else:
        depth_ratio = rand.uniform(
            max(0.05, 0.65 * ratio),
            min(0.95, ratio),
        )
        span_ratio = rand.uniform(0.65, 1.0)
        if side in (0, 1):
            depth, span = bh * depth_ratio, bw * span_ratio
            x1 = bx1 + rand.uniform(0.0, max(0.0, bw - span))
            y1 = by1 if side == 0 else by2 - depth
            x2, y2 = x1 + span, y1 + depth
        else:
            depth, span = bw * depth_ratio, bh * span_ratio
            x1 = bx1 if side == 2 else bx2 - depth
            y1 = by1 + rand.uniform(0.0, max(0.0, bh - span))
            x2, y2 = x1 + depth, y1 + span

    # Biên phải/dưới exclusive, bảo đảm vùng vẽ có ít nhất một pixel.
    ox1 = max(0, min(width - 1, math.floor(x1)))
    oy1 = max(0, min(height - 1, math.floor(y1)))
    ox2 = min(width, max(ox1 + 1, math.ceil(x2)))
    oy2 = min(height, max(oy1 + 1, math.ceil(y2)))

    region = out_img[oy1:oy2, ox1:ox2]
    fill_type = rand.choice(("rect", "ellipse", "noise"))

    if fill_type == "rect":
        region[:] = color
    elif fill_type == "ellipse":
        patch_height, patch_width = region.shape[:2]
        center = ((patch_width - 1) // 2, (patch_height - 1) // 2)
        axes = (max(1, patch_width // 2), max(1, patch_height // 2))
        cv2.ellipse(
            region, center, axes, rand.uniform(-30.0, 30.0),
            0, 360, color, -1,
        )
    else:
        # RNG NumPy riêng lấy seed từ RNG của worker, không dùng trạng thái global.
        noise_rng = np.random.default_rng(rand.getrandbits(64))
        noise = noise_rng.normal(
            0.0, 10.0, size=region.shape[:2] + (1,)
        ).astype(np.float32)
        patch = np.clip(
            np.asarray(color, dtype=np.float32) + noise,
            0.0,
            255.0,
        ).astype(np.uint8)
        region[:] = cv2.GaussianBlur(patch, (5, 5), 0)


def apply_partial_occlusion(
    img: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    rng: random.Random | Any = None,
    occlusion_ratio: float = 0.30,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Che một góc/cạnh biển báo, giữ nguyên ảnh đầu vào và box nhãn.

    Ảnh dùng thứ tự kênh BGR hoặc RGB của bên gọi, không chuyển đổi màu.
    ``occlusion_ratio`` điều chỉnh kích thước tuyến tính của vật cản,
    không biểu diễn chính xác tỷ lệ diện tích bị che.
    RNG mặc định là module random, sử dụng seed của tiến trình/worker.

    Raises:
        ValueError: Ảnh, box hoặc tỷ lệ che khuất không hợp lệ.
    """
    _validate_image(img)
    height, width = img.shape[:2]
    box = _validate_box_xyxy(box_xyxy, width, height)
    ratio = _validate_occlusion_ratio(occlusion_ratio)
    rand = random if rng is None else rng

    out_img = img.copy()
    _draw_occluder_inplace(out_img, box, rand, ratio)
    return out_img, box_xyxy


def apply_partial_occlusion_to_boxes(
    img: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]] | np.ndarray,
    rng: Any = None,
    occlusion_p: float = 0.30,
    occlusion_ratio: float = 0.30,
    *,
    copy_on_skip: bool = False,
) -> np.ndarray:
    """Che ngẫu nhiên các box hợp lệ với xác suất áp dụng trên mỗi ảnh.

    Clip box vào ảnh; bỏ box lỗi, không hữu hạn hoặc nhỏ hơn 4 pixel
    trên một trong hai trục. Khi áp dụng, chỉ sao chép ảnh đúng một lần.
    Khi bỏ qua, trả ảnh gốc hoặc bản sao theo ``copy_on_skip``.
    Nhánh bỏ qua sớm không kiểm tra nội dung ảnh và box.

    Raises:
        ValueError: Xác suất, tỷ lệ hoặc ảnh cần xử lý không hợp lệ.
    """
    probability = _validate_occlusion_p(occlusion_p)
    ratio = _validate_occlusion_ratio(occlusion_ratio)
    rand = random if rng is None else rng

    if (
        probability <= 0.0
        or len(boxes_xyxy) == 0
        or rand.random() >= probability
    ):
        return img.copy() if copy_on_skip else img

    _validate_image(img)
    height, width = img.shape[:2]
    valid_boxes: list[BoxXYXY] = []

    for raw_box in boxes_xyxy:
        try:
            if len(raw_box) != 4:
                continue
            x1, y1, x2, y2 = (
                _finite_float(value, "Tọa độ box") for value in raw_box
            )
        except (TypeError, ValueError, OverflowError):
            continue

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 - x1 < 4.0 or y2 - y1 < 4.0:
            continue
        valid_boxes.append((x1, y1, x2, y2))

    if not valid_boxes:
        return img.copy() if copy_on_skip else img

    out_img = img.copy()
    k = rand.randint(1, len(valid_boxes))
    for index in rand.sample(range(len(valid_boxes)), k):
        _draw_occluder_inplace(out_img, valid_boxes[index], rand, ratio)
    return out_img


def get_online_occlusion_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Đọc cấu hình che khuất riêng, không thêm kwargs train Ultralytics.

    Khi bật, ``occlusion_p`` và ``occlusion_ratio`` mặc định là 0.30
    nếu thiếu hoặc bằng None. Xác suất dạng số, kể cả boolean,
    được chuyển sang float. Tỷ lệ được giới hạn vào [0.05, 0.90].

    Raises:
        ValueError: Xác suất ngoài [0, 1] hoặc tham số không hữu hạn.
    """
    op = cfg.get("online_preprocess") or {}
    enabled = bool(op.get("enabled", False))
    if not enabled:
        return {"enabled": False, "p": 0.0, "ratio": 0.0}

    probability_value = op.get("occlusion_p")
    if probability_value is None:
        probability_value = 0.30

    ratio_value = op.get("occlusion_ratio")
    if ratio_value is None:
        ratio_value = 0.30

    return {
        "enabled": True,
        "p": _validate_occlusion_p(probability_value),
        "ratio": _validate_occlusion_ratio(ratio_value),
    }
