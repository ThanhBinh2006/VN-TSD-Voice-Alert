"""Vẽ bounding box + nhãn lên khung hình. Phụ trách: Thành viên 4"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_vietnamese_font() -> Path:
    """Tìm font hỗ trợ Unicode tiếng Việt trên Windows/Linux/macOS."""

    candidates = []

    # Cho phép người dùng chỉ định font riêng qua biến môi trường.
    custom_font = os.environ.get("TSD_VOICE_ALERT_FONT")
    if custom_font:
        candidates.append(Path(custom_font))

    candidates.extend([
        # Kaggle / Ubuntu
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),

        # Windows
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),

        # macOS
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ])

    for font_path in candidates:
        if font_path.is_file():
            return font_path

    raise FileNotFoundError(
        "Không tìm thấy font Unicode hỗ trợ tiếng Việt. "
        "Hãy cài DejaVu Sans hoặc đặt TSD_VOICE_ALERT_FONT."
    )


def draw_detections(frame, detections: list):
    """Vẽ bounding box và nhãn tiếng Việt lên frame BGR.

    Args:
        frame: Ảnh đầu vào dạng OpenCV BGR.
        detections: Danh sách detection gồm bbox, label, confidence.

    Returns:
        Ảnh BGR đã được vẽ. Không thay đổi ảnh gốc.
    """

    if frame is None:
        raise ValueError("frame không được là None")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame phải có kích thước (H, W, 3)")

    if not detections:
        return frame.copy()

    height, width = frame.shape[:2]

    # OpenCV BGR -> Pillow RGB.
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    image = Image.fromarray(rgb)

    draw = ImageDraw.Draw(image)
    font_path = _find_vietnamese_font()

    # Kích thước chữ tự thay đổi theo ảnh.
    base_font_size = max(14, min(28, int(height * 0.035)))
    line_width = max(2, int(height * 0.004))
    padding = 5

    for det in detections:
        bbox = det.get("bbox")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox phải có dạng [x1, y1, x2, y2]")

        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

        # Giới hạn tọa độ trong ảnh.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 < x1 or y2 < y1:
            continue

        # Vẽ bounding box màu xanh lá.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0),
            width=line_width,
        )

        # Nội dung nhãn.
        label = str(det.get("label", "Unknown"))
        confidence = det.get("confidence")

        if confidence is not None:
            text = f"{label} {float(confidence):.2f}"
        else:
            text = label

        # Tự giảm font nếu tên biển quá dài.
        font_size = base_font_size
        max_text_width = max(20, width - 2 * padding - 4)

        while True:
            font = ImageFont.truetype(str(font_path), font_size)
            text_bbox = draw.textbbox((0, 0), text, font=font)

            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width <= max_text_width or font_size <= 10:
                break

            font_size -= 1

        # Nền nhãn màu xanh đậm.
        box_width = text_width + padding * 2
        box_height = text_height + padding * 2

        label_left = min(x1, max(0, width - box_width))
        label_top = y1 - box_height

        # Nếu không đủ chỗ phía trên, chuyển nhãn vào trong box.
        if label_top < 0:
            label_top = y1

        label_top = min(label_top, height - box_height)

        draw.rounded_rectangle(
            (
                label_left,
                label_top,
                label_left + box_width,
                label_top + box_height,
            ),
            radius=4,
            fill=(0, 120, 0),
        )

        # Vẽ chữ trắng, hỗ trợ tiếng Việt có dấu.
        draw.text(
            (
                label_left + padding - text_bbox[0],
                label_top + padding - text_bbox[1],
            ),
            text,
            font=font,
            fill=(255, 255, 255),
        )

    # Pillow RGB -> OpenCV BGR.
    result_rgb = np.asarray(image)
    result_bgr = np.ascontiguousarray(result_rgb[:, :, ::-1])

    return result_bgr