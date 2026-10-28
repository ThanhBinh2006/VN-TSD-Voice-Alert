"""Vẽ bounding box + nhãn lên khung hình.   Phụ trách: Thành viên 4"""
from __future__ import annotations

import cv2


def draw_detections(frame, detections: list):
    """Vẽ box + "<nhãn> <độ tin cậy>" cho từng detection lên frame (BGR), trả frame đã vẽ.

    Lưu ý: cv2.putText không hiển thị tiếng Việt có dấu -> dùng Pillow (ImageFont TTF) hoặc
    hiển thị nhãn không dấu. Không sửa frame gốc nếu cần dùng lại (frame.copy()).
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        text = f"{label} {confidence:.2f}"

        cv2.putText(
            output,
            text,
            (int(x1), max(int(y1) - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output