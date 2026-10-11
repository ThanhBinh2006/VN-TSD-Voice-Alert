"""Vẽ bounding box + nhãn lên khung hình.   Phụ trách: Thành viên 4"""
from __future__ import annotations


def draw_detections(frame, detections: list):
    """Vẽ box + "<nhãn> <độ tin cậy>" cho từng detection lên frame (BGR), trả frame đã vẽ.

    Lưu ý: cv2.putText không hiển thị tiếng Việt có dấu -> dùng Pillow (ImageFont TTF) hoặc
    hiển thị nhãn không dấu. Không sửa frame gốc nếu cần dùng lại (frame.copy()).
    """
    # TODO: Thành viên 4
    raise NotImplementedError("TODO: Thành viên 4")
