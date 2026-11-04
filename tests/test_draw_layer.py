import numpy as np

from ts_alert.postprocess.draw_layer import draw_detections


def test_draw_detections():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        {
            "bbox": [100, 100, 200, 200],
            "label": "stop",
            "class_id": 0,
            "confidence": 0.95,
        }
    ]

    original = frame.copy()

    result = draw_detections(frame, detections)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype

    # Hàm phải tạo bản sao, không sửa frame gốc.
    assert np.array_equal(frame, original)

    # Kết quả phải có pixel khác 0 do bbox/text đã được vẽ.
    assert np.any(result != 0)