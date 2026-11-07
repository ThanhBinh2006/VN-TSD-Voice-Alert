def test_draw_vietnamese_label():
    import numpy as np

    from ts_alert.postprocess.draw_layer import draw_detections

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    original = frame.copy()

    detections = [{
        "bbox": [30, 40, 180, 190],
        "label": "Đường người đi bộ cắt ngang",
        "confidence": 0.95,
    }]

    result = draw_detections(frame, detections)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert not np.array_equal(result, original)
    assert np.array_equal(frame, original)