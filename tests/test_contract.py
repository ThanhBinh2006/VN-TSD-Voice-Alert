from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ts_alert.config.resolve import resolve_config
from ts_alert.data.preprocess_online import build_online_transforms
from ts_alert.pipelines.yolo.infer import YoloDetector


class DummyBox:
    def __init__(self, xyxy, cls_id, conf):
        self._xyxy = np.array([xyxy], dtype=np.float32)
        self._cls = np.array([cls_id], dtype=np.float32)
        self._conf = np.array([conf], dtype=np.float32)

    @property
    def xyxy(self):
        m = MagicMock()
        m.__getitem__.return_value.tolist.return_value = self._xyxy[0].tolist()
        return m

    @property
    def cls(self):
        m = MagicMock()
        m.__getitem__.return_value.item.return_value = int(self._cls[0])
        return m

    @property
    def conf(self):
        m = MagicMock()
        m.__getitem__.return_value.item.return_value = float(self._conf[0])
        return m


class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_yolo_contract():
    """Kiểm tra YoloDetector.predict() tuân thủ chặt chẽ docs/DATA_CONTRACT.md:
    [{"bbox": [x1, y1, x2, y2], "label": str, "class_id": int, "confidence": float}]
    """
    cfg = resolve_config("yolo", "local")

    mock_model = MagicMock()
    # Giả lập 2 boxes phát hiện
    boxes = [
        DummyBox([50.0, 60.0, 150.0, 180.0], 0, 0.88),
        DummyBox([200.0, 100.0, 320.0, 240.0], 10, 0.92),
    ]
    mock_model.predict.return_value = [DummyResult(boxes)]

    with patch("ts_alert.pipelines.yolo.infer.load_model", return_value=mock_model):
        detector = YoloDetector("dummy_weights.pt", cfg)

        # 1. Khung hình rỗng -> trả []
        assert detector.predict(None) == []
        assert detector.predict(np.array([])) == []

        # 2. Khung hình hợp lệ
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(frame)

        assert len(detections) == 2
        for d in detections:
            assert "bbox" in d
            assert "label" in d
            assert "class_id" in d
            assert "confidence" in d

            bbox = d["bbox"]
            assert isinstance(bbox, list)
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox
            assert x1 < x2 and y1 < y2

            assert isinstance(d["label"], str) and len(d["label"]) > 0
            assert isinstance(d["class_id"], int)
            assert isinstance(d["confidence"], float)
            assert 0.0 <= d["confidence"] <= 1.0

        # Kiểm tra nhãn khớp configs/classes.csv
        assert detections[0]["class_id"] == 0
        assert detections[0]["label"] == "Đường người đi bộ cắt ngang"
        assert abs(detections[0]["confidence"] - 0.88) < 1e-4

        assert detections[1]["class_id"] == 10
        assert detections[1]["label"] == "Cấm dừng và đỗ xe"
        assert abs(detections[1]["confidence"] - 0.92) < 1e-4


def test_online_transforms_yolo():
    """Kiểm tra build_online_transforms tắt đúng các augment mặc định khi enabled=false."""
    cfg = resolve_config("yolo", "local")

    # Khi enabled = False
    args_disabled = build_online_transforms(cfg, "ultralytics")
    assert args_disabled["mosaic"] == 0.0
    assert args_disabled["mixup"] == 0.0
    assert args_disabled["fliplr"] == 0.0

    # Khi enabled = True
    cfg_online = dict(cfg)
    cfg_online["online_preprocess"] = {"enabled": True, "mosaic": True, "mixup": True, "hflip": True}
    args_enabled = build_online_transforms(cfg_online, "ultralytics")
    assert args_enabled["mosaic"] == 1.0
    assert args_enabled["mixup"] == 0.15
    assert args_enabled["fliplr"] == 0.5
