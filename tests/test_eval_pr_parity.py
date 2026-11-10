"""Kiểm thử PR curve, ngưỡng confidence và parity hai pipeline trên CPU.

Chạy: pytest -q tests/test_eval_pr_parity.py
Không tải trọng số, sử dụng GPU hoặc kết nối W&B.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from ts_alert.eval import evaluate, metrics
from ts_alert.pipelines import tracking
from ts_alert.utils.io import write_yaml

Config = dict[str, Any]
Detection = dict[str, Any]
TensorDict = dict[str, torch.Tensor]

BOX = (4.0, 4.0, 12.0, 12.0)
SECOND_BOX = (0.0, 0.0, 3.0, 3.0)
FALSE_BOX = (13.0, 13.0, 16.0, 16.0)

MAP_VALUES = {
    "map": 0.5,
    "map_50": 0.75,
    "map_75": 0.45,
    "mar_100": 0.8,
}
SCORE_KEYS = ("mAP50-95", "mAP50", "mAP75", "AR100", "precision", "recall", "f1")
PLOT_FILES = {
    "training_curves.png",
    "pr_curve.png",
    "eval_comparison.png",
    "confusion_matrix.png",
    "confusion_matrix_test.png",
    "confusion_matrix_val.png",
}
EVAL_FILES = {
    "metrics.json",
    "metrics_test.json",
    "metrics_val.json",
    "pr_curve_data.json",
    "pr_curve.png",
    "eval_comparison.png",
    "confusion_matrix.png",
    "confusion_matrix.json",
    "confusion_matrix_test.png",
    "confusion_matrix_test.json",
    "confusion_matrix_val.png",
    "confusion_matrix_val.json",
}
STANDARD_FILES = EVAL_FILES | {
    "results.csv",
    "metrics_history.json",
    "summary.json",
    "training_curves.png",
}


def detection(
    bbox: tuple[float, float, float, float] = BOX,
    class_id: int = 0,
    confidence: float = 1.0,
) -> Detection:
    """Tạo detection mới theo hợp đồng dữ liệu của ứng dụng."""
    return {
        "bbox": list(bbox),
        "class_id": class_id,
        "confidence": confidence,
    }


def read_json(path: Path) -> Any:
    """Đọc JSON, từ chối hằng số NaN và Infinity."""

    def reject_constant(value: str) -> None:
        """Báo lỗi khi artifact chứa số ngoài chuẩn JSON."""
        raise AssertionError(f"JSON chứa hằng số không hợp lệ: {value}")

    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
    )


def assert_png(path: Path) -> None:
    """Xác minh chữ ký, định dạng và cấu trúc PNG."""
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0
        image.verify()


def assert_artifacts(directory: Path, filenames: set[str]) -> None:
    """Kiểm tra sự tồn tại và tính hợp lệ của các artifact yêu cầu."""
    for filename in sorted(filenames):
        path = directory / filename
        assert path.is_file(), filename
        if path.suffix == ".png":
            assert_png(path)
        elif path.suffix == ".json":
            read_json(path)


class FakeMAP:
    """Thay backend COCO; giữ nguyên matching P/R/F1 của ứng dụng."""

    def __init__(self, **kwargs: Any) -> None:
        """Lưu tùy chọn khởi tạo của TorchMetrics."""
        self.options = kwargs

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        """Kiểm tra số ảnh mà không tính COCO metrics."""
        assert len(predictions) == len(targets)

    def compute(self) -> dict[str, torch.Tensor]:
        """Trả metric hữu hạn với schema tương thích TorchMetrics."""
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in MAP_VALUES.items()
        }


class FakeRun:
    """W&B run giả ghi toàn bộ dữ liệu trong bộ nhớ."""

    def __init__(self) -> None:
        """Khởi tạo summary, log và trạng thái đóng run."""
        self.id = "pr-parity-cpu"
        self.config: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self.logs: list[dict[str, Any]] = []
        self.finished = False

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """Chấp nhận khai báo metric mà không truy cập mạng."""

    def log(self, payload: dict[str, Any], **kwargs: Any) -> None:
        """Lưu payload độc lập để kiểm tra."""
        self.logs.append(dict(payload))

    def finish(self, **kwargs: Any) -> None:
        """Đánh dấu kết thúc run."""
        self.finished = True


class StaticDetector:
    """Detector cố định khớp GT mặc định của fixture."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Chấp nhận đối số của cả hai detector thực."""

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Trả một TP khi confidence đạt ngưỡng."""
        threshold = 0.25 if conf_threshold is None else float(conf_threshold)
        return [detection(confidence=0.9)] if threshold <= 0.9 else []


class VariableConfDetector:
    """Phân biệt test/val bằng giá trị pixel và ghi các lượt suy luận."""

    def __init__(self) -> None:
        """Khởi tạo danh sách lời gọi."""
        self.calls: list[tuple[int, float | None]] = []

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Trả test TP=.70; val TP=.85/.55; FP=.40 ở cả hai split."""
        pixel = int(frame[0, 0, 0])
        self.calls.append((pixel, conf_threshold))
        threshold = 0.25 if conf_threshold is None else float(conf_threshold)
        if pixel == 10:
            items = [
                detection(confidence=0.70),
                detection(FALSE_BOX, confidence=0.40),
            ]
        elif pixel == 20:
            items = [
                detection(confidence=0.85),
                detection(SECOND_BOX, confidence=0.55),
                detection(FALSE_BOX, confidence=0.40),
            ]
        else:
            raise AssertionError(f"Pixel không mong đợi: {pixel}")
        return [item for item in items if item["confidence"] >= threshold]


class TinyDetector(nn.Module):
    """Detector nhỏ dùng optimizer CPU thật và API torchvision."""

    def __init__(self) -> None:
        """Tạo Linear có tham số xác định."""
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.constant_(self.linear.weight, 0.2)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[TensorDict] | None = None,
    ) -> dict[str, torch.Tensor] | list[TensorDict]:
        """Trả loss khả vi khi train và detection khi eval."""
        assert all(image.device.type == "cpu" for image in images)
        if self.training:
            assert targets is not None
            inputs = images[0].new_ones((len(images), 1))
            return {"loss_classifier": self.linear(inputs).square().mean()}
        return [
            {
                "boxes": image.new_tensor([BOX]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": image.new_tensor([0.9]),
            }
            for image in images
        ]


def make_trainer() -> SimpleNamespace:
    """Tạo trainer giả chứa loss, learning rate và native YOLO metrics."""
    return SimpleNamespace(
        epoch=0,
        tloss=torch.tensor([0.8, 0.4, 0.3]),
        loss_names=("box_loss", "cls_loss", "dfl_loss"),
        lr={"lr/pg0": 0.002},
        metrics={
            "metrics/mAP50-95(B)": 0.5,
            "metrics/mAP50(B)": 0.75,
            "metrics/precision(B)": 0.2,
            "metrics/recall(B)": 0.3,
            "val/box_loss": 0.2,
            "val/cls_loss": 0.1,
        },
        validator=SimpleNamespace(
            training=True,
            metrics=SimpleNamespace(box=SimpleNamespace(map75=0.45)),
        ),
    )


class FakeYOLO:
    """YOLO giả tạo checkpoint và chạy các callback tracking thật."""

    def __init__(self) -> None:
        """Khởi tạo các callback độc lập."""
        self.callbacks: dict[str, list[Callable[..., Any]]] = {}

    def add_callback(self, event: str, callback: Callable[..., Any]) -> None:
        """Đăng ký callback theo API Ultralytics."""
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs: Any) -> None:
        """Mô phỏng huấn luyện CPU và callback final_eval."""
        assert kwargs["device"] == "cpu"
        checkpoint = (
            Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake-yolo-checkpoint")
        trainer = make_trainer()
        trainer.callbacks = self.callbacks

        for callback in self.callbacks.get("on_train_start", []):
            callback(trainer)
        for epoch in range(int(kwargs["epochs"])):
            trainer.epoch = epoch
            trainer.metrics["metrics/mAP50-95(B)"] = 0.5 + 0.1 * epoch
            for callback in self.callbacks.get("on_fit_epoch_end", []):
                callback(trainer)

        trainer.validator.training = False
        for callback in self.callbacks.get("on_fit_epoch_end", []):
            callback(trainer)
        for callback in self.callbacks.get("on_train_end", []):
            callback(trainer)


class FakeValidator:
    """Validator có lifecycle và phép chuẩn bị batch thật cho wrapper."""

    def __init__(self) -> None:
        """Khởi tạo bộ đếm để chứng minh wrapper gọi phương thức gốc."""
        self.training = True
        self.calls = {"init": 0, "update": 0, "gather": 0, "stats": 0}

    def init_metrics(self, model: Any = None) -> None:
        """Bắt đầu một lượt validation."""
        self.calls["init"] += 1

    def update_metrics(self, preds: list[TensorDict], batch: dict) -> None:
        """Kiểm tra số ảnh trước khi wrapper thu thập dữ liệu."""
        assert len(preds) == len(batch["targets"])
        self.calls["update"] += 1

    def gather_stats(self) -> None:
        """Mô phỏng bước gom thống kê trong chế độ một rank."""
        self.calls["gather"] += 1

    def get_stats(self) -> dict[str, float]:
        """Trả metric gốc khác với metric chung để phát hiện dùng nhầm."""
        self.calls["stats"] += 1
        return {"fitness": -1.0, "metrics/mAP50-95(B)": 0.99}

    def _prepare_batch(self, index: int, batch: dict) -> dict[str, Any]:
        """Trả tọa độ pixel với phép scale đồng nhất 16x16."""
        return {
            **batch["targets"][index],
            "im_file": batch["im_file"][index],
            "imgsz": (16, 16),
            "ori_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        }


class ScoreThresholdModel:
    """Mô phỏng lọc score bên trong roi_heads của torchvision."""

    def __init__(self) -> None:
        """Tạo score threshold và khả năng phát lỗi có kiểm soát."""
        self.roi_heads = SimpleNamespace(score_thresh=0.05)
        self.seen_thresholds: list[float] = []
        self.fail = False

    def __call__(self, images: list[torch.Tensor]) -> list[TensorDict]:
        """Lọc bằng ngưỡng nội bộ đang có hiệu lực trước khi trả kết quả."""
        assert images[0].device.type == "cpu"
        threshold = float(self.roi_heads.score_thresh)
        self.seen_thresholds.append(threshold)
        if self.fail:
            raise RuntimeError("intentional inference failure")
        scores = torch.tensor([0.03, 0.8, 1.0, 0.9], dtype=torch.float64)
        labels = torch.tensor([1, 1, 1, 0], dtype=torch.int64)
        boxes = torch.tensor([BOX, SECOND_BOX, FALSE_BOX, BOX])
        keep = scores >= threshold
        return [{
            "boxes": boxes[keep],
            "labels": labels[keep],
            "scores": scores[keep],
        }]


@pytest.fixture(autouse=True)
def isolated_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Cô lập CUDA, W&B, backend COCO và giảm chi phí render PNG."""
    for key in (
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "TS_ALERT_WANDB_RUN_ID",
        "TS_ALERT_WANDB_RESUME",
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "RANK": "-1",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "WANDB_MODE": "disabled",
    }.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(metrics, "MeanAveragePrecision", FakeMAP)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)

    wandb = ModuleType("wandb")
    wandb.run = None
    wandb.init = Mock(side_effect=AssertionError("Không được mở W&B run."))
    wandb.Image = Mock(side_effect=lambda path: {"image": path})
    wandb.plot = SimpleNamespace(confusion_matrix=Mock(return_value="cm-chart"))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    original_savefig = Figure.savefig

    def small_savefig(
        figure: Figure,
        destination: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Render PNG thật với số pixel nhỏ."""
        figure.set_size_inches(6, 5)
        kwargs["dpi"] = 40
        kwargs["bbox_inches"] = None
        return original_savefig(figure, destination, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", small_savefig)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Tạo một ảnh 16x16 và một GT cho mỗi split train/test/val."""
    root = tmp_path / "processed"
    for split, pixel in (("train", 30), ("test", 10), ("val", 20)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (16, 16), (pixel, pixel, pixel)).save(
            image_dir / "sample.png"
        )
        (label_dir / "sample.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )

    classes = tmp_path / "classes.csv"
    classes.write_text(
        "id,name,tts_text\n0,Stop,Stop\n1,Yield,Yield\n2,Limit,Limit\n",
        encoding="utf-8",
    )
    write_yaml(
        {
            "path": str(root),
            "train": "images/train",
            "test": "images/test",
            "val": "images/val",
            "names": ["Stop", "Yield", "Limit"],
        },
        root / "data.yaml",
    )
    return {
        "seed": 42,
        "pipeline": "test",
        "device": "cpu",
        "workers": 0,
        "num_gpus": 0,
        "runtime": {"device": "cpu", "workers": 0, "num_gpus": 0},
        "paths": {
            "processed_dir": str(root),
            "outputs_dir": str(tmp_path / "outputs"),
            "classes_csv": str(classes),
        },
        "data": {
            "root": str(root),
            "num_classes": 3,
            "image_size": 16,
            "pad_value": 114,
            "split": [0.8, 0.1, 0.1],
        },
        "model": {
            "pretrained": False,
            "weights": str(tmp_path / "weights" / "detector.pth"),
        },
        "train": {
            "epochs": 2,
            "batch_size": 1,
            "num_workers": 0,
            "early_stopping_patience": 10,
            "amp": False,
            "primary_metric": "mAP50-95",
            "optimizer": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "warmup_epochs": 0.0,
            "require_wandb": False,
            "run_name": "pr_parity",
        },
        "online_preprocess": {"enabled": False},
        "inference": {"conf_threshold": 0.25, "iou_threshold": 0.5},
        "eval": {
            "batch_size": 1,
            "split": "test",
            "auto_eval_test": True,
            "metrics": list(SCORE_KEYS),
            "benchmark": {"num_frames": 1, "warmup": 0},
        },
        "project": {"wandb_project": "offline-pr-parity"},
    }


def test_compute_pr_curve_matches_compute_metrics_and_tiebreak() -> None:
    """Mọi ngưỡng khớp metric chung; tie chọn cao nhất, không TP dùng mặc định."""
    thresholds = [round(index / 100, 2) for index in range(101)]
    predictions = [[
        detection(confidence=0.65),
        detection(FALSE_BOX, confidence=0.35),
    ]]
    targets = [[detection()]]
    curve = metrics.compute_pr_curve(
        predictions, targets, 3, thresholds=thresholds
    )

    for index, threshold in enumerate(thresholds):
        expected = metrics.compute_metrics(
            predictions, targets, 3, conf_threshold=threshold
        )
        for key in ("precision", "recall", "f1"):
            assert curve[key][index] == expected[key]
        assert curve["tp"][index] == int(threshold <= 0.65)
        assert curve["fp"][index] == int(threshold <= 0.35)
        assert curve["fn"][index] == int(threshold > 0.65)

    tied = [
        threshold
        for threshold, f1 in zip(curve["thresholds"], curve["f1"])
        if f1 == 1.0
    ]
    assert tied == thresholds[36:66]
    assert curve["best_threshold"] == 0.65
    assert curve["best_precision"] == curve["best_recall"] == curve["best_f1"] == 1.0
    assert curve["pr_auc"] == 1.0

    unordered = [0.65, 0.1, 0.66, 0.36, 0.65, 1.0, 0.0]
    reordered = metrics.compute_pr_curve(
        predictions, targets, 3, thresholds=unordered
    )
    assert reordered["thresholds"] == unordered
    assert reordered["best_threshold"] == 0.65
    for index, threshold in enumerate(unordered):
        expected = metrics.compute_metrics(
            predictions, targets, 3, conf_threshold=threshold
        )
        for key in ("precision", "recall", "f1"):
            assert reordered[key][index] == expected[key]

    for no_tp_predictions, no_tp_targets in (
        ([[detection(FALSE_BOX)]], targets),
        ([[]], targets),
        ([], []),
    ):
        fallback = metrics.compute_pr_curve(
            no_tp_predictions,
            no_tp_targets,
            3,
            thresholds=[0.1, 0.6, 0.9],
            default_conf=0.27,
        )
        assert fallback["best_threshold"] == 0.27
        assert fallback["best_f1"] == fallback["pr_auc"] == 0.0

    for invalid in (
        [],
        [[0.1, 0.5]],
        [1.01],
        [-0.01],
        [float("nan")],
        [float("inf")],
    ):
        with pytest.raises(ValueError):
            metrics.compute_pr_curve(
                predictions, targets, 3, thresholds=invalid
            )

    for invalid in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            metrics.compute_pr_curve(
                predictions, targets, 3, default_conf=invalid
            )


@pytest.mark.parametrize("splits", [("test",), ("test", "val")])
def test_save_pr_curve_and_eval_comparison_plots(
    tmp_path: Path,
    splits: tuple[str, ...],
) -> None:
    """Cả một và hai split tạo PNG hợp lệ, dữ liệu sai bị từ chối."""
    curve = metrics.compute_pr_curve(
        [[detection(confidence=0.65)]],
        [[detection()]],
        3,
        thresholds=[0.0, 0.25, 0.65, 0.9, 1.0],
    )
    curves = {split: deepcopy(curve) for split in splits}
    split_metrics = {
        split: {
            "mAP50-95": 0.5,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "conf_threshold": 0.65,
        }
        for split in splits
    }

    assert_png(metrics.save_pr_curve_plot(
        curves, tmp_path / "pr_curve.png", selected_threshold=0.65
    ))
    assert_png(metrics.save_pr_curve_plot(
        curves, tmp_path / "pr_curve_auto.png"
    ))
    assert_png(metrics.save_eval_comparison_plot(
        split_metrics, tmp_path / "comparison.png"
    ))

    with pytest.raises(ValueError):
        metrics.save_pr_curve_plot({}, tmp_path / "empty_pr.png")
    with pytest.raises(ValueError):
        metrics.save_eval_comparison_plot({}, tmp_path / "empty_eval.png")
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            metrics.save_pr_curve_plot(
                curves, tmp_path / "invalid_pr.png", selected_threshold=invalid
            )
    with pytest.raises(ValueError):
        metrics.save_eval_comparison_plot(
            {
                "test": {
                    "mAP50-95": float("nan"),
                    "precision": float("inf"),
                    "recall": True,
                    "f1": "0.8",
                    "fps": 100.0,
                }
            },
            tmp_path / "invalid_comparison.png",
        )


def test_detectors_conf_threshold_override_and_restore() -> None:
    """Hai detector hỗ trợ override; Faster R-CNN luôn khôi phục ngưỡng ROI."""
    yolo_module = importlib.import_module("ts_alert.pipelines.yolo.infer")
    rcnn_module = importlib.import_module("ts_alert.pipelines.faster_rcnn.infer")
    frame = np.full((16, 16, 3), 10, dtype=np.uint8)

    yolo = yolo_module.YoloDetector.__new__(yolo_module.YoloDetector)
    yolo.conf, yolo.iou, yolo.img_size = 0.25, 0.5, 16
    yolo.classes = {}
    yolo.model = SimpleNamespace(predict=Mock(return_value=[]))

    for threshold in (None, 0, 1, 0.01, np.float32(0.01)):
        assert yolo.predict(frame, conf_threshold=threshold) == []
        kwargs = yolo.model.predict.call_args.kwargs
        assert kwargs["source"] is frame
        assert kwargs["conf"] == (0.25 if threshold is None else float(threshold))
        assert kwargs["iou"] == 0.5
        assert kwargs["imgsz"] == 16
        assert yolo.conf == 0.25

    for invalid in (True, False, np.bool_(True), np.nan, -0.1, 1.1, "bad"):
        before = yolo.model.predict.call_count
        with pytest.raises(ValueError):
            yolo.predict(frame, conf_threshold=invalid)
        assert yolo.model.predict.call_count == before

    rcnn = rcnn_module.FasterRCNNDetector.__new__(rcnn_module.FasterRCNNDetector)
    rcnn.conf, rcnn.iou = 0.25, 0.5
    rcnn.device = torch.device("cpu")
    rcnn.classes = {}
    model = ScoreThresholdModel()
    rcnn.model = model

    for threshold in (None, 0, 1, 0.01):
        effective = 0.25 if threshold is None else float(threshold)
        result = rcnn.predict(frame, conf_threshold=threshold)
        expected = [score for score in (0.03, 0.8, 1.0) if score >= effective]
        assert [item["confidence"] for item in result] == pytest.approx(expected)
        assert all(item["class_id"] == 0 for item in result)
        assert model.seen_thresholds[-1] == min(0.05, effective)
        assert model.roi_heads.score_thresh == 0.05
        assert rcnn.conf == 0.25

    for invalid in (True, False, np.bool_(False), np.nan, -0.1, 1.1, "bad"):
        before = len(model.seen_thresholds)
        with pytest.raises(ValueError):
            rcnn.predict(frame, conf_threshold=invalid)
        assert len(model.seen_thresholds) == before
        assert model.roi_heads.score_thresh == 0.05

    model.fail = True
    for threshold in (0, 0.01, None):
        with pytest.raises(RuntimeError, match="intentional inference failure"):
            rcnn.predict(frame, conf_threshold=threshold)
        effective = 0.25 if threshold is None else float(threshold)
        assert model.seen_thresholds[-1] == min(0.05, effective)
        assert model.roi_heads.score_thresh == 0.05


def test_evaluate_best_model_tunes_on_test_and_applies_to_val(
    cfg: Config,
    tmp_path: Path,
) -> None:
    """Chọn .70 trên test và áp dụng đúng .70 cho val có optimum riêng .55."""
    root = Path(cfg["paths"]["processed_dir"])
    (root / "labels" / "val" / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n0 0.09375 0.09375 0.1875 0.1875\n",
        encoding="utf-8",
    )
    detector = VariableConfDetector()
    output = tmp_path / "evaluation"
    report = evaluate.evaluate_best_model(detector, cfg, output)

    assert report["best_conf_threshold"] == 0.70
    assert report["tuned_on_split"] == "test"
    assert report["curves"]["test"]["best_threshold"] == 0.70
    assert report["curves"]["val"]["best_threshold"] == 0.55
    assert report["test_metrics"] == report["splits"]["test"]
    assert report["val_metrics"] == report["splits"]["val"]

    for split, expected in (
        ("test", {"precision": 1.0, "recall": 1.0, "f1": 1.0}),
        ("val", {"precision": 1.0, "recall": 0.5, "f1": 0.6667}),
    ):
        result = report["splits"][split]
        assert result["conf_threshold"] == 0.70
        assert result["tuned_on_split"] == "test"
        for key, value in expected.items():
            assert result[key] == pytest.approx(value)
        stored = read_json(output / f"metrics_{split}.json")
        assert all(stored[key] == result[key] for key in result)

    assert_artifacts(output, EVAL_FILES)
    assert read_json(output / "metrics.json") == read_json(
        output / "metrics_test.json"
    )
    data = read_json(output / "pr_curve_data.json")
    assert data["selected_conf_threshold"] == 0.70
    assert data["tuned_on_split"] == "test"
    assert data["curves"] == report["curves"]

    for suffix in (".png", ".json"):
        assert (output / f"confusion_matrix{suffix}").read_bytes() == (
            output / f"confusion_matrix_test{suffix}"
        ).read_bytes()

    val_matrix = read_json(output / "confusion_matrix_val.json")
    assert val_matrix["conf_threshold"] == 0.70
    assert val_matrix["matrix"][0][0] == 1
    assert val_matrix["matrix"][0][3] == 1
    assert val_matrix["preds"][0] == [detection(confidence=0.85)]
    assert sum(map(sum, val_matrix["matrix"])) == 2
    assert detector.calls.count((10, 0.01)) == 1
    assert detector.calls.count((20, 0.01)) == 1


def test_evaluate_best_model_skips_missing_val_with_warning(
    cfg: Config,
    tmp_path: Path,
) -> None:
    """Thiếu images/val chỉ cảnh báo, test và các biểu đồ vẫn được lưu."""
    root = Path(cfg["paths"]["processed_dir"])
    shutil.rmtree(root / "images" / "val")
    output = tmp_path / "missing_val"

    with pytest.warns(UserWarning, match="val"):
        report = evaluate.evaluate_best_model(StaticDetector(), cfg, output)

    assert report["val_metrics"] is None
    assert set(report["splits"]) == {"test"}
    assert set(report["curves"]) == {"test"}
    assert_artifacts(output, {
        "metrics_test.json",
        "metrics.json",
        "pr_curve.png",
        "eval_comparison.png",
    })
    assert not (output / "metrics_val.json").exists()
    assert not (output / "confusion_matrix_val.json").exists()


def test_yolo_shared_validator_ts_alert_metrics_and_final_eval_suppression(
    cfg: Config,
    tmp_path: Path,
) -> None:
    """Wrapper xuất metric chung; hook bỏ final_eval và epoch vượt cấu hình."""
    module = importlib.import_module("ts_alert.pipelines.yolo.train")
    validator = FakeValidator()
    module._attach_shared_validator(validator, cfg)
    wrapped_update = validator.update_metrics
    module._attach_shared_validator(validator, cfg)
    assert validator.update_metrics is wrapped_update

    predictions = [{
        "bboxes": torch.tensor([BOX, BOX], dtype=torch.float32),
        "cls": torch.tensor([0.0, 0.0]),
        "conf": torch.tensor([0.9, 0.8]),
    }]
    batch = {
        "im_file": ["sample.png"],
        "targets": [{
            "bboxes": torch.tensor([BOX], dtype=torch.float32),
            "cls": torch.tensor([0.0]),
        }],
    }
    validator.init_metrics(None)
    validator.update_metrics(predictions, batch)
    validator.gather_stats()
    stats = validator.get_stats()

    expected = metrics.compute_metrics(
        [[detection(confidence=0.9), detection(confidence=0.8)]],
        [[detection()]],
        3,
    )
    assert {f"ts_alert/{key}" for key in SCORE_KEYS} <= stats.keys()
    for key in SCORE_KEYS:
        assert stats[f"ts_alert/{key}"] == expected[key]
    assert stats["fitness"] == float(stats["ts_alert/mAP50-95"])
    assert stats["metrics/mAP50-95(B)"] == 0.99
    assert validator.calls == {"init": 1, "update": 1, "gather": 1, "stats": 1}
    assert not hasattr(validator, "_ts_shared_records")
    assert not hasattr(validator, "_ts_shared_preds")
    assert not hasattr(validator, "_ts_shared_gts")

    hook = module._YoloTrackingHook(cfg, tmp_path, "mAP50-95")
    trainer = make_trainer()
    trainer.validator = validator
    trainer.metrics.update(stats)
    hook.on_fit_epoch_end(trainer)

    assert len(hook.history) == 1
    for key in SCORE_KEYS:
        assert hook.history[0][f"val/{key}"] == expected[key]
    assert hook.history[0]["best/metric"] == expected["mAP50-95"]

    history_path = tmp_path / "metrics_history.json"
    before = history_path.read_bytes()
    trainer.epoch = 1
    validator.training = False
    hook.on_fit_epoch_end(trainer)
    assert len(hook.history) == 1
    assert history_path.read_bytes() == before

    validator.training = True
    for epoch in (cfg["train"]["epochs"], cfg["train"]["epochs"] + 1):
        trainer.epoch = epoch
        hook.on_fit_epoch_end(trainer)
        assert len(hook.history) == 1
        assert history_path.read_bytes() == before

    trainer.epoch = 1
    hook.on_fit_epoch_end(trainer)
    assert [row["epoch"] for row in hook.history] == [1, 2]


def test_yolo_vs_faster_rcnn_output_parity_and_wandb(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hai pipeline có schema/artifact chung và tải đủ ảnh W&B không trùng path."""
    yolo_module = importlib.import_module("ts_alert.pipelines.yolo.train")
    rcnn_module = importlib.import_module("ts_alert.pipelines.faster_rcnn.train")
    fake_yolo = FakeYOLO()
    tiny = TinyDetector()
    initial_weight = tiny.linear.weight.detach().clone()

    monkeypatch.setattr(yolo_module, "load_model", Mock(return_value=fake_yolo))
    monkeypatch.setattr(yolo_module, "YoloDetector", StaticDetector)
    monkeypatch.setattr(rcnn_module, "build_model", Mock(return_value=tiny))
    monkeypatch.setattr(rcnn_module, "FasterRCNNDetector", StaticDetector)

    directories: dict[str, Path] = {}
    for pipeline, pipeline_type, extension in (
        ("yolo", yolo_module.YoloPipeline, ".pt"),
        ("faster_rcnn", rcnn_module.FasterRCNNPipeline, ".pth"),
    ):
        config = deepcopy(cfg)
        config["pipeline"] = pipeline
        config["paths"]["outputs_dir"] = str(tmp_path / pipeline / "outputs")
        config["model"]["weights"] = str(
            tmp_path / pipeline / f"detector{extension}"
        )
        directories[pipeline] = pipeline_type(config).fit()

    assert not torch.equal(initial_weight, tiny.linear.weight.detach())
    assert (directories["yolo"] / "weights" / "best.pt").is_file()
    assert (directories["faster_rcnn"] / "best.pth").is_file()

    summaries: dict[str, dict[str, Any]] = {}
    for pipeline, directory in directories.items():
        assert_artifacts(directory, STANDARD_FILES)
        summaries[pipeline] = read_json(directory / "summary.json")
        assert summaries[pipeline]["pipeline"] == pipeline
        assert summaries[pipeline]["epochs_completed"] == 2
        assert summaries[pipeline]["best_conf_threshold"] == 0.9
        assert {
            path.name for path in directory.iterdir()
            if path.name in STANDARD_FILES
        } == STANDARD_FILES

        history = read_json(directory / "metrics_history.json")
        assert [row["epoch"] for row in history] == [1, 2]
        for split in ("test", "val"):
            result = read_json(directory / f"metrics_{split}.json")
            assert result["precision"] == result["recall"] == result["f1"] == 1.0
            assert result["conf_threshold"] == 0.9
            assert result["tuned_on_split"] == "test"

    assert summaries["yolo"].keys() == summaries["faster_rcnn"].keys()
    assert {
        "pipeline", "primary_metric", "best_epoch", "best_metric",
        "epochs_completed", "best_checkpoint", "weights", "wandb_run_id",
        "best_conf_threshold", "test_metrics", "val_metrics",
    } <= summaries["yolo"].keys()

    for filename in ("metrics.json", "metrics_test.json", "metrics_val.json"):
        payloads = [
            {
                key: value
                for key, value in read_json(directory / filename).items()
                if key != "pipeline"
            }
            for directory in directories.values()
        ]
        assert payloads[0].keys() == payloads[1].keys()
        for key in SCORE_KEYS:
            assert payloads[0][key] == payloads[1][key]

    wandb = sys.modules["wandb"]
    wandb.init.assert_not_called()

    for directory in directories.values():
        wandb.Image.reset_mock()
        wandb.plot.confusion_matrix.reset_mock()
        run = FakeRun()
        test_metrics = read_json(directory / "metrics_test.json")
        val_metrics = read_json(directory / "metrics_val.json")
        tracking.log_final_to_wandb(
            run,
            directory,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            cm_json_path=directory / "confusion_matrix.json",
            extra_plots=[
                directory / "pr_curve.png",
                directory / "." / "confusion_matrix_val.png",
            ],
        )

        for key, value in test_metrics.items():
            assert run.summary[f"test/{key}"] == value
            assert any(log.get(f"test/{key}") == value for log in run.logs)
        for key, value in val_metrics.items():
            assert run.summary[f"val_final/{key}"] == value
            assert any(log.get(f"val_final/{key}") == value for log in run.logs)
        assert run.summary["eval/best_conf_threshold"] == 0.9

        uploaded = [
            Path(call.args[0]).resolve()
            for call in wandb.Image.call_args_list
        ]
        assert len(uploaded) == len(set(uploaded)) == len(PLOT_FILES)
        assert set(uploaded) == {
            (directory / filename).resolve() for filename in PLOT_FILES
        }
        chart_keys = {
            key
            for log in run.logs
            for key in log
            if key.startswith("charts/")
        }
        assert {f"charts/{Path(name).stem}" for name in PLOT_FILES} <= chart_keys
        assert "charts/confusion_matrix_interactive" in chart_keys

        matrix = read_json(directory / "confusion_matrix.json")
        wandb.plot.confusion_matrix.assert_called_once_with(
            y_true=matrix["y_true"],
            preds=matrix["y_pred"],
            class_names=matrix["labels"],
        )
