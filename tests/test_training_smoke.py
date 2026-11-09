"""Unit test và CPU smoke test cho evaluation, tracking và training.

Chạy: pytest -q tests/test_training_smoke.py
Không tải model, gọi W&B hoặc sử dụng API key thật.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from tools import run_sweep
from ts_alert.eval import evaluate, metrics
from ts_alert.pipelines import tracking
from ts_alert.utils.io import write_yaml

Config = dict[str, Any]

STANDARD_KEYS = {
    "system/learning_rate",
    "val/precision",
    "val/recall",
    "val/f1",
    "best/epoch",
    "best/metric",
    "is_best",
}
CM_KEYS = {"labels", "class_names", "matrix", "y_true", "y_pred", "preds"}
PROTECTED_KEYS = (
    "seed",
    "data",
    "data.image_size",
    "eval",
    "train.batch_size",
    "train.epochs",
    "train.early_stopping_patience",
    "train.amp",
    "train.primary_metric",
)


class FakeMAP:
    """Thay backend COCO; giữ matching P/R/F1 của ứng dụng chạy thật."""

    updates: list[tuple[list[dict], list[dict]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Chấp nhận các tùy chọn của MeanAveragePrecision."""
        self.options = kwargs

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        """Ghi dữ liệu để kiểm tra confidence không bị lọc trước mAP."""
        type(self).updates.append((predictions, targets))

    def compute(self) -> dict[str, torch.Tensor]:
        """Trả các scalar có cùng schema với TorchMetrics."""
        return {
            "map": torch.tensor(0.5),
            "map_50": torch.tensor(0.75),
            "map_75": torch.tensor(0.45),
            "mar_100": torch.tensor(0.8),
        }


class FakeRun:
    """W&B run chỉ lưu dữ liệu trong bộ nhớ."""

    def __init__(self) -> None:
        """Khởi tạo trạng thái run giả."""
        self.id = "cpu_smoke"
        self.config: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self.logs: list[dict[str, Any]] = []
        self.finished = False

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """Chấp nhận khai báo metric mà không kết nối dịch vụ."""

    def log(self, payload: dict[str, Any], **kwargs: Any) -> None:
        """Lưu payload để kiểm tra."""
        self.logs.append(payload)

    def finish(self, **kwargs: Any) -> None:
        """Đánh dấu kết thúc run."""
        self.finished = True


class StaticDetector:
    """Detector cố định khớp nhãn của ảnh 16x16."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Chấp nhận cùng kiểu đối số với detector thực."""

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Trả một bbox foreground có confidence cao."""
        return [detection((4, 4, 12, 12), confidence=0.9)]


class TinyDetector(nn.Module):
    """Detector CPU có Linear trainable và API train/eval giống torchvision."""

    def __init__(self) -> None:
        """Tạo tham số nhỏ, xác định để kiểm tra optimizer thực."""
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.constant_(self.linear.weight, 0.2)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Trả loss khả vi khi train và detections khi eval."""
        if self.training:
            assert targets is not None
            inputs = torch.ones((len(images), 1), device=images[0].device)
            return {"loss_classifier": self.linear(inputs).square().mean()}

        return [
            {
                "boxes": image.new_tensor([[4, 4, 12, 12]]),
                "labels": torch.tensor([1], device=image.device),
                "scores": image.new_tensor([0.9]),
            }
            for image in images
        ]


class FakeYOLO:
    """YOLO giả chạy callback thật và tạo checkpoint trong tmp_path."""

    def __init__(self) -> None:
        """Khởi tạo callback có thể được pipeline sửa trực tiếp."""
        self.callbacks: dict[str, list[Callable[..., Any]]] = {}

    def add_callback(self, event: str, handler: Callable[..., Any]) -> None:
        """Đăng ký callback theo giao diện Ultralytics."""
        self.callbacks.setdefault(event, []).append(handler)

    def train(self, **kwargs: Any) -> None:
        """Mô phỏng hai epoch, final_eval và checkpoint best.pt."""
        assert kwargs["device"] in ("cpu", [0, 1])
        directory = Path(kwargs["project"]) / kwargs["name"]
        checkpoint = directory / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake-yolo-checkpoint")

        trainer = make_yolo_trainer()
        trainer.callbacks = self.callbacks
        for callback in list(self.callbacks.get("on_train_start", [])):
            callback(trainer)

        for epoch in range(int(kwargs["epochs"])):
            trainer.epoch = epoch
            trainer.metrics["metrics/mAP50-95(B)"] = 0.5 + epoch * 0.1
            for callback in list(self.callbacks.get("on_fit_epoch_end", [])):
                callback(trainer)

        # Ultralytics có thể gọi lại callback sau final_eval.
        for callback in list(self.callbacks.get("on_fit_epoch_end", [])):
            callback(trainer)
        for callback in list(self.callbacks.get("on_train_end", [])):
            callback(trainer)


def detection(
    bbox: tuple[float, float, float, float],
    class_id: int = 0,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Tạo một detection độc lập."""
    return {
        "bbox": list(bbox),
        "class_id": class_id,
        "confidence": confidence,
    }


def make_yolo_trainer() -> SimpleNamespace:
    """Tạo trainer chứa loss mapping và metric YOLO điển hình."""
    return SimpleNamespace(
        epoch=0,
        tloss={
            "box_loss": torch.tensor(0.8),
            "cls_loss": torch.tensor(0.4),
            "dfl_loss": torch.tensor(0.3),
        },
        lr={"lr/pg0": 0.0025},
        metrics={
            "metrics/precision(B)": 0.75,
            "metrics/recall(B)": 0.5,
            "metrics/mAP50(B)": 0.7,
            "metrics/mAP50-95(B)": 0.5,
            "val/box_loss": 0.2,
            "val/cls_loss": 0.1,
        },
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=SimpleNamespace(map75=0.45))
        ),
    )


def read_json(path: Path) -> Any:
    """Đọc JSON và từ chối NaN/Infinity ngoài chuẩn."""

    def reject_constant(value: str) -> None:
        """Báo lỗi khi JSON chứa hằng số không hữu hạn."""
        raise AssertionError(f"JSON chứa giá trị không hợp lệ: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def assert_png(path: Path) -> None:
    """Kiểm tra chữ ký và cấu trúc PNG bằng Pillow."""
    assert path.read_bytes().startswith(b"\x89PNG")
    with Image.open(path) as image:
        assert image.format == "PNG"
        image.verify()


def epoch_payload(epoch: int = 1) -> dict[str, Any]:
    """Tạo payload chuẩn có F1 được tính từ precision và recall."""
    return tracking.standardize_epoch_payload(
        epoch=epoch,
        train_loss=1.5,
        val_loss=0.3,
        val_metrics={"precision": 0.75, "recall": 0.5, "mAP50-95": 0.5},
        lr=0.0025,
        best_epoch=1,
        best_metric=0.5,
        is_best=epoch == 1,
    )


@pytest.fixture(autouse=True)
def isolated_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Cô lập môi trường, backend mAP và kích thước render PNG."""
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
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    FakeMAP.updates = []
    monkeypatch.setattr(metrics, "MeanAveragePrecision", FakeMAP)

    # Render thật nhưng giảm số pixel để test nhanh trên CPU.
    original_savefig = Figure.savefig

    def small_savefig(
        figure: Figure,
        destination: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Giữ PNG hợp lệ với chi phí raster thấp."""
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
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Cài module W&B giả, không thể gọi mạng."""
    module = ModuleType("wandb")
    module.run = None
    module.init = Mock(side_effect=AssertionError("Không được mở W&B run."))
    module.Image = Mock(side_effect=lambda path: {"image": path})
    module.plot = SimpleNamespace(confusion_matrix=Mock(return_value="cm-chart"))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Tạo hai ảnh YOLO cho mỗi split train/val/test."""
    root = tmp_path / "processed"
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (16, 16), (40 + index, 80, 120)).save(
                image_dir / f"{index}.png"
            )
            (label_dir / f"{index}.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )

    classes = tmp_path / "classes.csv"
    classes.write_text(
        "id,name,tts_text\n0,Stop,Stop\n1, ,\n2,Yield,Yield\n",
        encoding="utf-8",
    )
    write_yaml(
        {"path": str(root), "train": "images/train", "val": "images/val",
         "test": "images/test", "names": ["Stop", "Class_1", "Yield"]},
        root / "data.yaml",
    )
    return {
        "seed": 42,
        "device": "cpu",
        "workers": 0,
        "num_gpus": 0,
        "runtime": {"device": "cpu", "workers": 0, "num_workers": 0,
                    "num_gpus": 0},
        "paths": {
            "processed_dir": str(root),
            "outputs_dir": str(tmp_path / "outputs"),
            "classes_csv": str(classes),
        },
        "data": {
            "root": str(root), "num_classes": 3, "image_size": 16,
            "pad_value": 114, "split": [0.8, 0.1, 0.1],
        },
        "model": {
            "pretrained": False,
            "weights": str(tmp_path / "weights" / "detector.pth"),
        },
        "train": {
            "epochs": 2, "batch_size": 2, "num_workers": 0,
            "early_stopping_patience": 10, "amp": False,
            "primary_metric": "mAP50-95", "optimizer": "SGD",
            "lr": 0.01, "lr0": 0.01, "momentum": 0.9,
            "weight_decay": 0.0, "warmup_epochs": 0.0,
            "require_wandb": True, "run_name": "cpu_smoke",
        },
        "online_preprocess": {"enabled": False},
        "inference": {"conf_threshold": 0.25, "iou_threshold": 0.5},
        "eval": {
            "batch_size": 2, "split": "test", "auto_eval_test": True,
            "metrics": ["mAP50-95"],
            "benchmark": {"num_frames": 2, "warmup": 0},
        },
        "project": {"wandb_project": "offline-test"},
        "sweep": {
            "method": "random", "count": 1,
            "metric": {"name": "best/metric", "goal": "maximize"},
            "parameters": {"lr": {"values": [0.001, 0.01]}},
        },
    }


def test_metrics_greedy_one_to_one_and_confidence() -> None:
    """Duplicate, sai lớp và confidence thấp không tạo thêm TP."""
    a, b, c = (0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)
    predictions = [[
        detection(a, confidence=0.9),
        detection(a, confidence=0.8),
        detection(b, class_id=1),
        detection(c, confidence=0.249),
    ]]
    targets = [[detection(a), detection(b), detection(c)]]
    result = metrics.compute_metrics(predictions, targets, 3)

    assert result["precision"] == pytest.approx(0.3333)
    assert result["recall"] == pytest.approx(0.3333)
    assert result["f1"] == pytest.approx(0.3333)
    assert result["mAP50-95"] == 0.5
    assert result["mAP50"] == 0.75
    assert len(FakeMAP.updates[-1][0][0]["scores"]) == 4

    boundary = metrics.compute_metrics(
        [[detection(a, confidence=0.25)]], [[detection(a)]], 3
    )
    assert boundary["precision"] == boundary["recall"] == boundary["f1"] == 1.0


def test_metrics_match_by_iou_instead_of_confidence() -> None:
    """IoU cao nhất chiếm GT trước, dù confidence thấp hơn."""
    result = metrics.compute_metrics(
        [[
            detection((1, 0, 11, 10), confidence=0.99),
            detection((0, 0, 10, 10), confidence=0.3),
        ]],
        [[detection((0, 0, 10, 10)), detection((4, 0, 14, 10))]],
        1,
        iou_threshold=0.5,
    )
    assert result["precision"] == result["recall"] == result["f1"] == 1.0


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        ([], []),
        ([[]], [[]]),
        ([[]], [[detection((0, 0, 10, 10))]]),
        ([[detection((0, 0, 10, 10))]], [[]]),
    ],
)
def test_metrics_empty_cases(
    predictions: list,
    targets: list,
) -> None:
    """Không chia cho 0 khi thiếu ảnh, prediction hoặc GT."""
    result = metrics.compute_metrics(predictions, targets, 3)
    assert all(math.isfinite(value) for value in result.values())
    assert result["precision"] == result["recall"] == result["f1"] == 0.0


def test_confusion_matrix_background_and_wrong_class() -> None:
    """Ma trận dùng hàng GT, cột prediction và background ở index 3."""
    predictions = [[
        detection((0, 0, 10, 10), 1),
        detection((40, 0, 50, 10), 2),
        detection((60, 0, 70, 10), 0, 0.1),
    ]]
    targets = [[
        detection((0, 0, 10, 10), 0),
        detection((20, 0, 30, 10), 2),
    ]]
    matrix, truth, predicted = metrics.compute_confusion_matrix(
        predictions, targets, 3
    )
    expected = np.zeros((4, 4), dtype=np.int64)
    expected[0, 1] = 1
    expected[3, 2] = 1
    expected[2, 3] = 1
    np.testing.assert_array_equal(matrix, expected)
    rebuilt = np.zeros_like(matrix)
    np.add.at(rebuilt, (truth, predicted), 1)
    np.testing.assert_array_equal(rebuilt, matrix)
    assert matrix[3, 3] == 0


@pytest.mark.parametrize("num_classes", [3, 52])
def test_confusion_plot_png(tmp_path: Path, num_classes: int) -> None:
    """Render được PNG thật cho danh mục nhỏ và 52 lớp."""
    matrix = np.eye(num_classes + 1, dtype=np.int64)
    matrix[-1] = 0  # Hàng tổng bằng 0 vẫn chuẩn hóa an toàn.
    path = metrics.save_confusion_matrix_plot(
        matrix,
        [f"Class {index}" for index in range(num_classes)],
        tmp_path / f"cm_{num_classes}.png",
    )
    assert_png(path)


def test_evaluate_detector_artifacts(cfg: Config, tmp_path: Path) -> None:
    """Evaluator thật tạo đầy đủ schema JSON, PNG và benchmark."""
    output = tmp_path / "evaluation"
    result = evaluate.evaluate_detector(StaticDetector(), cfg, out_dir=output)
    assert {"fps", "latency_ms", "precision", "recall", "f1"} <= result.keys()
    assert all(math.isfinite(float(value)) for value in result.values())
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert result["fps"] >= 0
    assert result["latency_ms"] >= 0
    assert_png(output / "confusion_matrix.png")
    matrix = read_json(output / "confusion_matrix.json")
    assert CM_KEYS <= matrix.keys()
    assert matrix["class_names"] == ["Stop", "Class_1", "Yield"]
    assert matrix["labels"][-1] == "background"
    assert np.asarray(matrix["matrix"]).shape == (4, 4)
    assert matrix["matrix"][0][0] == 2
    assert matrix["y_true"] == matrix["y_pred"] == [0, 0]
    assert len(matrix["preds"]) == 2
    assert matrix["preds"][0][0]["class_id"] == 0


def test_standardize_epoch_payload() -> None:
    """Payload có các khóa chuẩn và tự tính F1 khi thiếu."""
    payload = epoch_payload()
    assert STANDARD_KEYS <= payload.keys()
    assert payload["system/learning_rate"] == payload["lr"] == 0.0025
    assert payload["val/precision"] == 0.75
    assert payload["val/recall"] == 0.5
    assert payload["val/f1"] == pytest.approx(0.6)
    assert payload["best/epoch"] == 1
    assert payload["best/metric"] == 0.5
    assert payload["is_best"] == 1
    assert epoch_payload(2)["is_best"] == 0


def test_save_epoch_artifacts_clean_merge_and_none(tmp_path: Path) -> None:
    """Gộp epoch an toàn, làm sạch NaN và chấp nhận payload=None."""
    first = epoch_payload()
    first["extra/nonfinite"] = float("nan")
    tracking.save_epoch_artifacts(tmp_path, [first], payload=None)
    tracking.save_epoch_artifacts(tmp_path, [epoch_payload(2)], payload=None)
    tracking.save_epoch_artifacts(tmp_path, [], payload=epoch_payload(2))

    history = read_json(tmp_path / "metrics_history.json")
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[0]["extra/nonfinite"] is None
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert STANDARD_KEYS <= set(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 2
    assert "nan" not in (tmp_path / "results.csv").read_text().lower()

    empty = tmp_path / "empty"
    tracking.save_epoch_artifacts(empty, [], payload=None)
    assert not (empty / "results.csv").exists()


@pytest.mark.parametrize("epochs", [1, 3])
def test_training_curves_png_and_best_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    epochs: int,
) -> None:
    """Vẽ cả một/nhiều epoch và đánh dấu đúng best_epoch."""
    from matplotlib.axes import Axes

    original = Axes.axvline
    positions: list[int] = []

    def record_best(axis: Axes, x: int, *args: Any, **kwargs: Any) -> Any:
        """Ghi vị trí đánh dấu nhưng vẫn vẽ đường thật."""
        positions.append(x)
        return original(axis, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", record_best)
    path = tracking.plot_training_curves(
        [epoch_payload(epoch) for epoch in range(1, epochs + 1)],
        tmp_path / "curves.png",
        best_epoch=1,
    )
    assert path is not None
    assert_png(path)
    assert positions == [1]


@pytest.mark.parametrize("lr_key", ["lr", "lr0", "train.lr", "train.lr0"])
def test_sweep_overrides_aliases_and_copy(cfg: Config, lr_key: str) -> None:
    """Đồng bộ LR, ánh xạ occlusion và không sửa cấu hình gốc."""
    before = deepcopy(cfg)
    updated = tracking.apply_sweep_overrides(
        cfg, {lr_key: 0.002, "occlusion_p": 0.4, "occlusion_ratio": 0.35}
    )
    assert updated["train"]["lr"] == updated["train"]["lr0"] == 0.002
    assert updated["online_preprocess"]["occlusion_p"] == 0.4
    assert updated["online_preprocess"]["occlusion_ratio"] == 0.35
    assert cfg == before


@pytest.mark.parametrize("key", PROTECTED_KEYS)
def test_sweep_overrides_block_immutable(cfg: Config, key: str) -> None:
    """Mỗi khóa bất biến, kể cả khóa cha data/eval, đều bị chặn."""
    before = deepcopy(cfg)
    assert tracking.apply_sweep_overrides(cfg, {key: "forbidden"}) == before
    assert cfg == before


def test_missing_wandb_key_falls_back_without_outputting_secret(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Thiếu key trả (None, False), không thử init hoặc lộ nội dung .env."""
    secret = "fake_key_not_real"
    (tmp_path / ".env").write_text(
        f"UNRELATED_SECRET={secret}\n", encoding="utf-8"
    )
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    tracking.load_wandb_env(cfg)
    assert "WANDB_API_KEY" not in tracking.os.environ
    assert tracking.init_wandb_run(cfg, tmp_path / "run") == (None, False)
    fake_wandb.init.assert_not_called()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_load_wandb_env_does_not_print_key(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Key giả đọc được từ .env và biến môi trường có quyền ưu tiên."""
    secret = "fake_key_not_real"
    (tmp_path / ".env").write_text(
        f'export WANDB_API_KEY="{secret}"\n', encoding="utf-8"
    )
    tracking.load_wandb_env(cfg)
    assert tracking.os.environ["WANDB_API_KEY"] == secret

    monkeypatch.setenv("WANDB_API_KEY", "fake_existing_key")
    tracking.load_wandb_env(cfg)
    assert tracking.os.environ["WANDB_API_KEY"] == "fake_existing_key"
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "fake_existing_key" not in captured.out + captured.err


@pytest.mark.parametrize("with_labels", [False, True])
def test_log_final_wandb_background(
    tmp_path: Path,
    fake_wandb: ModuleType,
    with_labels: bool,
) -> None:
    """Chart hỗ trợ index background cả khi JSON chỉ có class_names."""
    payload: dict[str, Any] = {
        "class_names": ["Stop", "Yield", "Limit"],
        "y_true": [0, 3, 2],
        "y_pred": [3, 1, 2],
    }
    if with_labels:
        payload["labels"] = payload["class_names"] + ["background"]
    path = tmp_path / "confusion_matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    Image.new("RGB", (2, 2)).save(tmp_path / "confusion_matrix.png")

    run = FakeRun()
    tracking.log_final_to_wandb(
        run, tmp_path, {"precision": 0.75}, cm_json_path=path
    )
    assert run.summary["test/precision"] == 0.75
    fake_wandb.Image.assert_called_once()
    fake_wandb.plot.confusion_matrix.assert_called_once_with(
        y_true=[0, 3, 2],
        preds=[3, 1, 2],
        class_names=["Stop", "Yield", "Limit", "background"],
    )
    assert any(
        row.get("charts/confusion_matrix_interactive") == "cm-chart"
        for row in run.logs
    )


def test_build_sweep_spec_blocks_immutable(cfg: Config) -> None:
    """Spec loại toàn bộ khóa bất biến và giữ tham số hợp lệ."""
    cfg["sweep"]["parameters"].update(
        {key: {"values": [1]} for key in PROTECTED_KEYS}
    )
    before = deepcopy(cfg)
    spec = run_sweep.build_sweep_spec(cfg)
    assert spec["parameters"] == {"lr": {"values": [0.001, 0.01]}}
    assert spec["metric"] == {"name": "best/metric", "goal": "maximize"}
    assert cfg == before


@pytest.mark.parametrize("pipeline", ["yolo", "faster_rcnn"])
def test_sweep_dry_run(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_wandb: ModuleType,
    pipeline: str,
) -> None:
    """Dry-run CLI xuất JSON hợp lệ cho cả hai pipeline, không mở W&B."""
    monkeypatch.setattr(
        run_sweep, "resolve_config", lambda *args, **kwargs: deepcopy(cfg)
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_sweep", "--pipeline", pipeline, "--dry-run", "--count", "2"],
    )
    run_sweep.main()
    result = json.loads(capsys.readouterr().out)
    assert result["pipeline"] == pipeline
    assert result["count"] == 2
    assert result["method"] == "random"
    assert result["parameters"] == {"lr": {"values": [0.001, 0.01]}}
    assert result["metric"]["name"] == "best/metric"
    fake_wandb.init.assert_not_called()


def assert_training_artifacts(directory: Path, checkpoint: str) -> None:
    """Kiểm tra artifact, schema và đủ hai epoch sau fit."""
    for filename in (
        checkpoint, "results.csv", "metrics_history.json", "metrics.json",
        "summary.json", "training_curves.png", "confusion_matrix.png",
        "confusion_matrix.json",
    ):
        assert (directory / filename).is_file(), filename

    for filename in ("training_curves.png", "confusion_matrix.png"):
        assert_png(directory / filename)

    history = read_json(directory / "metrics_history.json")
    assert [row["epoch"] for row in history] == [1, 2]
    assert all(STANDARD_KEYS <= row.keys() for row in history)
    with (directory / "results.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        assert STANDARD_KEYS <= set(reader.fieldnames or [])
        assert len(list(reader)) == 2

    result = read_json(directory / "metrics.json")
    assert {"fps", "latency_ms", "precision", "recall", "f1"} <= result.keys()
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
    assert CM_KEYS <= read_json(directory / "confusion_matrix.json").keys()
    assert read_json(directory / "summary.json")["epochs_completed"] == 2


def test_faster_rcnn_two_epoch_cpu_smoke(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fit thật hai epoch CPU với dataset, optimizer và artifact thật."""
    module = importlib.import_module("ts_alert.pipelines.faster_rcnn.train")
    tiny = TinyDetector()
    initial_weight = tiny.linear.weight.detach().clone()
    monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: tiny)
    monkeypatch.setattr(module, "FasterRCNNDetector", StaticDetector)

    directory = module.FasterRCNNPipeline(cfg).fit()
    assert_training_artifacts(directory, "best.pth")
    destination = Path(cfg["model"]["weights"])
    assert destination.read_bytes() == (directory / "best.pth").read_bytes()
    assert not torch.equal(initial_weight, tiny.linear.weight.detach())
    assert read_json(directory / "summary.json")["primary_metric"] == "mAP50-95"
    fake_wandb.init.assert_not_called()

    captured = capsys.readouterr()
    assert "* BEST" in captured.out
    assert "BEST epoch" in captured.out
    assert "WANDB_API_KEY" in captured.err


def test_yolo_tracking_hook_mapping_losses_and_pickle(
    cfg: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Loss dict được cộng đúng, final_eval không ghi trùng, hook pickle được."""
    module = importlib.import_module("ts_alert.pipelines.yolo.train")
    hook = module._YoloTrackingHook(cfg, tmp_path, "mAP50-95")
    trainer = make_yolo_trainer()
    hook.on_fit_epoch_end(trainer)

    payload = hook.history[0]
    assert payload["train/loss"] == pytest.approx(1.5)
    assert payload["train/box_loss"] == pytest.approx(0.8)
    assert payload["val/f1"] == pytest.approx(0.6)
    assert payload["val/mAP75"] == pytest.approx(0.45)
    assert payload["system/learning_rate"] == 0.0025
    assert payload["is_best"] == 1
    assert "* BEST" in capsys.readouterr().out

    before = (tmp_path / "metrics_history.json").read_bytes()
    hook.on_fit_epoch_end(trainer)
    assert len(hook.history) == 1
    assert (tmp_path / "metrics_history.json").read_bytes() == before
    assert capsys.readouterr().out == ""

    # Đối tượng kết nối không pickle được phải bị loại khỏi trạng thái.
    hook._wandb_run = lambda: None
    restored = pickle.loads(pickle.dumps(hook))
    assert restored.history == hook.history
    assert restored.best_epoch == 1
    assert restored.best_metric == 0.5
    assert restored._wandb_run is None
    assert restored._owns_run is False

    trainer.epoch = 1
    trainer.metrics["metrics/mAP50-95(B)"] = 0.6
    restored.on_fit_epoch_end(trainer)
    assert [row["epoch"] for row in restored.history] == [1, 2]
    assert restored.best_epoch == 2


def test_yolo_pipeline_cpu_smoke(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fake YOLO gọi callback thật và evaluator tạo đầy đủ artifact."""
    module = importlib.import_module("ts_alert.pipelines.yolo.train")
    model = FakeYOLO()
    cfg["model"]["weights"] = str(Path(cfg["model"]["weights"]).with_suffix(".pt"))
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "YoloDetector", StaticDetector)

    directory = module.YoloPipeline(cfg).fit()
    assert_training_artifacts(directory, "weights/best.pt")
    assert Path(cfg["model"]["weights"]).read_bytes() == b"fake-yolo-checkpoint"
    assert all(
        model.callbacks.get(event)
        for event in ("on_train_start", "on_fit_epoch_end", "on_train_end")
    )
    assert read_json(directory / "summary.json")["best_epoch"] == 2
    fake_wandb.init.assert_not_called()

    captured = capsys.readouterr()
    assert "* BEST" in captured.out
    assert "BEST epoch" in captured.out
    assert "WANDB_API_KEY" in captured.err


def test_string_runtime_option_both_pipelines(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: ModuleType,
) -> None:
    """Runtime dạng chuỗi vẫn đọc đúng tùy chọn và fit Faster R-CNN trên CPU."""
    yolo = importlib.import_module("ts_alert.pipelines.yolo.train")
    cfg.update(runtime="kaggle", num_gpus=2, device="cpu", workers=0)
    args = yolo.YoloPipeline(cfg)._train_args(
        Path(cfg["paths"]["outputs_dir"]) / "runtime_options",
        Path(cfg["paths"]["processed_dir"]) / "data.yaml",
    )
    assert args["device"] == "cpu"
    assert args["workers"] == 0

    module = importlib.import_module("ts_alert.pipelines.faster_rcnn.train")
    cfg.update(runtime="local", num_gpus=0, device="cpu", workers=0)
    tiny = TinyDetector()
    monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: tiny)
    monkeypatch.setattr(module, "FasterRCNNDetector", StaticDetector)

    directory = module.FasterRCNNPipeline(cfg).fit()
    assert_training_artifacts(directory, "best.pth")
    fake_wandb.init.assert_not_called()


def test_yolo_multi_gpu_missing_wandb_key_falls_back(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """YOLO cấu hình hai GPU thiếu key vẫn cảnh báo và lưu đầy đủ artifact."""
    module = importlib.import_module("ts_alert.pipelines.yolo.train")
    cfg.update(runtime="kaggle", num_gpus=2, device="cpu", workers=0)
    cfg["train"]["require_wandb"] = True
    cfg["model"]["weights"] = str(Path(cfg["model"]["weights"]).with_suffix(".pt"))
    model = FakeYOLO()
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "YoloDetector", StaticDetector)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    directory = module.YoloPipeline(cfg).fit()
    assert_training_artifacts(directory, "weights/best.pt")
    assert "WANDB_API_KEY" not in tracking.os.environ
    assert "WANDB_API_KEY" in capsys.readouterr().err
    fake_wandb.init.assert_not_called()
