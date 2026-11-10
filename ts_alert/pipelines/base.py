"""Lớp cơ sở cho detector và pipeline huấn luyện."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ts_alert.config.resolve import resolve_path
from ts_alert.eval.evaluate import evaluate_best_model
from ts_alert.types import Detection
from ts_alert.utils.io import read_json, write_json


class BaseDetector(ABC):
    """Giao diện chung cho bộ phát hiện biển báo."""

    name: str = ""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Phát hiện đối tượng trên ảnh gốc."""
        raise NotImplementedError


class BasePipeline(ABC):
    """Giao diện chung cho huấn luyện và đánh giá."""

    name: str = ""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Lưu cấu hình pipeline."""
        self.cfg = cfg

    @abstractmethod
    def fit(self) -> Path:
        """Huấn luyện và trả về thư mục kết quả."""
        raise NotImplementedError

    @abstractmethod
    def load_detector(
        self,
        weights: Path | str | None = None,
    ) -> BaseDetector:
        """Nạp detector từ trọng số chỉ định hoặc mặc định."""
        raise NotImplementedError

    def best_checkpoint(self, run_dir: Path) -> Path:
        """Tìm checkpoint tốt nhất theo thứ tự đường dẫn."""
        checkpoints = sorted(Path(run_dir).rglob("best.*"))
        for checkpoint in checkpoints:
            if checkpoint.is_file():
                return checkpoint
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint best.* trong: {run_dir}"
        )

    def evaluate(self, run_dir: Path) -> dict[str, Any]:
        """Đánh giá checkpoint tốt nhất, lưu metrics và cập nhật tổng kết."""
        run_dir = Path(run_dir)
        detector = self.load_detector(self.best_checkpoint(run_dir))
        eval_split = str((self.cfg.get("eval") or {}).get("split", "test"))

        report = evaluate_best_model(
            detector,
            self.cfg,
            out_dir=run_dir,
            splits=("test", "val"),
            tune_split=eval_split,
            raw_conf=0.01,
        )

        test_metrics = {"pipeline": self.name, **report["test_metrics"]}
        write_json(test_metrics, run_dir / "metrics.json")
        write_json(test_metrics, run_dir / f"metrics_{eval_split}.json")

        val_metrics = report.get("val_metrics")
        if val_metrics is not None:
            write_json(
                {"pipeline": self.name, **val_metrics},
                run_dir / "metrics_val.json",
            )

        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            summary.update(
                {
                    "best_conf_threshold": report["best_conf_threshold"],
                    "test_metrics": report["test_metrics"],
                }
            )
            if val_metrics is not None:
                summary["val_metrics"] = val_metrics
            write_json(summary, summary_path)

        return dict(report["test_metrics"])

    def default_weights(self) -> Path:
        """Trả về đường dẫn trọng số mặc định."""
        return resolve_path(self.cfg["model"]["weights"])
