"""Lớp cơ sở cho detector và pipeline huấn luyện."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ts_alert.config.resolve import resolve_path
from ts_alert.eval.evaluate import evaluate_detector
from ts_alert.types import Detection
from ts_alert.utils.io import write_json


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
        """Đánh giá checkpoint tốt nhất và lưu metrics.json."""
        run_dir = Path(run_dir)
        detector = self.load_detector(self.best_checkpoint(run_dir))
        metrics = evaluate_detector(
            detector,
            self.cfg,
            split=self.cfg["eval"]["split"],
            out_dir=run_dir,
        )
        write_json({"pipeline": self.name, **metrics}, run_dir / "metrics.json")
        return metrics

    def default_weights(self) -> Path:
        """Trả về đường dẫn trọng số mặc định."""
        return resolve_path(self.cfg["model"]["weights"])
