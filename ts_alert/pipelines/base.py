"""Khuôn mẫu chung cho hai mô hình. Cả hai PHẢI trả về cùng định dạng (docs/DATA_CONTRACT.md)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ts_alert.config.resolve import resolve_path
from ts_alert.types import Detection
from ts_alert.utils.io import write_json


class BaseDetector(ABC):
    """Đối tượng suy luận: nạp trọng số một lần, gọi predict() trên từng khung hình."""

    name: str = ""

    @abstractmethod
    def predict(self, frame) -> list[Detection]:
        """frame: ndarray BGR (H, W, 3), ảnh gốc. Trả [] nếu không phát hiện được gì."""


class BasePipeline(ABC):
    name: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit(self) -> Path:
        """Huấn luyện, lưu checkpoint + log vào run_dir (ts_alert.runtime.run_dir.make_run_dir). Trả run_dir."""

    @abstractmethod
    def load_detector(self, weights=None) -> BaseDetector:
        """Nạp checkpoint (mặc định cfg["model"]["weights"]) và trả về detector."""

    def best_checkpoint(self, run_dir: Path) -> Path:
        found = sorted(Path(run_dir).rglob("best.*"))
        if not found:
            raise FileNotFoundError(f"Không thấy best.* trong {run_dir}")
        return found[0]

    def evaluate(self, run_dir: Path) -> dict:
        """Đánh giá trên tập Test với đúng bộ chỉ số chung, ghi metrics.json vào run_dir."""
        from ts_alert.eval.evaluate import evaluate_detector

        detector = self.load_detector(self.best_checkpoint(run_dir))
        metrics = evaluate_detector(detector, self.cfg, split=self.cfg["eval"]["split"])
        write_json({"pipeline": self.name, **metrics}, Path(run_dir) / "metrics.json")
        return metrics

    def default_weights(self) -> Path:
        return resolve_path(self.cfg["model"]["weights"])
