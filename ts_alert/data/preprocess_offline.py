"""Tiền xử lý offline dùng chung cho cả hai mô hình.   Phụ trách: Thành viên 1

Luồng (run_preprocess) đã code sẵn; các hàm bên dưới là vỏ để điền.
"""
from __future__ import annotations

from pathlib import Path

from ts_alert.config.resolve import get_num_classes, resolve_path
from ts_alert.utils.io import load_classes, write_json
from ts_alert.utils.seed import set_seed


def list_raw_samples(raw_dir: Path) -> list[dict]:
    """Quét thư mục dữ liệu gốc, trả [{"image": Path, "label": Path | None}, ...] (ảnh + file nhãn YOLO .txt).
    Dataset Kaggle có thể dùng định dạng nhãn khác YOLO -> chuyển đổi ở đây."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def split_samples(samples: list[dict], ratios: dict, seed: int) -> dict:
    """Chia {"train": [...], "val": [...], "test": [...]} theo ratios (tổng = 1), cố định theo seed."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def letterbox_sample(sample: dict, out_dir: Path, split: str, image_size: int, pad_value: int = 114) -> None:
    """Letterbox ảnh về image_size x image_size, biến đổi lại toạ độ nhãn tương ứng,
    ghi vào out_dir/images/<split>/ và out_dir/labels/<split>/."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    """Ghi out_dir/data.yaml (path, train/val/test, nc, names) cho Ultralytics."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def run_preprocess(cfg: dict, data_root=None) -> dict:
    data = cfg["data"]
    raw_dir = resolve_path(data_root or cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    set_seed(cfg["seed"])

    samples = list_raw_samples(raw_dir)
    splits = split_samples(samples, data["split"], cfg["seed"])
    for split, items in splits.items():
        for sample in items:
            letterbox_sample(sample, out_dir, split, data["image_size"], data["pad_value"])

    classes = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
    write_data_yaml(out_dir, [classes[i]["name"] for i in sorted(classes)])
    meta = {"image_size": data["image_size"], "seed": cfg["seed"], "num_classes": get_num_classes(cfg),
            "counts": {k: len(v) for k, v in splits.items()}}
    write_json(meta, out_dir / "meta.json")
    return meta
