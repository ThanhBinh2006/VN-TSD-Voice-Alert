"""Các hàm đọc nhãn YOLO, nạp danh mục lớp và duyệt ảnh dùng chung trong EDA."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_classes(csv_path: str | Path) -> dict[int, str]:
    """Đọc configs/classes.csv trả về dict {class_id: name_vi}."""
    p = Path(csv_path)
    if not p.exists():
        return {}
    classes: dict[int, str] = {}
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row.get("id") or row.get("class_id") or 0)
                name = row.get("name_vi") or row.get("name") or f"Class_{cid}"
                classes[cid] = name.strip()
            except ValueError:
                continue
    return classes


def read_yolo_labels(
    label_dir: Path,
    image_size_px: int | None = None,
) -> list[dict[str, Any]]:
    """Đọc toàn bộ file .txt định dạng YOLO trong label_dir (hoặc tìm đệ quy)."""
    boxes: list[dict[str, Any]] = []
    if not label_dir.exists():
        return boxes

    txt_files = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
    for f in txt_files:
        stem = f.stem
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid, cx, cy, bw, bh = (
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    continue
                entry: dict[str, Any] = dict(
                    class_id=cid, cx=cx, cy=cy, w=bw, h=bh, stem=stem
                )
                if image_size_px is not None:
                    entry["w_px"] = bw * image_size_px
                    entry["h_px"] = bh * image_size_px
                boxes.append(entry)
    return boxes


def get_image_paths(data_root: Path) -> list[Path]:
    """Tìm toàn bộ file ảnh trong data_root (hoặc data_root/images)."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root

    paths = [p for p in search_dir.rglob("*.*") if p.suffix.lower() in img_extensions]
    return sorted(paths)
