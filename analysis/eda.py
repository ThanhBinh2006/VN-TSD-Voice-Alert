"""Module điều phối trung tâm phân tích khám phá dữ liệu (EDA Router).

Tự động nhận diện định dạng dữ liệu (RAW hay PROCESSED) và chuyển tiếp tới module chuyên trách:
  - analysis.eda_raw: Chuyên phân tích tập thô (8 biểu đồ chuẩn, Figure 01 - 08)
  - analysis.eda_processed: Chuyên phân tích tập sau tiền xử lý (9 biểu đồ chuẩn, Figure 01 - 09)

Đồng thời tương thích hoàn toàn với notebook 01_data_analysis.ipynb thông qua các hàm:
  - class_distribution(data_root)
  - box_size_stats(data_root)
  - center_heatmap(data_root)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FIG_DIR = Path(__file__).resolve().parent / "figures"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from analysis.eda_raw import analyze_raw_dataset
from analysis.eda_processed import analyze_processed_dataset
from analysis.helpers.common import load_classes, read_yolo_labels
from analysis.helpers.class_distribution import plot_class_distribution
from analysis.helpers.box_metrics import plot_box_dimensions, plot_box_center_heatmap

__all__ = [
    "FIG_DIR",
    "class_distribution",
    "box_size_stats",
    "center_heatmap",
    "analyze_raw_dataset",
    "analyze_processed_dataset",
    "main",
]


def _get_class_table(class_table: str | Path | None = None) -> Path:
    if class_table:
        return Path(class_table).resolve()
    default_csv = PROJECT_ROOT / "configs" / "classes.csv"
    if default_csv.exists():
        return default_csv
    return Path("configs/classes.csv").resolve()


def class_distribution(data_root: Path, class_table: str | Path | None = None) -> Path:
    """Đếm số box theo lớp, vẽ biểu đồ cột -> figures/class_distribution.png (và figures/raw/01_class_distribution.png)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c_table = _get_class_table(class_table)
    classes = load_classes(c_table)
    lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    boxes = read_yolo_labels(lbl_dir)
    out_path = FIG_DIR / "class_distribution.png"
    plot_class_distribution(boxes, classes, out_path, is_raw=True)
    return out_path


def box_size_stats(data_root: Path, image_size: int = 640) -> Path:
    """Thống kê w/h/diện tích box (tương đối theo ảnh) -> figures/box_sizes.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    boxes = read_yolo_labels(lbl_dir, image_size_px=image_size)
    out_path = FIG_DIR / "box_sizes.png"
    plot_box_dimensions(boxes, out_path, image_size=image_size)
    return out_path


def center_heatmap(data_root: Path) -> Path:
    """Heatmap vị trí tâm box -> figures/center_heatmap.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    boxes = read_yolo_labels(lbl_dir)
    out_path = FIG_DIR / "center_heatmap.png"
    plot_box_center_heatmap(boxes, out_path)
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Offline EDA toàn diện chuẩn 8 biểu đồ (RAW) hoặc 9 biểu đồ (PROCESSED) cho TSD-Voice-Alert"
    )
    p.add_argument("--data-root", default=None, help="Thư mục gốc dataset (raw hoặc processed)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    p.add_argument("--num-samples", type=int, default=9, help="Số ảnh mẫu vẽ demo nhãn (default: 9)")
    p.add_argument("--quality-samples", type=int, default=1000, help="Số ảnh lấy mẫu đo độ sáng/độ nét (default: 1000)")
    p.add_argument("--dhash-size", type=int, default=32, help="Kích thước cạnh dHash (default: 32 -> 1024 bits)")
    p.add_argument("--dhash-hamming", type=int, default=380, help="Ngưỡng khoảng cách Hamming dHash gom cụm (default: 380)")
    args = p.parse_args(argv)

    data_root_str = args.data_root or os.environ.get("DATA_ROOT") or ""
    if not data_root_str:
        print("[EDA] Chưa có --data-root. Dùng: python -m analysis.eda --data-root data/raw")
        sys.exit(1)
    data_root = Path(data_root_str).resolve()
    if not data_root.exists():
        print(f"[EDA] Không tìm thấy thư mục: {data_root}")
        sys.exit(1)

    is_processed = (data_root / "labels" / "train").is_dir() or "processed" in data_root.parts

    if is_processed:
        dhash_h = args.dhash_hamming
        analyze_processed_dataset(
            data_root=data_root,
            figures_dir=args.figures_dir,
            class_table=args.class_table,
            image_size=args.image_size,
            min_box_px=args.min_box_px,
            num_samples=args.num_samples,
            quality_samples=args.quality_samples,
            dhash_size=args.dhash_size,
            dhash_hamming=dhash_h,
        )
    else:
        analyze_raw_dataset(
            data_root=data_root,
            figures_dir=args.figures_dir,
            class_table=args.class_table,
            image_size=args.image_size,
            min_box_px=args.min_box_px,
            num_samples=args.num_samples,
            quality_samples=args.quality_samples,
            dhash_size=args.dhash_size,
            dhash_hamming=args.dhash_hamming,
        )
        # Đồng bộ các biểu đồ chính ra analysis/figures/ cho notebook 01_data_analysis.ipynb
        raw_fig_dir = FIG_DIR / "raw"
        if raw_fig_dir.is_dir():
            if (raw_fig_dir / "01_class_distribution.png").exists():
                shutil.copy2(raw_fig_dir / "01_class_distribution.png", FIG_DIR / "class_distribution.png")
            if (raw_fig_dir / "03_box_dimensions.png").exists():
                shutil.copy2(raw_fig_dir / "03_box_dimensions.png", FIG_DIR / "box_sizes.png")
            if (raw_fig_dir / "04_box_center_heatmap.png").exists():
                shutil.copy2(raw_fig_dir / "04_box_center_heatmap.png", FIG_DIR / "center_heatmap.png")


if __name__ == "__main__":
    main()
