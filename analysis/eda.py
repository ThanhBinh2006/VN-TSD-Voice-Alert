"""EDA: thống kê nhãn, kích thước box, heatmap tâm.   Phụ trách: Thành viên 1

Chạy: python -m analysis.eda --data-root <DATA_PATH>
"""
from __future__ import annotations

import argparse
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"


def class_distribution(data_root: Path):
    """Đếm số box theo lớp, vẽ biểu đồ cột -> figures/class_distribution.png."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def box_size_stats(data_root: Path):
    """Thống kê w/h/diện tích box (tương đối theo ảnh) -> figures/box_sizes.png."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def center_heatmap(data_root: Path):
    """Heatmap vị trí tâm box -> figures/center_heatmap.png (gợi ý cho quy tắc trái/phải/trước mặt)."""
    # TODO: Thành viên 1
    raise NotImplementedError("TODO: Thành viên 1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    class_distribution(root)
    box_size_stats(root)
    center_heatmap(root)


if __name__ == "__main__":
    main()
