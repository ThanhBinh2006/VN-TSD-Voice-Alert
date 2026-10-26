"""Phân tích khám phá dữ liệu (EDA) chuyên biệt cho tập dữ liệu thô ban đầu (data/raw).

Xuất chuẩn 8 biểu đồ lưu tại analysis/figures/raw/:
  01_class_distribution: Phân bố 52 lớp trên tập dữ liệu thô (sắp xếp giảm dần, trục Y symlog)
  02_boxes_per_image: Phân bố số lượng box trên mỗi ảnh
  03_box_dimensions: Kích thước box (chiều rộng vs chiều cao và phân loại MS-COCO)
  04_box_center_heatmap: Bản đồ nhiệt mật độ vị trí tâm box
  05_dhash_clusters: Gom cụm tương đồng bối cảnh bằng dHash 1024-bit
  06_tiny_box_candidates: Khảo sát các box nhỏ < min_box_px
  07_brightness_sharpness: Đánh giá quang học độ sáng & độ sắc nét (layout 2x2)
  08_sample_labels: Demo 9 ảnh mẫu kèm nhãn bounding box
File dữ liệu tổng hợp: analysis/figures/raw/eda_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analysis.helpers.common import load_classes, read_yolo_labels, get_image_paths
from analysis.helpers.class_distribution import plot_class_distribution
from analysis.helpers.box_metrics import (
    plot_boxes_per_image,
    plot_box_dimensions,
    plot_box_center_heatmap,
    plot_tiny_box_candidates,
)
from analysis.helpers.clustering import cluster_images_dhash, plot_dhash_clusters
from analysis.helpers.quality import analyze_and_plot_brightness_sharpness_raw
from analysis.helpers.sample_visualizer import plot_sample_labeled_images


def analyze_raw_dataset(
    data_root: str | Path = "data/raw",
    figures_dir: str | Path | None = None,
    class_table: str | Path | None = None,
    image_size: int = 640,
    min_box_px: int = 8,
    num_samples: int = 9,
    quality_samples: int = 1000,
    dhash_size: int = 32,
    dhash_hamming: int = 380,
    seed: int = 42,
) -> dict[str, Any]:
    """Quy trình phân tích chuyên biệt và xuất đủ 8 biểu đồ cho dataset thô."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root).resolve()
    target_dir = Path(figures_dir).resolve() if figures_dir else repo_root / "analysis" / "figures" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    c_table = Path(class_table).resolve() if class_table else repo_root / "configs" / "classes.csv"
    classes = load_classes(c_table)

    print(f"[EDA Raw] ========================================================")
    print(f"[EDA Raw] Phân tích dataset thô tại: {data_root}")
    print(f"[EDA Raw] Thư mục xuất biểu đồ: {target_dir}")
    print(f"[EDA Raw] ========================================================")

    img_paths = get_image_paths(data_root)
    flat_lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    all_boxes = read_yolo_labels(flat_lbl_dir, image_size_px=image_size)
    stems_all = sorted(list({b["stem"] for b in all_boxes}))

    summary_meta: dict[str, Any] = {
        "tag": "raw",
        "data_root": str(data_root),
        "n_images": len(img_paths),
        "n_boxes": len(all_boxes),
        "splits": {"all": (len(img_paths) if img_paths else len(stems_all), len(all_boxes))},
    }

    # 01. Class distribution
    print("\n[EDA Raw] 01 Class distribution (toàn bộ 52 lớp sắp xếp giảm dần)...")
    summary_meta["classes"] = plot_class_distribution(all_boxes, classes, target_dir / "01_class_distribution.png", tag="raw")

    # 02. Boxes per Image
    print("\n[EDA Raw] 02 Boxes per label / image distribution...")
    plot_boxes_per_image(data_root, all_boxes, target_dir, tag="raw")

    # 03. Box dimensions
    print("\n[EDA Raw] 03 Box dimensions...")
    plot_box_dimensions(all_boxes, target_dir, image_size=image_size, tag="raw")

    # 04. Box center heatmap
    print("\n[EDA Raw] 04 Box center heatmap...")
    plot_box_center_heatmap(all_boxes, target_dir, tag="raw")

    # 05. dHash Clusters (thuần túy dHash 1024-bit)
    print(f"\n[EDA Raw] 05 Gom cụm dHash 1024-bit (size={dhash_size}, Hamming <= {dhash_hamming})...")
    cluster_res = cluster_images_dhash(
        img_paths,
        hash_size=dhash_size,
        hamming_thresh=dhash_hamming,
    )
    plot_dhash_clusters(cluster_res, target_dir / "05_dhash_clusters.png", tag="raw")
    summary_meta["clustering"] = {k: v for k, v in cluster_res.items() if k != "clusters"}

    # 06. Tiny box candidates
    print("\n[EDA Raw] 06 Tiny box candidates...")
    plot_tiny_box_candidates(all_boxes, target_dir, min_box_px=min_box_px, image_size=image_size, tag="raw")

    # 07. Brightness & Sharpness
    print(f"\n[EDA Raw] 07 Brightness & Sharpness (mẫu {min(quality_samples, len(img_paths))} ảnh)...")
    summary_meta["quality"] = analyze_and_plot_brightness_sharpness_raw(
        img_paths, target_dir, max_samples=quality_samples, fig_num=7
    )

    # 08. Demo n ảnh mẫu
    print(f"\n[EDA Raw] 08 Demo {num_samples} ảnh mẫu kèm label...")
    plot_sample_labeled_images(data_root, all_boxes, classes, target_dir, num_samples=num_samples, seed=seed, tag="raw", fig_num=8)

    summary_path = target_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[EDA Raw] Đã lưu tóm tắt thống kê: {summary_path}")
    print(f"[EDA Raw] Hoàn thành phân tích! Toàn bộ 8 biểu đồ đã ghi tại: {target_dir}")

    return summary_meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Chương trình phân tích EDA chuyên biệt cho dữ liệu thô (RAW)")
    p.add_argument("--data-root", default="data/raw", help="Thư mục dataset thô (default: data/raw)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/raw/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    p.add_argument("--num-samples", type=int, default=9, help="Số ảnh mẫu vẽ demo nhãn (default: 9)")
    p.add_argument("--quality-samples", type=int, default=1000, help="Số ảnh lấy mẫu đo độ sáng/độ nét (default: 1000)")
    p.add_argument("--dhash-size", type=int, default=32, help="Kích thước cạnh dHash (default: 32 -> 1024 bits)")
    p.add_argument("--dhash-hamming", type=int, default=380, help="Ngưỡng khoảng cách Hamming dHash gom cụm (default: 380)")
    args = p.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[EDA Raw] Không tìm thấy thư mục: {data_root}")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
