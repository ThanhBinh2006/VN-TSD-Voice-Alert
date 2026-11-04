"""Phân tích khám phá dữ liệu (EDA) chuyên biệt cho tập dữ liệu sau tiền xử lý (data/processed).

Xuất chuẩn 9 biểu đồ lưu tại analysis/figures/processed/:
  01_class_distribution: Phân bố 52 lớp trên 3 subplot (Train Real vs Aug, Val, Test) với trục Y symlog
  02_boxes_per_image: Phân bố số lượng box trên mỗi ảnh sau letterbox
  03_box_dimensions: Kích thước box (chiều rộng vs chiều cao và MS-COCO scale trên canvas 640x640)
  04_box_center_heatmap: Mật độ vị trí tâm box trên canvas 640x640
  05_images_per_split: Số lượng ảnh và box theo từng tập Train, Val, Test
  06_tiny_box_candidates: Xác nhận 0% box nhỏ < min_box_px
  07_data_leakage: Ma trận trùng lặp và khoảng cách Hamming dHash (<=362)
  08_brightness_sharpness: Đánh giá quang học độ sáng & độ nét phân tách từng tập (layout 3x3)
  09_sample_labels: Demo 9 ảnh mẫu sau tiền xử lý kèm bounding box
File dữ liệu tổng hợp: analysis/figures/processed/eda_summary.json
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
from analysis.helpers.split_metrics import plot_images_per_split
from analysis.helpers.leakage import check_data_leakage, plot_data_leakage
from analysis.helpers.quality import analyze_and_plot_brightness_sharpness_processed
from analysis.helpers.sample_visualizer import plot_sample_labeled_images


def analyze_processed_dataset(
    data_root: str | Path = "data/processed",
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
    """Quy trình phân tích chuyên sâu độc lập cho dataset sau tiền xử lý (9 biểu đồ)."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root).resolve()
    target_dir = Path(figures_dir).resolve() if figures_dir else repo_root / "analysis" / "figures" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)

    c_table = Path(class_table).resolve() if class_table else repo_root / "configs" / "classes.csv"
    classes = load_classes(c_table)

    print(f"[EDA Processed] ========================================================")
    print(f"[EDA Processed] Phân tích dataset sau tiền xử lý tại: {data_root}")
    print(f"[EDA Processed] Thư mục xuất biểu đồ: {target_dir}")
    print(f"[EDA Processed] ========================================================")

    img_paths = get_image_paths(data_root)
    stem_to_path = {p.stem: p for p in img_paths}

    split_names = ["train", "val", "test"]
    split_stems: dict[str, list[str]] = {}
    split_stats: dict[str, tuple[int, int]] = {}
    all_boxes: list[dict[str, Any]] = []

    for sname in split_names:
        s_lbl_dir = data_root / "labels" / sname
        s_img_dir = data_root / "images" / sname
        s_imgs = list(s_img_dir.glob("*.jpg")) if s_img_dir.is_dir() else []
        sboxes = read_yolo_labels(s_lbl_dir, image_size_px=image_size)
        for b in sboxes:
            b["split"] = sname
            b["is_augmented"] = ("_aug" in b.get("stem", ""))
        all_boxes.extend(sboxes)
        stems = [p.stem for p in s_imgs] if s_imgs else list({b["stem"] for b in sboxes})
        split_stems[sname] = sorted(stems)
        split_stats[sname] = (len(stems), len(sboxes))

    summary_meta: dict[str, Any] = {
        "tag": "processed",
        "data_root": str(data_root),
        "n_images": len(img_paths),
        "n_boxes": len(all_boxes),
        "splits": split_stats,
    }

    # 01. Class distribution (3 subplot)
    print("\n[EDA Processed] 01 Class distribution (3 subplot: Train Real vs Aug, Val, Test)...")
    summary_meta["classes"] = plot_class_distribution(all_boxes, classes, target_dir / "01_class_distribution.png", tag="processed")

    # 02. Boxes per Image
    print("\n[EDA Processed] 02 Boxes per label / image distribution...")
    plot_boxes_per_image(data_root, all_boxes, target_dir, tag="processed")

    # 03. Box dimensions
    print("\n[EDA Processed] 03 Box dimensions...")
    plot_box_dimensions(all_boxes, target_dir, image_size=image_size, tag="processed")

    # 04. Box center heatmap
    print("\n[EDA Processed] 04 Box center heatmap...")
    plot_box_center_heatmap(all_boxes, target_dir, tag="processed")

    # 05. Images per split
    print("\n[EDA Processed] 05 Images per split...")
    plot_images_per_split(split_stats, target_dir, tag="processed")

    # 06. Tiny box candidates
    print("\n[EDA Processed] 06 Tiny box candidates...")
    plot_tiny_box_candidates(all_boxes, target_dir, min_box_px=min_box_px, image_size=image_size, tag="processed")

    # 07. Data leakage check (dHash thuần túy, loại bỏ padding xám 114)
    print(f"\n[EDA Processed] 07 Data leakage check (dHash <= {dhash_hamming}, bỏ viền xám)...")
    leak_data = check_data_leakage(
        split_stems,
        stem_to_path,
        hash_size=dhash_size,
        hamming_thresh=dhash_hamming,
    )
    plot_data_leakage(leak_data, target_dir / "07_data_leakage.png", tag="processed")
    summary_meta["leakage"] = {k: v for k, v in leak_data.items() if k != "min_dists"}

    # 08. Brightness & Sharpness (phân tách theo từng tập 3x3)
    print(f"\n[EDA Processed] 08 Brightness & Sharpness theo từng tập (mẫu {min(quality_samples, len(img_paths))} ảnh)...")
    summary_meta["quality"] = analyze_and_plot_brightness_sharpness_processed(
        img_paths=img_paths,
        split_stems=split_stems,
        out_dir=target_dir,
        max_samples=quality_samples,
        fig_num=8,
    )

    # 09. Demo n ảnh mẫu
    print(f"\n[EDA Processed] 09 Demo {num_samples} ảnh mẫu kèm label...")
    plot_sample_labeled_images(data_root, all_boxes, classes, target_dir, num_samples=num_samples, seed=seed, tag="processed", fig_num=9)

    summary_path = target_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[EDA Processed] Đã lưu tóm tắt thống kê: {summary_path}")
    print(f"[EDA Processed] Hoàn thành phân tích! Toàn bộ 9 biểu đồ đã ghi tại: {target_dir}")

    return summary_meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Chương trình phân tích EDA chuyên biệt cho dữ liệu sau tiền xử lý (PROCESSED)")
    p.add_argument("--data-root", default="data/processed", help="Thư mục dataset processed (default: data/processed)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/processed/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    p.add_argument("--num-samples", type=int, default=9, help="Số ảnh mẫu vẽ demo nhãn (default: 9)")
    p.add_argument("--quality-samples", type=int, default=1000, help="Số ảnh lấy mẫu đo độ sáng/độ nét (default: 1000)")
    p.add_argument("--dhash-size", type=int, default=32, help="Kích thước cạnh dHash (default: 32 -> 1024 bits)")
    p.add_argument("--dhash-hamming", type=int, default=380, help="Ngưỡng khoảng cách Hamming dHash (default: 380)")
    args = p.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[EDA Processed] Không tìm thấy thư mục: {data_root}")
        sys.exit(1)

    analyze_processed_dataset(
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
