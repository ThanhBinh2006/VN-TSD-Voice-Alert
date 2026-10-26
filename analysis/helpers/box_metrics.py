"""Module trực quan hóa thống kê hình học Bounding Box: số lượng, kích thước, tâm box, tiny boxes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_boxes_per_image(
    data_root: Path,
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 02: Phân bố số lượng bounding box trên mỗi file nhãn / ảnh."""
    counts = Counter(b["stem"] for b in boxes)
    values = list(counts.values()) if counts else [0]
    total_imgs = len(counts)
    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    max_v = max(values) if values else 0

    val_counts = Counter(values)
    xs = sorted(val_counts.keys())
    ys = [val_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(xs, ys, color="#2B5B84", edgecolor="#1D3557", width=0.65, alpha=0.9)

    for bar, y_val in zip(bars, ys):
        pct = (y_val / total_imgs * 100) if total_imgs else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ys) * 0.015,
            f"{y_val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1D3557",
        )

    ax.axvline(mean_v, color="#E63946", linestyle="--", linewidth=2, label=f"Mean = {mean_v:.2f} box/ảnh")
    ax.axvline(median_v, color="#FB8500", linestyle="-.", linewidth=2, label=f"Median = {median_v:.1f} box/ảnh")

    title_suffix = f" – {tag.upper()}" if tag else ""
    ax.set_title(f"02. Phân bố Số lượng Bounding Box trên mỗi Ảnh{title_suffix}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Số lượng Bounding Box trên 1 ảnh", fontsize=11)
    ax.set_ylabel("Số lượng ảnh", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(ys) * 1.18 if ys else 10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "02_boxes_per_image.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_box_dimensions(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 03: Phân bố kích thước Bounding Box (Width vs Height theo pixel và tỉ lệ)."""
    if not boxes:
        return

    w_px = np.array([b.get("w_px", b["w"] * image_size) for b in boxes])
    h_px = np.array([b.get("h_px", b["h"] * image_size) for b in boxes])
    areas_px = w_px * h_px

    small_mask = areas_px < (32 * 32)
    med_mask = (areas_px >= (32 * 32)) & (areas_px < (96 * 96))
    large_mask = areas_px >= (96 * 96)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"03. Phân bố Kích thước Bounding Box (Khuôn hình {image_size}x{image_size}){title_suffix}", fontsize=14, fontweight="bold")

    # Panel 0: 2D Hexbin
    ax0 = axes[0]
    max_dim = min(image_size, max(np.percentile(w_px, 99.5), np.percentile(h_px, 99.5)) * 1.1)
    hb = ax0.hexbin(
        w_px, h_px,
        gridsize=45, cmap="YlGnBu", mincnt=1,
        extent=[0, max_dim, 0, max_dim],
        norm=mcolors.LogNorm(),
    )
    cb = fig.colorbar(hb, ax=ax0, label="Số lượng box (Log scale)")
    ax0.axline((0, 0), slope=1, color="#E63946", linestyle="--", linewidth=1.5, label="Tỉ lệ 1:1 (Vuông)")
    ax0.set_title("1. Mật độ Kích thước Chiều rộng vs Chiều cao", fontsize=12, fontweight="bold")
    ax0.set_xlabel(f"Chiều rộng Box (px trên canvas {image_size})", fontsize=10.5)
    ax0.set_ylabel(f"Chiều cao Box (px trên canvas {image_size})", fontsize=10.5)
    ax0.set_xlim(0, max_dim)
    ax0.set_ylim(0, max_dim)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="upper left")

    # Panel 1: Thống kê diện tích theo chuẩn MS-COCO
    ax1 = axes[1]
    coco_counts = [int(small_mask.sum()), int(med_mask.sum()), int(large_mask.sum())]
    coco_labels = [
        f"Small\n(< 32² px)\n{coco_counts[0]:,} ({coco_counts[0]/len(boxes)*100:.1f}%)",
        f"Medium\n(32² - 96² px)\n{coco_counts[1]:,} ({coco_counts[1]/len(boxes)*100:.1f}%)",
        f"Large\n(≥ 96² px)\n{coco_counts[2]:,} ({coco_counts[2]/len(boxes)*100:.1f}%)",
    ]
    coco_colors = ["#457B9D", "#2A9D8F", "#E76F51"]
    bars1 = ax1.bar(coco_labels, coco_counts, color=coco_colors, edgecolor="#1D3557", width=0.55, alpha=0.9)

    for bar, c in zip(bars1, coco_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(coco_counts) * 0.015,
            f"{c:,}\n({c/len(boxes)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax1.set_title("2. Phân loại Quy mô Box theo Chuẩn MS-COCO", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Số lượng bounding box", fontsize=10.5)
    ax1.set_ylim(0, max(coco_counts) * 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fname = out_dir / "03_box_dimensions.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_box_center_heatmap(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 04: Bản đồ nhiệt vị trí tâm bounding box trên khung ảnh chuẩn hóa."""
    if not boxes:
        return

    cx = np.array([b["cx"] for b in boxes])
    cy = np.array([b["cy"] for b in boxes])

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    bins = 50
    h, xedges, yedges = np.histogram2d(cx, cy, bins=bins, range=[[0, 1], [0, 1]])

    im = ax.imshow(
        h.T, origin="upper", extent=[0, 1, 1, 0],
        cmap="magma", interpolation="gaussian",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mật độ tâm box", fontsize=10.5)

    ax.axvline(0.5, color="white", linestyle="--", alpha=0.6, linewidth=1)
    ax.axhline(0.5, color="white", linestyle="--", alpha=0.6, linewidth=1)

    title_suffix = f" – {tag.upper()}" if tag else ""
    ax.set_title(f"04. Bản đồ Nhiệt Vị trí Tâm Bounding Box (Khuôn hình chuẩn hóa [0, 1]){title_suffix}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tọa độ tâm X (chuẩn hóa)", fontsize=11)
    ax.set_ylabel("Tọa độ tâm Y (chuẩn hóa)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    fig.tight_layout()
    fname = out_dir / "04_box_center_heatmap.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_tiny_box_candidates(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    min_box_px: int = 8,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 06: Kiểm tra các ứng viên box siêu nhỏ (tiny boxes < min_box_px)."""
    if not boxes:
        return

    w_px = np.array([b.get("w_px", b["w"] * image_size) for b in boxes])
    h_px = np.array([b.get("h_px", b["h"] * image_size) for b in boxes])
    min_dim = np.minimum(w_px, h_px)

    tiny_mask = min_dim < min_box_px
    n_tiny = int(tiny_mask.sum())
    pct_tiny = float(n_tiny / len(boxes) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"06. Khảo sát Box Siêu nhỏ (Tiny Box < {min_box_px}px trên {image_size}x{image_size}){title_suffix}", fontsize=14, fontweight="bold")

    # Panel 0: Histogram cạnh nhỏ nhất
    ax0 = axes[0]
    bins = np.linspace(0, 80, 41)
    counts, edges, patches_list = ax0.hist(min_dim, bins=bins, edgecolor="white", alpha=0.88)
    for edge, patch in zip(edges[:-1], patches_list):
        if edge < min_box_px:
            patch.set_facecolor("#E63946")
        else:
            patch.set_facecolor("#2A9D8F")

    ax0.axvline(min_box_px, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng tiny = {min_box_px} px")
    ax0.set_title(f"1. Phân bố Cạnh Nhỏ Nhất của Box (Tiny: {n_tiny:,} box = {pct_tiny:.2f}%)", fontsize=11.5, fontweight="bold")
    ax0.set_xlabel(f"Kích thước cạnh nhỏ nhất (min(w, h) theo px trên canvas {image_size})")
    ax0.set_ylabel("Số lượng box")
    ax0.legend()
    ax0.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 1: Bar chart so sánh Tiny vs Hợp lệ
    ax1 = axes[1]
    n_valid = len(boxes) - n_tiny
    cats = [f"Hợp lệ (≥ {min_box_px}px)\n{n_valid:,} ({n_valid/len(boxes)*100:.1f}%)",
            f"Tiny Box (< {min_box_px}px)\n{n_tiny:,} ({pct_tiny:.2f}%)"]
    vals = [n_valid, n_tiny]
    cols = ["#2A9D8F", "#E63946" if n_tiny > 0 else "#6C757D"]
    bars1 = ax1.bar(cats, vals, color=cols, edgecolor="#1D3557", width=0.45, alpha=0.88)

    for bar, val in zip(bars1, vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{val:,} box\n({val/len(boxes)*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax1.set_title(f"2. Tỉ lệ Phân loại Hợp lệ vs Tiny (< {min_box_px}px)", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Số lượng bounding box")
    ax1.set_ylim(0, max(vals) * 1.18 if vals else 10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fname = out_dir / "06_tiny_box_candidates.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
