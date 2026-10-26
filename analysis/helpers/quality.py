"""Module khảo sát chất lượng quang học: độ sáng (Grayscale mean) và độ sắc nét (Laplacian variance).

Tách biệt rõ ràng:
  - analyze_and_plot_brightness_sharpness_raw: Dành riêng cho tập ảnh thô (RAW), layout 2x2.
  - analyze_and_plot_brightness_sharpness_processed: Dành riêng cho các tập split (Train/Val/Test), layout 3x3 từng tập.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from analysis.helpers.image_utils import calculate_brightness_sharpness


def analyze_and_plot_brightness_sharpness_raw(
    img_paths: list[Path],
    out_dir: Path,
    max_samples: int = 1000,
    blur_threshold: float = 100.0,
    fig_num: int = 7,
) -> dict[str, Any]:
    """Figure 07 (RAW): Đánh giá Quang học Độ sáng & Độ sắc nét trên tập dữ liệu thô (2x2 grid)."""
    if not img_paths:
        return {}

    rng = random.Random(42)
    sample_paths = list(img_paths)
    if len(sample_paths) > max_samples:
        sample_paths = rng.sample(sample_paths, max_samples)

    brights, sharps = [], []
    for p in sample_paths:
        res = calculate_brightness_sharpness(p)
        if res is None:
            continue
        b, s_val = res
        brights.append(b)
        sharps.append(s_val)

    if not brights:
        return {}

    b_arr = np.array(brights)
    s_arr = np.array(sharps)
    blur_pct = float((s_arr < blur_threshold).sum() / len(s_arr) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{fig_num:02d}. Đánh giá Quang học: Độ sáng & Độ sắc nét – RAW", fontsize=14, fontweight="bold")

    # Panel 0: Brightness
    axes[0, 0].hist(b_arr, bins=40, range=(0, 255), color="#E9C46A", edgecolor="white", alpha=0.9)
    axes[0, 0].axvline(b_arr.mean(), color="#1D3557", linestyle="--", linewidth=2, label=f"Mean = {b_arr.mean():.1f}")
    axes[0, 0].set_title("1. Phân bố Độ sáng (Grayscale Mean [0, 255])", fontsize=11.5, fontweight="bold")
    axes[0, 0].set_xlabel("Độ sáng")
    axes[0, 0].set_ylabel("Số lượng ảnh")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 1: Sharpness
    s_plot = np.clip(s_arr, 0, 2000)
    axes[0, 1].hist(s_plot, bins=40, color="#2A9D8F", edgecolor="white", alpha=0.9)
    axes[0, 1].axvline(blur_threshold, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng mờ = {blur_threshold:.0f}")
    axes[0, 1].set_title(f"2. Phân bố Độ sắc nét (Laplacian Var | Ảnh mờ: {blur_pct:.1f}%)", fontsize=11.5, fontweight="bold")
    axes[0, 1].set_xlabel("Độ sắc nét")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 2: Scatter
    scatter = axes[1, 0].scatter(b_arr, s_plot, c=b_arr, cmap="plasma", alpha=0.5, s=16)
    axes[1, 0].axhline(blur_threshold, color="#E63946", linestyle="--", label="Ngưỡng mờ")
    axes[1, 0].set_title("3. Tương quan Độ sáng vs Độ sắc nét", fontsize=11.5, fontweight="bold")
    axes[1, 0].set_xlabel("Độ sáng")
    axes[1, 0].set_ylabel("Độ nét")
    axes[1, 0].legend()
    axes[1, 0].grid(linestyle=":", alpha=0.5)
    plt.colorbar(scatter, ax=axes[1, 0], label="Độ sáng")

    # Panel 3: Donut chart
    ax_d = axes[1, 1]
    clear_cnt = int(((s_arr >= blur_threshold) & (b_arr >= 60) & (b_arr <= 190)).sum())
    blur_cnt = int((s_arr < blur_threshold).sum())
    dark_cnt = int(((s_arr >= blur_threshold) & (b_arr < 60)).sum())
    bright_cnt = int(((s_arr >= blur_threshold) & (b_arr > 190)).sum())

    labels_q = ["Rõ nét & Chuẩn sáng", "Bị mờ (Blur)", "Thiếu sáng (Tối)", "Chói sáng"]
    counts_q = [clear_cnt, blur_cnt, dark_cnt, bright_cnt]
    colors_q = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261"]

    valid_q = [(l, c, col) for l, c, col in zip(labels_q, counts_q, colors_q) if c > 0]
    if valid_q:
        vl, vc, vcol = zip(*valid_q)
        tot_q = sum(vc)
        wedges, texts, autotexts = ax_d.pie(
            vc, labels=None, colors=vcol,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5.0 else "",
            startangle=140, pctdistance=0.72,
            textprops={"fontsize": 11, "fontweight": "bold", "color": "white"},
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.42},
        )
        ax_d.text(0, 0.08, f"{clear_cnt/tot_q*100:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold", color="#1D3557")
        ax_d.text(0, -0.14, "Đạt chuẩn", ha="center", va="center", fontsize=10.5, color="#6C757D")
        leg_labels = [f"{l}: {c:,} ({c/tot_q*100:.1f}%)" for l, c in zip(vl, vc)]
        ax_d.legend(wedges, leg_labels, title=f"Phân loại ({tot_q:,} ảnh)", loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=9.5)
    ax_d.set_title("4. Tỉ lệ Phân loại Chất lượng Quang học", fontsize=11.5, fontweight="bold")

    fig.tight_layout()
    fname = out_dir / f"{fig_num:02d}_brightness_sharpness.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    return {
        "samples_analyzed": len(b_arr),
        "mean_brightness": round(float(b_arr.mean()), 2),
        "median_sharpness": round(float(np.median(s_arr)), 2),
        "blur_pct": round(blur_pct, 2),
    }


def analyze_and_plot_brightness_sharpness_processed(
    img_paths: list[Path],
    split_stems: dict[str, list[str]],
    out_dir: Path,
    max_samples: int = 1200,
    blur_threshold: float = 100.0,
    fig_num: int = 8,
) -> dict[str, Any]:
    """Figure 08 (PROCESSED): Đánh giá Quang học phân tách độc lập theo từng tập (Train, Val, Test) (3x3 grid)."""
    if not img_paths:
        return {}

    stem_to_p = {p.stem: p for p in img_paths}
    rng = random.Random(42)

    active_splits = [s for s in ("train", "val", "test") if s in split_stems and split_stems[s]]
    split_metrics: dict[str, dict[str, list[float]]] = {}
    all_brights, all_sharps = [], []

    samples_per_split = max(200, max_samples // max(len(active_splits), 1))
    for s in active_splits:
        stems_s = split_stems[s]
        chosen_stems = rng.sample(stems_s, min(len(stems_s), samples_per_split))
        b_list, s_list = [], []
        for st in chosen_stems:
            p = stem_to_p.get(st)
            if not p or not p.exists():
                continue
            res = calculate_brightness_sharpness(p)
            if res is None:
                continue
            b, s_val = res
            b_list.append(b)
            s_list.append(s_val)
            all_brights.append(b)
            all_sharps.append(s_val)
        split_metrics[s] = {"brightness": b_list, "sharpness": s_list}

    if not all_brights:
        return {}

    b_arr = np.array(all_brights)
    s_arr = np.array(all_sharps)
    blur_pct = float((s_arr < blur_threshold).sum() / len(s_arr) * 100)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5))
    fig.suptitle(
        f"{fig_num:02d}. Đánh giá Quang học: Độ sáng & Độ sắc nét trên từng Tập – PROCESSED",
        fontsize=14,
        fontweight="bold",
    )

    split_colors = {"train": "#1D3557", "val": "#2A9D8F", "test": "#7209B7"}
    split_titles = {"train": "Tập Huấn luyện (Train)", "val": "Tập Kiểm định (Val)", "test": "Tập Kiểm tra (Test)"}

    # Hàng 1: Độ sáng từng tập
    for j, s in enumerate(active_splits[:3]):
        ax = axes[0, j]
        b_s = np.array(split_metrics[s]["brightness"])
        ax.hist(b_s, bins=35, range=(0, 255), color=split_colors.get(s, "#2A9D8F"), edgecolor="white", alpha=0.88)
        ax.axvline(b_s.mean(), color="#E63946", linestyle="--", linewidth=2, label=f"Mean = {b_s.mean():.1f}")
        ax.axvline(np.median(b_s), color="#FB8500", linestyle=":", linewidth=1.5, label=f"Median = {np.median(b_s):.1f}")
        ax.set_title(f"1{chr(97+j)}. Độ sáng – {split_titles.get(s, s)}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Độ sáng Grayscale [0, 255]", fontsize=10)
        ax.set_ylabel("Số lượng ảnh", fontsize=10)
        ax.legend(fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    # Hàng 2: Độ sắc nét từng tập
    for j, s in enumerate(active_splits[:3]):
        ax = axes[1, j]
        s_s = np.clip(np.array(split_metrics[s]["sharpness"]), 0, 2000)
        blur_pct_s = float((s_s < blur_threshold).sum() / max(len(s_s), 1) * 100)
        ax.hist(s_s, bins=35, color=split_colors.get(s, "#2A9D8F"), edgecolor="white", alpha=0.88)
        ax.axvline(blur_threshold, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng mờ = {blur_threshold:.0f}")
        ax.axvline(np.median(s_s), color="#FB8500", linestyle=":", linewidth=1.5, label=f"Median = {np.median(s_s):.0f}")
        ax.set_title(f"2{chr(97+j)}. Độ nét – {split_titles.get(s, s)} (Mờ: {blur_pct_s:.1f}%)", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Độ nét Laplacian Variance", fontsize=10)
        ax.set_ylabel("Số lượng ảnh", fontsize=10)
        ax.legend(fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    # Hàng 3a: Đối chuẩn Độ sáng giữa các tập
    ax3a = axes[2, 0]
    for s in active_splits[:3]:
        b_s = np.array(split_metrics[s]["brightness"])
        ax3a.hist(
            b_s, bins=35, range=(0, 255), density=True, histtype="step",
            linewidth=2.4, color=split_colors.get(s, "#2A9D8F"),
            label=f"{s.upper()} (Mean: {b_s.mean():.1f})",
        )
    ax3a.set_title("3a. Đối chuẩn Phân bố Độ sáng giữa 3 Tập", fontsize=11.5, fontweight="bold")
    ax3a.set_xlabel("Độ sáng Grayscale [0, 255]", fontsize=10)
    ax3a.set_ylabel("Mật độ (Density)", fontsize=10)
    ax3a.legend(fontsize=9.5)
    ax3a.grid(linestyle=":", alpha=0.45)

    # Hàng 3b: Đối chuẩn Độ nét giữa các tập
    ax3b = axes[2, 1]
    for s in active_splits[:3]:
        s_s = np.clip(np.array(split_metrics[s]["sharpness"]), 0, 2000)
        ax3b.hist(
            s_s, bins=35, density=True, histtype="step",
            linewidth=2.4, color=split_colors.get(s, "#2A9D8F"),
            label=f"{s.upper()} (Med: {np.median(s_s):.0f})",
        )
    ax3b.axvline(blur_threshold, color="#E63946", linestyle="--", linewidth=1.5, label=f"Ngưỡng mờ ({blur_threshold:.0f})")
    ax3b.set_title("3b. Đối chuẩn Phân bố Độ sắc nét giữa 3 Tập", fontsize=11.5, fontweight="bold")
    ax3b.set_xlabel("Độ nét Laplacian Variance", fontsize=10)
    ax3b.set_ylabel("Mật độ (Density)", fontsize=10)
    ax3b.legend(fontsize=9.5)
    ax3b.grid(linestyle=":", alpha=0.45)

    # Hàng 3c: Grouped bar chart phân loại chất lượng
    ax3c = axes[2, 2]
    cats = ["Đạt chuẩn", "Mờ (<100)", "Tối (<60)", "Chói (>190)"]
    x_idx = np.arange(len(cats))
    bar_w = 0.25
    for k, s in enumerate(active_splits[:3]):
        b_s = np.array(split_metrics[s]["brightness"])
        s_s = np.array(split_metrics[s]["sharpness"])
        tot = max(len(b_s), 1)
        p_clear = ((s_s >= blur_threshold) & (b_s >= 60) & (b_s <= 190)).sum() / tot * 100
        p_blur = (s_s < blur_threshold).sum() / tot * 100
        p_dark = ((s_s >= blur_threshold) & (b_s < 60)).sum() / tot * 100
        p_bright = ((s_s >= blur_threshold) & (b_s > 190)).sum() / tot * 100
        vals = [p_clear, p_blur, p_dark, p_bright]
        bars = ax3c.bar(
            x_idx + (k - 1) * bar_w, vals, bar_w,
            label=s.upper(), color=split_colors.get(s, "#2A9D8F"), alpha=0.88, edgecolor="white"
        )
        for bar in bars:
            h = bar.get_height()
            if h >= 2.0:
                ax3c.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%",
                          ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax3c.set_xticks(x_idx)
    ax3c.set_xticklabels(cats, fontsize=9.5, fontweight="bold")
    ax3c.set_ylabel("Tỉ lệ (%)", fontsize=10)
    ax3c.set_ylim(0, 105)
    ax3c.set_title("3c. Tỉ lệ Phân loại Chất lượng Quang học (%)", fontsize=11.5, fontweight="bold")
    ax3c.legend(fontsize=9.5)
    ax3c.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fname = out_dir / f"{fig_num:02d}_brightness_sharpness.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    return {
        "samples_analyzed": len(b_arr),
        "mean_brightness": round(float(b_arr.mean()), 2),
        "median_sharpness": round(float(np.median(s_arr)), 2),
        "blur_pct": round(blur_pct, 2),
    }
