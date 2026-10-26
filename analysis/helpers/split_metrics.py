"""Module thống kê số lượng ảnh và box theo các tập Train, Validation, Test."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def plot_images_per_split(
    split_stats: dict[str, tuple[int, int]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 05: Số lượng ảnh và bounding box trên mỗi split (train, val, test)."""
    splits = [s for s in ("train", "val", "test") if s in split_stats]
    if not splits:
        splits = list(split_stats.keys())

    n_imgs = [split_stats[s][0] for s in splits]
    n_bxs = [split_stats[s][1] for s in splits]
    tot_imgs = sum(n_imgs) or 1
    tot_bxs = sum(n_bxs) or 1

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6.5))
    rects1 = ax.bar(x - width / 2, n_imgs, width, label="Số lượng Ảnh", color="#1D3557", edgecolor="white", alpha=0.9)
    rects2 = ax.bar(x + width / 2, n_bxs, width, label="Số lượng Bounding Box", color="#2A9D8F", edgecolor="white", alpha=0.9)

    for r in rects1:
        h = r.get_height()
        ax.text(
            r.get_x() + r.get_width() / 2, h + max(n_imgs) * 0.015,
            f"{int(h):,}\n({h/tot_imgs*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1D3557",
        )

    for r in rects2:
        h = r.get_height()
        ax.text(
            r.get_x() + r.get_width() / 2, h + max(n_bxs) * 0.015,
            f"{int(h):,}\n({h/tot_bxs*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2A9D8F",
        )

    title_suffix = f" – {tag.upper()}" if tag else ""
    ax.set_title(f"05. Phân bố Ảnh và Bounding Box trên các Phân vùng dữ liệu (Splits){title_suffix}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax.set_ylabel("Số lượng", fontsize=11)
    ax.set_ylim(0, max(max(n_imgs), max(n_bxs)) * 1.18 if (n_imgs and n_bxs) else 10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fname = out_dir / "05_images_per_split.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
