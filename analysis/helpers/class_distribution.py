"""Module vẽ biểu đồ phân bố 52 lớp biển báo giao thông (Figure 01).

Đáp ứng yêu cầu:
  - Tách thành 3 biểu đồ (subplot) riêng biệt cho từng tập: Train, Validation, Test.
  - Trên tập Train: Phân biệt rõ Ảnh Thật (Real) và Ảnh Tự tạo / Tăng cường (Augmented) bằng 2 màu sắc khác nhau.
  - Loại bỏ hoàn toàn các câu chữ kết luận dài dòng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_path: Path,
    tag: str = "",
) -> dict[str, Any]:
    """Vẽ Figure 01: Phân bố toàn bộ 52 lớp biển báo giao thông."""
    all_class_ids = sorted(classes.keys())
    total_boxes = len(boxes)

    splits_present = set(b.get("split") for b in boxes if b.get("split"))
    is_processed_split = (tag != "raw") and bool(splits_present & {"train", "val", "test"})

    if is_processed_split:
        # ──────────────────────────────────────────────────────────────────────
        # Chế độ Processed: Vẽ 3 biểu đồ riêng biệt (Train, Val, Test)
        # ──────────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
        title_suffix = f" – {tag.upper()}" if tag else ""
        fig.suptitle(
            f"01. Phân bố 52 Lớp Biển báo theo từng Tập Dữ liệu (Train, Validation, Test){title_suffix}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # 1. Thống kê theo từng split
        train_real_counts: Counter[int] = Counter()
        train_aug_counts: Counter[int] = Counter()
        val_counts: Counter[int] = Counter()
        test_counts: Counter[int] = Counter()
        overall_counts: Counter[int] = Counter()

        for b in boxes:
            cid = b["class_id"]
            sname = b.get("split", "train")
            is_aug = b.get("is_augmented", False) or ("_aug" in b.get("stem", ""))

            overall_counts[cid] += 1
            if sname == "train":
                if is_aug:
                    train_aug_counts[cid] += 1
                else:
                    train_real_counts[cid] += 1
            elif sname == "val":
                val_counts[cid] += 1
            elif sname == "test":
                test_counts[cid] += 1

        # Sắp xếp các lớp theo tổng số lượng giảm dần để trục X đồng bộ và dễ quan sát
        sorted_cids = sorted(all_class_ids, key=lambda c: overall_counts[c], reverse=True)
        x_indices = np.arange(len(sorted_cids))
        x_labels = [f"[{cid}] {classes.get(cid, '')[:14]}" for cid in sorted_cids]

        # ── Panel 0: Tập Train (Stacked bar: Real vs Augmented) ──
        ax_tr = axes[0]
        y_tr_real = [train_real_counts[c] for c in sorted_cids]
        y_tr_aug = [train_aug_counts[c] for c in sorted_cids]

        ax_tr.bar(x_indices, y_tr_real, color="#1D3557", label="Ảnh Thật (Real)", width=0.72)
        ax_tr.bar(x_indices, y_tr_aug, bottom=y_tr_real, color="#E76F51", label="Ảnh Tự tạo (Augmented)", width=0.72)

        tot_tr_real = sum(y_tr_real)
        tot_tr_aug = sum(y_tr_aug)
        ax_tr.set_title(
            f"1. Tập Huấn luyện (Train Split): Tổng {tot_tr_real + tot_tr_aug:,} box  |  Ảnh Thật: {tot_tr_real:,} box,  Ảnh Tự tạo: {tot_tr_aug:,} box",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax_tr.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # ── Panel 1: Tập Validation ──
        ax_val = axes[1]
        y_val = [val_counts[c] for c in sorted_cids]
        ax_val.bar(x_indices, y_val, color="#2A9D8F", label="Validation Box", width=0.72)
        tot_val = sum(y_val)
        ax_val.set_title(
            f"2. Tập Kiểm định (Validation Split): Tổng {tot_val:,} box",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax_val.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # ── Panel 2: Tập Test ──
        ax_te = axes[2]
        y_te = [test_counts[c] for c in sorted_cids]
        ax_te.bar(x_indices, y_te, color="#7209B7", label="Test Box", width=0.72)
        tot_te = sum(y_te)
        ax_te.set_title(
            f"3. Tập Kiểm thử (Test Split): Tổng {tot_te:,} box",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax_te.set_xticks(x_indices)
        ax_te.set_xticklabels(x_labels, rotation=90, fontsize=9.5)
        ax_te.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # Cài đặt trục Y theo thang log (symlog) cho cả 3 panel
        y_ticks = [0, 1, 5, 20, 100, 500]
        for ax in axes:
            ax.set_yscale("symlog", linthresh=1, linscale=0.5)
            ax.set_yticks(y_ticks)
            ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_ylim(bottom=0, top=700)
            ax.set_ylabel("Số lượng Box (Log scale)", fontsize=11)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    else:
        # ──────────────────────────────────────────────────────────────────────
        # Chế độ Raw: 1 biểu đồ chuẩn toàn bộ 52 lớp (nguyên vẹn dữ liệu gốc)
        # ──────────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(24, 9.5))
        class_counts = Counter(b["class_id"] for b in boxes)
        sorted_cids = sorted(all_class_ids, key=lambda c: class_counts[c], reverse=True)
        counts = [class_counts[c] for c in sorted_cids]
        labels = [f"[{c:02d}] {classes.get(c, '')[:20]}" for c in sorted_cids]

        norm_counts = np.array(counts, dtype=float)
        max_cnt = max(norm_counts) if max(norm_counts) > 0 else 1.0
        colors = plt.cm.viridis(norm_counts / max_cnt)

        x_coords = np.arange(len(sorted_cids))
        bars = ax.bar(x_coords, counts, color=colors, edgecolor="#2B3A42", linewidth=0.7, alpha=0.9, width=0.75)

        y_max = max(counts) if len(counts) else 0
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        y_ticks_raw = [0, 1, 5, 20, 100, 500, 1000] if y_max >= 500 else [0, 1, 5, 20, 100, 500]
        ax.set_yticks(y_ticks_raw)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(bottom=0, top=y_max * 1.35)

        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    cnt * 1.08,
                    str(cnt),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    rotation=90,
                )

        title_suffix = f" – {tag.upper()}" if tag else ""
        ax.set_title(
            f"01. Phân bố 52 Lớp Biển báo Giao thông (Tổng {total_boxes:,} box){title_suffix}",
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Lớp biển báo ([ID] Tên tiếng Việt)", fontsize=12)
        ax.set_ylabel("Số lượng Bounding Box (Log scale)", fontsize=12)
        ax.set_xticks(x_coords)
        ax.set_xticklabels(labels, rotation=90, fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    class_counts = Counter(b["class_id"] for b in boxes)
    non_zero = [c for c in class_counts.values() if c > 0]
    return {
        "total_boxes": total_boxes,
        "classes_present": len(non_zero),
        "total_classes": len(classes),
        "max_class_count": max(class_counts.values()) if class_counts else 0,
        "min_nonzero_count": min(non_zero) if non_zero else 0,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }
