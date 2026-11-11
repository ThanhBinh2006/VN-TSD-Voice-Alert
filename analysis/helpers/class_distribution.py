"""Vẽ phân bố đầy đủ các lớp biển báo cho dữ liệu RAW và PROCESSED."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter


def _configure_axis(ax: Axes, max_count: int, padding: float) -> None:
    """Thiết lập trục số lượng symlog và mở rộng tick theo dữ liệu."""
    upper_limit = max(1, max_count) * padding
    ticks = [0, 1, 5, 20, 100, 500]

    magnitude = 1_000
    while magnitude <= upper_limit:
        ticks.append(magnitude)
        if magnitude * 5 <= upper_limit:
            ticks.append(magnitude * 5)
        magnitude *= 10

    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xticks([tick for tick in ticks if tick <= upper_limit])

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(0, upper_limit)
    ax.set_xlabel("Số lượng Bounding Box (thang symlog)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)


def _annotate_count(
    ax: Axes,
    row: int,
    count: int,
    label: str,
    fontsize: float = 9.0,
) -> None:
    """Ghi số lượng bên phải thanh, kể cả thanh có giá trị bằng không."""
    ax.annotate(
        label,
        xy=(count, row),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=fontsize,
        color="#999999" if count == 0 else "#222222",
        annotation_clip=False,
        clip_on=False,
    )


def plot_class_distribution(
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_path: Path,
    tag: str = "",
) -> dict[str, Any]:
    """Vẽ biểu đồ ngang cho mọi lớp và trả về thống kê bounding box.

    RAW sử dụng một biểu đồ. PROCESSED sử dụng ba panel chung trục Y
    cho Train, Validation và Test; Train phân biệt box thật và tăng cường.
    Các lớp được sắp xếp giảm dần theo tổng số box, cùng số lượng thì
    sắp xếp tăng dần theo ID. Lớp không có box vẫn được hiển thị.
    """
    all_class_ids = sorted(classes)
    total_boxes = len(boxes)
    class_counts: Counter[int] = Counter(box["class_id"] for box in boxes)

    splits_present = {
        box.get("split") for box in boxes if box.get("split")
    }
    is_processed_split = (
        tag != "raw"
        and bool(splits_present & {"train", "val", "test"})
    )

    sorted_cids = sorted(
        all_class_ids,
        key=lambda cid: (-class_counts[cid], cid),
    )
    positions = list(range(len(sorted_cids)))
    labels = [f"[{cid:02d}] {classes[cid]}" for cid in sorted_cids]
    title_suffix = f" – {tag.upper()}" if tag else ""
    class_total = len(classes)

    if is_processed_split:
        train_real_counts: Counter[int] = Counter()
        train_aug_counts: Counter[int] = Counter()
        val_counts: Counter[int] = Counter()
        test_counts: Counter[int] = Counter()

        for box in boxes:
            cid = box["class_id"]
            split = box.get("split", "train")
            is_augmented = bool(
                box.get("is_augmented", False)
                or "_aug" in (box.get("stem") or "")
            )

            if split == "train":
                if is_augmented:
                    train_aug_counts[cid] += 1
                else:
                    train_real_counts[cid] += 1
            elif split == "val":
                val_counts[cid] += 1
            elif split == "test":
                test_counts[cid] += 1

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(26, max(19.5, class_total * 0.375)),
            sharey=True,
        )
        fig.suptitle(
            f"01. Phân bố {class_total} lớp biển báo theo tập dữ liệu"
            f"{title_suffix}",
            fontsize=15,
            fontweight="bold",
        )

        ax_train, ax_val, ax_test = axes
        real_values = [train_real_counts[cid] for cid in sorted_cids]
        aug_values = [train_aug_counts[cid] for cid in sorted_cids]
        train_values = [
            real + augmented
            for real, augmented in zip(real_values, aug_values)
        ]
        val_values = [val_counts[cid] for cid in sorted_cids]
        test_values = [test_counts[cid] for cid in sorted_cids]

        ax_train.barh(
            positions,
            real_values,
            height=0.72,
            color="#1D3557",
            label="Ảnh Thật (Real)",
        )
        ax_train.barh(
            positions,
            aug_values,
            left=real_values,
            height=0.72,
            color="#E76F51",
            label="Ảnh Tự tạo (Augmented)",
        )
        ax_train.set_title(
            f"Train: {sum(train_values):,} box\n"
            f"Real: {sum(real_values):,} | Augmented: {sum(aug_values):,}",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax_train.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.045),
            fontsize=10,
            framealpha=0.9,
            borderaxespad=0,
        )

        ax_val.barh(
            positions,
            val_values,
            height=0.72,
            color="#2A9D8F",
        )
        ax_val.set_title(
            f"Validation: {sum(val_values):,} box",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        ax_test.barh(
            positions,
            test_values,
            height=0.72,
            color="#7209B7",
        )
        ax_test.set_title(
            f"Test: {sum(test_values):,} box",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        ax_train.set_yticks(positions)
        ax_train.set_yticklabels(labels, fontsize=9.5)
        ax_train.set_ylim(-0.75, max(class_total - 0.25, 0.75))
        ax_train.invert_yaxis()
        ax_train.set_ylabel("Lớp biển báo", fontsize=11)

        for ax in axes:
            ax.tick_params(axis="y", labelsize=9.5, length=3)

        ax_val.tick_params(axis="y", labelleft=False)
        ax_test.tick_params(axis="y", labelleft=False)

        _configure_axis(ax_train, max(train_values, default=0), padding=2.1)
        _configure_axis(ax_val, max(val_values, default=0), padding=1.5)
        _configure_axis(ax_test, max(test_values, default=0), padding=1.5)

        for row, (real, augmented) in enumerate(zip(real_values, aug_values)):
            total = real + augmented
            label = (
                f"{total:,} ({real:,}+{augmented:,})"
                if augmented > 0
                else f"{real:,}"
            )
            _annotate_count(ax_train, row, total, label)

        for ax, values in ((ax_val, val_values), (ax_test, test_values)):
            for row, count in enumerate(values):
                _annotate_count(ax, row, count, f"{count:,}")

        fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.0)

    else:
        counts = [class_counts[cid] for cid in sorted_cids]
        max_count = max(counts, default=0)
        normalizer = max(max_count, 1)
        colormap = plt.get_cmap("viridis")
        colors = [colormap(count / normalizer) for count in counts]

        fig, ax = plt.subplots(
            figsize=(16, max(18.5, class_total * 0.35)),
        )
        ax.barh(
            positions,
            counts,
            height=0.75,
            color=colors,
            edgecolor="#2B3A42",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_ylim(-0.75, max(class_total - 0.25, 0.75))
        ax.invert_yaxis()
        ax.set_ylabel("Lớp biển báo", fontsize=12)
        ax.set_title(
            f"01. Phân bố {class_total} lớp biển báo giao thông"
            f" (Tổng {total_boxes:,} box){title_suffix}",
            fontsize=15,
            fontweight="bold",
            pad=12,
        )

        _configure_axis(ax, max_count, padding=1.5)

        for row, count in enumerate(counts):
            _annotate_count(ax, row, count, f"{count:,}", fontsize=9.5)

        fig.tight_layout()

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)

    print(f"  Saved: {out_path}")

    non_zero = [count for count in class_counts.values() if count > 0]
    return {
        "total_boxes": total_boxes,
        "classes_present": len(non_zero),
        "total_classes": class_total,
        "max_class_count": max(class_counts.values(), default=0),
        "min_nonzero_count": min(non_zero, default=0),
        "class_counts": {
            int(cid): int(count) for cid, count in class_counts.items()
        },
    }
