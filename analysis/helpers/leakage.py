"""Module chuyên trách phân tích và vẽ biểu đồ rò rỉ dữ liệu (Data Leakage) bằng dHash.

Đặc điểm quan trọng:
  - Tự động loại bỏ hoàn toàn viền xám (114) sinh ra khi letterbox-resize
    để không nhận nhầm vùng xám là tương đồng nội dung.
  - Sử dụng dHash thuần túy (size=32 -> 1024-bit) để đánh giá tương đồng bối cảnh cục bộ.
  - Loại bỏ các câu chữ kết luận dài dòng trong biểu đồ theo yêu cầu người dùng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from analysis.helpers.image_utils import calculate_dhash


def check_data_leakage(
    split_stems: dict[str, list[str]],
    stem_to_path: dict[str, Path],
    hash_size: int = 32,
    hamming_thresh: int = 380,
) -> dict[str, Any]:
    """Khảo sát rò rỉ dữ liệu sử dụng dHash thuần túy giữa tất cả các cặp split (train, val, test).

    Ngưỡng dHash mặc định 380 bit để đánh giá rò rỉ dữ liệu giữa tất cả các cặp split.
    """
    splits = [s for s in ("train", "val", "test") if s in split_stems and split_stems[s]]
    n_splits = len(splits)
    if n_splits < 2:
        return {"status": "SKIPPED_NOT_ENOUGH_SPLITS"}

    # 1. Tính mã băm dHash (bỏ viền xám 114) cho từng split
    dhash_cache: dict[str, np.ndarray | None] = {}

    def _get_hash(stem: str) -> np.ndarray | None:
        if stem in dhash_cache:
            return dhash_cache[stem]
        p = stem_to_path.get(stem)
        if not p or not p.exists():
            dhash_cache[stem] = None
            return None
        hd = calculate_dhash(p, hash_size=hash_size, strip_padding=True)
        dhash_cache[stem] = hd
        return hd

    split_valid_stems: dict[str, list[str]] = {}
    split_packed_d: dict[str, np.ndarray] = {}
    max_check_per_split = 2500

    for s in splits:
        stems_to_eval = split_stems[s]
        if len(stems_to_eval) > max_check_per_split:
            real_stems = [st for st in stems_to_eval if not ("_aug" in st)]
            aug_stems = [st for st in stems_to_eval if "_aug" in st]
            remain = max(0, max_check_per_split - len(real_stems))
            if remain > 0 and aug_stems:
                step = max(1, len(aug_stems) // remain)
                sampled_aug = aug_stems[::step][:remain]
                stems_to_eval = real_stems + sampled_aug
            else:
                stems_to_eval = real_stems[:max_check_per_split]

        valid: list[str] = []
        d_list: list[np.ndarray] = []
        for st in stems_to_eval:
            hd = _get_hash(st)
            if hd is not None:
                valid.append(st)
                d_list.append(hd)

        split_valid_stems[s] = valid
        if valid:
            split_packed_d[s] = np.packbits(np.array(d_list, dtype=np.uint8), axis=1)
        else:
            split_packed_d[s] = np.zeros((0, (hash_size * hash_size) // 8), dtype=np.uint8)

    # 2. Xây dựng ma trận khoảng cách Hamming dHash
    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)
    leak_matrix = np.zeros((n_splits, n_splits), dtype=int)
    leak_pair_matrix = np.zeros((n_splits, n_splits), dtype=int)
    all_leak_pairs: list[dict[str, Any]] = []
    leak_summary_by_split: dict[str, Any] = {}

    for i, s1 in enumerate(splits):
        for j, s2 in enumerate(splits):
            if i == j:
                leak_matrix[i, j] = len(split_stems[s1])
                leak_pair_matrix[i, j] = len(split_stems[s1])
                continue

            p1_d = split_packed_d.get(s1)
            p2_d = split_packed_d.get(s2)
            if p1_d is None or p2_d is None or len(p1_d) == 0 or len(p2_d) == 0:
                continue

            xor_d = np.bitwise_xor(p1_d[:, None, :], p2_d[None, :, :])
            dist_d = popcount[xor_d].sum(axis=2)

            leaked_mask = (dist_d <= hamming_thresh)
            leaked_i, leaked_j = np.where(leaked_mask)
            n_pairs = len(leaked_i)
            n_unique_s1 = len(np.unique(leaked_i))
            n_unique_s2 = len(np.unique(leaked_j))

            leak_matrix[i, j] = n_unique_s1
            leak_pair_matrix[i, j] = n_pairs

            if i < j:
                split_pair_key = f"{s1}_vs_{s2}"
                leak_summary_by_split[split_pair_key] = {
                    "leak_pairs_count": int(n_pairs),
                    f"{s1}_images_leaked": int(n_unique_s1),
                    f"{s2}_images_leaked": int(n_unique_s2),
                    f"{s1}_leak_pct": round(float(n_unique_s1 / max(len(split_stems[s1]), 1) * 100), 2),
                    f"{s2}_leak_pct": round(float(n_unique_s2 / max(len(split_stems[s2]), 1) * 100), 2),
                }

                for k in range(n_pairs):
                    idx1 = leaked_i[k]
                    idx2 = leaked_j[k]
                    st1 = split_valid_stems[s1][idx1]
                    st2 = split_valid_stems[s2][idx2]
                    all_leak_pairs.append({
                        "split_a": s1,
                        "image_a": stem_to_path[st1].name if st1 in stem_to_path else f"{st1}.jpg",
                        "split_b": s2,
                        "image_b": stem_to_path[st2].name if st2 in stem_to_path else f"{st2}.jpg",
                        "hamming_dhash": int(dist_d[idx1, idx2]),
                    })

    # 3. Đánh giá riêng chi tiết Test vs Train
    dhash_test_to_train: dict[str, Any] = {}
    min_dists_d: np.ndarray | None = None
    if "train" in split_packed_d and "test" in split_packed_d:
        p_tr_d = split_packed_d["train"]
        p_te_d = split_packed_d["test"]
        if len(p_tr_d) > 0 and len(p_te_d) > 0:
            xor_te_tr_d = np.bitwise_xor(p_te_d[:, None, :], p_tr_d[None, :, :])
            dist_te_tr_d = popcount[xor_te_tr_d].sum(axis=2)
            min_dists_d = dist_te_tr_d.min(axis=1)

            is_near_dup = (dist_te_tr_d <= hamming_thresh)
            n_near_thresh = int(np.count_nonzero(is_near_dup.any(axis=1)))
            pct_near_thresh = float(n_near_thresh / len(min_dists_d) * 100)
            dhash_test_to_train = {
                "min_hamming_dhash": int(min_dists_d.min()),
                "median_hamming_dhash": float(np.median(min_dists_d)),
                "near_duplicates_count": n_near_thresh,
                "near_duplicates_pct": round(pct_near_thresh, 2),
                "dhash_threshold": hamming_thresh,
            }

    total_leaks = len(all_leak_pairs)
    status = "PASSED" if total_leaks == 0 else "LEAK_DETECTED"

    return {
        "status": status,
        "hash_size": hash_size,
        "dhash_threshold": hamming_thresh,
        "splits": splits,
        "split_stems": split_stems,
        "total_leak_pairs": total_leaks,
        "leak_matrix": leak_matrix.tolist(),
        "leak_pair_matrix": leak_pair_matrix.tolist(),
        "leak_summary_by_split": leak_summary_by_split,
        "dhash_test_to_train": dhash_test_to_train,
        "leak_pairs": all_leak_pairs,
        "min_dists": min_dists_d,
    }


def find_leaked_test_stems(
    train_stems: list[str],
    test_stems: list[str],
    stem_to_path: dict[str, Path],
    hash_size: int = 32,
    hamming_thresh: int = 362,
) -> tuple[list[str], list[str]]:
    """Phát hiện và phân tách các stem trong test split thành clean và leaked dựa trên dHash."""
    tr_d = []
    for s in train_stems:
        if s in stem_to_path:
            hd = calculate_dhash(stem_to_path[s], hash_size=hash_size, strip_padding=True)
            if hd is not None:
                tr_d.append(hd)

    te_d, valid_te_stems = [], []
    for s in test_stems:
        if s in stem_to_path:
            hd = calculate_dhash(stem_to_path[s], hash_size=hash_size, strip_padding=True)
            if hd is not None:
                te_d.append(hd)
                valid_te_stems.append(s)

    if not tr_d or not te_d:
        return list(test_stems), []

    tr_packed_d = np.packbits(np.array(tr_d, dtype=np.uint8), axis=1)
    te_packed_d = np.packbits(np.array(te_d, dtype=np.uint8), axis=1)
    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)

    xor_d = np.bitwise_xor(te_packed_d[:, None, :], tr_packed_d[None, :, :])
    dist_d = popcount[xor_d].sum(axis=2)

    is_leaked_matrix = (dist_d <= hamming_thresh)
    is_leaked_test = is_leaked_matrix.any(axis=1)

    clean_stems: list[str] = []
    leaked_stems: list[str] = []
    for stem, leaked in zip(valid_te_stems, is_leaked_test):
        if leaked:
            leaked_stems.append(stem)
        else:
            clean_stems.append(stem)

    return clean_stems, leaked_stems


def plot_data_leakage(
    leak_data: dict[str, Any],
    out_path: Path,
    tag: str = "",
) -> None:
    """Vẽ Figure 07: Ma trận Trùng lặp Overlap và Histogram khoảng cách Hamming dHash."""
    splits = leak_data.get("splits", ["train", "val", "test"])
    n_splits = len(splits)
    leak_pair_matrix = np.array(leak_data.get("leak_pair_matrix", np.zeros((n_splits, n_splits))))
    leak_matrix = np.array(leak_data.get("leak_matrix", np.zeros((n_splits, n_splits))))
    split_stems = leak_data.get("split_stems", {})
    min_dists = leak_data.get("min_dists")
    hamming_thresh = leak_data.get("dhash_threshold", 380)
    dhash_test_to_train = leak_data.get("dhash_test_to_train", {})

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(
        f"07. Đánh giá Rò rỉ Dữ liệu (dHash ≤ {hamming_thresh}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 0: Ma trận trùng lặp
    ax0 = axes[0]
    rgba_grid = np.zeros((n_splits, n_splits, 4), dtype=float)
    for i in range(n_splits):
        for j in range(n_splits):
            if i == j:
                rgba_grid[i, j] = [0.89, 0.95, 0.99, 1.0]      # Light blue (Đường chéo)
            elif leak_pair_matrix[i, j] == 0:
                rgba_grid[i, j] = [0.91, 0.96, 0.91, 1.0]      # Light green (Sạch 100%)
            else:
                rgba_grid[i, j] = [1.0, 0.92, 0.93, 1.0]        # Light red (Phát hiện rò rỉ)

    ax0.imshow(rgba_grid, interpolation="nearest")
    ax0.set_xticks(range(n_splits))
    ax0.set_yticks(range(n_splits))
    ax0.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax0.set_yticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax0.set_title(
        f"1. Ma trận Trùng lặp giữa các Split (dHash ≤ {hamming_thresh})",
        fontsize=12,
        fontweight="bold",
    )

    for i in range(n_splits):
        for j in range(n_splits):
            if i == j:
                txt = f"{len(split_stems.get(splits[i], [])):,}\n(Tổng ảnh)"
                color = "#1D3557"
            else:
                n_pairs = leak_pair_matrix[i, j]
                if n_pairs == 0:
                    txt = "0\n(Sạch)"
                    color = "#2A9D8F"
                else:
                    n_u = leak_matrix[i, j]
                    pct = n_u / max(len(split_stems.get(splits[i], [])), 1) * 100
                    txt = f"Leak: {n_pairs:,} cặp\n({pct:.1f}% ảnh {splits[i].upper()})"
                    color = "#E63946"
            ax0.text(j, i, txt, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)

    # Panel 1: Histogram khoảng cách Test vs Train
    ax1 = axes[1]
    if min_dists is not None and len(min_dists) > 0:
        bin_width = max(1, int(round((max(min_dists) - min(min_dists)) / 30)))
        bins = np.arange(min(min_dists) - 0.5, max(min_dists) + bin_width + 0.5, bin_width)
        counts, edges, patches_list = ax1.hist(min_dists, bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85)

        for edge, p in zip(edges[:-1], patches_list):
            if edge <= hamming_thresh:
                p.set_facecolor("#E63946")  # Đỏ rò rỉ
            elif edge <= hamming_thresh + 30:
                p.set_facecolor("#FB8500")  # Cam gần ngưỡng
            else:
                p.set_facecolor("#2A9D8F")  # Xanh an toàn

        ax1.axvline(hamming_thresh + 0.5, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng dHash (≤ {hamming_thresh})")
        ax1.axvline(np.median(min_dists), color="#1D3557", linestyle="-", linewidth=2,
                    label=f"Trung vị dHash = {np.median(min_dists):.0f}")

        pct_val = dhash_test_to_train.get("near_duplicates_pct", 0)
        ax1.set_xlabel("Khoảng cách Hamming dHash giữa Test và Train (bit)", fontsize=11)
        ax1.set_ylabel("Số lượng ảnh Test", fontsize=11)
        ax1.set_title(
            f"2. Phân bố Độ tương đồng Test vs Train (Rò rỉ dHash: {pct_val:.2f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, "Không đủ dữ liệu Test/Train", ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
