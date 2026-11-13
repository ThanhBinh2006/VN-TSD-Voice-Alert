"""Phân tích và vẽ biểu đồ rò rỉ dữ liệu bằng dHash.

Loại bỏ viền xám letterbox khi tính dHash và kiểm tra toàn bộ ảnh theo
từng khối để giới hạn bộ nhớ dùng cho phép tính khoảng cách Hamming.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers.image_utils import calculate_dhash


def _iter_hamming_chunks(
    packed_a: np.ndarray,
    packed_b: np.ndarray,
    chunk_size: int = 256,
) -> Iterator[tuple[int, np.ndarray]]:
    """Sinh khoảng cách Hamming của từng khối hàng so với toàn bộ tập B.

    Chia cả tập B thành các khối để tensor XOR và bảng tra popcount
    không tăng theo tổng số ảnh. Ma trận khoảng cách trả về có tối đa
    ``chunk_size`` hàng; tổng khoảng cách dùng uint64 để tránh tràn số.
    """
    popcount = np.array([k.bit_count() for k in range(256)], dtype=np.uint8)

    for start_a in range(0, len(packed_a), chunk_size):
        block_a = packed_a[start_a:start_a + chunk_size]
        distances = np.empty((len(block_a), len(packed_b)), dtype=np.uint64)

        for start_b in range(0, len(packed_b), chunk_size):
            block_b = packed_b[start_b:start_b + chunk_size]
            xor = np.bitwise_xor(block_a[:, None, :], block_b[None, :, :])
            distances[:, start_b:start_b + len(block_b)] = popcount[xor].sum(
                axis=2,
                dtype=np.uint64,
            )
            del xor

        yield start_a, distances
        del distances


def check_data_leakage(
    split_stems: dict[str, list[str]],
    stem_to_path: dict[str, Path],
    hash_size: int = 32,
    hamming_thresh: int = 50,
) -> dict[str, Any]:
    """Kiểm tra toàn bộ ảnh giữa các split bằng dHash theo từng khối.

    Ảnh không tồn tại hoặc không tính được dHash được bỏ qua khi so sánh.
    Ngưỡng mặc định là 50 bit; không giới hạn số lượng stem được xét.
    """
    splits = [s for s in ("train", "val", "test") if s in split_stems and split_stems[s]]
    n_splits = len(splits)
    if n_splits < 2:
        return {"status": "SKIPPED_NOT_ENOUGH_SPLITS"}

    chunk_size = 256
    dhash_cache: dict[str, np.ndarray | None] = {}

    def _get_hash(stem: str) -> np.ndarray | None:
        """Lấy dHash từ bộ nhớ đệm hoặc tính từ ảnh đã loại viền xám."""
        if stem in dhash_cache:
            return dhash_cache[stem]
        path = stem_to_path.get(stem)
        if path is None or not path.exists():
            dhash_cache[stem] = None
            return None
        hashed = calculate_dhash(path, hash_size=hash_size, strip_padding=True)
        dhash_cache[stem] = hashed
        return hashed

    split_valid_stems: dict[str, list[str]] = {}
    split_packed_d: dict[str, np.ndarray] = {}

    for split in splits:
        valid: list[str] = []
        hashes: list[np.ndarray] = []
        for stem in split_stems[split]:
            hashed = _get_hash(stem)
            if hashed is not None:
                valid.append(stem)
                hashes.append(hashed)

        split_valid_stems[split] = valid
        if valid:
            split_packed_d[split] = np.packbits(
                np.array(hashes, dtype=np.uint8),
                axis=1,
            )
        else:
            split_packed_d[split] = np.zeros(
                (0, (hash_size * hash_size + 7) // 8),
                dtype=np.uint8,
            )

    dhash_cache.clear()
    del hashes

    leak_matrix = np.zeros((n_splits, n_splits), dtype=int)
    leak_pair_matrix = np.zeros((n_splits, n_splits), dtype=int)
    all_leak_pairs: list[dict[str, Any]] = []
    leak_summary_by_split: dict[str, Any] = {}

    for i, s1 in enumerate(splits):
        leak_matrix[i, i] = len(split_stems[s1])
        leak_pair_matrix[i, i] = len(split_stems[s1])

        for j in range(i + 1, n_splits):
            s2 = splits[j]
            p1_d = split_packed_d[s1]
            p2_d = split_packed_d[s2]
            if len(p1_d) == 0 or len(p2_d) == 0:
                continue

            unique_i = np.zeros(len(p1_d), dtype=bool)
            unique_j = np.zeros(len(p2_d), dtype=bool)
            n_pairs = 0

            for start, dist_d in _iter_hamming_chunks(p1_d, p2_d, chunk_size):
                leaked_mask = dist_d <= hamming_thresh
                unique_i[start:start + len(dist_d)] = leaked_mask.any(axis=1)
                unique_j |= leaked_mask.any(axis=0)

                # Duyệt từng hàng để tránh giữ mảng chỉ số cho cả khối
                # khi phần lớn các cặp đều nằm dưới ngưỡng.
                for local_i, row_mask in enumerate(leaked_mask):
                    matched_j = np.flatnonzero(row_mask)
                    n_pairs += int(matched_j.size)
                    if matched_j.size == 0:
                        continue

                    st1 = split_valid_stems[s1][start + local_i]
                    image_a = (
                        stem_to_path[st1].name
                        if st1 in stem_to_path
                        else f"{st1}.jpg"
                    )
                    for idx2 in matched_j:
                        st2 = split_valid_stems[s2][idx2]
                        all_leak_pairs.append({
                            "split_a": s1,
                            "image_a": image_a,
                            "split_b": s2,
                            "image_b": (
                                stem_to_path[st2].name
                                if st2 in stem_to_path
                                else f"{st2}.jpg"
                            ),
                            "hamming_dhash": int(dist_d[local_i, idx2]),
                        })

                del dist_d, leaked_mask

            n_unique_s1 = int(np.count_nonzero(unique_i))
            n_unique_s2 = int(np.count_nonzero(unique_j))

            leak_matrix[i, j] = n_unique_s1
            leak_matrix[j, i] = n_unique_s2
            leak_pair_matrix[i, j] = n_pairs
            leak_pair_matrix[j, i] = n_pairs

            split_pair_key = f"{s1}_vs_{s2}"
            leak_summary_by_split[split_pair_key] = {
                "leak_pairs_count": int(n_pairs),
                f"{s1}_images_leaked": n_unique_s1,
                f"{s2}_images_leaked": n_unique_s2,
                f"{s1}_leak_pct": round(
                    float(n_unique_s1 / max(len(split_stems[s1]), 1) * 100),
                    2,
                ),
                f"{s2}_leak_pct": round(
                    float(n_unique_s2 / max(len(split_stems[s2]), 1) * 100),
                    2,
                ),
            }

    dhash_test_to_train: dict[str, Any] = {}
    min_dists_d: np.ndarray | None = None
    if "train" in split_packed_d and "test" in split_packed_d:
        p_tr_d = split_packed_d["train"]
        p_te_d = split_packed_d["test"]
        if len(p_tr_d) > 0 and len(p_te_d) > 0:
            min_dists_d = np.empty(len(p_te_d), dtype=np.uint64)
            for start, dist_te_tr_d in _iter_hamming_chunks(
                p_te_d,
                p_tr_d,
                chunk_size,
            ):
                min_dists_d[start:start + len(dist_te_tr_d)] = dist_te_tr_d.min(
                    axis=1,
                )
                del dist_te_tr_d

            n_near_thresh = int(np.count_nonzero(min_dists_d <= hamming_thresh))
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
    hamming_thresh: int = 50,
) -> tuple[list[str], list[str]]:
    """Phân tách stem Test thành clean và leaked bằng dHash theo từng khối."""
    tr_d: list[np.ndarray] = []
    for stem in train_stems:
        if stem in stem_to_path:
            hashed = calculate_dhash(
                stem_to_path[stem],
                hash_size=hash_size,
                strip_padding=True,
            )
            if hashed is not None:
                tr_d.append(hashed)

    te_d: list[np.ndarray] = []
    valid_te_stems: list[str] = []
    for stem in test_stems:
        if stem in stem_to_path:
            hashed = calculate_dhash(
                stem_to_path[stem],
                hash_size=hash_size,
                strip_padding=True,
            )
            if hashed is not None:
                te_d.append(hashed)
                valid_te_stems.append(stem)

    if not tr_d or not te_d:
        return list(test_stems), []

    tr_packed_d = np.packbits(np.array(tr_d, dtype=np.uint8), axis=1)
    te_packed_d = np.packbits(np.array(te_d, dtype=np.uint8), axis=1)
    del tr_d, te_d

    chunk_size = 256
    clean_stems: list[str] = []
    leaked_stems: list[str] = []

    for start, dist_d in _iter_hamming_chunks(
        te_packed_d,
        tr_packed_d,
        chunk_size,
    ):
        is_leaked_test = (dist_d <= hamming_thresh).any(axis=1)
        for stem, leaked in zip(
            valid_te_stems[start:start + len(dist_d)],
            is_leaked_test,
        ):
            if leaked:
                leaked_stems.append(stem)
            else:
                clean_stems.append(stem)
        del dist_d

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
    hamming_thresh = leak_data.get("dhash_threshold", 50)
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
