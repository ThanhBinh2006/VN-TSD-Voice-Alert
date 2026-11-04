"""Module gom cụm ảnh bằng mã băm Difference Hash (Figure 05).

Sử dụng:
  - dHash 1024-bit (size=32)
  - Ngưỡng Hamming distance <= 380
  - Tự động loại bỏ viền padding xám 114 khi tính hash
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers.image_utils import calculate_dhash


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def cluster_images_dhash(
    img_paths: list[Path],
    hash_size: int = 32,
    hamming_thresh: int = 380,
) -> dict[str, Any]:
    """Gom cụm ảnh bằng mã băm dHash thuần túy (1024-bit).

    Gom cụm các ảnh cùng chuỗi dashcam/cùng địa điểm nếu khoảng cách Hamming dHash <= hamming_thresh.
    """
    if not img_paths:
        return {}

    d_hashes = []
    valid_stems = []

    for p in img_paths:
        hd = calculate_dhash(p, hash_size=hash_size, strip_padding=True)
        if hd is not None:
            d_hashes.append(hd)
            valid_stems.append(p.stem)

    if not d_hashes:
        return {}

    mat_d = np.array(d_hashes, dtype=np.uint8)
    packed_d = np.packbits(mat_d, axis=1)
    N = len(packed_d)
    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)

    # Gom cụm bằng dHash
    dsu_d = _DisjointSet(N)
    chunk_size = 500
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xor_d = np.bitwise_xor(packed_d[start:end, None, :], packed_d[None, :, :])
        dist_d = popcount[xor_d].sum(axis=2)
        for local_i in range(end - start):
            global_i = start + local_i
            match_j = np.where((dist_d[local_i] <= hamming_thresh) & (np.arange(N) > global_i))[0]
            for j in match_j:
                dsu_d.union(global_i, j)

    raw_clusters: dict[int, list[int]] = {}
    for idx in range(N):
        root = dsu_d.find(idx)
        raw_clusters.setdefault(root, []).append(idx)

    final_raw_groups = [[valid_stems[idx] for idx in group] for group in raw_clusters.values()]
    sorted_groups = sorted(final_raw_groups, key=len, reverse=True)
    clusters: dict[str, list[str]] = {}
    stem_to_cluster: dict[str, int] = {}
    cluster_sizes = []

    for new_id, group in enumerate(sorted_groups, start=1):
        cname = f"cluster_{new_id:04d}"
        clusters[cname] = group
        cluster_sizes.append(len(group))
        for st in group:
            stem_to_cluster[st] = new_id

    n_singletons = sum(1 for sz in cluster_sizes if sz == 1)
    n_multi = len(cluster_sizes) - n_singletons

    print(
        f"[cluster] dHash 1024-bit (Hamming <= {hamming_thresh}): "
        f"Gom {N:,} ảnh thành {len(clusters):,} cụm "
        f"({n_multi:,} cụm đa ảnh, {n_singletons:,} ảnh đơn lẻ)."
    )

    return {
        "hash_size": hash_size,
        "hamming_threshold": hamming_thresh,
        "total_images": N,
        "total_clusters": len(clusters),
        "singletons": n_singletons,
        "multi_image_clusters": n_multi,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "cluster_sizes": cluster_sizes,
        "clusters": clusters,
        "stem_to_cluster": stem_to_cluster,
    }


def plot_dhash_clusters(
    cluster_res: dict[str, Any],
    out_path: Path,
    tag: str = "",
) -> None:
    """Vẽ Figure 05: Phân tích phân bố kích thước cụm dHash."""
    cluster_sizes = cluster_res.get("cluster_sizes", [])
    if not cluster_sizes:
        return

    hash_size = cluster_res.get("hash_size", 32)
    hamming_thresh = cluster_res.get("hamming_threshold", 380)
    N = cluster_res.get("total_images", 0)

    sub_h = f"dHash {hash_size*hash_size}b (≤{hamming_thresh})"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(
        f"05. Phân bố Gom cụm Bối cảnh bằng {sub_h}{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 0: Biểu đồ cột phân nhóm quy mô cụm
    ax0 = axes[0]
    bins_def = [
        ("1 ảnh (đơn lẻ)", lambda s: s == 1, "#2A9D8F"),
        ("2-5 ảnh", lambda s: 2 <= s <= 5, "#4EA8DE"),
        ("6-20 ảnh", lambda s: 6 <= s <= 20, "#FB8500"),
        ("21-50 ảnh", lambda s: 21 <= s <= 50, "#E76F51"),
        ("> 50 ảnh (chuỗi video dài)", lambda s: s > 50, "#E63946"),
    ]
    labels_b = [b[0] for b in bins_def]
    n_clusters_in_bin = [sum(1 for sz in cluster_sizes if b[1](sz)) for b in bins_def]
    n_imgs_in_bin = [sum(sz for sz in cluster_sizes if b[1](sz)) for b in bins_def]
    colors_b = [b[2] for b in bins_def]

    x_b = np.arange(len(labels_b))
    bars0 = ax0.bar(x_b, n_clusters_in_bin, color=colors_b, width=0.6, edgecolor="white")
    ax0.set_xticks(x_b)
    ax0.set_xticklabels(labels_b, rotation=20, ha="right", fontsize=9.5)
    ax0.set_ylabel("Số lượng cụm", fontsize=11)
    ax0.set_title(f"1. Phân bố Quy mô Cụm (Tổng: {len(cluster_sizes):,} cụm từ {N:,} ảnh)", fontsize=11.5, fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, n_c, n_im in zip(bars0, n_clusters_in_bin, n_imgs_in_bin):
        if n_c > 0:
            pct_im = n_im / max(N, 1) * 100
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(n_clusters_in_bin) * 0.02,
                f"{n_c:,} cụm\n({n_im:,} ảnh | {pct_im:.1f}%)",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    # Panel 1: Top 15 cụm lớn nhất
    ax1 = axes[1]
    top_k = min(15, len(cluster_sizes))
    top_sizes = cluster_sizes[:top_k]
    top_labels = [f"Cụm {i+1}" for i in range(top_k)]

    bars1 = ax1.bar(range(top_k), top_sizes, color="#E63946", width=0.65, edgecolor="white")
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=9.5)
    ax1.set_ylabel("Số lượng ảnh trong cụm", fontsize=11)
    ax1.set_title(f"2. Top {top_k} Cụm Trùng lặp Lớn nhất (Chuỗi Video liên tục)", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, sz in zip(bars1, top_sizes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(top_sizes) * 0.015,
            f"{sz:,}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
