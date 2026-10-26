"""Phân chia dataset theo cụm dHash, bảo toàn 52 lớp cho tập train."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections import defaultdict
import numpy as np

from PIL import Image

from analysis.helpers.image_utils import calculate_brightness_sharpness
from analysis.helpers.clustering import cluster_images_dhash


def _read_labels_for_stems(
    lbl_dir: Path,
    stems: list[str],
    stem_to_path: dict[str, Path] | None = None,
    min_box_px: int = 8,
    image_size: tuple[int, int] = (640, 640),
) -> dict[str, list[int]]:
    """Đọc class_id cho từng file nhãn, bỏ qua box < min_box_px."""
    stem_to_classes: dict[str, list[int]] = defaultdict(list)
    tw, th = image_size

    for s in stems:
        p = lbl_dir / f"{s}.txt"
        if not p.exists():
            continue

        orig_w, orig_h = 960, 540
        if stem_to_path and s in stem_to_path:
            img_p = stem_to_path[s]
            if img_p.exists():
                try:
                    with Image.open(img_p) as im:
                        orig_w, orig_h = im.size
                except Exception:
                    pass

        scale = min(tw / orig_w, th / orig_h)

        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        bw, bh = float(parts[3]), float(parts[4])
                        px_w = bw * orig_w * scale
                        px_h = bh * orig_h * scale
                        if px_w >= min_box_px and px_h >= min_box_px:
                            stem_to_classes[s].append(cid)
        except Exception:
            continue
    return stem_to_classes


def split_clusters_for_classes(
    clusters: dict[str, list[str]],
    stem_to_classes: dict[str, list[int]],
    img_paths: list[Path],
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Phân bổ cụm vào Train/Val/Test (Train đủ 52 lớp, cô lập cụm 100%)."""
    total_images = len(img_paths)
    all_stems = [p.stem for p in img_paths]
    stem_to_path = {p.stem: p for p in img_paths}

    # Đo độ sáng và độ nét để cân bằng
    stem_quality: dict[str, dict[str, bool]] = {}
    for s in all_stems:
        p = stem_to_path.get(s)
        if not p:
            continue
        res = calculate_brightness_sharpness(p)
        if res is not None:
            b, s_val = res
            stem_quality[s] = {"dark": b < 85, "blur": s_val < 100}
        else:
            stem_quality[s] = {"dark": False, "blur": False}

    cluster_to_classes: dict[str, list[int]] = defaultdict(list)
    cluster_quality: dict[str, dict[str, int]] = defaultdict(lambda: {"dark": 0, "blur": 0, "total": 0})

    for cname, c_stems in clusters.items():
        for s in c_stems:
            cluster_to_classes[cname].extend(stem_to_classes.get(s, []))
            q = stem_quality.get(s, {})
            if q.get("dark"):
                cluster_quality[cname]["dark"] += 1
            if q.get("blur"):
                cluster_quality[cname]["blur"] += 1
            cluster_quality[cname]["total"] += 1

    tot_dark = sum(cluster_quality[c]["dark"] for c in clusters)
    tot_blur = sum(cluster_quality[c]["blur"] for c in clusters)

    target_counts = {
        "train": int(total_images * ratio[0]),
        "val": int(total_images * ratio[1]),
        "test": total_images - int(total_images * ratio[0]) - int(total_images * ratio[1]),
    }

    # Xác định số lớp duy nhất có trong dữ liệu
    all_dataset_classes = set.union(*[set(cls) for cls in stem_to_classes.values()]) if stem_to_classes else set()
    required_classes = len(all_dataset_classes)

    # Cụm bắt buộc vào train: chứa lớp hiếm hoặc kích thước lớn
    class_to_clusters: dict[int, set[str]] = defaultdict(set)
    for cname, cls_list in cluster_to_classes.items():
        for cid in cls_list:
            class_to_clusters[cid].add(cname)

    mandatory_train: set[str] = set()
    for cid, cset in class_to_clusters.items():
        if len(cset) == 1:
            mandatory_train.update(cset)
        else:
            tot_b = sum(len(clusters[c]) for c in cset)
            if tot_b <= 30:
                largest_c = max(cset, key=lambda c: len(clusters[c]))
                mandatory_train.add(largest_c)

    for cname, c_stems in clusters.items():
        if len(c_stems) >= 300:
            mandatory_train.add(cname)

    print(f"[split] Cụm bắt buộc vào Train (bảo toàn 52 lớp): {len(mandatory_train)} cụm ({sum(len(clusters[c]) for c in mandatory_train):,} ảnh).")

    best_score = float("inf")
    best_splits: dict[str, list[str]] | None = None
    best_info: Any = None

    mandatory_stems: list[str] = []
    mandatory_sz = 0
    mandatory_classes: set[int] = set()
    mandatory_dark = 0
    mandatory_blur = 0
    for cname in mandatory_train:
        c_stems = clusters[cname]
        mandatory_stems.extend(c_stems)
        mandatory_sz += len(c_stems)
        mandatory_classes.update(cluster_to_classes[cname])
        mandatory_dark += cluster_quality[cname]["dark"]
        mandatory_blur += cluster_quality[cname]["blur"]

    remaining_clusters = [c for c in clusters if c not in mandatory_train]
    cluster_cls_set = {c: set(cluster_to_classes[c]) for c in remaining_clusters}
    cluster_sz = {c: len(clusters[c]) for c in remaining_clusters}
    cluster_dk = {c: cluster_quality[c]["dark"] for c in remaining_clusters}
    cluster_bl = {c: cluster_quality[c]["blur"] for c in remaining_clusters}

    for trial in range(300):
        rng = random.Random(seed + trial)
        splits: dict[str, list[str]] = {"train": list(mandatory_stems), "val": [], "test": []}
        split_sizes = {"train": mandatory_sz, "val": 0, "test": 0}
        split_classes: dict[str, set[int]] = {"train": set(mandatory_classes), "val": set(), "test": set()}
        split_dark = {"train": mandatory_dark, "val": 0, "test": 0}
        split_blur = {"train": mandatory_blur, "val": 0, "test": 0}

        remaining = list(remaining_clusters)
        rng.shuffle(remaining)

        for cname in remaining:
            c_stems = clusters[cname]
            c_s = cluster_sz[cname]
            c_cls = cluster_cls_set[cname]
            c_d = cluster_dk[cname]
            c_b = cluster_bl[cname]

            missing_train = c_cls - split_classes["train"]
            if missing_train and split_sizes["train"] < target_counts["train"] + 30:
                chosen = "train"
            else:
                scores = {}
                for s in ("val", "test", "train"):
                    fill = (split_sizes[s] + c_s) / target_counts[s]
                    new_c = len(c_cls - split_classes[s])
                    dark_fill = (split_dark[s] + c_d) / max(1, tot_dark * (ratio[0] if s == "train" else ratio[1]))
                    blur_fill = (split_blur[s] + c_b) / max(1, tot_blur * (ratio[0] if s == "train" else ratio[1]))
                    scores[s] = fill + 0.35 * dark_fill + 0.35 * blur_fill - 0.2 * new_c + rng.uniform(-0.02, 0.02)
                chosen = min(scores.keys(), key=lambda s: scores[s])

            splits[chosen].extend(c_stems)
            split_sizes[chosen] += c_s
            split_classes[chosen].update(c_cls)
            split_dark[chosen] += c_d
            split_blur[chosen] += c_b

        if len(split_classes["train"]) < required_classes:
            continue

        err_size = sum(abs(split_sizes[s] - target_counts[s]) / target_counts[s] for s in splits)
        err_dark = sum(abs(split_dark[s] / max(1, tot_dark) - (ratio[0] if s == "train" else ratio[1])) for s in splits)
        err_blur = sum(abs(split_blur[s] / max(1, tot_blur) - (ratio[0] if s == "train" else ratio[1])) for s in splits)
        cov = len(split_classes["val"]) + len(split_classes["test"])

        score = err_size * 15 + err_dark * 20 + err_blur * 20 - cov
        if score < best_score:
            best_score = score
            best_splits = {s: sorted(splits[s]) for s in splits}
            best_info = (split_sizes, split_classes, split_dark, split_blur)

    if best_splits is None:
        raise RuntimeError("Không tìm thấy cấu hình split thỏa mãn điều kiện {required_classes} lớp trong train.")

    sizes, classes, s_dark, s_blur = best_info
    print(f"[split] Hoàn thành phân bổ split theo cụm (Cô lập cụm 100%):")
    print(f"[split]   Train: {sizes['train']:,} ảnh ({sizes['train']/total_images*100:.1f}%) | {len(classes['train'])}/52 lớp | {s_dark['train']} ảnh tối | {s_blur['train']} ảnh mờ")
    print(f"[split]   Val  : {sizes['val']:,} ảnh ({sizes['val']/total_images*100:.1f}%) | {len(classes['val'])}/52 lớp | {s_dark['val']} ảnh tối | {s_blur['val']} ảnh mờ")
    print(f"[split]   Test : {sizes['test']:,} ảnh ({sizes['test']/total_images*100:.1f}%) | {len(classes['test'])}/52 lớp | {s_dark['test']} ảnh tối | {s_blur['test']} ảnh mờ")

    return best_splits


def split_by_dhash_clusters(
    img_paths: list[Path],
    lbl_dir: Path,
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    hash_size: int = 32,
    hamming_thresh: int = 380,
    min_box_px: int = 8,
    image_size: tuple[int, int] = (640, 640),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Gom cụm dHash thuần túy và phân chia split."""
    if not img_paths:
        return {"train": [], "val": [], "test": []}

    cluster_res = cluster_images_dhash(
        img_paths,
        hash_size=hash_size,
        hamming_thresh=hamming_thresh,
    )
    clusters = cluster_res.get("clusters", {})
    all_stems = [p.stem for p in img_paths]
    stem_to_path = {p.stem: p for p in img_paths}

    stem_to_classes = _read_labels_for_stems(
        lbl_dir=lbl_dir,
        stems=all_stems,
        stem_to_path=stem_to_path,
        min_box_px=min_box_px,
        image_size=image_size,
    )

    return split_clusters_for_classes(
        clusters=clusters,
        stem_to_classes=stem_to_classes,
        img_paths=img_paths,
        ratio=ratio,
        seed=seed,
    )
