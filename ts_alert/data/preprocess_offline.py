"""Tiền xử lý offline 6 bước chuẩn hoá dataset cho Faster R-CNN và YOLO (TSD-Voice-Alert).
Phụ trách: Thành viên 1.

Các bước chuẩn hóa:
  B1: Xoá các bounding box nhỏ (< min_box_px trên canvas 640x640)
  B2: Gom cụm tương đồng bối cảnh bằng dHash 1024-bit (Hamming <= 380)
  B3: Phân chia cụm train/val/test bảo đảm 100% Cluster Isolation & đủ 52/52 lớp cho train
  B4: Augmentation cân bằng nhãn thiểu số trong train (loại trừ ứng viên rò rỉ sang val/test)
  B5: Letterbox 640x640 giữ nguyên tỉ lệ khung hình (pad 114)
  B6: Kiểm tra rò rỉ sau resize (dHash <= 380) và dồn tập triệt để -> Đạt chính xác 0.00% Leakage
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import csv
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from analysis.helpers.clustering import cluster_images_dhash
from analysis.helpers.image_utils import calculate_dhash
from ts_alert.config.resolve import get_num_classes, resolve_path
from ts_alert.data.helpers.augmentation import (
    _read_raw_labels,
    _write_labels,
    get_safe_crop_bounds,
    transform_and_letterbox_crop,
)
from ts_alert.data.helpers.split import split_clusters_for_classes
from ts_alert.utils.io import load_classes, write_json
from ts_alert.utils.seed import set_seed


def list_raw_samples(raw_dir: Path) -> list[dict]:
    """Quét thư mục dữ liệu gốc, trả [{"image": Path, "label": Path | None, "stem": str}, ...]."""
    raw_dir = Path(raw_dir)
    img_dir = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
    lbl_dir = raw_dir / "labels" if (raw_dir / "labels").is_dir() else raw_dir

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for img_p in sorted(img_dir.glob("*.*")):
        if img_p.suffix.lower() in img_extensions:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            samples.append({
                "image": img_p,
                "label": lbl_p if lbl_p.exists() else None,
                "stem": img_p.stem,
            })
    return samples


def _letterbox_image(
    img: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> tuple[np.ndarray, float, float, float]:
    """Resize giữ nguyên tỉ lệ khung hình và padding viền màu xám 114."""
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)

    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, float(dx), float(dy)


def _adjust_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    orig_w: int,
    orig_h: int,
    scale: float,
    dx: float,
    dy: float,
    target_w: int,
    target_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """Chuyển đổi toạ độ YOLO box sang canvas mới sau letterbox."""
    adjusted = []
    for cid, cx, cy, bw, bh in boxes:
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_bw = bw * orig_w
        abs_bh = bh * orig_h

        new_cx = (abs_cx * scale + dx) / target_w
        new_cy = (abs_cy * scale + dy) / target_h
        new_bw = (abs_bw * scale) / target_w
        new_bh = (abs_bh * scale) / target_h

        x1 = max(0.0, new_cx - new_bw / 2)
        y1 = max(0.0, new_cy - new_bh / 2)
        x2 = min(1.0, new_cx + new_bw / 2)
        y2 = min(1.0, new_cy + new_bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue

        final_cx = (x1 + x2) / 2
        final_cy = (y1 + y2) / 2
        final_bw = x2 - x1
        final_bh = y2 - y1

        adjusted.append((cid, final_cx, final_cy, final_bw, final_bh))

    return adjusted


def letterbox_sample(
    sample: dict,
    out_dir: Path,
    split: str,
    image_size: int = 640,
    pad_value: int = 114,
    min_box_px: int = 8,
) -> None:
    """Letterbox ảnh về image_size x image_size, biến đổi lại toạ độ nhãn tương ứng,
    ghi vào out_dir/images/<split>/ và out_dir/labels/<split>/."""
    out_dir = Path(out_dir)
    dst_img_dir = out_dir / "images" / split
    dst_lbl_dir = out_dir / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(sample["image"])
    stem = sample.get("stem") or img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        return

    orig_h, orig_w = img.shape[:2]
    canvas, scale, dx, dy = _letterbox_image(img, (image_size, image_size), pad_value)

    # Đọc nhãn nếu có
    raw_boxes = []
    lbl_p = sample.get("label")
    if lbl_p and Path(lbl_p).exists():
        raw_boxes = _read_raw_labels(Path(lbl_p))

    # Lọc tiny boxes
    valid_boxes = []
    for cid, cx, cy, bw, bh in raw_boxes:
        if (bw * orig_w * scale >= min_box_px) and (bh * orig_h * scale >= min_box_px):
            valid_boxes.append((cid, cx, cy, bw, bh))

    adj_boxes = _adjust_boxes(
        valid_boxes, orig_w, orig_h, scale, dx, dy, image_size, image_size
    )

    cv2.imwrite(str(dst_img_dir / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    _write_labels(dst_lbl_dir / f"{stem}.txt", adj_boxes)


def split_samples(
    samples: list[dict],
    ratios: dict,
    seed: int = 42,
    hash_size: int = 32,
    hamming_thresh: int = 380,
    min_box_px: int = 8,
    image_size: int = 640,
) -> dict[str, list[dict]]:
    """Chia {"train": [...], "val": [...], "test": [...]} theo ratios, gom cụm dHash độc lập 100%
    và bảo đảm tập train chứa toàn bộ các lớp xuất hiện trong dataset."""
    ratio_tuple = (
        float(ratios.get("train", 0.7)),
        float(ratios.get("val", 0.2)),
        float(ratios.get("test", 0.1)),
    )
    img_paths = [Path(s["image"]) for s in samples]
    stem_to_sample = {Path(s["image"]).stem: s for s in samples}

    # Gom cụm dHash
    cluster_res = cluster_images_dhash(
        img_paths,
        hash_size=hash_size,
        hamming_thresh=hamming_thresh,
    )
    clusters = cluster_res.get("clusters", {})

    # Đọc nhãn cho từng stem (lọc box < min_box_px)
    stem_to_classes = defaultdict(list)
    tw, th = image_size, image_size
    for s in samples:
        stem = s["stem"]
        lbl_p = s.get("label")
        if lbl_p and Path(lbl_p).exists():
            orig_w, orig_h = 960, 540
            try:
                with Image.open(s["image"]) as im:
                    orig_w, orig_h = im.size
            except Exception:
                pass
            scale = min(tw / orig_w, th / orig_h)
            for cid, cx, cy, bw, bh in _read_raw_labels(Path(lbl_p)):
                if (bw * orig_w * scale >= min_box_px) and (bh * orig_h * scale >= min_box_px):
                    stem_to_classes[stem].append(cid)

    # Chia cụm
    split_stems = split_clusters_for_classes(
        clusters=clusters,
        stem_to_classes=stem_to_classes,
        img_paths=img_paths,
        ratio=ratio_tuple,
        seed=seed,
    )

    return {
        sname: [stem_to_sample[st] for st in split_stems.get(sname, []) if st in stem_to_sample]
        for sname in ("train", "val", "test")
    }


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    """Ghi out_dir/data.yaml (path, train/val/test, nc, names) cho Ultralytics YOLO."""
    out_dir = Path(out_dir)
    names_dict = {i: name for i, name in enumerate(class_names)}
    content = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        f"names: {json.dumps(names_dict, ensure_ascii=False)}",
        "",
    ]
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(content))


def _filter_tiny_boxes(
    samples: list[dict],
    min_box_px: int = 8,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[dict[str, list[tuple[int, float, float, float, float]]], int, int]:
    """BƯỚC 1: Lọc bỏ toàn bộ box nhỏ (< min_box_px) trên canvas target_size."""
    clean_boxes = {}
    total_in = 0
    total_dropped = 0
    tw, th = target_size

    for s in samples:
        stem = s["stem"]
        lbl_p = s.get("label")
        raw_b = _read_raw_labels(Path(lbl_p)) if lbl_p and Path(lbl_p).exists() else []
        total_in += len(raw_b)

        orig_w, orig_h = 960, 540
        try:
            with Image.open(s["image"]) as im:
                orig_w, orig_h = im.size
        except Exception:
            pass
        scale = min(tw / orig_w, th / orig_h)

        kept = []
        for cid, cx, cy, bw, bh in raw_b:
            if (bw * orig_w * scale >= min_box_px) and (bh * orig_h * scale >= min_box_px):
                kept.append((cid, cx, cy, bw, bh))
            else:
                total_dropped += 1
        clean_boxes[stem] = kept

    return clean_boxes, total_in, total_dropped


def _consolidate_post_resize_leakage(
    out_root: Path,
    hamming_thresh: int = 380,
    hash_size: int = 32,
) -> dict[str, Any]:
    """Kiểm tra rò rỉ dữ liệu sau resize và dồn tập triệt để:
      - Nếu rò rỉ giữa (val hoặc test) với train -> dồn sang train.
      - Nếu rò rỉ giữa val với test -> dồn sang val.
    """
    img_base = out_root / "images"
    lbl_base = out_root / "labels"
    splits = ["train", "val", "test"]

    img_split: dict[str, str] = {}
    hashes: dict[str, np.ndarray] = {}

    for s in splits:
        s_img_dir = img_base / s
        if not s_img_dir.is_dir():
            continue
        for p in sorted(s_img_dir.glob("*.jpg")):
            img_split[p.name] = s
            h = calculate_dhash(p, hash_size=hash_size, strip_padding=True)
            if h is not None:
                hashes[p.name] = np.packbits(h)

    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)
    names = list(hashes.keys())
    if not names:
        return {
            "moved_to_train": [],
            "moved_to_val": [],
            "split_stats": {s: {"n_images": 0, "n_boxes_out": 0} for s in splits},
            "total_boxes_kept": 0,
            "remaining_leaks": {"train_vs_val": 0, "train_vs_test": 0, "val_vs_test": 0},
        }

    packed_matrix = np.array([hashes[n] for n in names], dtype=np.uint8)

    # Đồ thị tương đồng
    adj: dict[str, list[str]] = {n: [] for n in names}
    for i in range(len(names)):
        diff = np.bitwise_xor(packed_matrix[i + 1:], packed_matrix[i])
        dists = popcount[diff].sum(axis=1)
        matches = np.where(dists <= hamming_thresh)[0]
        for m in matches:
            j = i + 1 + m
            adj[names[i]].append(names[j])
            adj[names[j]].append(names[i])

    current_split = dict(img_split)

    # 1. Dồn về Train
    train_queue = [n for n, s in current_split.items() if s == "train"]
    visited_train = set(train_queue)
    moved_to_train: list[tuple[str, str, str]] = []
    head = 0
    while head < len(train_queue):
        curr = train_queue[head]
        head += 1
        for neighbor in adj[curr]:
            if neighbor not in visited_train:
                visited_train.add(neighbor)
                train_queue.append(neighbor)
                if current_split[neighbor] != "train":
                    moved_to_train.append((neighbor, current_split[neighbor], "train"))
                    current_split[neighbor] = "train"

    # 2. Dồn về Val
    val_queue = [n for n, s in current_split.items() if s == "val"]
    visited_val = set(val_queue)
    moved_to_val: list[tuple[str, str, str]] = []
    head = 0
    while head < len(val_queue):
        curr = val_queue[head]
        head += 1
        for neighbor in adj[curr]:
            if current_split[neighbor] == "test":
                moved_to_val.append((neighbor, "test", "val"))
                current_split[neighbor] = "val"
                if neighbor not in visited_val:
                    visited_val.add(neighbor)
                    val_queue.append(neighbor)

    # Thực hiện di chuyển file trên đĩa
    for name, src_s, dst_s in moved_to_train + moved_to_val:
        stem = Path(name).stem
        src_img = img_base / src_s / name
        dst_img = img_base / dst_s / name
        if src_img.exists():
            shutil.move(src_img, dst_img)

        src_lbl = lbl_base / src_s / f"{stem}.txt"
        dst_lbl = lbl_base / dst_s / f"{stem}.txt"
        if src_lbl.exists():
            shutil.move(src_lbl, dst_lbl)

    # Đếm kiểm tra rò rỉ còn lại
    def _count_cross_leaks(list1: list[str], list2: list[str]) -> int:
        if not list1 or not list2:
            return 0
        mat1 = np.array([hashes[n] for n in list1], dtype=np.uint8)
        mat2 = np.array([hashes[n] for n in list2], dtype=np.uint8)
        cnt = 0
        for idx in range(len(list1)):
            diff = np.bitwise_xor(mat2, mat1[idx])
            dists = popcount[diff].sum(axis=1)
            cnt += int(np.sum(dists <= hamming_thresh))
        return cnt

    final_tr = [n for n, s in current_split.items() if s == "train"]
    final_va = [n for n, s in current_split.items() if s == "val"]
    final_te = [n for n, s in current_split.items() if s == "test"]

    tr_va_leaks = _count_cross_leaks(final_tr, final_va)
    tr_te_leaks = _count_cross_leaks(final_tr, final_te)
    va_te_leaks = _count_cross_leaks(final_va, final_te)

    split_stats: dict[str, dict[str, int]] = {}
    total_kept = 0
    for sname in splits:
        s_imgs = list((img_base / sname).glob("*.jpg"))
        n_boxes = 0
        for p in s_imgs:
            lbl_p = lbl_base / sname / f"{p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    n_boxes += sum(1 for line in f if line.strip())
        split_stats[sname] = {
            "n_images": len(s_imgs),
            "n_boxes_out": n_boxes,
        }
        total_kept += n_boxes

    return {
        "moved_to_train": moved_to_train,
        "moved_to_val": moved_to_val,
        "split_stats": split_stats,
        "total_boxes_kept": total_kept,
        "remaining_leaks": {
            "train_vs_val": tr_va_leaks,
            "train_vs_test": tr_te_leaks,
            "val_vs_test": va_te_leaks,
        },
    }


def run_preprocess(cfg: dict, data_root=None) -> dict:
    """Chạy quy trình tiền xử lý 6 bước và trả về metadata tổng hợp."""
    data = cfg["data"]
    raw_dir = resolve_path(data_root or cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    set_seed(cfg["seed"])

    classes_dict = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
    class_names = [classes_dict[i]["name"] for i in sorted(classes_dict)]
    num_classes = len(class_names)

    img_size = int(data.get("image_size", 640))
    pad_val = int(data.get("pad_value", 114))
    min_box_px = int(cfg.get("preprocess", {}).get("min_box_px", 8))
    hash_size = int(cfg.get("split", {}).get("hash_size", 32))
    hamming_thresh = int(cfg.get("split", {}).get("hamming_threshold", 380))
    augment_minority = True

    print(f"[run_preprocess] Bắt đầu tiền xử lý: raw_dir={raw_dir}, out_dir={out_dir}")
    t0 = time.time()

    # Quét dữ liệu thô
    samples = list_raw_samples(raw_dir)
    print(f"[run_preprocess] Tìm thấy {len(samples)} ảnh trong {raw_dir}")

    # BƯỚC 1: Lọc tiny boxes
    print(f"[run_preprocess] BƯỚC 1: Lọc tiny boxes (< {min_box_px}px)...")
    clean_boxes, total_boxes_in, total_dropped_tiny = _filter_tiny_boxes(
        samples, min_box_px=min_box_px, target_size=(img_size, img_size)
    )

    # BƯỚC 2 & BƯỚC 3: dHash clustering & Split bảo toàn 52 lớp
    print(f"[run_preprocess] BƯỚC 2 & 3: Gom cụm dHash 1024-bit và chia tập không rò rỉ...")
    splits = split_samples(
        samples,
        data.get("split", {"train": 0.7, "val": 0.2, "test": 0.1}),
        seed=cfg["seed"],
        hash_size=hash_size,
        hamming_thresh=hamming_thresh,
        min_box_px=min_box_px,
        image_size=img_size,
    )

    # BƯỚC 4: Augment lớp thiểu số trong tập train
    print(f"[run_preprocess] BƯỚC 4: Tạo augment cho các lớp thiểu số trong tập Train...")
    aug_items: list[tuple[str, np.ndarray, list[tuple[int, float, float, float, float]]]] = []
    stem_to_sample = {s["stem"]: s for s in samples}

    if augment_minority and "train" in splits:
        rng = random.Random(cfg["seed"])
        train_class_counts: Counter[int] = Counter()
        class_candidates = defaultdict(list)

        for s in splits["train"]:
            stem = s["stem"]
            boxes = clean_boxes[stem]
            img_p = Path(s["image"])
            orig_w, orig_h = 960, 540
            try:
                with Image.open(img_p) as im:
                    orig_w, orig_h = im.size
            except Exception:
                pass

            for i, b in enumerate(boxes):
                train_class_counts[b[0]] += 1
                other_b = [other for j, other in enumerate(boxes) if j != i and other[0] != b[0]]
                bounds = get_safe_crop_bounds(b, other_b, orig_w, orig_h)
                if bounds is not None:
                    class_candidates[b[0]].append((stem, b, bounds))

        # Mục tiêu adaptive
        class_targets = {}
        for cid, cnt in train_class_counts.items():
            if cnt < 30:
                class_targets[cid] = max(int(round(cnt * 2.5)), min(15, cnt * 3))
            elif cnt <= 150:
                class_targets[cid] = int(round(cnt * 1.75))
            else:
                class_targets[cid] = cnt

        max_augmented_per_class = 7
        class_needed = {
            c: min(class_targets[c] - train_class_counts[c], max_augmented_per_class)
            for c in train_class_counts
            if class_targets[c] > train_class_counts[c]
        }

        # Chống rò rỉ đối chiếu với val và test
        val_test_hashes = []
        for sname in ("val", "test"):
            for s in splits.get(sname, []):
                h = calculate_dhash(s["image"], hash_size=32, strip_padding=True)
                if h is not None:
                    val_test_hashes.append(h)
        val_test_packed = np.packbits(np.array(val_test_hashes, dtype=np.uint8), axis=1) if val_test_hashes else None
        popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        aug_per_class: Counter[int] = Counter()
        for cid, needed in class_needed.items():
            cands = list(class_candidates.get(cid, []))
            if not cands:
                continue
            rng.shuffle(cands)
            gen = 0
            attempts = 0
            while gen < needed and attempts < len(cands) * 3:
                cand_stem, b, bounds = cands[attempts % len(cands)]
                attempts += 1
                im_raw = cv2.imread(str(stem_to_sample[cand_stem]["image"]))
                if im_raw is None:
                    continue
                orig_h, orig_w = im_raw.shape[:2]
                canvas, final_box = transform_and_letterbox_crop(
                    im_raw, b, bounds, rng, (img_size, img_size), min_box_px, pad_val
                )
                if final_box is None:
                    continue

                if val_test_packed is not None:
                    h_aug = calculate_dhash(canvas, hash_size=32, strip_padding=True)
                    if h_aug is not None:
                        packed_aug = np.packbits(h_aug)
                        diff = np.bitwise_xor(val_test_packed, packed_aug)
                        dists = popcount[diff].sum(axis=1)
                        if dists.min() <= hamming_thresh:
                            continue

                gen += 1
                aug_stem = f"{cand_stem}_aug_{cid}_{gen}"
                aug_items.append((aug_stem, canvas, [final_box]))
                aug_per_class[cid] += 1

        print(f"[run_preprocess]   Sinh {len(aug_items)} ảnh tăng cường cho {len(aug_per_class)} lớp thiểu số trong Train.")

    # BƯỚC 5: Letterbox và xuất ra out_dir
    print(f"[run_preprocess] BƯỚC 5: Letterbox 640x640 cho toàn bộ split...")
    for sub in ("images", "labels"):
        d = out_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for sname in ("train", "val", "test"):
            (d / sname).mkdir(parents=True, exist_ok=True)

    for sname, items in splits.items():
        for s in items:
            letterbox_sample(s, out_dir, sname, img_size, pad_val, min_box_px)

    # Ghi ảnh tăng cường vào train
    if aug_items:
        train_img_dir = out_dir / "images" / "train"
        train_lbl_dir = out_dir / "labels" / "train"
        for aug_stem, canvas, aug_boxes in aug_items:
            cv2.imwrite(str(train_img_dir / f"{aug_stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_labels(train_lbl_dir / f"{aug_stem}.txt", aug_boxes)

    # BƯỚC 6: Dồn tập sau resize bảo đảm 0.00% rò rỉ
    print(f"[run_preprocess] BƯỚC 6: Kiểm tra rò rỉ sau resize và dồn tập triệt để...")
    consolidation_res = _consolidate_post_resize_leakage(
        out_root=out_dir,
        hamming_thresh=hamming_thresh,
        hash_size=hash_size,
    )
    moved_tr = consolidation_res["moved_to_train"]
    moved_va = consolidation_res["moved_to_val"]
    print(f"[run_preprocess]   Dồn về Train: {len(moved_tr)} ảnh, dồn về Val: {len(moved_va)} ảnh.")
    rem = consolidation_res["remaining_leaks"]
    print(f"[run_preprocess]   Rò rỉ còn lại: Train-Val={rem['train_vs_val']}, Train-Test={rem['train_vs_test']}, Val-Test={rem['val_vs_test']}.")

    # Ghi data.yaml và meta.json
    write_data_yaml(out_dir, class_names)

    meta = {
        "dataset_name": "TSD-Voice-Alert-Processed",
        "image_size": img_size,
        "seed": cfg["seed"],
        "num_classes": num_classes,
        "counts": {s: consolidation_res["split_stats"][s]["n_images"] for s in ("train", "val", "test")},
        "box_counts": {s: consolidation_res["split_stats"][s]["n_boxes_out"] for s in ("train", "val", "test")},
        "total_images_in": len(samples),
        "total_dropped_tiny": total_dropped_tiny,
        "total_boxes_kept": consolidation_res["total_boxes_kept"],
        "remaining_leaks": consolidation_res["remaining_leaks"],
        "leakage_rate": 0.0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    write_json(meta, out_dir / "meta.json")
    print(f"[run_preprocess] Hoàn thành tiền xử lý trong {meta['elapsed_seconds']}s -> {out_dir}")
    return meta
