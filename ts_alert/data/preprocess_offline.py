"""Tiền xử lý offline dataset với presplit train/val/test cố định.

Lọc box nhỏ, tăng cường lớp thiểu số trong train không làm mờ,
letterbox và thống kê rò rỉ dHash mà không chuyển ảnh giữa split.
Mọi split phải có ít nhất một ảnh nguồn.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

from analysis.helpers.image_utils import calculate_dhash
from ts_alert.config.resolve import resolve_path
from ts_alert.data.helpers.augmentation import (
    _read_raw_labels,
    _write_labels,
    compute_adaptive_class_targets,
    get_safe_crop_bounds,
    transform_and_letterbox_crop,
)
from ts_alert.utils.io import load_classes, write_json
from ts_alert.utils.seed import set_seed


def list_raw_samples(raw_dir: Path) -> list[dict]:
    """Quét layout có sẵn split hoặc layout phẳng, bảo đảm stem đầu ra duy nhất.

    Mỗi sample chứa image, label, stem và split. Đường dẫn ảnh/nhãn gốc được
    giữ nguyên; chỉ stem đầu ra được đổi khi trùng tên.
    """
    raw_dir = Path(raw_dir)
    split_names = ("train", "val", "test")
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    samples: list[dict] = []

    def _collect(img_dir: Path, lbl_dir: Path, split: str | None) -> None:
        """Thu thập ảnh trực tiếp trong thư mục theo thứ tự tên file."""
        for img_p in sorted(img_dir.glob("*.*"), key=lambda path: path.name):
            if not img_p.is_file() or img_p.suffix.lower() not in img_extensions:
                continue
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            samples.append(
                {
                    "image": img_p,
                    "label": lbl_p if lbl_p.exists() else None,
                    "stem": img_p.stem,
                    "split": split,
                }
            )

    if any((raw_dir / "images" / split).is_dir() for split in split_names):
        for split in split_names:
            _collect(raw_dir / "images" / split, raw_dir / "labels" / split, split)
    elif any((raw_dir / split / "images").is_dir() for split in split_names):
        for split in split_names:
            _collect(raw_dir / split / "images", raw_dir / split / "labels", split)
    else:
        img_dir = raw_dir / "images" if (raw_dir / "images").is_dir() else raw_dir
        lbl_dir = raw_dir / "labels" if (raw_dir / "labels").is_dir() else raw_dir
        _collect(img_dir, lbl_dir, None)

    stem_counts: Counter[str] = Counter(sample["stem"] for sample in samples)
    reserved_stems = {
        stem for stem, count in stem_counts.items() if count == 1
    }
    used_stems = set(reserved_stems)

    for sample in samples:
        raw_stem = sample["stem"]
        if stem_counts[raw_stem] == 1:
            continue

        split = sample["split"]
        base_stem = f"{split}_{raw_stem}" if split is not None else raw_stem
        candidate = base_stem
        suffix = 1
        while candidate in used_stems:
            candidate = f"{base_stem}_{suffix}"
            suffix += 1

        sample["stem"] = candidate
        used_stems.add(candidate)

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
    adjusted: list[tuple[int, float, float, float, float]] = []
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

        adjusted.append(
            (
                cid,
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            )
        )

    return adjusted


def letterbox_sample(
    sample: dict,
    out_dir: Path,
    split: str,
    image_size: int = 640,
    pad_value: int = 114,
    min_box_px: int = 8,
) -> None:
    """Letterbox ảnh, biến đổi nhãn và ghi vào images/<split>, labels/<split>."""
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
    canvas, scale, dx, dy = _letterbox_image(
        img, (image_size, image_size), pad_value
    )

    raw_boxes: list[tuple[int, float, float, float, float]] = []
    lbl_p = sample.get("label")
    if lbl_p and Path(lbl_p).exists():
        raw_boxes = _read_raw_labels(Path(lbl_p))

    valid_boxes = [
        (cid, cx, cy, bw, bh)
        for cid, cx, cy, bw, bh in raw_boxes
        if bw * orig_w * scale >= min_box_px
        and bh * orig_h * scale >= min_box_px
    ]

    adj_boxes = _adjust_boxes(
        valid_boxes, orig_w, orig_h, scale, dx, dy, image_size, image_size
    )

    cv2.imwrite(
        str(dst_img_dir / f"{stem}.jpg"),
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    _write_labels(dst_lbl_dir / f"{stem}.txt", adj_boxes)






def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    """Ghi data.yaml chứa đường dẫn, các split và danh sách lớp cho YOLO."""
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
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as file:
        file.write("\n".join(content))


def _filter_tiny_boxes(
    samples: list[dict],
    min_box_px: int = 8,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[dict[str, list[tuple[int, float, float, float, float]]], int, int]:
    """Lọc bỏ box có chiều rộng hoặc chiều cao nhỏ hơn ngưỡng trên canvas."""
    clean_boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}
    total_in = 0
    total_dropped = 0
    tw, th = target_size

    for sample in samples:
        stem = sample["stem"]
        lbl_p = sample.get("label")
        raw_boxes = (
            _read_raw_labels(Path(lbl_p))
            if lbl_p and Path(lbl_p).exists()
            else []
        )
        total_in += len(raw_boxes)

        orig_w, orig_h = 960, 540
        try:
            with Image.open(sample["image"]) as im:
                orig_w, orig_h = im.size
        except Exception:
            pass
        scale = min(tw / orig_w, th / orig_h)

        kept: list[tuple[int, float, float, float, float]] = []
        for cid, cx, cy, bw, bh in raw_boxes:
            if (
                bw * orig_w * scale >= min_box_px
                and bh * orig_h * scale >= min_box_px
            ):
                kept.append((cid, cx, cy, bw, bh))
            else:
                total_dropped += 1
        clean_boxes[stem] = kept

    return clean_boxes, total_in, total_dropped


def _consolidate_post_resize_leakage(
    out_root: Path,
    hamming_thresh: int = 50,
    hash_size: int = 32,
) -> dict[str, Any]:
    """Đếm rò rỉ dHash sau resize, giữ nguyên mọi ảnh trong split hiện tại.

    So sánh Hamming theo khối 256x256 để giới hạn bộ nhớ tạm.
    Ảnh không tính được dHash vẫn được tính trong thống kê đầu ra.
    """
    out_root = Path(out_root)
    img_base = out_root / "images"
    lbl_base = out_root / "labels"
    split_names = ("train", "val", "test")
    chunk_size = 256
    hashes: dict[str, list[np.ndarray]] = {
        name: [] for name in split_names
    }
    split_stats: dict[str, dict[str, int]] = {}
    total_boxes_kept = 0

    for name in split_names:
        images = sorted((img_base / name).glob("*.jpg"))
        n_boxes = 0
        for path in images:
            dhash = calculate_dhash(
                path, hash_size=hash_size, strip_padding=True
            )
            if dhash is not None:
                hashes[name].append(np.packbits(dhash))

            label_path = lbl_base / name / f"{path.stem}.txt"
            if label_path.is_file():
                with label_path.open(encoding="utf-8") as file:
                    n_boxes += sum(1 for line in file if line.strip())

        split_stats[name] = {
            "n_images": len(images),
            "n_boxes_out": n_boxes,
        }
        total_boxes_kept += n_boxes

    matrices = {
        name: np.stack(items) if items else None
        for name, items in hashes.items()
    }
    popcount = np.array(
        [value.bit_count() for value in range(256)],
        dtype=np.uint8,
    )

    def _count_cross_leaks(first: str, second: str) -> int:
        """Đếm cặp ảnh giữa hai split có Hamming không vượt ngưỡng."""
        mat1 = matrices[first]
        mat2 = matrices[second]
        if mat1 is None or mat2 is None:
            return 0

        count = 0
        for start_a in range(0, len(mat1), chunk_size):
            block_a = mat1[start_a : start_a + chunk_size]
            for start_b in range(0, len(mat2), chunk_size):
                block_b = mat2[start_b : start_b + chunk_size]
                xor = np.bitwise_xor(
                    block_a[:, None, :], block_b[None, :, :]
                )
                distances = popcount[xor].sum(axis=2, dtype=np.uint64)
                del xor
                count += int(
                    np.count_nonzero(distances <= hamming_thresh)
                )
        return count

    return {
        "moved_to_train": [],
        "moved_to_val": [],
        "split_stats": split_stats,
        "total_boxes_kept": total_boxes_kept,
        "remaining_leaks": {
            "train_vs_val": _count_cross_leaks("train", "val"),
            "train_vs_test": _count_cross_leaks("train", "test"),
            "val_vs_test": _count_cross_leaks("val", "test"),
        },
    }


def run_preprocess(
    cfg: dict,
    data_root: str | Path | None = None,
) -> dict:
    """Tiền xử lý presplit cố định và trả metadata chất lượng đầu ra."""
    data = cfg["data"]
    raw_dir = resolve_path(data_root or cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    set_seed(cfg["seed"])

    classes_dict = load_classes(resolve_path(cfg["paths"]["classes_csv"]))
    class_names = [classes_dict[i]["name"] for i in sorted(classes_dict)]
    num_classes = len(class_names)

    img_size = int(data.get("image_size", 640))
    pad_val = int(data.get("pad_value", 114))
    preprocess_cfg = cfg.get("preprocess", {})
    max_augmented_per_class = preprocess_cfg.get(
        "max_augmented_per_class", 10
    )
    if max_augmented_per_class is not None:
        max_augmented_per_class = int(max_augmented_per_class)
    min_box_px = int(preprocess_cfg.get("min_box_px", 8))
    blur_threshold = float(
        data.get("blur_threshold", preprocess_cfg.get("blur_threshold", 50.0))
    )
    dark_threshold = float(
        data.get("dark_threshold", preprocess_cfg.get("dark_threshold", 85.0))
    )
    hash_size = int(cfg.get("split", {}).get("hash_size", 32))
    hamming_thresh = int(
        cfg.get("split", {}).get("hamming_threshold", 50)
    )
    augment_minority = True
    split_names = ("train", "val", "test")

    print(
        f"[run_preprocess] Bắt đầu tiền xử lý: "
        f"raw_dir={raw_dir}, out_dir={out_dir}"
    )
    print(
        "[run_preprocess]   Cấu hình chất lượng ảnh: "
        f"blur_threshold={blur_threshold}, dark_threshold={dark_threshold}"
    )
    t0 = time.time()

    samples = list_raw_samples(raw_dir)
    print(f"[run_preprocess] Tìm thấy {len(samples)} ảnh trong {raw_dir}")

    print(f"[run_preprocess] Lọc tiny boxes (< {min_box_px}px)...")
    clean_boxes, total_boxes_in, total_dropped_tiny = _filter_tiny_boxes(
        samples,
        min_box_px=min_box_px,
        target_size=(img_size, img_size),
    )

    split_mode = "existing"
    if any(sample.get("split") not in split_names for sample in samples):
        raise ValueError(
            "Mọi ảnh phải thuộc presplit train, val hoặc test có sẵn."
        )

    splits = {
        name: [sample for sample in samples if sample["split"] == name]
        for name in split_names
    }
    empty_splits = [name for name in split_names if not splits[name]]
    if empty_splits:
        raise ValueError(
            "Các presplit phải có ít nhất một ảnh; split rỗng: "
            + ", ".join(empty_splits)
        )

    print("[run_preprocess] B1: Giữ phân chia Train/Val/Test có sẵn.")

    augmentation_cap = (
        "không giới hạn"
        if max_augmented_per_class is None
        else f"{max_augmented_per_class} ảnh/lớp"
    )
    print(
        "[run_preprocess] B2: Tăng cường lớp thiểu số trong Train "
        f"(không làm mờ, trần: {augmentation_cap}, "
        "chỉ lớp có dưới 100 box)..."
    )
    aug_items: list[
        tuple[
            str,
            np.ndarray,
            list[tuple[int, float, float, float, float]],
        ]
    ] = []
    stem_to_sample = {sample["stem"]: sample for sample in samples}
    used_output_stems = set(stem_to_sample)

    if augment_minority and "train" in splits:
        rng = random.Random(cfg["seed"])
        train_class_counts: Counter[int] = Counter()
        class_candidates: dict[
            int,
            list[
                tuple[
                    str,
                    tuple[int, float, float, float, float],
                    tuple[float, float, float, float],
                ]
            ],
        ] = defaultdict(list)

        for sample in splits["train"]:
            stem = sample["stem"]
            boxes = clean_boxes[stem]
            if not boxes:
                continue

            orig_w, orig_h = 960, 540
            try:
                with Image.open(sample["image"]) as im:
                    orig_w, orig_h = im.size
            except Exception:
                pass

            for index, box in enumerate(boxes):
                train_class_counts[box[0]] += 1
                other_boxes = [
                    other
                    for other_index, other in enumerate(boxes)
                    if other_index != index and other[0] != box[0]
                ]
                bounds = get_safe_crop_bounds(
                    box, other_boxes, orig_w, orig_h
                )
                if bounds is not None:
                    class_candidates[box[0]].append((stem, box, bounds))

        class_targets = compute_adaptive_class_targets(train_class_counts)
        class_needed: dict[int, int] = {}
        for cid, count in train_class_counts.items():
            if count >= 100 or class_targets[cid] <= count:
                continue

            needed = class_targets[cid] - count
            if max_augmented_per_class is not None:
                needed = min(needed, max_augmented_per_class)
            if needed <= 0:
                continue

            class_needed[cid] = needed

        val_test_hashes: list[np.ndarray] = []
        for sname in ("val", "test"):
            for sample in splits.get(sname, []):
                dhash = calculate_dhash(
                    sample["image"],
                    hash_size=hash_size,
                    strip_padding=True,
                )
                if dhash is not None:
                    val_test_hashes.append(dhash)

        val_test_packed = (
            np.packbits(np.array(val_test_hashes, dtype=np.uint8), axis=1)
            if val_test_hashes
            else None
        )
        popcount = np.array(
            [k.bit_count() for k in range(256)], dtype=np.uint8
        )

        aug_per_class: Counter[int] = Counter()
        for cid, needed in sorted(
            class_needed.items(),
            key=lambda item: train_class_counts[item[0]],
        ):
            candidates = list(class_candidates.get(cid, []))
            if not candidates:
                print(
                    f"[run_preprocess]   Class {cid}: cần thêm {needed} ảnh "
                    "nhưng không có candidate cắt an toàn."
                )
                continue

            rng.shuffle(candidates)
            generated = 0
            attempts = 0
            max_attempts = max(needed * 25, len(candidates) * 10, 100)

            while generated < needed and attempts < max_attempts:
                cand_stem, box, bounds = candidates[
                    attempts % len(candidates)
                ]
                attempts += 1

                im_raw = cv2.imread(
                    str(stem_to_sample[cand_stem]["image"])
                )
                if im_raw is None:
                    continue

                canvas, final_box = transform_and_letterbox_crop(
                    im_raw,
                    box,
                    bounds,
                    rng,
                    (img_size, img_size),
                    min_box_px,
                    pad_val,
                )
                if canvas is None or final_box is None:
                    continue

                if val_test_packed is not None:
                    h_aug = calculate_dhash(
                        canvas,
                        hash_size=hash_size,
                        strip_padding=True,
                    )
                    if h_aug is None:
                        continue
                    packed_aug = np.packbits(h_aug)
                    diff = np.bitwise_xor(val_test_packed, packed_aug)
                    distances = popcount[diff].sum(
                        axis=1, dtype=np.uint16
                    )
                    if distances.min() <= hamming_thresh:
                        continue

                generated += 1
                base_stem = f"{cand_stem}_aug_{cid}_{generated}"
                aug_stem = base_stem
                suffix = 1
                while aug_stem in used_output_stems:
                    aug_stem = f"{base_stem}_{suffix}"
                    suffix += 1
                used_output_stems.add(aug_stem)

                aug_items.append((aug_stem, canvas, [final_box]))
                aug_per_class[cid] += 1

            print(
                f"[run_preprocess]   Class {cid}: "
                f"train={train_class_counts[cid]} box, "
                f"mục tiêu={class_targets[cid]} box, "
                f"đã sinh={generated}/{needed} ảnh, "
                f"số lần thử={attempts}/{max_attempts}."
            )

        print(
            f"[run_preprocess]   Sinh {len(aug_items)} ảnh tăng cường "
            f"cho {len(aug_per_class)} lớp thiểu số trong Train."
        )

    print(
        f"[run_preprocess] B3: Letterbox {img_size}x{img_size}, "
        f"padding {pad_val} cho toàn bộ split..."
    )
    for sub in ("images", "labels"):
        directory = out_dir / sub
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for sname in split_names:
            (directory / sname).mkdir(parents=True, exist_ok=True)

    for sname, items in splits.items():
        for sample in items:
            letterbox_sample(
                sample, out_dir, sname, img_size, pad_val, min_box_px
            )

    train_img_dir = out_dir / "images" / "train"
    train_lbl_dir = out_dir / "labels" / "train"
    for aug_stem, canvas, aug_boxes in aug_items:
        cv2.imwrite(
            str(train_img_dir / f"{aug_stem}.jpg"),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        _write_labels(train_lbl_dir / f"{aug_stem}.txt", aug_boxes)

    print(
        "[run_preprocess] B4: Kiểm tra rò rỉ dHash sau resize..."
    )
    consolidation_res = _consolidate_post_resize_leakage(
        out_root=out_dir,
        hamming_thresh=hamming_thresh,
        hash_size=hash_size,
    )
    rem = consolidation_res["remaining_leaks"]
    print(
        "[run_preprocess]   Rò rỉ còn lại: "
        f"Train-Val={rem['train_vs_val']}, "
        f"Train-Test={rem['train_vs_test']}, "
        f"Val-Test={rem['val_vs_test']}."
    )

    write_data_yaml(out_dir, class_names)

    counts = {
        sname: consolidation_res["split_stats"][sname]["n_images"]
        for sname in split_names
    }
    remaining_leak_count = (
        rem["train_vs_val"] + rem["train_vs_test"] + rem["val_vs_test"]
    )
    cross_split_pairs = (
        counts["train"] * counts["val"]
        + counts["train"] * counts["test"]
        + counts["val"] * counts["test"]
    )
    leakage_rate = (
        0.0
        if remaining_leak_count == 0
        else 100.0 * remaining_leak_count / max(1, cross_split_pairs)
    )

    meta = {
        "dataset_name": "TSD-Voice-Alert-Processed",
        "image_size": img_size,
        "seed": cfg["seed"],
        "num_classes": num_classes,
        "blur_threshold": blur_threshold,
        "dark_threshold": dark_threshold,
        "counts": counts,
        "box_counts": {
            sname: consolidation_res["split_stats"][sname]["n_boxes_out"]
            for sname in split_names
        },
        "total_images_in": len(samples),
        "total_dropped_tiny": total_dropped_tiny,
        "total_boxes_kept": consolidation_res["total_boxes_kept"],
        "remaining_leaks": rem,
        "leakage_rate": leakage_rate,
        "elapsed_seconds": round(time.time() - t0, 2),
        "rare_classes_promoted": {
            "min_train_boxes": 0,
            "promoted_classes": [],
            "moved_total": 0,
        },
        "n_augmented_images": len(aug_items),
        "max_augmented_per_class": max_augmented_per_class,
        "split_mode": split_mode,
        "aug_blur": False,
    }
    write_json(meta, out_dir / "meta.json")
    print(
        f"[run_preprocess] Hoàn thành tiền xử lý trong "
        f"{meta['elapsed_seconds']}s -> {out_dir}"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    """Chạy CLI qua python -m ts_alert.data.preprocess_offline."""
    import argparse

    from ts_alert.config.resolve import (
        _load_with_inherits,
        _runtime_path,
        deep_merge,
        parse_overrides,
        resolve_config,
    )
    from ts_alert.utils.io import read_yaml

    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu offline (TSD-Voice-Alert)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Đường dẫn file YAML cấu hình (vd: configs/shared.yaml)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Thư mục dữ liệu gốc (mặc định paths.raw_dir)",
    )
    parser.add_argument(
        "--runtime",
        default="local",
        help="Runtime preset trong configs/runtime/ (default: local)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Ghi đè cấu hình, vd. data.image_size=640",
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.set)
    if args.config is not None:
        cfg_path = resolve_path(args.config)
        cfg = _load_with_inherits(cfg_path)
        if args.runtime:
            cfg = deep_merge(cfg, read_yaml(_runtime_path(args.runtime)))
        cfg = deep_merge(cfg, overrides)
    else:
        cfg = resolve_config(None, args.runtime, overrides)

    print(
        json.dumps(
            run_preprocess(cfg, args.data_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
