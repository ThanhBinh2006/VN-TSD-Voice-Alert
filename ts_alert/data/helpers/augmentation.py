"""Augmentation nhãn thiểu số trong tập train (cắt an toàn, zoom, blur nhẹ, không xoay)."""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter
import cv2
import numpy as np


def _read_raw_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    """Đọc nhãn YOLO [class_id, cx, cy, bw, bh]."""
    boxes = []
    if not p.exists():
        return boxes
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append((cid, cx, cy, bw, bh))
                except ValueError:
                    continue
    return boxes


def _write_labels(p: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """Ghi nhãn YOLO format chuẩn."""
    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes]
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_safe_crop_bounds(
    target_box: tuple[int, float, float, float, float],
    other_boxes: list[tuple[int, float, float, float, float]],
    W: int = 640,
    H: int = 640,
) -> tuple[float, float, float, float] | None:
    """Xác định giới hạn cắt an toàn không chứa box của class khác."""
    t_cid, t_cx, t_cy, t_bw, t_bh = target_box
    bx1 = (t_cx - t_bw / 2.0) * W
    by1 = (t_cy - t_bh / 2.0) * H
    bx2 = (t_cx + t_bw / 2.0) * W
    by2 = (t_cy + t_bh / 2.0) * H

    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    safe_x1, safe_y1 = 0.0, 0.0
    safe_x2, safe_y2 = float(W), float(H)

    for o_cid, o_cx, o_cy, o_bw, o_bh in other_boxes:
        ox1 = (o_cx - o_bw / 2.0) * W
        oy1 = (o_cy - o_bh / 2.0) * H
        ox2 = (o_cx + o_bw / 2.0) * W
        oy2 = (o_cy + o_bh / 2.0) * H
        o_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))

        iw = max(0.0, min(bx2, ox2) - max(bx1, ox1))
        ih = max(0.0, min(by2, oy2) - max(by1, oy1))
        inter_area = iw * ih

        if inter_area > 0:
            if inter_area / min(b_area, o_area) > 0.15:
                return None  # Đè lên nhau quá nhiều
            # Biển cắm chung cột chạm mép nhẹ: cắt ranh giới phân tách
            if ih <= iw:
                if by1 < oy1:
                    safe_y2 = min(safe_y2, max(by1 + 5, oy1))
                else:
                    safe_y1 = max(safe_y1, min(by2 - 5, oy2))
            else:
                if bx1 < ox1:
                    safe_x2 = min(safe_x2, max(bx1 + 5, ox1))
                else:
                    safe_x1 = max(safe_x1, min(bx2 - 5, ox2))
        else:
            if bx1 >= ox2:
                safe_x1 = max(safe_x1, ox2)
            elif bx2 <= ox1:
                safe_x2 = min(safe_x2, ox1)
            elif by1 >= oy2:
                safe_y1 = max(safe_y1, oy2)
            elif by2 <= oy1:
                safe_y2 = min(safe_y2, oy1)

    if safe_x1 <= bx1 + 1e-3 and bx2 - 1e-3 <= safe_x2 and safe_y1 <= by1 + 1e-3 and by2 - 1e-3 <= safe_y2:
        return (safe_x1, safe_y1, safe_x2, safe_y2)
    return None


def transform_and_letterbox_crop(
    img: np.ndarray,
    target_box: tuple[int, float, float, float, float],
    safe_bounds: tuple[float, float, float, float],
    rng: random.Random,
    target_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
) -> tuple[np.ndarray, tuple[int, float, float, float, float]] | tuple[None, None]:
    """Cắt vùng biển báo, zoom, blur nhẹ và letterbox về target_size (không xoay)."""
    H, W = img.shape[:2]
    cid, t_cx, t_cy, t_bw, t_bh = target_box
    bx1 = (t_cx - t_bw / 2.0) * W
    by1 = (t_cy - t_bh / 2.0) * H
    bx2 = (t_cx + t_bw / 2.0) * W
    by2 = (t_cy + t_bh / 2.0) * H
    bw = bx2 - bx1
    bh = by2 - by1

    safe_x1, safe_y1, safe_x2, safe_y2 = safe_bounds

    # Mở rộng ngẫu nhiên vùng đệm quanh biển báo trong safe_bounds
    pad_factor_x = rng.uniform(0.6, 2.5)
    pad_factor_y = rng.uniform(0.6, 2.5)

    crop_x1 = int(max(safe_x1, bx1 - bw * pad_factor_x))
    crop_y1 = int(max(safe_y1, by1 - bh * pad_factor_y))
    crop_x2 = int(min(safe_x2, bx2 + bw * pad_factor_x))
    crop_y2 = int(min(safe_y2, by2 + bh * pad_factor_y))

    if crop_x2 <= crop_x1 + 10 or crop_y2 <= crop_y1 + 10:
        return None, None

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    ch, cw = crop.shape[:2]

    cbx1 = bx1 - crop_x1
    cby1 = by1 - crop_y1
    cbx2 = bx2 - crop_x1
    cby2 = by2 - crop_y1

    # Zoom in / Zoom out (co dãn tỉ lệ 0.85 - 1.25, không xoay ảnh)
    zoom = rng.uniform(0.85, 1.25)
    new_cw = max(20, int(round(cw * zoom)))
    new_ch = max(20, int(round(ch * zoom)))
    transformed_crop = cv2.resize(crop, (new_cw, new_ch), interpolation=cv2.INTER_LINEAR)

    nbx1 = cbx1 * (new_cw / cw)
    nby1 = cby1 * (new_ch / ch)
    nbx2 = cbx2 * (new_cw / cw)
    nby2 = cby2 * (new_ch / ch)
    cw, ch = new_cw, new_ch

    if (nbx2 - nbx1) < min_box_px or (nby2 - nby1) < min_box_px:
        return None, None

    # Làm mờ nhẹ (xác suất 35%, kernel 3x3) & chỉnh sáng nhẹ
    if rng.random() < 0.35:
        sigma = rng.uniform(0.6, 1.0)
        transformed_crop = cv2.GaussianBlur(transformed_crop, (3, 3), sigmaX=sigma, sigmaY=sigma)

    alpha = rng.uniform(0.88, 1.15)
    beta = rng.uniform(-18, 18)
    transformed_crop = cv2.convertScaleAbs(transformed_crop, alpha=alpha, beta=beta)

    # Letterbox về target_size (640, 640)
    tw, th = target_size
    scale = min(tw / cw, th / ch)
    rw = int(round(cw * scale))
    rh = int(round(ch * scale))
    resized_patch = cv2.resize(transformed_crop, (rw, rh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    dx = (tw - rw) // 2
    dy = (th - rh) // 2
    canvas[dy : dy + rh, dx : dx + rw] = resized_patch

    final_x1 = nbx1 * scale + dx
    final_y1 = nby1 * scale + dy
    final_x2 = nbx2 * scale + dx
    final_y2 = nby2 * scale + dy

    final_w = final_x2 - final_x1
    final_h = final_y2 - final_y1
    if final_w < min_box_px or final_h < min_box_px:
        return None, None

    final_cx = (final_x1 + final_x2) / 2.0 / tw
    final_cy = (final_y1 + final_y2) / 2.0 / th
    final_bw = final_w / tw
    final_bh = final_h / th

    final_box = (cid, final_cx, final_cy, final_bw, final_bh)
    return canvas, final_box


def augment_minority_classes(
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_test_packed: np.ndarray | None = None,
    target_min_boxes: int | str | None = "adaptive",
    image_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
    max_augmented_per_class: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    """Cân bằng dữ liệu cho các lớp trong tập Train (tối đa 7 ảnh/lớp, không xoay)."""
    rng = random.Random(seed)
    if not train_img_dir.is_dir() or not train_lbl_dir.is_dir():
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    base_lbl_paths = sorted([p for p in train_lbl_dir.glob("*.txt") if not ("_aug" in p.stem)])
    
    img_boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}
    class_candidates: dict[int, list[tuple[str, tuple[int, float, float, float, float], tuple[float, float, float, float]]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()

    for lbl_p in base_lbl_paths:
        stem = lbl_p.stem
        boxes = _read_raw_labels(lbl_p)
        if not boxes:
            continue
        img_boxes[stem] = boxes
        for i, b in enumerate(boxes):
            class_counts[b[0]] += 1
            other_b = [other for j, other in enumerate(boxes) if j != i and other[0] != b[0]]
            bounds = get_safe_crop_bounds(b, other_b, image_size[0], image_size[1])
            if bounds is not None:
                class_candidates[b[0]].append((stem, b, bounds))

    if not class_counts:
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    class_targets: dict[int, int] = {}
    policy_str = str(target_min_boxes).strip().lower() if target_min_boxes is not None else "adaptive"

    if policy_str in ("adaptive", "ratio", "tier", "multiplier"):
        for cid, cnt in class_counts.items():
            if cnt < 30:
                target_cnt = max(int(round(cnt * 2.5)), min(15, cnt * 3))
            elif cnt <= 150:
                target_cnt = int(round(cnt * 1.75))
            else:
                target_cnt = cnt
            class_targets[cid] = target_cnt
    elif policy_str in ("max", "-1", "none"):
        max_count = max(class_counts.values())
        for cid in class_counts:
            class_targets[cid] = max_count
    else:
        fixed_target = int(target_min_boxes)
        for cid, cnt in class_counts.items():
            class_targets[cid] = max(cnt, fixed_target)

    # Giới hạn tối đa không quá max_augmented_per_class (mặc định 7) ảnh cộng thêm mỗi class
    class_needed = {
        c: min(class_targets[c] - class_counts[c], max_augmented_per_class)
        for c in class_counts
        if class_targets[c] > class_counts[c]
    }
    if not class_needed:
        print("[augment] Tất cả các lớp trong train đã đạt mục tiêu, không cần augment.")
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    total_needed = sum(class_needed.values())
    print(f"[augment] Bắt đầu tăng cường cho {len(class_needed)} lớp trong Train (trần: {max_augmented_per_class} ảnh/lớp, cần sinh thêm ~{total_needed:,} box)...")

    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    img_cache: dict[str, np.ndarray] = {}
    def get_img(st: str) -> np.ndarray | None:
        if st in img_cache:
            return img_cache[st]
        p = train_img_dir / f"{st}.jpg"
        if not p.exists():
            return None
        im = cv2.imread(str(p))
        if len(img_cache) < 250:
            img_cache[st] = im
        return im

    n_augmented_images = 0
    n_augmented_boxes = 0
    t_start = time.time()

    sorted_classes = sorted(class_needed.items(), key=lambda x: class_counts[x[0]])
    stem_aug_count: Counter[str] = Counter()

    for cid, needed in sorted_classes:
        candidates = class_candidates.get(cid, [])
        if not candidates:
            continue

        c_created = 0
        candidate_pool: list[tuple[str, tuple[int, float, float, float, float], tuple[float, float, float, float]]] = []
        attempts_total = 0
        max_attempts_class = max(needed * 30, 200)

        while c_created < needed and attempts_total < max_attempts_class:
            if not candidate_pool:
                candidate_pool = list(candidates)
                rng.shuffle(candidate_pool)

            stem, target_b, safe_bounds = candidate_pool.pop()
            attempts_total += 1

            src_img = get_img(stem)
            if src_img is None:
                continue

            passed = False
            for attempt in range(4):
                canvas, final_box = transform_and_letterbox_crop(
                    img=src_img,
                    target_box=target_b,
                    safe_bounds=safe_bounds,
                    rng=rng,
                    target_size=image_size,
                    min_box_px=min_box_px,
                    pad_value=pad_value,
                )
                if canvas is None or final_box is None:
                    continue

                if val_test_packed is not None:
                    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    res = cv2.resize(gray, (33, 32), interpolation=cv2.INTER_AREA)
                    diff = res[:, 1:] > res[:, :-1]
                    cand_packed = np.packbits(diff.flatten()[None, :], axis=1)
                    xor = np.bitwise_xor(cand_packed, val_test_packed)
                    dist = popcount[xor].sum(axis=1)
                    if dist.min() <= 374:
                        continue

                passed = True
                break

            if not passed:
                continue

            c_created += 1
            n_augmented_images += 1
            n_augmented_boxes += 1

            stem_aug_count[stem] += 1
            aug_stem = f"{stem}_c{cid}_aug{stem_aug_count[stem]}"
            dst_img = train_img_dir / f"{aug_stem}.jpg"
            dst_lbl = train_lbl_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(dst_img), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _write_labels(dst_lbl, [final_box])

            if n_augmented_images % 1000 == 0:
                elapsed = time.time() - t_start
                rate = n_augmented_images / elapsed if elapsed > 0 else 0
                print(f"[augment]   Đã sinh {n_augmented_images:,}/{total_needed:,} ảnh ({rate:.1f} ảnh/s)...")

    elapsed_total = time.time() - t_start
    print(f"[augment]   Hoàn tất: Đã sinh thêm {n_augmented_images:,} ảnh cho {len(class_needed)} lớp ({elapsed_total:.1f}s).")

    return {
        "n_augmented_images": n_augmented_images,
        "n_augmented_boxes": n_augmented_boxes,
        "minority_classes_augmented": list(class_needed.keys()),
        "target_boxes": policy_str,
    }
