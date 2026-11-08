"""Module tăng cường dữ liệu che khuất một phần biển báo (góc, cạnh) cho tập Train.
Cắt biển báo kèm background ngữ cảnh, tạo che khuất thực tế và ghép đôi vào canvas 640x640.
Đảm bảo sinh đủ số lượng box (mặc định 30 box/lớp) cho mỗi class trong tập Train.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def apply_partial_occlusion(
    crop_img: np.ndarray,
    box_in_crop: tuple[float, float, float, float],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Tạo mảng che khuất một phần ở góc hoặc cạnh của biển báo.
    
    box_in_crop: (bx1, by1, bx2, by2) pixel trong crop_img.
    Trả về: (ảnh crop đã che, box_in_crop giữ nguyên tọa độ gốc của biển báo).
    """
    img = crop_img.copy()
    bx1, by1, bx2, by2 = box_in_crop
    bw = max(2.0, bx2 - bx1)
    bh = max(2.0, by2 - by1)

    # Chọn hình thức che: 0 = Che góc (Corner), 1 = Che cạnh (Edge)
    mode = rng.choice(["corner", "edge"])
    
    # Bảng màu vật cản thực tế (cành lá cây xanh, thân cây nâu, cột kim loại xám/đen, mép tôn trắng xám)
    colors = [
        (25, 75, 35),    # Xanh lá cây đậm
        (35, 50, 80),    # Nâu thân cây
        (40, 40, 40),    # Xám đen (cột đèn, thanh kim loại)
        (70, 70, 70),    # Xám bê tông
        (210, 210, 215), # Tôn trắng bạc
    ]
    color = rng.choice(colors)

    if mode == "corner":
        # Chọn 1 trong 4 góc: top-left (0), top-right (1), bottom-left (2), bottom-right (3)
        corner = rng.randint(0, 3)
        # Tỉ lệ che góc: 22% - 38% chiều rộng và chiều cao (khoảng 5% - 15% diện tích)
        rw = bw * rng.uniform(0.22, 0.38)
        rh = bh * rng.uniform(0.22, 0.38)
        
        if corner == 0:  # Top-Left
            ox1, oy1 = int(round(bx1)), int(round(by1))
            ox2, oy2 = int(round(bx1 + rw)), int(round(by1 + rh))
        elif corner == 1:  # Top-Right
            ox1, oy1 = int(round(bx2 - rw)), int(round(by1))
            ox2, oy2 = int(round(bx2)), int(round(by1 + rh))
        elif corner == 2:  # Bottom-Left
            ox1, oy1 = int(round(bx1)), int(round(by2 - rh))
            ox2, oy2 = int(round(bx1 + rw)), int(round(by2))
        else:  # Bottom-Right
            ox1, oy1 = int(round(bx2 - rw)), int(round(by2 - rh))
            ox2, oy2 = int(round(bx2)), int(round(by2))

    else:  # mode == "edge"
        # Chọn 1 trong 4 cạnh: top (0), bottom (1), left (2), right (3)
        edge = rng.randint(0, 3)
        if edge == 0:  # Top edge
            depth = bh * rng.uniform(0.15, 0.30)
            span = bw * rng.uniform(0.65, 1.0)
            start_x = bx1 + rng.uniform(0, max(0.0, bw - span))
            ox1, oy1 = int(round(start_x)), int(round(by1))
            ox2, oy2 = int(round(start_x + span)), int(round(by1 + depth))
        elif edge == 1:  # Bottom edge
            depth = bh * rng.uniform(0.15, 0.30)
            span = bw * rng.uniform(0.65, 1.0)
            start_x = bx1 + rng.uniform(0, max(0.0, bw - span))
            ox1, oy1 = int(round(start_x)), int(round(by2 - depth))
            ox2, oy2 = int(round(start_x + span)), int(round(by2))
        elif edge == 2:  # Left edge
            depth = bw * rng.uniform(0.15, 0.30)
            span = bh * rng.uniform(0.65, 1.0)
            start_y = by1 + rng.uniform(0, max(0.0, bh - span))
            ox1, oy1 = int(round(bx1)), int(round(start_y))
            ox2, oy2 = int(round(bx1 + depth)), int(round(start_y + span))
        else:  # Right edge
            depth = bw * rng.uniform(0.15, 0.30)
            span = bh * rng.uniform(0.65, 1.0)
            start_y = by1 + rng.uniform(0, max(0.0, bh - span))
            ox1, oy1 = int(round(bx2 - depth)), int(round(start_y))
            ox2, oy2 = int(round(bx2)), int(round(start_y + span))

    # Đảm bảo tọa độ không vượt quá biên crop
    H_crop, W_crop = img.shape[:2]
    ox1 = max(0, min(W_crop - 1, ox1))
    oy1 = max(0, min(H_crop - 1, oy1))
    ox2 = max(ox1 + 2, min(W_crop, ox2))
    oy2 = max(oy1 + 2, min(H_crop, oy2))

    # Vẽ vật che khuất (hình chữ nhật hoặc elip giả lập cành cây)
    fill_type = rng.choice(["rect", "ellipse", "noise"])
    if fill_type == "rect":
        cv2.rectangle(img, (ox1, oy1), (ox2, oy2), color, -1)
    elif fill_type == "ellipse":
        center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
        axes = (max(2, (ox2 - ox1) // 2), max(2, (oy2 - oy1) // 2))
        angle = rng.uniform(-30, 30)
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    else:  # noise / texture
        patch_shape = (oy2 - oy1, ox2 - ox1, 3)
        noise = rng.randint(0, 255)
        patch = np.full(patch_shape, noise, dtype=np.uint8)
        if rng.random() < 0.5:
            patch = cv2.GaussianBlur(patch, (5, 5), 0)
        img[oy1:oy2, ox1:ox2] = patch

    return img, (bx1, by1, bx2, by2)


def extract_sign_crop_with_background(
    img: np.ndarray,
    box_xywh_norm: tuple[int, float, float, float, float],
    rng: random.Random,
    pad_ratio_min: float = 0.20,
    pad_ratio_max: float = 0.50,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Cắt biển báo kèm vùng đệm background xung quanh."""
    H, W = img.shape[:2]
    cid, cx, cy, bw, bh = box_xywh_norm
    bx1 = (cx - bw / 2.0) * W
    by1 = (cy - bh / 2.0) * H
    bx2 = (cx + bw / 2.0) * W
    by2 = (cy + bh / 2.0) * H

    w = bx2 - bx1
    h = by2 - by1
    if w < 4 or h < 4:
        return None

    # Mở rộng background xung quanh biển
    pad_x = rng.uniform(pad_ratio_min, pad_ratio_max) * w
    pad_y = rng.uniform(pad_ratio_min, pad_ratio_max) * h

    cx1 = int(max(0, bx1 - pad_x))
    cy1 = int(max(0, by1 - pad_y))
    cx2 = int(min(W, bx2 + pad_x))
    cy2 = int(min(H, by2 + pad_y))

    if cx2 <= cx1 + 10 or cy2 <= cy1 + 10:
        return None

    crop = img[cy1:cy2, cx1:cx2].copy()

    # Tọa độ biển báo bên trong crop
    cbx1 = float(bx1 - cx1)
    cby1 = float(by1 - cy1)
    cbx2 = float(bx2 - cx1)
    cby2 = float(by2 - cy1)

    return crop, (cbx1, cby1, cbx2, cby2)


def compose_crops_on_canvas(
    crops_and_boxes: list[tuple[int, np.ndarray, tuple[float, float, float, float]]],
    target_size: tuple[int, int] = (640, 640),
    pad_value: int = 114,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Ghép 1 hoặc 2 crop biển báo đã che vào canvas 640x640 và tính lại tọa độ chuẩn YOLO.
    
    crops_and_boxes: danh sách [(class_id, crop_img, (bx1, by1, bx2, by2))] (tối đa 2 item).
    """
    if rng is None:
        rng = random.Random(42)

    tw, th = target_size
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    final_boxes: list[tuple[int, float, float, float, float]] = []

    n = len(crops_and_boxes)
    if n == 0:
        return canvas, final_boxes

    if n == 1:
        # 1 crop: scale và đặt ngẫu nhiên vào trung tâm canvas
        cid, crop, (bx1, by1, bx2, by2) = crops_and_boxes[0]
        ch, cw = crop.shape[:2]
        
        # Scale vừa trong canvas (0.50 đến 0.85 kích thước canvas)
        max_dim = rng.uniform(0.50, 0.85) * min(tw, th)
        scale = max_dim / max(cw, ch)
        rw, rh = int(round(cw * scale)), int(round(ch * scale))
        resized_crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)

        dx = rng.randint(0, max(0, tw - rw))
        dy = rng.randint(0, max(0, th - rh))
        canvas[dy : dy + rh, dx : dx + rw] = resized_crop

        fbx1 = bx1 * scale + dx
        fby1 = by1 * scale + dy
        fbx2 = bx2 * scale + dx
        fby2 = by2 * scale + dy

        fcx = ((fbx1 + fbx2) / 2.0) / tw
        fcy = ((fby1 + fby2) / 2.0) / th
        fbw = (fbx2 - fbx1) / tw
        fbh = (fby2 - fby1) / th
        final_boxes.append((cid, float(fcx), float(fcy), float(fbw), float(fbh)))

    else:
        # 2 crops: ghép 2 vùng không chồng lấn (trái-phải hoặc trên-dưới)
        split_dir = rng.choice(["horizontal", "vertical"])
        
        if split_dir == "horizontal":
            slots = [
                (0, 0, tw // 2 - 10, th),           # Nửa trái
                (tw // 2 + 10, 0, tw // 2 - 10, th), # Nửa phải
            ]
        else:
            slots = [
                (0, 0, tw, th // 2 - 10),           # Nửa trên
                (0, th // 2 + 10, tw, th // 2 - 10), # Nửa dưới
            ]

        for i, (cid, crop, (bx1, by1, bx2, by2)) in enumerate(crops_and_boxes[:2]):
            sx, sy, sw, sh = slots[i]
            ch, cw = crop.shape[:2]
            
            # Scale vừa vào slot (0.7 đến 0.95 slot)
            scale = min((sw * 0.92) / cw, (sh * 0.92) / ch)
            rw = max(10, int(round(cw * scale)))
            rh = max(10, int(round(ch * scale)))
            resized_crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)

            dx = sx + (sw - rw) // 2
            dy = sy + (sh - rh) // 2
            canvas[dy : dy + rh, dx : dx + rw] = resized_crop

            fbx1 = bx1 * scale + dx
            fby1 = by1 * scale + dy
            fbx2 = bx2 * scale + dx
            fby2 = by2 * scale + dy

            fcx = ((fbx1 + fbx2) / 2.0) / tw
            fcy = ((fby1 + fby2) / 2.0) / th
            fbw = (fbx2 - fbx1) / tw
            fbh = (fby2 - fby1) / th
            final_boxes.append((cid, float(fcx), float(fcy), float(fbw), float(fbh)))

    return canvas, final_boxes


def augment_occluded_signs_per_class(
    train_samples: list[dict[str, Any]],
    target_boxes_per_class: int = 30,
    target_size: tuple[int, int] = (640, 640),
    pad_value: int = 114,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, list[tuple[int, float, float, float, float]]]]:
    """Sinh đúng `target_boxes_per_class` box biển báo bị che khuất (ở góc/cạnh) cho mỗi class trong tập Train.
    
    Ghép 2 box vào cùng 1 canvas 640x640 để tạo bối cảnh giao thông tự nhiên và tiết kiệm số lượng file.
    Trả về danh sách: [(aug_stem, canvas_image, [(cid, cx, cy, bw, bh), ...])]
    """
    rng = random.Random(seed)
    
    # 1. Gom toàn bộ box theo class từ tập train
    class_candidates: dict[int, list[tuple[Any, tuple[int, float, float, float, float]]]] = defaultdict(list)
    for s in train_samples:
        img_path = s["image"]
        for b in s.get("boxes", []):
            cid = int(b[0])
            class_candidates[cid].append((img_path, b))

    if not class_candidates:
        return []

    # Cache ảnh để tránh đọc lại đĩa liên tục
    img_cache: dict[str, np.ndarray] = {}
    def load_img(p: Any) -> np.ndarray | None:
        key = str(p)
        if key in img_cache:
            return img_cache[key]
        im = cv2.imread(key)
        if im is not None and len(img_cache) < 300:
            img_cache[key] = im
        return im

    # 2. Sinh chính xác `target_boxes_per_class` box cho từng class
    items_to_pair: list[tuple[int, np.ndarray, tuple[float, float, float, float]]] = []
    
    for cid in sorted(class_candidates.keys()):
        cands = class_candidates[cid]
        if not cands:
            continue
        
        boxes_generated_for_class = 0
        pool = []
        attempts = 0
        max_attempts = target_boxes_per_class * 15

        while boxes_generated_for_class < target_boxes_per_class and attempts < max_attempts:
            attempts += 1
            if not pool:
                pool = list(cands)
                rng.shuffle(pool)

            img_path, raw_box = pool.pop()
            src_img = load_img(img_path)
            if src_img is None:
                continue

            # Cắt biển kèm background
            crop_res = extract_sign_crop_with_background(src_img, raw_box, rng)
            if crop_res is None:
                continue
            crop_img, crop_box = crop_res

            # Tạo che khuất 1 phần ở góc hoặc cạnh
            occluded_crop, occ_box = apply_partial_occlusion(crop_img, crop_box, rng)

            items_to_pair.append((cid, occluded_crop, occ_box))
            boxes_generated_for_class += 1

    # 3. Trộn ngẫu nhiên các crop giữa các class và ghép đôi vào canvas 640x640
    rng.shuffle(items_to_pair)

    augmented_results: list[tuple[str, np.ndarray, list[tuple[int, float, float, float, float]]]] = []
    pair_idx = 0

    # Lấy từng 2 item ghép vào 1 canvas
    for i in range(0, len(items_to_pair), 2):
        pair = items_to_pair[i : i + 2]
        canvas, boxes = compose_crops_on_canvas(pair, target_size=target_size, pad_value=pad_value, rng=rng)
        
        cids_str = "_".join(str(b[0]) for b in boxes)
        aug_stem = f"train_occluded_aug_{pair_idx:05d}_c{cids_str}"
        augmented_results.append((aug_stem, canvas, boxes))
        pair_idx += 1

    return augmented_results

