"""Kiểm thử tính năng tăng cường che khuất một phần góc, cạnh cho biển báo."""
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from ts_alert.data.helpers.occlusion import (
    apply_partial_occlusion,
    extract_sign_crop_with_background,
    compose_crops_on_canvas,
    augment_occluded_signs_per_class,
)


def test_apply_partial_occlusion():
    rng = random.Random(42)
    crop = np.full((120, 120, 3), 200, dtype=np.uint8)
    box = (20.0, 20.0, 100.0, 100.0)

    # Chạy nhiều lần để test cả mode corner và edge
    for _ in range(10):
        occ_crop, out_box = apply_partial_occlusion(crop, box, rng)
        assert occ_crop.shape == (120, 120, 3)
        assert out_box == box
        # Đảm bảo có pixel bị thay đổi màu do vật che khuất
        diff = np.abs(occ_crop.astype(int) - crop.astype(int)).sum()
        assert diff > 0, "Vật che khuất phải vẽ lên ảnh"


def test_extract_sign_crop_with_background():
    rng = random.Random(42)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Vẽ biển báo giả ở trung tâm
    img[200:260, 300:360] = 255
    # box format: [cid, cx, cy, bw, bh] normalized
    box_norm = (0, 330.0 / 640.0, 230.0 / 480.0, 60.0 / 640.0, 60.0 / 480.0)

    res = extract_sign_crop_with_background(img, box_norm, rng, pad_ratio_min=0.2, pad_ratio_max=0.4)
    assert res is not None
    crop, crop_box = res
    ch, cw = crop.shape[:2]
    cbx1, cby1, cbx2, cby2 = crop_box

    assert 0 <= cbx1 < cbx2 <= cw
    assert 0 <= cby1 < cby2 <= ch
    assert cw > 60 and ch > 60, "Crop phải bao gồm cả padding background xung quanh"


def test_compose_crops_on_canvas_single_and_pair():
    rng = random.Random(42)
    crop1 = np.full((80, 80, 3), 150, dtype=np.uint8)
    crop2 = np.full((100, 90, 3), 180, dtype=np.uint8)
    box1 = (10.0, 10.0, 70.0, 70.0)
    box2 = (15.0, 15.0, 75.0, 75.0)

    # Test 1: Ghép 1 crop
    canvas1, boxes1 = compose_crops_on_canvas([(1, crop1, box1)], target_size=(640, 640), rng=rng)
    assert canvas1.shape == (640, 640, 3)
    assert len(boxes1) == 1
    cid, cx, cy, bw, bh = boxes1[0]
    assert cid == 1
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
    assert 0.0 < bw < 1.0 and 0.0 < bh < 1.0

    # Test 2: Ghép 2 crops
    canvas2, boxes2 = compose_crops_on_canvas([(2, crop1, box1), (3, crop2, box2)], target_size=(640, 640), rng=rng)
    assert canvas2.shape == (640, 640, 3)
    assert len(boxes2) == 2
    for cid, cx, cy, bw, bh in boxes2:
        assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
        assert 0.0 < bw < 1.0 and 0.0 < bh < 1.0
        # Box không được tràn mép
        assert cx - bw / 2.0 >= 0.0
        assert cx + bw / 2.0 <= 1.0
        assert cy - bh / 2.0 >= 0.0
        assert cy + bh / 2.0 <= 1.0


def test_augment_occluded_signs_per_class(tmp_path):
    # Tạo ảnh mẫu và dữ liệu train giả lập gồm 2 class (0 và 1)
    img_dir = tmp_path / "raw_images"
    img_dir.mkdir()
    
    img1_path = img_dir / "sample1.jpg"
    img2_path = img_dir / "sample2.jpg"
    
    dummy_img = np.full((400, 400, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img1_path), dummy_img)
    cv2.imwrite(str(img2_path), dummy_img)

    train_samples = [
        {
            "image": img1_path,
            "boxes": [
                [0, 0.3, 0.3, 0.2, 0.2],  # class 0
                [1, 0.7, 0.7, 0.15, 0.15], # class 1
            ]
        },
        {
            "image": img2_path,
            "boxes": [
                [0, 0.5, 0.5, 0.25, 0.25], # class 0
            ]
        }
    ]

    target_per_class = 6  # Test với số lượng 6 box mỗi class
    results = augment_occluded_signs_per_class(
        train_samples=train_samples,
        target_boxes_per_class=target_per_class,
        target_size=(640, 640),
        seed=42,
    )

    # Đếm số lượng box của từng class được tạo ra
    class_box_counts = {0: 0, 1: 0}
    for stem, canvas, boxes in results:
        assert canvas.shape == (640, 640, 3)
        assert len(boxes) in (1, 2)
        for b in boxes:
            cid = int(b[0])
            class_box_counts[cid] += 1

    # Kiểm tra bảo đảm đủ 6 box cho mỗi class
    assert class_box_counts[0] == target_per_class, f"Class 0 phải đủ {target_per_class} box, thực tế: {class_box_counts[0]}"
    assert class_box_counts[1] == target_per_class, f"Class 1 phải đủ {target_per_class} box, thực tế: {class_box_counts[1]}"


def test_augment_occluded_signs_with_clean_boxes(tmp_path):
    """Kiểm tra augment_occluded_signs_per_class với train_samples chuẩn pipeline (không chứa key 'boxes')."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    p1 = img_dir / "img1.jpg"
    p2 = img_dir / "img2.jpg"
    im = np.full((300, 300, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(p1), im)
    cv2.imwrite(str(p2), im)

    train_samples = [
        {"image": p1, "label": None, "stem": "img1"},
        {"image": p2, "label": None, "stem": "img2"},
    ]
    clean_boxes = {
        "img1": [(0, 0.4, 0.4, 0.2, 0.2)],
        "img2": [(1, 0.5, 0.5, 0.25, 0.25)],
    }

    results = augment_occluded_signs_per_class(
        train_samples=train_samples,
        target_boxes_per_class=4,
        clean_boxes=clean_boxes,
        seed=42,
    )
    counts = {0: 0, 1: 0}
    for stem, canvas, boxes in results:
        for b in boxes:
            counts[b[0]] += 1
    assert counts[0] == 4
    assert counts[1] == 4


