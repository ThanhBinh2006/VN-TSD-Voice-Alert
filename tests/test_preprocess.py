import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from ts_alert.data.preprocess_offline import (
    letterbox_sample,
    list_raw_samples,
    split_samples,
    write_data_yaml,
)


def test_letterbox_sample(tmp_path: Path):
    """Kiểm tra letterbox_sample giữ nguyên tỉ lệ, pad 114, chuyển đổi toạ độ YOLO chính xác."""
    # Tạo ảnh giả lập 800x400 (tỉ lệ 2:1)
    img_dir = tmp_path / "raw" / "images"
    lbl_dir = tmp_path / "raw" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    test_img = np.ones((400, 800, 3), dtype=np.uint8) * 200
    img_path = img_dir / "test_sample.jpg"
    cv2.imwrite(str(img_path), test_img)

    # Box ở giữa ảnh: class 0, cx=0.5, cy=0.5, w=0.25 (200px), h=0.25 (100px)
    lbl_path = lbl_dir / "test_sample.txt"
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")

    sample = {"image": img_path, "label": lbl_path, "stem": "test_sample"}
    out_dir = tmp_path / "processed"
    letterbox_sample(sample, out_dir, split="train", image_size=640, pad_value=114)

    # Kiểm tra ảnh đầu ra
    out_img_path = out_dir / "images" / "train" / "test_sample.jpg"
    assert out_img_path.exists()
    out_img = cv2.imread(str(out_img_path))
    assert out_img.shape == (640, 640, 3)

    # Kiểm tra padding màu 114 trên và dưới (vì ảnh rộng 800, cao 400 -> scale=0.8 -> nw=640, nh=320 -> pad dy=160)
    assert np.all(out_img[0:150, :, :] == 114)
    assert np.all(out_img[490:640, :, :] == 114)

    # Kiểm tra nhãn đầu ra
    out_lbl_path = out_dir / "labels" / "train" / "test_sample.txt"
    assert out_lbl_path.exists()
    with open(out_lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    parts = list(map(float, lines[0].strip().split()))
    cid, cx, cy, bw, bh = int(parts[0]), parts[1], parts[2], parts[3], parts[4]
    assert cid == 0
    assert abs(cx - 0.5) < 1e-3
    assert abs(cy - 0.5) < 1e-3
    # bw ban đầu = 200px -> trên 640 là 200*0.8/640 = 0.25
    assert abs(bw - 0.25) < 1e-3
    # bh ban đầu = 100px -> trên 640 là 100*0.8/640 = 0.125
    assert abs(bh - 0.125) < 1e-3


def test_split_samples_no_leakage(tmp_path: Path):
    """Kiểm tra split_samples phân chia không có rò rỉ (không có ảnh trùng giữa các tập)."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "images").mkdir(parents=True)
    (raw_dir / "labels").mkdir(parents=True)

    # Tạo 20 ảnh mẫu
    samples = []
    for i in range(20):
        stem = f"sample_{i:03d}"
        img_p = raw_dir / "images" / f"{stem}.jpg"
        lbl_p = raw_dir / "labels" / f"{stem}.txt"
        cv2.imwrite(str(img_p), np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        with open(lbl_p, "w", encoding="utf-8") as f:
            f.write(f"{i % 3} 0.5 0.5 0.4 0.4\n")
        samples.append({"image": img_p, "label": lbl_p, "stem": stem})

    splits = split_samples(samples, ratios={"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)

    tr_stems = {s["stem"] for s in splits["train"]}
    va_stems = {s["stem"] for s in splits["val"]}
    te_stems = {s["stem"] for s in splits["test"]}

    assert tr_stems.isdisjoint(va_stems), "Rò rỉ giữa train và val!"
    assert tr_stems.isdisjoint(te_stems), "Rò rỉ giữa train và test!"
    assert va_stems.isdisjoint(te_stems), "Rò rỉ giữa val và test!"
    assert len(tr_stems) + len(va_stems) + len(te_stems) == len(samples)


def test_write_data_yaml(tmp_path: Path):
    """Kiểm tra sinh file data.yaml đúng chuẩn Ultralytics YOLO."""
    names = ["bien_1", "bien_2", "bien_3"]
    write_data_yaml(tmp_path, names)
    yaml_p = tmp_path / "data.yaml"
    assert yaml_p.exists()
    content = yaml_p.read_text(encoding="utf-8")
    assert "nc: 3" in content
    assert "bien_1" in content
    assert "images/train" in content
