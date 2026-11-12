import tempfile
from copy import deepcopy
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure

import analysis.helpers.class_distribution as class_distribution
from ts_alert.data.helpers.augmentation import (
    _read_raw_labels,
    augment_minority_classes,
    compute_adaptive_class_target,
    compute_adaptive_class_targets,
    transform_and_letterbox_crop,
)
from ts_alert.data.preprocess_offline import (
    letterbox_sample,
    list_raw_samples,
    promote_rare_classes_to_train,
    run_preprocess,
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


def test_split_samples_custom_quality_thresholds(tmp_path: Path):
    """Kiểm tra split_samples hỗ trợ truyền tuỳ biến blur_threshold và dark_threshold."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "images").mkdir(parents=True)
    (raw_dir / "labels").mkdir(parents=True)

    samples = []
    for i in range(15):
        stem = f"sample_{i:03d}"
        img_p = raw_dir / "images" / f"{stem}.jpg"
        lbl_p = raw_dir / "labels" / f"{stem}.txt"
        img = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
        if i % 3 == 0:
            # Ảnh mờ (blur nhẹ)
            img = cv2.GaussianBlur(img, (9, 9), 3.0)
        cv2.imwrite(str(img_p), img)
        with open(lbl_p, "w", encoding="utf-8") as f:
            f.write(f"{i % 2} 0.5 0.5 0.3 0.3\n")
        samples.append({"image": img_p, "label": lbl_p, "stem": stem})

    splits = split_samples(
        samples,
        ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        seed=42,
        blur_threshold=50.0,
        dark_threshold=85.0,
    )
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    assert sum(len(v) for v in splits.values()) == len(samples)


def test_promote_rare_classes_to_train_moves_only_matching_images() -> None:
    """Chỉ chuyển ảnh chứa lớp hiếm cần gom, giữ nguyên ảnh cùng cụm còn lại."""
    box_type = tuple[int, float, float, float, float]

    def make_boxes(cid: int, count: int) -> list[box_type]:
        """Tạo số box xác định cho một lớp."""
        return [(cid, 0.5, 0.5, 0.25, 0.25) for _ in range(count)]

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [
            {"stem": "train_common"},
            {"stem": "train_rare"},
            {"stem": "train_only"},
        ],
        "val": [
            {"stem": "val_rare_a", "cluster": "val_cluster"},
            {"stem": "val_rare_b", "cluster": "val_cluster"},
            {"stem": "val_common", "cluster": "val_cluster"},
        ],
        "test": [
            {"stem": "test_rare", "cluster": "test_cluster"},
            {"stem": "test_common", "cluster": "test_cluster"},
        ],
    }
    clean_boxes = {
        "train_common": make_boxes(0, 10),
        "train_rare": make_boxes(1, 6),
        "train_only": make_boxes(2, 4),
        # Sau ảnh đầu tiên, class 1 đã đạt 10 box; ảnh còn lại vẫn phải chuyển.
        "val_rare_a": make_boxes(1, 4),
        "val_rare_b": make_boxes(1, 1) + make_boxes(0, 1),
        "val_common": make_boxes(0, 2),
        "test_rare": make_boxes(1, 2),
        "test_common": make_boxes(0, 3),
    }
    original_splits = deepcopy(splits)
    original_boxes = deepcopy(clean_boxes)

    result, info = promote_rare_classes_to_train(
        splits,
        clean_boxes,
        min_train_boxes=10,
    )

    assert splits == original_splits
    assert clean_boxes == original_boxes
    assert result is not splits
    for name in ("train", "val", "test"):
        assert result[name] is not splits[name]

    assert info["promoted_classes"] == [1]
    assert info["moved_from_val"] == 2
    assert info["moved_from_test"] == 1
    assert info["moved_total"] == 3

    assert [sample["stem"] for sample in result["train"]] == [
        "train_common",
        "train_rare",
        "train_only",
        "val_rare_a",
        "val_rare_b",
        "test_rare",
    ]
    assert [sample["stem"] for sample in result["val"]] == ["val_common"]
    assert [sample["stem"] for sample in result["test"]] == ["test_common"]

    for name in ("val", "test"):
        assert all(
            box[0] != 1
            for sample in result[name]
            for box in clean_boxes[sample["stem"]]
        )

    assert info["class_counts_before"] == {
        "train": {0: 10, 1: 6, 2: 4},
        "val": {0: 3, 1: 5, 2: 0},
        "test": {0: 3, 1: 2, 2: 0},
    }
    assert info["class_counts_after"] == {
        "train": {0: 11, 1: 13, 2: 4},
        "val": {0: 2, 1: 0, 2: 0},
        "test": {0: 3, 1: 0, 2: 0},
    }

    original_stems = [
        sample["stem"] for items in splits.values() for sample in items
    ]
    result_stems = [
        sample["stem"] for items in result.values() for sample in items
    ]
    assert sorted(result_stems) == sorted(original_stems)
    assert len(result_stems) == len(set(result_stems))


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (3, 8),
        (5, 12),
        (19, 48),
        (20, 35),
        (35, 61),
        (49, 86),
        (50, 75),
        (80, 120),
        (99, 148),
        (100, 100),
        (250, 250),
    ],
)
def test_compute_adaptive_class_targets_tiers(
    count: int,
    expected: int,
) -> None:
    """Kiểm tra mọi biên hệ số và quy tắc round của Python."""
    assert compute_adaptive_class_target(count) == expected

    class_counts = {33: count, 44: 100, 51: 250}
    original_counts = dict(class_counts)

    assert compute_adaptive_class_targets(class_counts) == {
        33: expected,
        44: 100,
        51: 250,
    }
    assert class_counts == original_counts
    assert compute_adaptive_class_targets({}) == {}


def test_augment_minority_classes_uncapped_and_no_blur(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sinh đủ 22 ảnh, bảo toàn nhãn và không gọi GaussianBlur."""
    img_dir = tmp_path / "train" / "images"
    lbl_dir = tmp_path / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    rows, columns = np.indices((160, 160))
    checkerboard = (((rows // 8 + columns // 8) % 2) * 255).astype(
        np.uint8
    )
    sharp_image = np.repeat(checkerboard[:, :, None], 3, axis=2)

    original_labels: dict[str, str] = {}
    for cid, count in ((0, 30), (1, 100)):
        for index in range(count):
            stem = f"class_{cid}_{index:03d}"
            assert cv2.imwrite(str(img_dir / f"{stem}.jpg"), sharp_image)
            label_text = f"{cid} 0.5 0.5 0.5 0.5\n"
            (lbl_dir / f"{stem}.txt").write_text(
                label_text,
                encoding="utf-8",
            )
            original_labels[stem] = label_text

    original_blur = cv2.GaussianBlur
    blur_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def capture_blur(*args: Any, **kwargs: Any) -> np.ndarray:
        """Ghi nhận mọi lời gọi GaussianBlur trong toàn bộ augmentation."""
        blur_calls.append((args, kwargs))
        return original_blur(*args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", capture_blur)

    result = augment_minority_classes(
        train_img_dir=img_dir,
        train_lbl_dir=lbl_dir,
        image_size=(160, 160),
        seed=42,
    )

    assert blur_calls == []
    assert result["n_augmented_images"] == 22
    assert result["n_augmented_boxes"] == 22
    assert result["minority_classes_augmented"] == [0]

    augmented_labels = sorted(lbl_dir.glob("*_aug*.txt"))
    augmented_images = sorted(img_dir.glob("*_aug*.jpg"))
    assert len(augmented_labels) == 22
    assert len(augmented_images) == 22
    assert {path.stem for path in augmented_labels} == {
        path.stem for path in augmented_images
    }
    assert len(list(img_dir.glob("*.jpg"))) == 152

    class_counts: dict[int, int] = {}
    for label_path in lbl_dir.glob("*.txt"):
        boxes = _read_raw_labels(label_path)
        for cid, _, _, _, _ in boxes:
            class_counts[cid] = class_counts.get(cid, 0) + 1

    assert class_counts == {0: 52, 1: 100}

    for label_path in augmented_labels:
        boxes = _read_raw_labels(label_path)
        assert len(boxes) == 1
        cid, cx, cy, bw, bh = boxes[0]
        assert cid == 0
        assert 0.0 <= cx - bw / 2 <= cx + bw / 2 <= 1.0
        assert 0.0 <= cy - bh / 2 <= cy + bh / 2 <= 1.0
        assert bw * 160 >= 8
        assert bh * 160 >= 8

        image = cv2.imread(str(img_dir / f"{label_path.stem}.jpg"))
        assert image is not None
        assert image.shape == (160, 160, 3)

    for stem, label_text in original_labels.items():
        assert (lbl_dir / f"{stem}.txt").read_text(
            encoding="utf-8"
        ) == label_text

    class ControlledRandom(random.Random):
        """Cố định hình học và ánh sáng, phát hiện nhánh xác suất ngoài ý muốn."""

        def __init__(self) -> None:
            super().__init__(42)
            self.random_calls = 0

        def random(self) -> float:
            """Ghi nhận lời gọi xác suất vốn không còn được sử dụng."""
            self.random_calls += 1
            return 0.0

        def uniform(self, a: float, b: float) -> float:
            """Giữ vùng cắt, zoom và ánh sáng cố định; từ chối khoảng sigma."""
            if (a, b) == (-18, 18):
                return 0.0
            if (a, b) in (
                (0.6, 2.5),
                (0.85, 1.25),
                (0.88, 1.15),
            ):
                return 1.0
            raise AssertionError(f"Khoảng uniform ngoài dự kiến: {(a, b)}")

    baseline_sharpness = float(
        cv2.Laplacian(checkerboard, cv2.CV_64F).var()
    )
    assert baseline_sharpness > 1_000

    rng = ControlledRandom()
    canvas, final_box = transform_and_letterbox_crop(
        img=sharp_image,
        target_box=(0, 0.5, 0.5, 0.5, 0.5),
        safe_bounds=(0.0, 0.0, 160.0, 160.0),
        rng=rng,
        target_size=(160, 160),
    )

    assert canvas is not None
    assert final_box is not None
    assert final_box == pytest.approx((0, 0.5, 0.5, 0.5, 0.5))
    assert rng.random_calls == 0
    assert blur_calls == []
    assert np.array_equal(canvas, sharp_image)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert sharpness > 1_000
    assert sharpness == pytest.approx(baseline_sharpness)


def test_list_raw_samples_existing_splits_and_stem_collisions(
    tmp_path: Path,
) -> None:
    """Giữ split và đường dẫn gốc, đổi stem trùng và hỗ trợ layout phẳng."""
    raw_dir = tmp_path / "presplit_raw"
    image = np.full((24, 32, 3), 200, dtype=np.uint8)
    label_text = "0 0.5 0.5 0.5 0.5\n"

    # val_dup là tên duy nhất đã được giữ chỗ trước khi đổi tên val/dup.
    split_stems = {
        "train": ("train_unique", "val_dup"),
        "val": ("val_unique", "dup"),
        "test": ("test_unique", "dup"),
    }
    expected_paths: dict[Path, tuple[str, Path]] = {}

    for split, stems in split_stems.items():
        img_dir = raw_dir / "images" / split
        lbl_dir = raw_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for stem in stems:
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"
            assert cv2.imwrite(str(img_path), image)
            lbl_path.write_text(label_text, encoding="utf-8")
            expected_paths[img_path] = (split, lbl_path)

    samples = list_raw_samples(raw_dir)

    assert len(samples) == len(expected_paths)
    assert {sample["image"] for sample in samples} == set(expected_paths)
    assert len({sample["stem"] for sample in samples}) == len(samples)

    samples_by_image = {sample["image"]: sample for sample in samples}
    for img_path, (split, lbl_path) in expected_paths.items():
        sample = samples_by_image[img_path]
        assert sample["split"] == split
        assert sample["label"] == lbl_path
        assert img_path.is_file()
        assert lbl_path.read_text(encoding="utf-8") == label_text

        if img_path.stem != "dup":
            assert sample["stem"] == img_path.stem
        else:
            assert sample["stem"].startswith(f"{split}_dup")
            assert sample["stem"] != "dup"

    # Việc đổi stem đầu ra không đổi tên ảnh hoặc nhãn nguồn.
    assert (raw_dir / "images" / "val" / "dup.jpg").is_file()
    assert (raw_dir / "images" / "test" / "dup.jpg").is_file()

    flat_dir = tmp_path / "flat_raw"
    flat_img_dir = flat_dir / "images"
    flat_lbl_dir = flat_dir / "labels"
    flat_img_dir.mkdir(parents=True)
    flat_lbl_dir.mkdir(parents=True)

    flat_img_path = flat_img_dir / "flat_unique.jpg"
    flat_lbl_path = flat_lbl_dir / "flat_unique.txt"
    assert cv2.imwrite(str(flat_img_path), image)
    flat_lbl_path.write_text(label_text, encoding="utf-8")

    assert list_raw_samples(flat_dir) == [
        {
            "image": flat_img_path,
            "label": flat_lbl_path,
            "stem": "flat_unique",
            "split": None,
        }
    ]


def test_run_preprocess_existing_splits_no_blur_dhash_consolidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giữ split có sẵn, augment không blur và dồn ảnh rò rỉ sau letterbox."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    classes_csv = tmp_path / "classes.csv"
    classes_csv.write_text(
        "id,name,tts_text\n"
        "0,Biển báo phổ biến,Cảnh báo biển báo phổ biến\n"
        "1,Biển báo hiếm,Cảnh báo biển báo hiếm\n",
        encoding="utf-8",
    )

    for split in ("train", "val", "test"):
        (raw_dir / "images" / split).mkdir(parents=True)
        (raw_dir / "labels" / split).mkdir(parents=True)

    def make_pattern(seed: int) -> np.ndarray:
        """Tạo ảnh 320x240 có ô tương phản ngẫu nhiên tái lập được."""
        rng = np.random.default_rng(seed)
        tiles = rng.integers(0, 256, size=(32, 40), dtype=np.uint8)
        gray = cv2.resize(
            tiles,
            (320, 240),
            interpolation=cv2.INTER_NEAREST,
        )
        return np.repeat(gray[:, :, None], 3, axis=2)

    def write_sample(
        split: str,
        stem: str,
        image: np.ndarray,
        labels: str,
    ) -> None:
        """Ghi một cặp ảnh và nhãn vào layout raw đã chia split."""
        img_path = raw_dir / "images" / split / f"{stem}.jpg"
        lbl_path = raw_dir / "labels" / split / f"{stem}.txt"
        assert image.shape == (240, 320, 3)
        assert cv2.imwrite(
            str(img_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        lbl_path.write_text(labels, encoding="utf-8")

    common_image = make_pattern(101)
    common_labels = (
        "0 0.25 0.25 0.20 0.20\n"
        "0 0.75 0.25 0.20 0.20\n"
        "0 0.25 0.75 0.20 0.20\n"
        "0 0.75 0.75 0.20 0.20\n"
    )
    write_sample("train", "train_common", common_image, common_labels)
    write_sample(
        "train",
        "train_rare",
        make_pattern(202),
        "1 0.5 0.5 0.3 0.3\n",
    )
    write_sample(
        "val",
        "val_unique",
        make_pattern(303),
        "0 0.5 0.5 0.3 0.3\n",
    )
    write_sample(
        "test",
        "test_unique",
        make_pattern(404),
        "1 0.5 0.5 0.3 0.3\n",
    )
    write_sample(
        "test",
        "leaked_from_train",
        common_image,
        common_labels,
    )

    cfg: dict[str, Any] = {
        "seed": 42,
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "classes_csv": str(classes_csv),
        },
        "data": {
            "image_size": 640,
            "pad_value": 114,
            "split": {"train": 0.7, "val": 0.2, "test": 0.1},
            "blur_threshold": 50.0,
            "dark_threshold": 85.0,
        },
        "preprocess": {
            "min_box_px": 8,
        },
        "split": {
            "hash_size": 32,
            "hamming_threshold": 100,
        },
    }

    original_blur = cv2.GaussianBlur
    blur_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def capture_blur(*args: Any, **kwargs: Any) -> np.ndarray:
        """Ghi nhận GaussianBlur nếu pipeline vô tình làm mờ ảnh."""
        blur_calls.append((args, kwargs))
        return original_blur(*args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", capture_blur)

    meta = run_preprocess(cfg)

    assert blur_calls == []
    assert meta["split_mode"] == "existing"
    assert meta["aug_blur"] is False
    assert meta["num_classes"] == 2
    assert meta["n_augmented_images"] > 0
    assert meta["leakage_rate"] == 0.0
    assert meta["remaining_leaks"] == {
        "train_vs_val": 0,
        "train_vs_test": 0,
        "val_vs_test": 0,
    }

    # Hai lớp đều đã có trong train: ảnh trùng được chuyển bởi bước dHash.
    assert meta["rare_classes_promoted"]["moved_total"] == 0
    assert (
        processed_dir / "images" / "train" / "leaked_from_train.jpg"
    ).exists()
    assert not (
        processed_dir / "images" / "test" / "leaked_from_train.jpg"
    ).exists()
    assert (
        processed_dir / "labels" / "train" / "leaked_from_train.txt"
    ).exists()
    assert not (
        processed_dir / "labels" / "test" / "leaked_from_train.txt"
    ).exists()

    # Các ảnh độc lập tiếp tục ở đúng split ban đầu.
    assert (
        processed_dir / "images" / "train" / "train_common.jpg"
    ).exists()
    assert (
        processed_dir / "images" / "train" / "train_rare.jpg"
    ).exists()
    assert (
        processed_dir / "images" / "val" / "val_unique.jpg"
    ).exists()
    assert (
        processed_dir / "images" / "test" / "test_unique.jpg"
    ).exists()

    output_images: list[Path] = []
    for split in ("train", "val", "test"):
        split_images = sorted(
            (processed_dir / "images" / split).glob("*.jpg")
        )
        assert len(split_images) == meta["counts"][split]
        output_images.extend(split_images)

        for img_path in split_images:
            image = cv2.imread(str(img_path))
            assert image is not None, str(img_path)
            assert image.shape == (640, 640, 3), str(img_path)
            assert (
                processed_dir / "labels" / split / f"{img_path.stem}.txt"
            ).is_file()

    assert len(output_images) == 5 + meta["n_augmented_images"]
    assert len({path.stem for path in output_images}) == len(output_images)
    assert (processed_dir / "data.yaml").is_file()
    assert (processed_dir / "meta.json").is_file()


def test_plot_class_distribution_full_names_horizontal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giữ đầy đủ tên tiếng Việt trên trục Y ở biểu đồ RAW và PROCESSED."""
    classes = {
        0: (
            "Cấm các loại phương tiện cơ giới đi vào khu vực "
            "được quy định trên biển báo"
        ),
        33: (
            "Đường giao nhau với đường không ưu tiên tại khu vực "
            "có nhiều phương tiện qua lại"
        ),
        44: "Nguy hiểm khác",
        51: (
            "Hướng đi bắt buộc dành cho các phương tiện khi "
            "di chuyển qua khu vực giao nhau"
        ),
    }
    assert len(classes[0]) > 45
    assert len(classes[33]) > 45
    assert len(classes[51]) > 45

    processed_boxes: list[dict[str, Any]] = [
        {"class_id": 33, "split": "train", "stem": "real_a"},
        {"class_id": 33, "split": "train", "stem": "real_b"},
        {
            "class_id": 33,
            "split": "train",
            "stem": "generated",
            "is_augmented": True,
        },
        {"class_id": 0, "split": "train", "stem": "real_c"},
        {"class_id": 44, "split": "val", "stem": "val_a"},
        {"class_id": 44, "split": "test", "stem": "test_a"},
    ]
    raw_boxes = [
        {"class_id": box["class_id"]} for box in processed_boxes
    ]

    snapshots: list[dict[str, Any]] = []
    original_close = class_distribution.plt.close

    def capture_close(fig: Any = None) -> None:
        """Đọc nhãn và thanh ngang trước khi figure được đóng thật."""
        if isinstance(fig, Figure):
            snapshots.append(
                {
                    "axis_count": len(fig.axes),
                    "labels": [
                        label.get_text()
                        for label in fig.axes[0].get_yticklabels()
                    ],
                    "bars": [
                        [
                            (patch.get_width(), patch.get_height())
                            for patch in ax.patches
                        ]
                        for ax in fig.axes
                    ],
                }
            )
        original_close(fig)

    monkeypatch.setattr(class_distribution.plt, "close", capture_close)

    expected_labels = [
        f"[33] {classes[33]}",
        "[44] Nguy hiểm khác",
        f"[00] {classes[0]}",
        f"[51] {classes[51]}",
    ]

    for tag, boxes, expected_axes in (
        ("raw", raw_boxes, 1),
        ("processed", processed_boxes, 3),
    ):
        output = tmp_path / f"class_distribution_{tag}.png"
        previous_snapshots = len(snapshots)

        stats = class_distribution.plot_class_distribution(
            boxes=boxes,
            classes=classes,
            out_path=output,
            tag=tag,
        )

        assert output.exists()
        assert output.stat().st_size > 0
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

        image = cv2.imread(str(output))
        assert image is not None
        assert image.shape[0] > 0
        assert image.shape[1] > 0

        assert stats["total_classes"] == 4
        assert stats["classes_present"] == 3
        assert stats["total_boxes"] == 6
        assert stats["class_counts"] == {33: 3, 0: 1, 44: 2}
        assert stats["class_counts"].get(51, 0) == 0

        assert len(snapshots) == previous_snapshots + 1
        snapshot = snapshots[-1]
        assert snapshot["axis_count"] == expected_axes
        assert snapshot["labels"] == expected_labels

        # Mỗi hàng dùng chiều cao cố định dưới 1; chiều rộng biểu diễn số box.
        for bars in snapshot["bars"]:
            assert all(0.0 < height < 1.0 for _, height in bars)
            assert all(width >= 0.0 for width, _ in bars)

        widths = [
            [width for width, _ in bars]
            for bars in snapshot["bars"]
        ]
        if tag == "raw":
            assert widths == [[3, 2, 1, 0]]
        else:
            assert widths == [
                [2, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]
