"""Kiểm thử online partial occlusion và xác nhận khôi phục offline preprocess."""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format
from ultralytics.utils.instance import Instances

from ts_alert.config.resolve import resolve_config
from ts_alert.data.preprocess_online import (
    apply_partial_occlusion,
    apply_partial_occlusion_to_boxes,
    build_online_transforms,
    get_online_occlusion_config,
)
from ts_alert.pipelines.faster_rcnn.train import _FasterRCNNDataset
from ts_alert.pipelines.yolo.train import (
    OcclusionDetectionTrainer,
    OcclusionYOLODataset,
    YoloOcclusionTransform,
    YoloPipeline,
    _OcclusionConfigHook,
)


def test_offline_config_restored_and_online_defaults():
    """Offline config không còn occlusion_aug (giữ blur_threshold=20); online tắt mặc định."""
    shared_cfg = resolve_config(None, "local")
    assert "occlusion_aug" not in shared_cfg["data"]
    assert shared_cfg["data"]["blur_threshold"] == 20.0

    for pipeline in ("yolo", "faster_rcnn"):
        cfg = resolve_config(pipeline, "local")
        assert cfg["online_preprocess"]["enabled"] is False
        assert cfg["online_preprocess"]["occlusion_p"] == pytest.approx(0.30)
        assert cfg["online_preprocess"]["occlusion_ratio"] == pytest.approx(0.30)
        assert get_online_occlusion_config(cfg) == {
            "enabled": False,
            "p": 0.0,
            "ratio": 0.0,
        }

    cfg_on = resolve_config("yolo", "local", {"online_preprocess": {"enabled": True}})
    ul_args = build_online_transforms(cfg_on, "ultralytics")
    assert "occlusion_p" not in ul_args
    assert "occlusion_ratio" not in ul_args


def test_apply_partial_occlusion():
    rng = random.Random(42)
    img = np.full((120, 120, 3), 200, dtype=np.uint8)
    orig = img.copy()
    box = (20.0, 20.0, 100.0, 100.0)

    for _ in range(10):
        occ_img, out_box = apply_partial_occlusion(img, box, rng, occlusion_ratio=0.30)
        assert np.array_equal(img, orig), "Không được sửa trực tiếp ảnh đầu vào"
        assert occ_img.shape == (120, 120, 3)
        assert out_box == box
        assert np.abs(occ_img.astype(int) - img.astype(int)).sum() > 0


def test_occlusion_ratio_scales_coverage():
    """Tăng occlusion_ratio phải tăng diện tích vùng che trên bounding box."""
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    box = (20.0, 20.0, 180.0, 180.0)

    changed_small = 0
    changed_large = 0
    for seed in range(20):
        occ_small, _ = apply_partial_occlusion(img, box, random.Random(seed), occlusion_ratio=0.10)
        occ_large, _ = apply_partial_occlusion(img, box, random.Random(seed), occlusion_ratio=0.65)
        changed_small += int(np.any(occ_small != img, axis=2).sum())
        changed_large += int(np.any(occ_large != img, axis=2).sum())

    assert changed_large > changed_small * 2


def test_apply_partial_occlusion_to_boxes_and_guards():
    img = np.full((120, 120, 3), 210, dtype=np.uint8)

    # p=0 trả về đúng ảnh gốc không copy
    skipped = apply_partial_occlusion_to_boxes(img, [[10, 10, 60, 60]], occlusion_p=0.0)
    assert skipped is img

    # Clip box ra ngoài biên và bỏ qua box tiny < 4px
    boxes = [
        [0.0, 0.0, 2.0, 2.0],         # tiny -> bỏ qua
        [-10.0, 10.0, 70.0, 90.0],    # clip về [0, 10, 70, 90] -> hợp lệ
    ]
    out = apply_partial_occlusion_to_boxes(
        img,
        boxes,
        rng=random.Random(7),
        occlusion_p=1.0,
        occlusion_ratio=0.35,
    )
    assert out is not img
    assert np.any(out[:, :70] != img[:, :70])
    assert np.array_equal(out[:, 70:], img[:, 70:])

    # Tham số sai báo ValueError
    with pytest.raises(ValueError):
        apply_partial_occlusion_to_boxes(img, boxes, occlusion_p=1.2)
    with pytest.raises(ValueError):
        get_online_occlusion_config({"online_preprocess": {"enabled": True, "occlusion_ratio": float("nan")}})


def test_faster_rcnn_dataset_online_occlusion(tmp_path: Path):
    """Kiểm tra _FasterRCNNDataset áp dụng che khuất online khi bật và bỏ qua khi tắt."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    raw = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "sample.jpg"), raw)
    (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    ds_off = _FasterRCNNDataset(tmp_path, "train", num_fg_classes=52, occlusion_cfg={"enabled": False, "p": 1.0})
    ds_on = _FasterRCNNDataset(
        tmp_path,
        "train",
        num_fg_classes=52,
        occlusion_cfg={"enabled": True, "p": 1.0, "ratio": 0.45},
    )

    sample_off = ds_off[0]
    sample_on = ds_on[0]

    assert torch.equal(sample_off["boxes"], sample_on["boxes"])
    assert torch.equal(sample_off["labels"], sample_on["labels"])
    assert not torch.equal(sample_off["image"], sample_on["image"])

    # Pickle compatibility cho DataLoader workers trên Windows
    restored = pickle.loads(pickle.dumps(ds_on))
    assert restored.occlusion_p == pytest.approx(1.0)
    assert restored.occlusion_ratio == pytest.approx(0.45)


def test_yolo_occlusion_transform_dataset_and_close_mosaic(tmp_path: Path):
    """Kiểm tra YoloOcclusionTransform, OcclusionYOLODataset, close_mosaic và pickling."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    raw = np.full((320, 320, 3), 230, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "s0.jpg"), raw)
    (lbl_dir / "s0.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    # 1. Transform trực tiếp giữ nguyên instances nhưng thay đổi pixel ảnh
    tf = YoloOcclusionTransform(p=1.0, ratio=0.40)
    instances = Instances(np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32), bbox_format="xywh", normalized=True)
    orig_bboxes = instances.bboxes.copy()
    labels = {"img": raw.copy(), "instances": instances}
    out_labels = tf(labels)
    assert np.allclose(out_labels["instances"].bboxes, orig_bboxes)
    assert out_labels["instances"].normalized is True
    assert out_labels["instances"]._bboxes.format == "xywh"
    assert np.any(out_labels["img"] != raw)

    # 2. OcclusionYOLODataset chèn transform trước Format cả trước & sau close_mosaic
    hyp = get_cfg(overrides={"mosaic": 0.0, "mixup": 0.0, "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})
    ds = OcclusionYOLODataset(
        img_path=str(img_dir),
        imgsz=320,
        augment=True,
        hyp=hyp,
        data={"names": {0: "a", 1: "b"}, "nc": 2},
        occlusion_p=1.0,
        occlusion_ratio=0.40,
    )
    assert isinstance(ds.transforms.transforms[-1], Format)
    assert any(isinstance(t, YoloOcclusionTransform) for t in ds.transforms.transforms[:-1])

    ds.close_mosaic(hyp)
    assert isinstance(ds.transforms.transforms[-1], Format)
    assert any(isinstance(t, YoloOcclusionTransform) for t in ds.transforms.transforms[:-1])

    sample = ds[0]
    assert tuple(sample["img"].shape) == (3, 320, 320)
    assert len(sample["bboxes"]) == 1

    # 3. Picklability cho DataLoader workers & Ultralytics DDP
    assert pickle.loads(pickle.dumps(ds)).occlusion_p == pytest.approx(1.0)
    hook = _OcclusionConfigHook(0.35, 0.45)
    assert pickle.loads(pickle.dumps(hook)).ratio == pytest.approx(0.45)


def test_yolo_pipeline_fit_wires_occlusion_trainer_only_when_enabled(tmp_path: Path):
    """Kiểm tra YoloPipeline.fit chỉ gắn OcclusionDetectionTrainer khi online occlusion bật."""
    processed = tmp_path / "processed"
    (processed / "images" / "train").mkdir(parents=True)
    (processed / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/train\nnc: 1\nnames: {0: a}\n", encoding="utf-8")

    # Disabled: dùng trainer mặc định
    cfg_off = resolve_config(
        "yolo",
        "local",
        {
            "data": {"root": str(processed)},
            "paths": {"outputs_dir": str(tmp_path / "out_off")},
            "model": {"weights": str(tmp_path / "best_off.pt")},
            "eval": {"auto_eval_test": False},
        },
    )
    mock_model_off = MagicMock()
    with (
        patch("ts_alert.pipelines.yolo.train.load_model", return_value=mock_model_off),
        patch.object(YoloPipeline, "best_checkpoint", return_value=tmp_path / "best_off_ckpt.pt"),
        patch("ts_alert.pipelines.yolo.train.shutil.copy2"),
    ):
        YoloPipeline(cfg_off).fit()
    assert "trainer" not in mock_model_off.train.call_args.kwargs
    mock_model_off.add_callback.assert_not_called()

    # Enabled: gắn OcclusionDetectionTrainer + _OcclusionConfigHook
    cfg_on = resolve_config(
        "yolo",
        "local",
        {
            "data": {"root": str(processed)},
            "paths": {"outputs_dir": str(tmp_path / "out_on")},
            "model": {"weights": str(tmp_path / "best_on.pt")},
            "eval": {"auto_eval_test": False},
            "online_preprocess": {"enabled": True, "occlusion_p": 0.4, "occlusion_ratio": 0.5},
        },
    )
    mock_model_on = MagicMock()
    with (
        patch("ts_alert.pipelines.yolo.train.load_model", return_value=mock_model_on),
        patch.object(YoloPipeline, "best_checkpoint", return_value=tmp_path / "best_on_ckpt.pt"),
        patch("ts_alert.pipelines.yolo.train.shutil.copy2"),
    ):
        YoloPipeline(cfg_on).fit()
    assert mock_model_on.train.call_args.kwargs["trainer"] is OcclusionDetectionTrainer
    hook = mock_model_on.add_callback.call_args.args[1]
    assert isinstance(hook, _OcclusionConfigHook)
    assert hook.p == pytest.approx(0.4)
    assert hook.ratio == pytest.approx(0.5)
