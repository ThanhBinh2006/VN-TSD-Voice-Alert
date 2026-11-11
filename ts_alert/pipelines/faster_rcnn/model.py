from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError

import torch
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(
    num_classes: int,
    pretrained: bool = True,
) -> FasterRCNN:
    """Tạo Faster R-CNN; số lớp bao gồm lớp nền."""
    if num_classes < 2:
        raise ValueError("num_classes phải >= 2, bao gồm lớp nền.")

    if pretrained:
        try:
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            )
        except (OSError, URLError) as exc:
            warnings.warn(
                f"Không tải được trọng số pretrained: {exc}. "
                "Khởi tạo mô hình không có trọng số pretrained.",
                RuntimeWarning,
                stacklevel=2,
            )
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
            )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )
    return model


def load_checkpoint(
    weights: str | Path,
    num_classes: int,
    device: str = "cpu",
) -> FasterRCNN:
    """Nạp checkpoint và chuyển mô hình sang chế độ suy luận."""
    weight_path = Path(weights).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tệp trọng số: {weight_path}"
        )

    model = build_model(num_classes, pretrained=False)
    checkpoint = torch.load(weight_path, map_location=device)

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint phải là state_dict hoặc dictionary.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict phải là một mapping.")

    normalized_state_dict = OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state_dict.items()
    )
    model.load_state_dict(normalized_state_dict, strict=True)
    return model.to(device).eval()
