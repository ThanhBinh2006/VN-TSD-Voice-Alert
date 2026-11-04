"""Khởi tạo Faster R-CNN ResNet-50 FPN (torchvision 0.24.1).   Phụ trách: Thành viên 2"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, pretrained: bool = True):
    """Trả về torchvision fasterrcnn_resnet50_fpn đã thay box predictor.

    num_classes = get_num_classes(cfg) + 1 (số lớp biển báo + background, background = index 0).
    Do đó class_id trong DATA_CONTRACT = nhãn của torchvision - 1 khi trả kết quả ra ngoài.

    pretrained=True: khởi tạo từ trọng số COCO (dùng khi fine-tune). Nếu không tải được (không có mạng)
    thì cảnh báo và huấn luyện từ đầu. pretrained=False: không tải gì cả (dùng khi nạp checkpoint).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes phải >= 2 (ít nhất 1 lớp + background), nhận {num_classes}")

    model = None
    if pretrained:
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception as exc:  # không có mạng / lỗi cache
            warnings.warn(f"Không tải được trọng số COCO ({exc}); khởi tạo ngẫu nhiên.")
    if model is None:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(weights, num_classes: int, device: str = "cpu"):
    """Dựng model bằng build_model rồi nạp state_dict từ file weights."""
    device = torch.device(device)
    model = build_model(num_classes, pretrained=False)

    weight_path = Path(weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Faster R-CNN: {weights}")

    state = torch.load(weight_path, map_location=device)

    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break

    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model