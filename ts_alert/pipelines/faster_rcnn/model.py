"""Khởi tạo Faster R-CNN ResNet-50 FPN (torchvision 0.24.1).   Phụ trách: Thành viên 2"""
from __future__ import annotations


def build_model(num_classes: int):
    """Trả về torchvision fasterrcnn_resnet50_fpn đã thay box predictor.

    num_classes = get_num_classes(cfg) + 1 (số lớp biển báo + background) (background, index 0). Do đó class_id trong DATA_CONTRACT
    = nhãn của torchvision - 1 khi trả kết quả ra ngoài.
    """
    # TODO: Thành viên 2
    raise NotImplementedError("TODO: Thành viên 2")


def load_checkpoint(weights, num_classes: int, device: str = "cpu"):
    """Dựng model bằng build_model rồi nạp state_dict từ file weights."""
    # TODO: Thành viên 2
    raise NotImplementedError("TODO: Thành viên 2")
