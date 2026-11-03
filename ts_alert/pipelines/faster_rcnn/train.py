"""FasterRCNNPipeline: vòng lặp huấn luyện PyTorch (DDP qua torchrun).   Phụ trách: Thành viên 2"""
from __future__ import annotations

from pathlib import Path

from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.faster_rcnn.infer import FasterRCNNDetector


def build_optimizer(model, cfg: dict):
    """Khởi tạo optimizer theo cấu hình train (SGD, Adam, AdamW, momentum, weight_decay)."""
    import torch

    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", train_cfg.get("lr0", 0.005)))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 0.0005))
    opt_name = str(train_cfg.get("optimizer", "SGD")).lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:  # sgd or auto
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


class FasterRCNNPipeline(BasePipeline):
    name = "faster_rcnn"

    def fit(self) -> Path:
        """Fine-tune Faster R-CNN trên ĐÚNG split của Thành viên 1 (get_processed_root(cfg)).

        Gợi ý các bước:
          1. set_seed; run_dir = make_run_dir(outputs_dir, "faster_rcnn", cfg["train"]["run_name"], cfg) (chỉ rank 0: is_main_process())
          2. Nếu WORLD_SIZE > 1: init_process_group("nccl"), DistributedSampler, bọc model bằng DDP
          3. DataLoader batch_size = per_rank_batch(cfg["train"]["batch_size"])  <-- KHÔNG dùng nguyên tổng cho mỗi GPU
          4. model = build_model(get_num_classes(cfg) + 1); optimizer = build_optimizer(model, cfg)
          5. Mỗi epoch: train -> đánh giá val -> lưu best.pth (theo cfg["train"]["primary_metric"]) vào run_dir;
             dừng sớm theo cfg["train"]["early_stopping_patience"]; dùng cfg["train"]["amp"] cho mixed precision
          6. copy best.pth sang cfg["model"]["weights"]
          7. nếu cfg["train"]["require_wandb"]: log lên W&B (cfg["project"]); nếu cfg["eval"]["auto_eval_test"]: self.evaluate(run_dir)
        Trả về run_dir.
        """
        # TODO: Thành viên 2
        raise NotImplementedError("TODO: Thành viên 2")

    def load_detector(self, weights=None) -> FasterRCNNDetector:
        return FasterRCNNDetector(weights or self.default_weights(), self.cfg)
