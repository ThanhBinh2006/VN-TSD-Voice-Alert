"""FasterRCNNPipeline: vòng lặp huấn luyện PyTorch (DDP qua torchrun).   Phụ trách: Thành viên 2"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_tensor

from ts_alert.config.resolve import get_num_classes, get_processed_root
from ts_alert.eval import metrics as eval_metrics
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.faster_rcnn.infer import FasterRCNNDetector
from ts_alert.pipelines.faster_rcnn.model import build_model
from ts_alert.runtime.batch import get_rank, get_world_size, is_main_process, per_rank_batch
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.io import write_json
from ts_alert.utils.seed import set_seed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_yolo_boxes(label_path: Path, width: int, height: int, num_fg_classes: int):
    """Đọc nhãn YOLO -> boxes pixel xyxy + labels torchvision (class_id + 1, 0 là background)."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if not 0 <= cls_id < num_fg_classes:
                    raise ValueError(f"class_id {cls_id} ngoài khoảng [0, {num_fg_classes}) trong {label_path}")
                xc, yc = float(parts[1]) * width, float(parts[2]) * height
                bw, bh = float(parts[3]) * width, float(parts[4]) * height
                x1, y1 = max(0.0, xc - bw / 2.0), max(0.0, yc - bh / 2.0)
                x2, y2 = min(float(width), xc + bw / 2.0), min(float(height), yc + bh / 2.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class _FasterRCNNDataset(Dataset):
    def __init__(self, processed_root: Path, split: str, num_fg_classes: int):
        self.image_dir = processed_root / "images" / split
        self.label_dir = processed_root / "labels" / split
        self.num_fg_classes = num_fg_classes
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu {split}: {self.image_dir}")
        self.samples = sorted(p for p in self.image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = _read_yolo_boxes(label_path, width, height, self.num_fg_classes)
        return {
            "image": to_tensor(image),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }


def _collate_fn(batch):
    images = [item["image"] for item in batch]
    targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]
    return images, targets


def _targets_to(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


@torch.no_grad()
def _validate_metrics(model, loader, device, amp: bool, num_fg_classes: int) -> dict:
    """Chạy suy luận trên val rồi tính chỉ số chung (metrics.compute_metrics). class_id = nhãn torchvision - 1."""
    model.eval()
    preds, gts = [], []
    for images, targets in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model([img.to(device) for img in images])
        for out, tgt in zip(outputs, targets):
            preds.append([
                {"bbox": b.tolist(), "label": str(int(l) - 1), "class_id": int(l) - 1, "confidence": float(s)}
                for b, l, s in zip(out["boxes"].float().cpu(), out["labels"].cpu(), out["scores"].float().cpu())
            ])
            gts.append([
                {"bbox": b.tolist(), "class_id": int(l) - 1}
                for b, l in zip(tgt["boxes"], tgt["labels"])
            ])
    return eval_metrics.compute_metrics(preds, gts, num_fg_classes)


@torch.no_grad()
def _validate_loss(model, loader, device, amp: bool) -> float:
    """Phương án dự phòng khi compute_metrics chưa xong: loss val (torchvision chỉ trả loss ở train mode)."""
    model.train()
    total = 0.0
    for images, targets in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            loss_dict = model([img.to(device) for img in images], _targets_to(targets, device))
        total += float(sum(loss_dict.values()).item())
    return total / max(1, len(loader))


class FasterRCNNPipeline(BasePipeline):
    name = "faster_rcnn"

    def fit(self) -> Path:
        """Fine-tune Faster R-CNN trên ĐÚNG split của Thành viên 1 (get_processed_root(cfg)). Trả về run_dir."""
        cfg = self.cfg
        tcfg = cfg["train"]
        processed_root = get_processed_root(cfg)
        if not (processed_root / "images" / "train").exists():
            raise FileNotFoundError(
                f"Thiếu dữ liệu train đã tiền xử lý ở {processed_root}. Hãy chạy preprocessing trước khi huấn luyện."
            )

        set_seed(cfg["seed"])
        world_size, rank = get_world_size(), get_rank()
        distributed = world_size > 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if distributed and not torch.cuda.is_available():
            raise RuntimeError("DDP (WORLD_SIZE > 1) cần GPU/NCCL nhưng không tìm thấy CUDA.")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if distributed and not dist.is_initialized():
            dist.init_process_group("nccl")

        try:
            # run_dir chỉ tạo ở rank 0 rồi chia sẻ đường dẫn cho các rank khác
            holder = [None]
            if is_main_process():
                holder[0] = str(make_run_dir(cfg["paths"]["outputs_dir"], self.name, tcfg.get("run_name"), cfg))
            if distributed:
                dist.broadcast_object_list(holder, src=0)
            run_dir = Path(holder[0])

            num_fg = get_num_classes(cfg)
            model = build_model(num_fg + 1)  # +1 background
            model = model.to(device)
            if distributed:
                model = DDP(model, device_ids=[local_rank])
            core = model.module if distributed else model

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=tcfg["lr"],
                momentum=tcfg.get("momentum", 0.9),
                weight_decay=tcfg.get("weight_decay", 0.0005),
            )
            amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)

            train_ds = _FasterRCNNDataset(processed_root, "train", num_fg)
            val_ds = _FasterRCNNDataset(processed_root, "val", num_fg)
            per_gpu_batch = per_rank_batch(tcfg["batch_size"])  # batch_size là TỔNG trên mọi GPU
            workers = cfg.get("runtime", {}).get("workers", 2)

            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
            train_loader = DataLoader(
                train_ds, batch_size=per_gpu_batch, shuffle=(train_sampler is None), sampler=train_sampler,
                collate_fn=_collate_fn, num_workers=workers, pin_memory=device.type == "cuda",
            )
            # Val chỉ chạy ở rank 0 trên toàn bộ tập val
            val_loader = DataLoader(
                val_ds, batch_size=per_gpu_batch, shuffle=False, collate_fn=_collate_fn, num_workers=workers,
            ) if is_main_process() else None

            primary = tcfg.get("primary_metric", "mAP50-95")  # càng cao càng tốt
            patience_limit = int(tcfg.get("early_stopping_patience", 10))
            best_metric, best_epoch, patience = float("-inf"), -1, 0
            best_checkpoint = run_dir / "best.pth"
            history: list[dict] = []
            use_loss_fallback = False

            wandb = None
            if is_main_process() and tcfg.get("require_wandb"):
                import wandb as _wandb
                wandb = _wandb
                wandb.init(project=cfg["project"].get("wandb_project", "TSD-Voice-Alert"),
                           entity=cfg["project"].get("wandb_entity"), config=cfg, dir=str(run_dir))

            for epoch in range(1, int(tcfg["epochs"]) + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                model.train()
                running = 0.0
                for images, targets in train_loader:
                    inputs = [img.to(device) for img in images]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        loss_dict = model(inputs, _targets_to(targets, device))
                        losses = sum(loss_dict.values())
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running += float(losses.item())
                train_loss = running / max(1, len(train_loader))

                stop = torch.zeros(1, device=device)
                if is_main_process():
                    val_scores: dict = {}
                    if not use_loss_fallback:
                        try:
                            val_scores = _validate_metrics(core, val_loader, device, amp, num_fg)
                            metric = float(val_scores[primary])
                        except NotImplementedError:
                            warnings.warn("metrics.compute_metrics chưa hoàn thành -> tạm chọn best theo -val_loss.")
                            use_loss_fallback = True
                    if use_loss_fallback:
                        val_loss = _validate_loss(core, val_loader, device, amp)
                        val_scores = {"val_loss": val_loss}
                        metric = -val_loss

                    if metric > best_metric:
                        best_metric, best_epoch, patience = metric, epoch, 0
                        torch.save(core.state_dict(), best_checkpoint)
                    else:
                        patience += 1
                    if patience >= patience_limit:
                        stop.fill_(1)

                    payload = {"epoch": epoch, "train_loss": train_loss, **val_scores,
                               "primary_metric": primary if not use_loss_fallback else "-val_loss",
                               "best_epoch": best_epoch, "best_metric": best_metric}
                    history.append(payload)
                    write_json(history, run_dir / "metrics_history.json")
                    if wandb is not None:
                        wandb.log(payload, step=epoch)

                if distributed:
                    dist.broadcast(stop, src=0)  # mọi rank dừng cùng lúc, tránh treo
                if stop.item() > 0:
                    break

            if is_main_process():
                if not best_checkpoint.exists():
                    raise RuntimeError("Không có checkpoint nào được lưu (kiểm tra epochs/metric).")
                target = Path(cfg["model"]["weights"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_checkpoint, target)
                write_json({"run_dir": str(run_dir), "best_checkpoint": str(best_checkpoint),
                            "best_epoch": best_epoch, "primary_metric": primary, "best_metric": best_metric,
                            "weights": str(target)}, run_dir / "summary.json")
                if wandb is not None:
                    wandb.finish()

            if distributed:
                dist.barrier()
        finally:
            if distributed and dist.is_initialized():
                dist.destroy_process_group()

        # Đánh giá test sau khi đã đóng nhóm DDP, chỉ rank 0 chạy
        if is_main_process() and cfg["eval"].get("auto_eval_test", False):
            self.evaluate(run_dir)

        return run_dir

    def load_detector(self, weights=None) -> FasterRCNNDetector:
        return FasterRCNNDetector(weights or self.default_weights(), self.cfg)