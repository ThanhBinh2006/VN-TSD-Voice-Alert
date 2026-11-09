"""Huấn luyện Faster R-CNN trên CPU, một GPU hoặc DDP."""

from __future__ import annotations

import math
import os
import random
import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

from ts_alert.config.resolve import get_num_classes, get_processed_root, resolve_path
from ts_alert.data.preprocess_online import (
    apply_partial_occlusion_to_boxes,
    build_online_transforms,
    get_online_occlusion_config,
)
from ts_alert.eval.metrics import compute_metrics
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.faster_rcnn.infer import FasterRCNNDetector
from ts_alert.pipelines.faster_rcnn.model import build_model
from ts_alert.runtime.batch import per_rank_batch
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.types import IMAGE_EXTS
from ts_alert.utils.io import write_json, write_yaml
from ts_alert.utils.seed import set_seed

TensorDict = dict[str, torch.Tensor]
Batch = tuple[list[torch.Tensor], list[TensorDict]]


def _read_yolo_boxes(
    label_path: Path | str,
    width: int,
    height: int,
    num_fg_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Đọc nhãn YOLO, clip bbox và chuyển lớp foreground sang chỉ số từ 1."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    path = Path(label_path)

    if path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("Mỗi dòng phải có đúng 5 giá trị.")

                cid = int(fields[0])
                cx, cy, bw, bh = map(float, fields[1:])

                if not 0 <= cid < num_fg_classes:
                    raise ValueError("class_id nằm ngoài danh mục lớp.")
                if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                    raise ValueError("Tọa độ phải hữu hạn.")
                if bw <= 0 or bh <= 0:
                    raise ValueError("Kích thước bbox phải dương.")

                x1 = max(0.0, min(float(width), (cx - bw / 2) * width))
                y1 = max(0.0, min(float(height), (cy - bh / 2) * height))
                x2 = max(0.0, min(float(width), (cx + bw / 2) * width))
                y2 = max(0.0, min(float(height), (cy + bh / 2) * height))

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cid + 1)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Nhãn lỗi tại {path}:{line_number}: {exc}"
                ) from exc

    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.int64),
    )


class _FasterRCNNDataset(Dataset[TensorDict]):
    """Dataset cấp module, hỗ trợ pickle và augment đồng bộ ảnh với bbox."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        num_fg_classes: int,
        transform: Any = None,
        occlusion_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Lưu đường dẫn và cấu hình; không giữ handle ảnh hoặc RNG riêng."""
        self.root = Path(root)
        self.split = split
        self.num_fg_classes = num_fg_classes
        self.transform = transform

        image_dir = self.root / "images" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )

        occlusion = occlusion_cfg or {}
        self.occlusion_p = (
            float(occlusion.get("p", 0.30))
            if occlusion.get("enabled", False)
            else 0.0
        )
        self.occlusion_ratio = float(occlusion.get("ratio", 0.30))

    def __len__(self) -> int:
        """Trả số ảnh trong split."""
        return len(self.images)

    def _augment(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        operation: Any,
    ) -> tuple[Image.Image, torch.Tensor]:
        """Áp dụng Compose, lật ngang có sửa bbox hoặc ColorJitter."""
        if isinstance(operation, transforms.Compose):
            for child in operation.transforms:
                image, boxes = self._augment(image, boxes, child)
        elif isinstance(operation, transforms.RandomHorizontalFlip):
            if random.random() < operation.p:
                width = image.width
                image = TF.hflip(image)
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        elif isinstance(operation, transforms.ColorJitter):
            image = operation(image)
        else:
            raise TypeError(
                f"Transform chưa hỗ trợ đồng bộ bbox: "
                f"{type(operation).__name__}"
            )

        return image, boxes

    def __getitem__(self, index: int) -> TensorDict:
        """Đọc ảnh RGB và nhãn; che khuất giữ nguyên bbox, labels."""
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert("RGB")

        boxes, labels = _read_yolo_boxes(
            self.root / "labels" / self.split / f"{path.stem}.txt",
            image.width,
            image.height,
            self.num_fg_classes,
        )

        if self.transform is not None:
            image, boxes = self._augment(image, boxes, self.transform)

        pixels = np.array(image, dtype=np.uint8, copy=True)
        if self.occlusion_p > 0:
            pixels = apply_partial_occlusion_to_boxes(
                pixels,
                boxes.numpy(),
                occlusion_p=self.occlusion_p,
                occlusion_ratio=self.occlusion_ratio,
            )

        return {
            "image": TF.to_tensor(pixels),
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes[:, 2:] - boxes[:, :2]).prod(dim=1),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }


def _collate_fn(samples: list[TensorDict]) -> Batch:
    """Gom ảnh kích thước khác nhau thành danh sách cho detector."""
    return (
        [sample["image"] for sample in samples],
        [
            {key: value for key, value in sample.items() if key != "image"}
            for sample in samples
        ],
    )


def _targets_to(
    targets: list[TensorDict],
    device: torch.device,
) -> list[TensorDict]:
    """Chuyển toàn bộ tensor nhãn sang thiết bị huấn luyện."""
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]


def _seed_worker(worker_id: int) -> None:
    """Đồng bộ seed Python và NumPy với seed worker do PyTorch cấp."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def _validate_metrics(
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    num_classes: int,
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> dict[str, float]:
    """Tính mAP, AR100, precision, recall và F1 với lớp nền được loại bỏ."""
    previous = {module: module.training for module in model.modules()}
    predictions: list[list[dict[str, Any]]] = []
    targets: list[list[dict[str, Any]]] = []

    try:
        model.eval()
        for images, batch_targets in loader:
            outputs = model(
                [image.to(device, non_blocking=True) for image in images]
            )
            for output, target in zip(outputs, batch_targets):
                boxes = output["boxes"].cpu().tolist()
                labels = output["labels"].cpu().tolist()
                scores = output["scores"].cpu().tolist()

                predictions.append([
                    {
                        "bbox": box,
                        "class_id": int(label) - 1,
                        "confidence": float(score),
                    }
                    for box, label, score in zip(boxes, labels, scores)
                    if 1 <= label <= num_classes
                ])
                targets.append([
                    {"bbox": box, "class_id": int(label) - 1}
                    for box, label in zip(
                        target["boxes"].tolist(),
                        target["labels"].tolist(),
                    )
                ])

        return compute_metrics(
            predictions,
            targets,
            num_classes,
            iou_threshold,
            conf_threshold,
        )
    finally:
        for module, training in previous.items():
            module.training = training


@torch.no_grad()
def _validate_loss(
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
) -> float:
    """Đo loss ở chế độ train, đóng băng BatchNorm và bảo toàn trạng thái."""
    previous = {module: module.training for module in model.modules()}
    total = 0.0
    count = 0

    try:
        # Validation không làm thay đổi RNG dùng cho huấn luyện.
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            model.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

            for images, targets in loader:
                losses = model(
                    [image.to(device, non_blocking=True) for image in images],
                    _targets_to(targets, device),
                )
                loss = float(sum(losses.values()).item())
                if not math.isfinite(loss):
                    raise FloatingPointError("Validation loss không hữu hạn.")

                total += loss * len(images)
                count += len(images)

        if count == 0:
            raise ValueError("Tập validation không có ảnh.")

        return total / count
    finally:
        for module, training in previous.items():
            module.training = training


def build_optimizer(
    model: nn.Module,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    """Tạo SGD, Adam hoặc AdamW; tách bias để hỗ trợ warmup bias LR."""
    train = cfg.get("train", cfg)
    lr = float(train.get("lr", train.get("lr0", 0.005)))
    momentum = float(train.get("momentum", 0.9))
    decay = float(train.get("weight_decay", 0.0005))

    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate phải hữu hạn và dương.")

    weights: list[nn.Parameter] = []
    biases: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (biases if name.endswith(".bias") else weights).append(parameter)

    groups = [
        {"params": weights, "is_bias": False},
        {"params": biases, "is_bias": True},
    ]
    name = str(train.get("optimizer", "SGD")).lower()

    if name == "sgd":
        return torch.optim.SGD(
            groups,
            lr=lr,
            momentum=momentum,
            weight_decay=decay,
        )

    if name in {"adam", "adamw"}:
        optimizer = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return optimizer(
            groups,
            lr=lr,
            betas=(momentum, 0.999),
            weight_decay=decay,
        )

    raise ValueError(f"Optimizer không hợp lệ: {name!r}")


class FasterRCNNPipeline(BasePipeline):
    """Quản lý huấn luyện, checkpoint, tracking và đánh giá Faster R-CNN."""

    name = "faster_rcnn"

    def load_detector(
        self,
        weights: Path | str | None = None,
    ) -> FasterRCNNDetector:
        """Nạp detector từ checkpoint chỉ định hoặc trọng số mặc định."""
        return FasterRCNNDetector(
            weights if weights is not None else self.default_weights(),
            self.cfg,
        )

    def fit(self) -> Path:
        """Huấn luyện đồng bộ các rank; chỉ rank 0 ghi tệp và đánh giá."""
        # Import muộn để dataset vẫn có thể được import/pickle độc lập.
        from ts_alert.pipelines.tracking import (
            format_epoch_line,
            init_wandb_run,
            log_epoch_to_wandb,
            log_final_to_wandb,
            plot_training_curves,
            print_best_banner,
            save_epoch_artifacts,
            standardize_epoch_payload,
        )

        cfg = self.cfg
        initial_root = get_processed_root(cfg)
        if not (initial_root / "images" / "train").is_dir():
            raise FileNotFoundError(initial_root / "images" / "train")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed = world_size > 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        runtime_cfg = cfg.get("runtime")
        requested = str(
            runtime_cfg.get("device", cfg.get("device", "auto"))
            if isinstance(runtime_cfg, Mapping)
            else cfg.get("device", "auto")
        )
        use_cuda = torch.cuda.is_available() and requested != "cpu"
        device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

        if use_cuda:
            torch.cuda.set_device(device)

        created_group = False
        wandb_run: Any = None
        owns_wandb = False
        run_dir: Path | None = None
        rank = 0
        summary: dict[str, Any] = {}

        try:
            if distributed:
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl" if use_cuda and os.name != "nt" else "gloo",
                        init_method="env://",
                    )
                    created_group = True

                world_size = dist.get_world_size()
                rank = dist.get_rank()

            def synchronize_error(error: str | None) -> None:
                """Truyền lỗi của rank 0 trước khi rank khác bước vào epoch."""
                message: list[Any] = [error]
                if distributed:
                    dist.broadcast_object_list(message, src=0)
                if message[0] is not None:
                    raise RuntimeError(message[0])

            try:
                setup_error: str | None = None
                if rank == 0:
                    try:
                        run_dir = make_run_dir(
                            resolve_path(cfg["paths"]["outputs_dir"]),
                            self.name,
                            cfg["train"].get("run_name"),
                            cfg,
                        )
                        wandb_run, owns_wandb = init_wandb_run(cfg, run_dir)
                    except Exception as exc:
                        setup_error = (
                            f"Khởi tạo run thất bại: {type(exc).__name__}"
                        )

                synchronize_error(setup_error)

                shared: list[Any] = [
                    str(run_dir) if rank == 0 else None,
                    cfg if rank == 0 else None,
                ]
                if distributed:
                    dist.broadcast_object_list(shared, src=0)

                run_dir = Path(shared[0])
                cfg = shared[1]
                self.cfg = cfg
                set_seed(int(cfg["seed"]))

                root = get_processed_root(cfg)
                if not (root / "images" / "train").is_dir():
                    raise FileNotFoundError(root / "images" / "train")

                train_cfg = cfg["train"]
                total_epochs = int(train_cfg["epochs"])
                if total_epochs < 1:
                    raise ValueError("train.epochs phải lớn hơn 0.")

                batch_size = per_rank_batch(
                    int(train_cfg["batch_size"]),
                    world_size,
                )
                if batch_size < 1:
                    raise ValueError("Batch mỗi rank phải lớn hơn 0.")

                num_classes = get_num_classes(cfg)
                train_ds = _FasterRCNNDataset(
                    root,
                    "train",
                    num_classes,
                    transform=build_online_transforms(cfg, "torchvision"),
                    occlusion_cfg=get_online_occlusion_config(cfg),
                )
                if not len(train_ds):
                    raise ValueError("Tập train không có ảnh.")

                runtime_cfg = cfg.get("runtime")
                default_workers = cfg.get(
                    "workers",
                    train_cfg.get("num_workers", 0),
                )
                workers = int(
                    runtime_cfg.get(
                        "workers",
                        runtime_cfg.get("num_workers", default_workers),
                    )
                    if isinstance(runtime_cfg, Mapping)
                    else default_workers
                )

                loader_kwargs: dict[str, Any] = {
                    "num_workers": workers,
                    "pin_memory": use_cuda,
                    "collate_fn": _collate_fn,
                    "worker_init_fn": _seed_worker,
                }
                if workers > 0:
                    loader_kwargs["multiprocessing_context"] = "spawn"

                sampler = (
                    DistributedSampler(
                        train_ds,
                        num_replicas=world_size,
                        rank=rank,
                        seed=int(cfg["seed"]),
                    )
                    if distributed
                    else None
                )
                generator = torch.Generator().manual_seed(
                    int(cfg["seed"]) + rank
                )
                train_loader: DataLoader[Batch] = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=sampler is None,
                    sampler=sampler,
                    generator=generator,
                    **loader_kwargs,
                )

                val_loader: DataLoader[Batch] | None = None
                validation_error: str | None = None
                if rank == 0:
                    try:
                        val_ds = _FasterRCNNDataset(root, "val", num_classes)
                        if not len(val_ds):
                            raise ValueError("Tập validation không có ảnh.")

                        val_loader = DataLoader(
                            val_ds,
                            batch_size=int(cfg["eval"]["batch_size"]),
                            shuffle=False,
                            generator=torch.Generator().manual_seed(
                                int(cfg["seed"])
                            ),
                            **loader_kwargs,
                        )
                        write_yaml(cfg, run_dir / "config.yaml")
                    except Exception as exc:
                        validation_error = (
                            f"Không thể tạo validation loader: {exc}"
                        )

                synchronize_error(validation_error)

                core = build_model(
                    num_classes + 1,
                    pretrained=bool(cfg["model"].get("pretrained", True)),
                ).to(device)
                iou_thresh = float(
                    (cfg.get("inference") or {}).get("iou_threshold", 0.5)
                )
                if (
                    0.0 < iou_thresh <= 1.0
                    and hasattr(core, "roi_heads")
                    and hasattr(core.roi_heads, "nms_thresh")
                ):
                    core.roi_heads.nms_thresh = iou_thresh

                model: nn.Module = core
                if distributed:
                    model = DDP(
                        core,
                        device_ids=[device.index] if use_cuda else None,
                        broadcast_buffers=True,
                    )

                optimizer = build_optimizer(core, cfg)
                amp_enabled = bool(train_cfg.get("amp", True)) and use_cuda
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

                steps_per_epoch = len(train_loader)
                total_steps = total_epochs * steps_per_epoch
                warmup_steps = min(
                    float(train_cfg.get("warmup_epochs", 3.0))
                    * steps_per_epoch,
                    max(0, total_steps - 1),
                )
                base_lrs = [
                    float(group["lr"])
                    for group in optimizer.param_groups
                ]
                momentum = float(train_cfg.get("momentum", 0.9))
                warmup_momentum = float(
                    train_cfg.get("warmup_momentum", 0.8)
                )
                warmup_bias_lr = float(train_cfg.get("warmup_bias_lr", 0.1))
                minimum_factor = float(train_cfg.get("lrf", 0.01))

                def schedule(step: int) -> None:
                    """Cập nhật LR theo bước: warmup tuyến tính rồi cosine."""
                    progress = max(0.0, step - warmup_steps) / max(
                        1.0,
                        total_steps - 1 - warmup_steps,
                    )
                    factor = minimum_factor + (1.0 - minimum_factor) * (
                        1.0 + math.cos(math.pi * min(1.0, progress))
                    ) / 2.0
                    fraction = (
                        min(1.0, step / warmup_steps)
                        if warmup_steps > 0
                        else 1.0
                    )

                    for group, base_lr in zip(
                        optimizer.param_groups,
                        base_lrs,
                    ):
                        start_lr = (
                            warmup_bias_lr if group["is_bias"] else 0.0
                        )
                        group["lr"] = (
                            start_lr + (base_lr - start_lr) * fraction
                            if step < warmup_steps
                            else base_lr * factor
                        )
                        current_momentum = (
                            warmup_momentum
                            + (momentum - warmup_momentum) * fraction
                        )
                        if "momentum" in group:
                            group["momentum"] = current_momentum
                        if "betas" in group:
                            group["betas"] = (
                                current_momentum,
                                group["betas"][1],
                            )

                primary = str(
                    train_cfg.get("primary_metric", "mAP50-95")
                ).removeprefix("val/")
                best_metric = -math.inf
                best_epoch = 0
                stale_epochs = 0
                patience = int(
                    train_cfg.get("early_stopping_patience", 10)
                )
                history: list[dict[str, Any]] = []
                inference_cfg = cfg.get("inference", {})

                for epoch in range(1, total_epochs + 1):
                    if sampler is not None:
                        sampler.set_epoch(epoch)

                    model.train()
                    accumulated = torch.zeros(
                        2,
                        dtype=torch.float64,
                        device=device,
                    )

                    for batch_index, (images, targets) in enumerate(
                        train_loader
                    ):
                        schedule(
                            (epoch - 1) * steps_per_epoch + batch_index
                        )
                        optimizer.zero_grad(set_to_none=True)

                        with torch.autocast(
                            device_type=device.type,
                            enabled=amp_enabled,
                        ):
                            loss_parts = model(
                                [
                                    image.to(device, non_blocking=True)
                                    for image in images
                                ],
                                _targets_to(targets, device),
                            )
                            loss = sum(loss_parts.values())

                        finite = torch.isfinite(loss.detach()).to(
                            torch.int32
                        )
                        if distributed:
                            dist.all_reduce(
                                finite,
                                op=dist.ReduceOp.MIN,
                            )
                        if not bool(finite.item()):
                            raise FloatingPointError(
                                "Train loss không hữu hạn."
                            )

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        accumulated[0] += (
                            loss.detach().double() * len(images)
                        )
                        accumulated[1] += len(images)

                    if distributed:
                        dist.all_reduce(accumulated)
                        # Validation dùng core, không gọi DDP trên riêng rank 0.
                        for buffer in core.buffers():
                            dist.broadcast(buffer, src=0)

                    epoch_error: str | None = None
                    stop = False

                    if rank == 0:
                        try:
                            assert val_loader is not None
                            val_loss = _validate_loss(
                                core,
                                val_loader,
                                device,
                            )
                            scores: dict[str, float] = {}

                            try:
                                scores = _validate_metrics(
                                    core,
                                    val_loader,
                                    device,
                                    num_classes,
                                    float(
                                        inference_cfg.get(
                                            "iou_threshold",
                                            0.5,
                                        )
                                    ),
                                    float(
                                        inference_cfg.get(
                                            "conf_threshold",
                                            0.25,
                                        )
                                    ),
                                )
                                if primary != "neg_loss" and (
                                    primary not in scores
                                    or not math.isfinite(
                                        float(scores[primary])
                                    )
                                ):
                                    raise ValueError(
                                        "Primary metric không hợp lệ."
                                    )
                            except Exception as exc:
                                warnings.warn(
                                    f"Validation metric lỗi "
                                    f"({type(exc).__name__}); "
                                    "chuyển sang tối ưu -val_loss.",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                                if primary != "neg_loss":
                                    # Reset khi đổi sang thang đo loss.
                                    primary = "neg_loss"
                                    best_metric = -math.inf
                                    stale_epochs = 0

                            scores["neg_loss"] = -val_loss
                            current_metric = float(scores[primary])
                            is_best = current_metric > best_metric

                            if is_best:
                                torch.save(
                                    core.state_dict(),
                                    run_dir / "best.pth",
                                )
                                best_metric = current_metric
                                best_epoch = epoch
                                stale_epochs = 0
                            else:
                                stale_epochs += 1

                            lr = float(optimizer.param_groups[0]["lr"])
                            train_loss = float(
                                (accumulated[0] / accumulated[1]).item()
                            )
                            payload = standardize_epoch_payload(
                                epoch,
                                train_loss,
                                val_loss,
                                scores,
                                lr,
                                best_epoch,
                                best_metric,
                                is_best,
                                extra={
                                    "val/neg_loss": -val_loss,
                                    "patience": stale_epochs,
                                },
                            )

                            history.append(payload)
                            save_epoch_artifacts(
                                run_dir,
                                history,
                                payload,
                            )
                            log_epoch_to_wandb(
                                wandb_run,
                                payload,
                                primary,
                            )
                            print(
                                format_epoch_line(
                                    payload,
                                    total_epochs=total_epochs,
                                    primary_metric=primary,
                                ),
                                flush=True,
                            )
                            stop = (
                                patience > 0
                                and stale_epochs >= patience
                            )
                        except Exception as exc:
                            epoch_error = (
                                f"Epoch {epoch} thất bại: {exc}"
                            )

                    synchronize_error(epoch_error)

                    control = torch.tensor(
                        int(stop),
                        dtype=torch.int32,
                        device=device,
                    )
                    if distributed:
                        dist.broadcast(control, src=0)
                    if bool(control.item()):
                        break

                final_error: str | None = None
                if rank == 0:
                    try:
                        destination = resolve_path(
                            cfg["model"]["weights"]
                        )
                        destination.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        checkpoint = run_dir / "best.pth"
                        if checkpoint.resolve() != destination.resolve():
                            shutil.copy2(checkpoint, destination)

                        plot_training_curves(
                            history,
                            run_dir / "training_curves.png",
                            best_epoch=best_epoch,
                            primary_metric=primary,
                        )
                        print_best_banner(
                            best_epoch,
                            best_metric,
                            primary,
                        )

                        summary = {
                            "pipeline": self.name,
                            "primary_metric": primary,
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "epochs_completed": len(history),
                            "best_checkpoint": str(checkpoint),
                            "weights": str(destination),
                            "wandb_run_id": getattr(
                                wandb_run,
                                "id",
                                None,
                            ),
                        }
                        write_json(
                            summary,
                            run_dir / "summary.json",
                        )
                    except Exception as exc:
                        final_error = (
                            f"Hoàn tất huấn luyện thất bại: {exc}"
                        )

                synchronize_error(final_error)
            finally:
                # Không barrier trong finally: rank lỗi có thể đã rời nhóm.
                if created_group and dist.is_initialized():
                    dist.destroy_process_group()

            assert run_dir is not None
            if rank == 0:
                test_metrics: dict[str, Any] | None = None
                if cfg.get("eval", {}).get("auto_eval_test", False):
                    test_metrics = self.evaluate(run_dir)
                    summary["test_metrics"] = test_metrics
                    write_json(
                        summary,
                        run_dir / "summary.json",
                    )

                log_final_to_wandb(
                    wandb_run,
                    run_dir,
                    test_metrics=test_metrics,
                    cm_json_path=run_dir / "confusion_matrix.json",
                )

            return run_dir
        finally:
            if rank == 0 and owns_wandb and wandb_run is not None:
                wandb_run.finish()
