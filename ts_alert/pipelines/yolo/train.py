"""Huấn luyện YOLO26m với che khuất online và tracking hỗ trợ DDP."""

from __future__ import annotations

import importlib
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from ultralytics.data.augment import BaseTransform, Format
from ultralytics.data.build import get_split_fraction
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr, ops
from ultralytics.utils.instance import Bboxes
from ultralytics.utils.torch_utils import unwrap_model

from ts_alert.config.resolve import (
    get_num_classes,
    get_processed_root,
    resolve_path,
)
from ts_alert.data.preprocess_online import (
    apply_partial_occlusion_to_boxes,
    build_online_transforms,
    get_online_occlusion_config,
)
from ts_alert.eval.metrics import compute_metrics
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.tracking import (
    apply_sweep_overrides,
    format_epoch_line,
    init_wandb_run,
    load_wandb_env,
    log_epoch_to_wandb,
    log_final_to_wandb,
    plot_training_curves,
    print_best_banner,
    save_epoch_artifacts,
    standardize_epoch_payload,
)
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.io import read_json, read_yaml, write_json, write_yaml
from ts_alert.utils.seed import set_seed


def _is_main_process() -> bool:
    """Chỉ cho phép tiến trình thường hoặc rank 0 ghi kết quả."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    try:
        return int(rank) in {-1, 0}
    except (TypeError, ValueError):
        return False


def _wandb_module() -> Any | None:
    """Nạp W&B nếu dependency đã được cài đặt."""
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def _number(value: Any, default: float = 0.0) -> float:
    """Chuyển scalar NumPy/Torch thành số thực hữu hạn."""
    if value is None:
        return default

    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return default
            value = value.detach().item()
        result = float(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return default

    return result if math.isfinite(result) else default


def _loss_values(value: Any) -> list[float]:
    """Đọc loss từ mapping, tensor hoặc sequence mà không giữ tensor."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [_number(item) for item in value.values()]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [_number(item) for item in value]


def _runtime_option(
    cfg: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    """Ưu tiên khóa cấp cao, sau đó đọc runtime nếu là mapping."""
    runtime = cfg.get("runtime")
    fallback = runtime.get(name, default) if isinstance(runtime, Mapping) else default
    return cfg.get(name, fallback)


def _remove_native_wandb_callbacks(callbacks: dict[str, Any]) -> None:
    """Loại callback W&B mặc định để tránh ghi trùng hoặc đóng Sweep run."""
    for handlers in callbacks.values():
        if isinstance(handlers, list):
            handlers[:] = [
                handler
                for handler in handlers
                if not getattr(handler, "__module__", "").startswith(
                    "ultralytics.utils.callbacks.wb"
                )
            ]


def _open_wandb_run(
    cfg: dict[str, Any],
    run_dir: Path,
    run_id: str | None = None,
    resume: str | None = None,
) -> tuple[Any | None, bool]:
    """Khởi tạo tracking và xác định quyền sở hữu W&B run."""
    wandb = _wandb_module()
    previous = getattr(wandb, "run", None) if wandb is not None else None
    result = init_wandb_run(
        cfg,
        run_dir,
        run_id=run_id,
        resume=resume,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], bool(result[1])
    return result, result is not None and previous is None


def _attach_shared_validator(
    validator: Any,
    cfg: Mapping[str, Any],
) -> None:
    """Bọc validator để dùng metric chung và khử ảnh trùng khi chạy DDP.

    Dữ liệu thu thập chỉ chứa kiểu Python, không giữ tensor trên GPU.
    Các wrapper được gắn sau khi worker khởi tạo, không tham gia pickle
    callback từ tiến trình cha.
    """
    if validator is None or getattr(validator, "_ts_alert_wrapped", False):
        return

    method_names = (
        "init_metrics",
        "update_metrics",
        "gather_stats",
        "get_stats",
    )
    methods = {
        name: getattr(validator, name, None)
        for name in method_names
    }
    if not all(callable(method) for method in methods.values()):
        return

    num_classes = get_num_classes(dict(cfg))
    if num_classes <= 0:
        raise ValueError("num_classes phải lớn hơn 0.")

    inference = cfg.get("inference") or {}
    eval_iou = float(inference.get("iou_threshold", 0.50))
    eval_conf = float(inference.get("conf_threshold", 0.25))
    if not math.isfinite(eval_iou) or not 0.0 <= eval_iou <= 1.0:
        raise ValueError("inference.iou_threshold phải thuộc [0, 1].")
    if not math.isfinite(eval_conf) or not 0.0 <= eval_conf <= 1.0:
        raise ValueError("inference.conf_threshold phải thuộc [0, 1].")

    primary = str(
        (cfg.get("train") or {}).get("primary_metric", "mAP50-95")
    )
    orig_init_metrics = methods["init_metrics"]
    orig_update_metrics = methods["update_metrics"]
    orig_gather_stats = methods["gather_stats"]
    orig_get_stats = methods["get_stats"]

    def prepare_items(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        pbatch: Mapping[str, Any],
        scores: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """Scale bản sao bbox về ảnh gốc và chuẩn hóa từng đối tượng."""
        scaled_boxes = boxes.detach().clone().to(dtype=torch.float32)
        if scaled_boxes.numel() > 0:
            scaled_boxes = ops.scale_boxes(
                pbatch["imgsz"],
                scaled_boxes,
                pbatch["ori_shape"],
                pbatch["ratio_pad"],
            )

        box_values = scaled_boxes.cpu().reshape(-1, 4).numpy()
        label_values = labels.detach().cpu().reshape(-1).tolist()
        score_values = (
            scores.detach().cpu().reshape(-1).tolist()
            if scores is not None
            else None
        )

        if len(box_values) != len(label_values):
            raise ValueError("Số bbox và class_id phải bằng nhau.")
        if score_values is not None and len(score_values) != len(box_values):
            raise ValueError("Số bbox và confidence phải bằng nhau.")

        ori_height, ori_width = (
            int(pbatch["ori_shape"][0]),
            int(pbatch["ori_shape"][1]),
        )
        items: list[dict[str, Any]] = []

        for index, (box, raw_label) in enumerate(
            zip(box_values, label_values)
        ):
            label = float(raw_label)
            if (
                not math.isfinite(label)
                or not label.is_integer()
                or not 0 <= label < num_classes
                or not np.isfinite(box).all()
            ):
                continue

            x1 = float(np.clip(box[0], 0.0, ori_width))
            y1 = float(np.clip(box[1], 0.0, ori_height))
            x2 = max(x1, float(np.clip(box[2], 0.0, ori_width)))
            y2 = max(y1, float(np.clip(box[3], 0.0, ori_height)))

            item: dict[str, Any] = {
                "bbox": [x1, y1, x2, y2],
                "class_id": int(label),
            }
            if score_values is not None:
                score = float(score_values[index])
                if math.isnan(score):
                    score = 0.0
                item["confidence"] = min(1.0, max(0.0, score))
            items.append(item)

        return items

    def init_metrics(*args: Any, **kwargs: Any) -> Any:
        """Khởi tạo metric gốc rồi reset dữ liệu của lượt validation."""
        result = orig_init_metrics(*args, **kwargs)
        validator._ts_shared_records = []
        for name in ("_ts_shared_preds", "_ts_shared_gts"):
            if hasattr(validator, name):
                delattr(validator, name)
        return result

    def update_metrics(
        preds: list[dict[str, torch.Tensor]],
        batch: dict[str, Any],
    ) -> Any:
        """Cập nhật metric gốc và thu thập prediction/GT theo từng ảnh."""
        result = orig_update_metrics(preds, batch)
        records = getattr(validator, "_ts_shared_records", None)
        if records is None:
            records = []
            validator._ts_shared_records = records

        for si, pred in enumerate(preds):
            pbatch = validator._prepare_batch(si, batch)
            pred_items = prepare_items(
                pred["bboxes"],
                pred["cls"],
                pbatch,
                pred["conf"],
            )
            gt_items = prepare_items(
                pbatch["bboxes"],
                pbatch["cls"],
                pbatch,
            )
            records.append(
                (str(pbatch["im_file"]), pred_items, gt_items)
            )

        return result

    def gather_stats(*args: Any, **kwargs: Any) -> Any:
        """Gom dữ liệu mọi rank và loại ảnh lặp do DDP padding."""
        result = orig_gather_stats(*args, **kwargs)
        records = getattr(validator, "_ts_shared_records", [])

        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if distributed:
            rank = dist.get_rank()
            gathered: list[Any] | None = (
                [None] * dist.get_world_size()
                if rank == 0
                else None
            )
            dist.gather_object(records, gathered, dst=0)
        else:
            rank = 0
            gathered = [records]

        try:
            if rank == 0:
                unique_records: dict[
                    str,
                    tuple[list[dict[str, Any]], list[dict[str, Any]]],
                ] = {}
                for rank_records in gathered or []:
                    for im_file, pred_items, gt_items in rank_records or []:
                        unique_records.setdefault(
                            im_file,
                            (pred_items, gt_items),
                        )

                validator._ts_shared_preds = [
                    pred_items
                    for pred_items, _ in unique_records.values()
                ]
                validator._ts_shared_gts = [
                    gt_items
                    for _, gt_items in unique_records.values()
                ]
        finally:
            if hasattr(validator, "_ts_shared_records"):
                delattr(validator, "_ts_shared_records")

        return result

    def get_stats(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Bổ sung metric chung và fitness dùng để chọn checkpoint tốt nhất."""
        stats = orig_get_stats(*args, **kwargs)
        if not (
            hasattr(validator, "_ts_shared_preds")
            and hasattr(validator, "_ts_shared_gts")
        ):
            return stats

        try:
            shared = compute_metrics(
                validator._ts_shared_preds,
                validator._ts_shared_gts,
                num_classes,
                iou_threshold=eval_iou,
                conf_threshold=eval_conf,
            )
        finally:
            for name in ("_ts_shared_preds", "_ts_shared_gts"):
                if hasattr(validator, name):
                    delattr(validator, name)

        stats.update(
            {
                f"ts_alert/{name}": float(shared[name])
                for name in (
                    "mAP50-95",
                    "mAP50",
                    "mAP75",
                    "AR100",
                    "precision",
                    "recall",
                    "f1",
                )
            }
        )
        if primary != "neg_loss" and primary in shared:
            stats["fitness"] = float(shared[primary])

        return stats

    validator.init_metrics = init_metrics
    validator.update_metrics = update_metrics
    validator.gather_stats = gather_stats
    validator.get_stats = get_stats
    validator._ts_alert_wrapped = True


class YoloOcclusionTransform(BaseTransform):
    """Che một phần đối tượng, giữ nguyên tọa độ và định dạng nhãn."""

    def __init__(self, p: float = 0.30, ratio: float = 0.30) -> None:
        """Kiểm tra và lưu xác suất cùng kích thước tuyến tính vật cản."""
        validated = get_online_occlusion_config(
            {
                "online_preprocess": {
                    "enabled": True,
                    "occlusion_p": p,
                    "occlusion_ratio": ratio,
                }
            }
        )
        self.p: float = validated["p"]
        self.ratio: float = validated["ratio"]

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Đổi bản sao box sang pixel xyxy trước khi che ảnh."""
        if self.p <= 0.0:
            return labels

        image = labels.get("img")
        instances = labels.get("instances")
        if image is None or instances is None or len(instances) == 0:
            return labels

        height, width = image.shape[:2]
        boxes = Bboxes(
            np.array(instances.bboxes, dtype=np.float32, copy=True),
            format=instances._bboxes.format,
        )
        boxes.convert("xyxy")
        if instances.normalized:
            boxes.mul((width, height, width, height))

        labels["img"] = apply_partial_occlusion_to_boxes(
            image,
            boxes.bboxes,
            occlusion_p=self.p,
            occlusion_ratio=self.ratio,
        )
        return labels


class OcclusionYOLODataset(YOLODataset):
    """Dataset giữ transform che khuất khi Ultralytics đóng mosaic."""

    def __init__(
        self,
        *args: Any,
        occlusion_p: float = 0.0,
        occlusion_ratio: float = 0.30,
        **kwargs: Any,
    ) -> None:
        """Lưu cấu hình trước khi lớp cha xây dựng transform."""
        transform = YoloOcclusionTransform(occlusion_p, occlusion_ratio)
        self.occlusion_p = transform.p
        self.occlusion_ratio = transform.ratio
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp: Any = None) -> Any:
        """Chèn che khuất ngay trước Format, kể cả sau close_mosaic."""
        transforms = super().build_transforms(hyp)
        if not self.augment or self.occlusion_p <= 0.0:
            return transforms

        for index, transform in enumerate(transforms.transforms):
            if isinstance(transform, Format):
                transforms.insert(
                    index,
                    YoloOcclusionTransform(
                        self.occlusion_p,
                        self.occlusion_ratio,
                    ),
                )
                break
        else:
            raise RuntimeError(
                "Chuỗi transform YOLO thiếu Format; không thể chèn che khuất."
            )
        return transforms


class _OcclusionConfigHook:
    """Callback picklable chuyển cấu hình che khuất đến các DDP worker."""

    def __init__(self, p: float, ratio: float) -> None:
        """Lưu tham số độc lập với đối tượng W&B và tiến trình cha."""
        transform = YoloOcclusionTransform(p, ratio)
        self.p = transform.p
        self.ratio = transform.ratio

    def __call__(self, trainer: Any) -> None:
        """Giữ callback làm nơi chứa cấu hình cho build_dataset."""
        return None


class OcclusionDetectionTrainer(DetectionTrainer):
    """Dùng dataset che khuất riêng cho tập train."""

    def _get_occlusion_params(self) -> tuple[float, float]:
        """Đọc cấu hình đã được pickle cùng callback của trainer."""
        for callback in self.callbacks.get("on_pretrain_routine_start", []):
            if isinstance(callback, _OcclusionConfigHook):
                return callback.p, callback.ratio
        return 0.0, 0.30

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ) -> YOLODataset:
        """Giữ dataset chuẩn cho validation và khi che khuất bị tắt."""
        probability, ratio = self._get_occlusion_params()
        if mode != "train" or probability <= 0.0:
            return super().build_dataset(img_path, mode=mode, batch=batch)

        args = self.args
        return OcclusionYOLODataset(
            img_path=img_path,
            imgsz=args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=args,
            rect=bool(args.rect),
            cache=args.cache or None,
            single_cls=bool(args.single_cls),
            stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0,
            prefix=colorstr("train: "),
            task=args.task,
            classes=args.classes,
            data=self.data,
            fraction=get_split_fraction(args.fraction, "train"),
            occlusion_p=probability,
            occlusion_ratio=ratio,
        )


class _YoloTrackingHook:
    """Theo dõi epoch với trạng thái có thể chuyển đến DDP worker."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        primary_metric: str,
        wandb_run_id: str | None = None,
    ) -> None:
        """Khởi tạo lịch sử và thông tin nhận diện run."""
        self.cfg = cfg
        self.run_dir_str = str(run_dir)
        self.primary_metric = primary_metric
        self.wandb_run_id = wandb_run_id
        self.history: list[dict[str, Any]] = []
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

        self._wandb_run: Any | None = None
        self._owns_run = False
        self._ended = False

    def __getstate__(self) -> dict[str, Any]:
        """Chỉ pickle cấu hình và dữ liệu thuần, không pickle wandb.Run."""
        return {
            "cfg": self.cfg,
            "run_dir_str": self.run_dir_str,
            "primary_metric": self.primary_metric,
            "wandb_run_id": self.wandb_run_id,
            "history": self.history,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Khôi phục worker với kết nối tracking riêng."""
        self.__dict__.update(state)
        self._wandb_run = None
        self._owns_run = False
        self._ended = False

    def on_train_start(self, trainer: Any) -> None:
        """Gắn metric trên mọi rank và khởi tạo W&B trên rank chính."""
        _attach_shared_validator(
            getattr(trainer, "validator", None),
            self.cfg,
        )
        if not _is_main_process():
            return

        callbacks = getattr(trainer, "callbacks", None)
        if isinstance(callbacks, dict):
            _remove_native_wandb_callbacks(callbacks)

        run_dir = Path(self.run_dir_str)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run, self._owns_run = _open_wandb_run(
            self.cfg,
            run_dir,
            run_id=self.wandb_run_id,
            resume="allow" if self.wandb_run_id else None,
        )
        if self._wandb_run is not None:
            self.wandb_run_id = str(self._wandb_run.id)

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Ghi metric chung theo epoch và bỏ callback từ final_eval."""
        if not _is_main_process():
            return

        validator = getattr(trainer, "validator", None)
        if getattr(validator, "training", None) is False:
            return

        epoch = int(trainer.epoch) + 1
        total_epochs = int(
            (self.cfg.get("train") or {}).get("epochs", 0)
        )
        if total_epochs > 0 and epoch > total_epochs:
            return

        # final_eval có thể gọi lại callback của epoch đã ghi.
        if any(int(row["epoch"]) == epoch for row in self.history):
            return

        primary = self.primary_metric
        raw_metrics = dict(getattr(trainer, "metrics", {}) or {})
        raw_losses = getattr(trainer, "tloss", None)
        train_losses = _loss_values(raw_losses)

        if "ts_alert/mAP50-95" in raw_metrics:
            metrics: dict[str, float] = {
                name: _number(raw_metrics.get(f"ts_alert/{name}"))
                for name in (
                    "mAP50-95",
                    "mAP50",
                    "mAP75",
                    "AR100",
                    "precision",
                    "recall",
                    "f1",
                )
            }
        else:
            precision = _number(raw_metrics.get("metrics/precision(B)"))
            recall = _number(raw_metrics.get("metrics/recall(B)"))
            metrics = {
                "mAP50": _number(raw_metrics.get("metrics/mAP50(B)")),
                "mAP50-95": _number(raw_metrics.get("metrics/mAP50-95(B)")),
                "precision": precision,
                "recall": recall,
                "f1": (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall > 0.0
                    else 0.0
                ),
            }

            validator_metrics = getattr(validator, "metrics", None)
            box_metrics = getattr(validator_metrics, "box", None)
            map75 = getattr(box_metrics, "map75", None)
            if map75 is not None:
                metrics["mAP75"] = _number(map75)

        train_cfg = self.cfg.get("train") or {}
        learning_rates = getattr(trainer, "lr", {}) or {}
        learning_rate = _number(
            learning_rates.get("lr/pg0"),
            _number(train_cfg.get("lr", train_cfg.get("lr0", 0.01))),
        )

        current = metrics.get(primary)
        if current is None:
            current = _number(raw_metrics.get(primary))
        is_best = self.best_metric is None or current > self.best_metric
        if is_best:
            self.best_metric = current
            self.best_epoch = epoch

        extra: dict[str, Any] = {}
        if isinstance(raw_losses, Mapping):
            for name, value in zip(raw_losses, train_losses):
                extra[f"train/{name}"] = value
        else:
            loss_names = tuple(getattr(trainer, "loss_names", ()) or ())
            for name, value in zip(loss_names, train_losses):
                extra[f"train/{name}"] = value

        val_losses = {
            str(key): _number(value)
            for key, value in raw_metrics.items()
            if str(key).startswith("val/") and str(key).endswith("loss")
        }
        extra.update(val_losses)

        payload = standardize_epoch_payload(
            epoch,
            sum(train_losses),
            sum(val_losses.values()),
            metrics,
            learning_rate,
            self.best_epoch,
            self.best_metric,
            is_best,
            extra=extra,
        )

        self.history.append(payload)
        write_json(
            self.history,
            Path(self.run_dir_str) / "metrics_history.json",
        )

        line = format_epoch_line(
            payload,
            total_epochs=total_epochs or None,
            primary_metric=primary,
        )
        print(line, flush=True)
        log_epoch_to_wandb(self._wandb_run, payload, primary)

    def restore_history(self) -> None:
        """Đọc lịch sử và summary do DDP rank 0 ghi."""
        run_dir = Path(self.run_dir_str)
        history_path = run_dir / "metrics_history.json"
        if history_path.is_file():
            rows = read_json(history_path)
            if not isinstance(rows, list):
                raise ValueError("metrics_history.json phải chứa một danh sách.")
            self.history = sorted(rows, key=lambda row: int(row["epoch"]))

        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            self.best_metric = summary.get("best_metric", self.best_metric)
            self.best_epoch = summary.get("best_epoch", self.best_epoch)
            self.wandb_run_id = summary.get("wandb_run_id") or self.wandb_run_id

    def save_artifacts(
        self,
        test_metrics: dict[str, Any] | None = None,
        val_metrics: Mapping[str, Any] | None = None,
        *,
        best_checkpoint: Path | None = None,
        weights: Path | None = None,
    ) -> None:
        """Lưu lịch sử, biểu đồ và summary theo schema chung hai pipeline."""
        if not _is_main_process():
            return

        primary = self.primary_metric
        run_dir = Path(self.run_dir_str)
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.history:
            save_epoch_artifacts(run_dir, self.history, self.history[-1])
            plot_training_curves(
                self.history,
                run_dir / "training_curves.png",
                best_epoch=self.best_epoch,
                primary_metric=primary,
            )

        summary_path = run_dir / "summary.json"
        previous: dict[str, Any] = {}
        if summary_path.is_file():
            loaded = read_json(summary_path)
            if not isinstance(loaded, Mapping):
                raise ValueError("summary.json phải chứa một mapping.")
            previous = dict(loaded)

        if test_metrics is None:
            test_metrics = previous.get("test_metrics")

        if val_metrics is None:
            val_path = run_dir / "metrics_val.json"
            if val_path.is_file():
                loaded_val = read_json(val_path)
                if not isinstance(loaded_val, Mapping):
                    raise ValueError("metrics_val.json phải chứa một mapping.")
                val_metrics = {
                    key: value
                    for key, value in loaded_val.items()
                    if key != "pipeline"
                }
            else:
                val_metrics = previous.get("val_metrics")

        checkpoint_value = previous.get("best_checkpoint")
        if best_checkpoint is not None:
            checkpoint_value = str(best_checkpoint)
        elif checkpoint_value is None:
            checkpoint_path = run_dir / "weights" / "best.pt"
            if checkpoint_path.is_file():
                checkpoint_value = str(checkpoint_path)

        weights_value = previous.get("weights")
        if weights is not None:
            weights_value = str(weights)
        elif weights_value is None:
            configured_weights = (self.cfg.get("model") or {}).get("weights")
            if configured_weights is not None:
                weights_value = str(resolve_path(configured_weights))

        best_conf_threshold = previous.get("best_conf_threshold")
        if test_metrics is not None and "conf_threshold" in test_metrics:
            best_conf_threshold = test_metrics["conf_threshold"]

        summary: dict[str, Any] = {
            "pipeline": "yolo",
            "primary_metric": primary,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "epochs_completed": len(self.history),
            "best_checkpoint": checkpoint_value,
            "weights": weights_value,
            "wandb_run_id": self.wandb_run_id,
            "best_conf_threshold": best_conf_threshold,
            "test_metrics": test_metrics,
            "val_metrics": dict(val_metrics) if val_metrics is not None else None,
        }
        write_json(summary, summary_path)

    def on_train_end(self, trainer: Any) -> None:
        """Hoàn tất artifact và đóng W&B run thuộc DDP child."""
        if not _is_main_process() or self._ended:
            return

        self._ended = True
        primary = self.primary_metric
        try:
            self.save_artifacts()
            if self.best_epoch is not None:
                print_best_banner(
                    self.best_epoch,
                    self.best_metric if self.best_metric is not None else 0.0,
                    primary,
                )
        finally:
            if (
                self._owns_run
                and int(os.environ.get("WORLD_SIZE", "1")) > 1
                and self._wandb_run is not None
            ):
                self._wandb_run.finish()
                self._wandb_run = None
                self._owns_run = False


class YoloPipeline(BasePipeline):
    """Pipeline fine-tune YOLO26m, lưu checkpoint và đánh giá tập test."""

    name = "yolo"

    def _prepare_runtime_data(self, run_dir: Path) -> Path:
        """Ghi YAML riêng bên ngoài thư mục Ultralytics có thể xóa."""
        root = get_processed_root(self.cfg).resolve()
        source = root / "data.yaml"
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy data.yaml tại {source}.")

        data = read_yaml(source)
        data["path"] = root.as_posix()

        runtime_dir = run_dir.parent / "_runtime"
        runtime_path = runtime_dir / f"{run_dir.name}.yaml"
        write_yaml(data, runtime_path)
        return runtime_path.resolve()

    def _train_args(
        self,
        run_dir: Path,
        runtime_yaml: Path,
    ) -> dict[str, Any]:
        """Tạo kwargs Ultralytics, hỗ trợ runtime dạng chuỗi hoặc mapping."""
        cfg = self.cfg
        train = cfg["train"]
        args: dict[str, Any] = {
            "data": runtime_yaml.as_posix(),
            "epochs": int(train["epochs"]),
            "batch": int(train["batch_size"]),
            "imgsz": int(cfg["data"]["image_size"]),
            "seed": int(cfg.get("seed", 42)),
            "workers": int(_runtime_option(cfg, "workers", 2)),
            "patience": int(train.get("early_stopping_patience", 10)),
            "amp": bool(train.get("amp", True)),
            "optimizer": train.get("optimizer", "auto"),
            "lr0": float(train.get("lr", train.get("lr0", 0.01))),
            "project": run_dir.parent.resolve().as_posix(),
            "name": run_dir.name,
            "exist_ok": True,
            "plots": True,
            **build_online_transforms(cfg, "ultralytics"),
        }
        for name in (
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
        ):
            if train.get(name) is not None:
                args[name] = float(train[name])

        num_gpus = int(_runtime_option(cfg, "num_gpus", 1))
        device = _runtime_option(cfg, "device", "auto")
        if device == "cpu" or num_gpus <= 0:
            args["device"] = "cpu"
        elif num_gpus > 1:
            args["device"] = list(range(num_gpus))
        elif device is None or (
            isinstance(device, str) and device in {"auto", "cuda"}
        ):
            args["device"] = 0 if torch.cuda.is_available() else "cpu"
        else:
            args["device"] = device
        return args

    @staticmethod
    def _attach_tracking(
        model: Any,
        hook: _YoloTrackingHook,
    ) -> None:
        """Gắn tracking vào dict callback, tương thích model MagicMock."""
        callbacks = getattr(model, "callbacks", None)
        if not isinstance(callbacks, dict):
            return

        _remove_native_wandb_callbacks(callbacks)
        for event, handler in (
            ("on_train_start", hook.on_train_start),
            ("on_fit_epoch_end", hook.on_fit_epoch_end),
            ("on_train_end", hook.on_train_end),
        ):
            callbacks.setdefault(event, []).append(handler)

    def fit(self) -> Path:
        """Huấn luyện, lưu artifact, đánh giá và đóng run thuộc pipeline."""
        if not _is_main_process():
            raise RuntimeError(
                "Hãy gọi YoloPipeline.fit() tại tiến trình cha; "
                "Ultralytics sẽ quản lý các DDP worker."
            )

        wandb = _wandb_module()
        existing_run = getattr(wandb, "run", None) if wandb is not None else None
        if existing_run is not None:
            updated = apply_sweep_overrides(
                self.cfg,
                dict(existing_run.config),
            )
            if isinstance(updated, Mapping):
                self.cfg = dict(updated)

        cfg = self.cfg
        set_seed(int(cfg.get("seed", 42)))
        load_wandb_env(cfg)

        run_dir = make_run_dir(
            resolve_path(cfg["paths"]["outputs_dir"]),
            self.name,
            cfg["train"].get("run_name"),
            cfg,
        ).resolve()
        runtime_yaml = self._prepare_runtime_data(run_dir)
        train_args = self._train_args(run_dir, runtime_yaml)
        occ_cfg = get_online_occlusion_config(cfg)

        primary_metric = str(cfg["train"].get("primary_metric", "mAP50-95"))
        require_wandb = bool(cfg["train"].get("require_wandb", False))
        num_gpus = int(_runtime_option(cfg, "num_gpus", 1))

        wandb_run: Any | None = None
        owns_wandb = False
        reattached = False
        wandb_run_id: str | None = None

        hook = _YoloTrackingHook(cfg, run_dir, primary_metric)
        try:
            if num_gpus > 1 and existing_run is None:
                if require_wandb:
                    import sys

                    if wandb is None:
                        print(
                            "Cảnh báo: Không thể nạp W&B; "
                            "tiếp tục ghi thống kê cục bộ.",
                            file=sys.stderr,
                        )
                    elif not os.environ.get("WANDB_API_KEY"):
                        print(
                            "Cảnh báo: Thiếu WANDB_API_KEY; "
                            "tiếp tục ghi thống kê cục bộ.",
                            file=sys.stderr,
                        )
                    else:
                        wandb_run_id = str(wandb.util.generate_id())
            else:
                wandb_run, owns_wandb = _open_wandb_run(cfg, run_dir)
                if wandb_run is not None:
                    wandb_run_id = str(wandb_run.id)

            hook.wandb_run_id = wandb_run_id
            hook._wandb_run = wandb_run

            model_cfg = cfg.get("model") or {}
            model_ref = model_cfg.get("pretrained") or (
                f"{model_cfg['name']}.pt"
                if model_cfg.get("name")
                else "yolo26m.pt"
            )
            model = load_model(model_ref)
            self._attach_tracking(model, hook)

            if occ_cfg["enabled"] and occ_cfg["p"] > 0.0:
                model.add_callback(
                    "on_pretrain_routine_start",
                    _OcclusionConfigHook(
                        occ_cfg["p"],
                        occ_cfg["ratio"],
                    ),
                )
                train_args["trainer"] = OcclusionDetectionTrainer

            model.train(**train_args)

            # DDP có thể xóa save_dir khi chuẩn bị worker.
            run_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(cfg, run_dir / "config.yaml")
            write_yaml(read_yaml(runtime_yaml), run_dir / "data.yaml")

            hook.restore_history()
            wandb_run_id = hook.wandb_run_id or wandb_run_id
            hook.save_artifacts()

            best_checkpoint = self.best_checkpoint(run_dir)
            target_weights = resolve_path(cfg["model"]["weights"])
            target_weights.parent.mkdir(parents=True, exist_ok=True)
            if best_checkpoint.resolve() != target_weights.resolve():
                shutil.copy2(best_checkpoint, target_weights)

            # Ghi đường dẫn trước evaluate để BasePipeline cập nhật summary.
            hook.save_artifacts(
                best_checkpoint=best_checkpoint,
                weights=target_weights,
            )

            test_metrics: dict[str, Any] | None = None
            if (cfg.get("eval") or {}).get("auto_eval_test", False):
                evaluation_cfg = self.cfg
                self.cfg = {
                    **evaluation_cfg,
                    "eval": {
                        **(evaluation_cfg.get("eval") or {}),
                        "split": "test",
                    },
                }
                try:
                    test_metrics = self.evaluate(run_dir)
                except Exception as exc:
                    print(
                        "[YoloPipeline] Cảnh báo: đánh giá tập test "
                        f"không hoàn tất ({type(exc).__name__}): {exc}",
                        flush=True,
                    )
                finally:
                    self.cfg = evaluation_cfg

            val_metrics: dict[str, Any] | None = None
            val_path = run_dir / "metrics_val.json"
            if val_path.is_file():
                loaded_val = read_json(val_path)
                if not isinstance(loaded_val, Mapping):
                    raise ValueError("metrics_val.json phải chứa một mapping.")
                val_metrics = {
                    key: value
                    for key, value in loaded_val.items()
                    if key != "pipeline"
                }

            # Chỉ bổ sung alias dự đoán; giữ nguyên schema của evaluator.
            confusion_path = run_dir / "confusion_matrix.json"
            if confusion_path.is_file():
                matrix = read_json(confusion_path)
                if (
                    isinstance(matrix, dict)
                    and "preds" not in matrix
                    and "y_pred" in matrix
                ):
                    matrix["preds"] = matrix["y_pred"]
                    write_json(matrix, confusion_path)

            if wandb_run_id and (
                wandb_run is None
                or (num_gpus > 1 and existing_run is None)
            ):
                wandb_run, reattached = _open_wandb_run(
                    cfg,
                    run_dir,
                    run_id=wandb_run_id,
                    resume="allow",
                )

            if wandb_run is not None:
                hook.wandb_run_id = str(wandb_run.id)

            hook.save_artifacts(
                test_metrics,
                val_metrics,
                best_checkpoint=best_checkpoint,
                weights=target_weights,
            )
            log_final_to_wandb(
                wandb_run,
                run_dir,
                test_metrics,
                confusion_path,
                val_metrics=val_metrics,
            )
            return run_dir
        finally:
            if wandb_run is not None and (owns_wandb or reattached):
                wandb_run.finish()

    def load_detector(
        self,
        weights: str | Path | None = None,
    ) -> YoloDetector:
        """Nạp detector từ checkpoint chỉ định hoặc trọng số mặc định."""
        return YoloDetector(
            weights if weights is not None else self.default_weights(),
            self.cfg,
        )
