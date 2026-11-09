"""YoloPipeline: huấn luyện YOLO qua Ultralytics API kèm hỗ trợ Weights & Biases.

Phụ trách: Thành viên 1
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics.data.augment import BaseTransform
from ultralytics.data.build import get_split_fraction
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Bboxes
from ultralytics.utils.torch_utils import unwrap_model

from ts_alert.config.resolve import get_processed_root, resolve_path
from ts_alert.data.preprocess_online import (
    apply_partial_occlusion_to_boxes,
    build_online_transforms,
    get_online_occlusion_config,
)
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.seed import set_seed


class YoloOcclusionTransform(BaseTransform):
    """Che khuất một phần đối tượng mà không thay đổi box nhãn."""

    def __init__(self, p: float = 0.30, ratio: float = 0.30) -> None:
        """Lưu xác suất áp dụng và tỷ lệ kích thước vật cản."""
        self.p = float(p)
        self.ratio = float(ratio)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Áp dụng che khuất bằng bản sao box ở hệ tọa độ pixel xyxy."""
        if self.p <= 0.0:
            return labels

        img = labels.get("img")
        instances = labels.get("instances")
        if img is None or instances is None or len(instances) == 0:
            return labels

        h, w = img.shape[:2]
        bbox_copy = Bboxes(
            np.array(instances.bboxes, dtype=np.float32, copy=True),
            format=instances._bboxes.format,
        )
        bbox_copy.convert("xyxy")
        if instances.normalized:
            bbox_copy.mul((w, h, w, h))

        labels["img"] = apply_partial_occlusion_to_boxes(
            img,
            bbox_copy.bboxes,
            occlusion_p=self.p,
            occlusion_ratio=self.ratio,
        )
        return labels


class OcclusionYOLODataset(YOLODataset):
    """Dataset YOLO duy trì che khuất khi xây dựng lại chuỗi augmentation."""

    def __init__(
        self,
        *args: Any,
        occlusion_p: float = 0.0,
        occlusion_ratio: float = 0.30,
        **kwargs: Any,
    ) -> None:
        """Lưu cấu hình trước khi lớp cha khởi tạo các transform."""
        self.occlusion_p = float(occlusion_p)
        self.occlusion_ratio = float(occlusion_ratio)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp: Any) -> Any:
        """Chèn che khuất trước Format, kể cả sau khi đóng mosaic."""
        transforms = super().build_transforms(hyp)
        if self.augment and self.occlusion_p > 0.0:
            insert_idx = max(0, len(transforms.transforms) - 1)
            transforms.insert(
                insert_idx,
                YoloOcclusionTransform(
                    p=self.occlusion_p,
                    ratio=self.occlusion_ratio,
                ),
            )
        return transforms


class _OcclusionConfigHook:
    """Callback picklable mang cấu hình che khuất tới các DDP rank."""

    def __init__(self, p: float, ratio: float) -> None:
        """Lưu cấu hình để trainer đọc từ danh sách callback."""
        self.p = float(p)
        self.ratio = float(ratio)

    def __call__(self, trainer: Any) -> None:
        """Giữ nguyên trạng thái trainer khi callback được thực thi."""
        pass


class OcclusionDetectionTrainer(DetectionTrainer):
    """Trainer sử dụng dataset có che khuất cho tập huấn luyện."""

    def _get_occlusion_params(self) -> tuple[float, float]:
        """Đọc cấu hình che khuất được chuyển qua callback, hỗ trợ DDP."""
        for cb in self.callbacks.get("on_pretrain_routine_start", []):
            if isinstance(cb, _OcclusionConfigHook):
                return cb.p, cb.ratio
        return 0.0, 0.30

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ) -> Any:
        """Tạo dataset che khuất cho train và giữ hành vi gốc cho tập khác."""
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)

        p, ratio = self._get_occlusion_params()
        if p <= 0.0 or self.args.task in {"depth", "semantic"}:
            return super().build_dataset(img_path, mode=mode, batch=batch)

        return OcclusionYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect or False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=get_split_fraction(self.args.fraction, "train"),
            occlusion_p=p,
            occlusion_ratio=ratio,
        )


class YoloPipeline(BasePipeline):
    name = "yolo"

    def fit(self) -> Path:
        """Fine-tune YOLO trên <processed root>/data.yaml (get_processed_root(cfg)).

        1. Tạo thư mục output run_dir.
        2. Chuẩn bị runtime data.yaml.
        3. Khởi tạo W&B (nếu train.require_wandb: true).
        4. Nạp model và gọi model.train(...), hỗ trợ che khuất online.
        5. Sao chép best.pt sang cfg["model"]["weights"].
        6. Đánh giá tập Test (nếu eval.auto_eval_test: true).
        7. Đóng W&B run (wandb.finish()).
        """
        cfg = self.cfg
        set_seed(cfg["seed"])

        # 1. Tạo thư mục output run_dir
        outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
        run_name = (
            cfg["train"].get("run_name")
            or f"yolo_s{cfg['seed']}_ep{cfg['train']['epochs']}_bs{cfg['train']['batch_size']}"
        )
        run_dir = make_run_dir(outputs_dir, "yolo", run_name, cfg)

        # 2. Chuẩn bị đường dẫn data.yaml với path chuẩn xác
        processed_root = get_processed_root(cfg)
        source_yaml = processed_root / "data.yaml"
        if not source_yaml.exists():
            raise FileNotFoundError(
                f"Không tìm thấy data.yaml tại {source_yaml}. Hãy kiểm tra đường dẫn dataset."
            )

        import yaml

        with open(source_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = processed_root.resolve().as_posix()

        runtime_yaml = outputs_dir / f"data_runtime_{cfg['seed']}.yaml"
        with open(runtime_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        try:
            shutil.copy2(runtime_yaml, run_dir / "data.yaml")
        except Exception:
            pass

        # 3. Khởi tạo W&B nếu cấu hình require_wandb: true
        require_wandb = bool(cfg.get("train", {}).get("require_wandb", False))
        if require_wandb:
            try:
                import wandb

                proj = cfg.get("project") or {}
                wandb.init(
                    project=proj.get("wandb_project", "TSD-Voice-Alert"),
                    name=run_name,
                    config=cfg,
                )
            except Exception as e:
                print(f"[YoloPipeline][W&B] Cảnh báo khi khởi tạo W&B: {e}")

        # 4. Nạp model ban đầu
        pretrained = cfg["model"].get("pretrained", "yolo26m.pt")
        model = load_model(pretrained)

        # 5. Tham số huấn luyện Ultralytics
        online_args = build_online_transforms(cfg, "ultralytics")
        occ_cfg = get_online_occlusion_config(cfg)
        train_args: dict[str, Any] = {
            "data": runtime_yaml.resolve().as_posix(),
            "epochs": int(cfg["train"]["epochs"]),
            "batch": int(cfg["train"]["batch_size"]),
            "imgsz": int(cfg["data"]["image_size"]),
            "seed": int(cfg["seed"]),
            "patience": int(cfg["train"].get("early_stopping_patience", 10)),
            "amp": bool(cfg["train"].get("amp", True)),
            "lr0": float(cfg["train"].get("lr0", cfg["train"].get("lr", 0.01))),
            "optimizer": cfg["train"].get("optimizer", "auto"),
            "project": run_dir.parent.resolve().as_posix(),
            "name": run_dir.name,
            "exist_ok": True,
            **online_args,
        }

        # Optimization hyperparameters (momentum, weight_decay, warmup_*)
        for opt_key in (
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
        ):
            if opt_key in cfg["train"] and cfg["train"][opt_key] is not None:
                train_args[opt_key] = float(cfg["train"][opt_key])

        # Thiết lập thiết bị tính toán (GPU / CPU)
        num_gpus = int(cfg.get("num_gpus", 1))
        if num_gpus == 2:
            train_args["device"] = [0, 1]
        elif num_gpus == 1:
            train_args["device"] = 0
        else:
            train_args["device"] = "cpu"

        if occ_cfg["enabled"] and occ_cfg["p"] > 0.0:
            model.add_callback(
                "on_pretrain_routine_start",
                _OcclusionConfigHook(occ_cfg["p"], occ_cfg["ratio"]),
            )
            train_args["trainer"] = OcclusionDetectionTrainer
            print(
                "[YoloPipeline] Đã bật che khuất một phần online: "
                f"p={occ_cfg['p']:.3f}, ratio={occ_cfg['ratio']:.3f}"
            )

        print(f"[YoloPipeline] Bắt đầu fine-tune YOLO26m tại: {run_dir}")
        model.train(**train_args)

        # 6. Sao chép best.pt sang đường dẫn weights được chỉ định
        best_pt = self.best_checkpoint(run_dir)
        target_weights = resolve_path(cfg["model"]["weights"])
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, target_weights)
        print(f"[YoloPipeline] Đã lưu checkpoint tốt nhất sang: {target_weights}")

        # 7. Tự động đánh giá trên tập Test nếu được kích hoạt
        test_metrics: dict[str, Any] | None = None
        if cfg.get("eval", {}).get("auto_eval_test", False):
            try:
                print("[YoloPipeline] Tự động đánh giá trên tập Test...")
                test_metrics = self.evaluate(run_dir)
            except Exception as e:
                print(f"[YoloPipeline] Chú ý: Đánh giá tập Test bị bỏ qua: {e}")

        # 8. Kết thúc W&B run
        if require_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    if test_metrics:
                        wandb.log(test_metrics)
                    wandb.finish()
            except Exception:
                pass

        return run_dir

    def load_detector(self, weights: str | Path | None = None) -> YoloDetector:
        """Nạp detector từ weights được chỉ định hoặc checkpoint mặc định."""
        return YoloDetector(weights or self.default_weights(), self.cfg)
