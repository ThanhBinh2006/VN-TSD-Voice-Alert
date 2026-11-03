"""YoloPipeline: huấn luyện YOLO qua Ultralytics API kèm hỗ trợ Weights & Biases.
Phụ trách: Thành viên 1
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ts_alert.config.resolve import get_processed_root, resolve_path
from ts_alert.data.preprocess_online import build_online_transforms
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.seed import set_seed


def _setup_wandb(cfg: dict[str, Any]) -> bool:
    """Cấu hình Weights & Biases trước khi huấn luyện."""
    require_wandb = bool(cfg.get("train", {}).get("require_wandb", False))

    if not require_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        try:
            from ultralytics import settings as ul_settings

            ul_settings.update({"wandb": False})
        except Exception:
            pass
        return False

    os.environ.pop("WANDB_DISABLED", None)
    proj = cfg.get("project") or {}
    if proj.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = str(proj["wandb_project"]).strip()
    if proj.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(proj["wandb_entity"]).strip()
    if proj.get("wandb_base_url"):
        os.environ["WANDB_BASE_URL"] = str(proj["wandb_base_url"]).strip()

    # Nạp API key nếu chưa có trong môi trường
    if not os.environ.get("WANDB_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient

            key = str(UserSecretsClient().get_secret("WANDB_API_KEY") or "").strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
        except Exception:
            pass

    try:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)

        from ultralytics import settings as ul_settings

        ul_settings.update({"wandb": True})
        print(f"[YoloPipeline][W&B] Đã bật tích hợp W&B (Project: {os.environ.get('WANDB_PROJECT', 'TSD-Voice-Alert')})")
        return True
    except Exception as exc:
        print(f"[YoloPipeline][W&B] Cảnh báo: không thể bật W&B ({exc}). Huấn luyện tiếp mà không dùng W&B.")
        return False


class YoloPipeline(BasePipeline):
    name = "yolo"

    def fit(self) -> Path:
        """Fine-tune YOLO trên <processed root>/data.yaml (get_processed_root(cfg)).

        1. Tạo thư mục output run_dir
        2. Sinh runtime data.yaml với đường dẫn tuyệt đối
        3. Cấu hình W&B (nếu train.require_wandb: true)
        4. Nạp model và gọi model.train(...)
        5. Sao chép best.pt sang cfg["model"]["weights"]
        6. Đánh giá tập Test (nếu eval.auto_eval_test: true) và đồng bộ kết quả lên W&B
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

        # Lưu runtime data.yaml tại outputs_dir để tránh bị DDP đa GPU xóa nhầm khi khởi tạo run_dir
        runtime_yaml = outputs_dir / f"data_runtime_{cfg['seed']}.yaml"
        with open(runtime_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        try:
            shutil.copy2(runtime_yaml, run_dir / "data.yaml")
        except Exception:
            pass

        # 3. Cấu hình Weights & Biases
        wandb_enabled = _setup_wandb(cfg)

        # 4. Nạp model ban đầu
        pretrained = cfg["model"].get("pretrained", "yolo26m.pt")
        model = load_model(pretrained)

        # 5. Tham số huấn luyện Ultralytics
        online_args = build_online_transforms(cfg, "ultralytics")
        train_args: dict[str, Any] = {
            "data": runtime_yaml.resolve().as_posix(),
            "epochs": int(cfg["train"]["epochs"]),
            "batch": int(cfg["train"]["batch_size"]),
            "imgsz": int(cfg["data"]["image_size"]),
            "seed": int(cfg["seed"]),
            "patience": int(cfg["train"].get("early_stopping_patience", 10)),
            "amp": bool(cfg["train"].get("amp", True)),
            "lr0": float(cfg["train"].get("lr0", 0.01)),
            "optimizer": cfg["train"].get("optimizer", "auto"),
            "project": run_dir.parent.resolve().as_posix(),
            "name": run_dir.name,
            "exist_ok": True,
            **online_args,
        }

        # Thiết lập thiết bị tính toán (GPU / CPU)
        num_gpus = int(cfg.get("num_gpus", 1))
        if num_gpus == 2:
            train_args["device"] = [0, 1]
        elif num_gpus == 1:
            train_args["device"] = 0
        else:
            train_args["device"] = "cpu"

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

        # 8. Đồng bộ kết quả đánh giá Test lên W&B nếu có
        if wandb_enabled:
            try:
                import wandb

                if test_metrics is None and (run_dir / "metrics.json").is_file():
                    test_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

                if test_metrics and wandb.run is not None:
                    test_payload = {
                        f"test/{k}": float(v)
                        for k, v in test_metrics.items()
                        if isinstance(v, (int, float))
                    }
                    wandb.log(test_payload)
                    for k, v in test_payload.items():
                        wandb.run.summary[k] = v

                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass

        return run_dir

    def load_detector(self, weights=None) -> YoloDetector:
        return YoloDetector(weights or self.default_weights(), self.cfg)
