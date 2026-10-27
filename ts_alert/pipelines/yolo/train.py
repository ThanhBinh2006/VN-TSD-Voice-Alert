"""YoloPipeline: huấn luyện YOLO qua Ultralytics API.   Phụ trách: Thành viên 1"""
from __future__ import annotations

import shutil
from pathlib import Path

from ts_alert.config.resolve import get_processed_root, resolve_path
from ts_alert.data.preprocess_online import build_online_transforms
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.seed import set_seed


class YoloPipeline(BasePipeline):
    name = "yolo"

    def fit(self) -> Path:
        """Fine-tune YOLO trên <processed root>/data.yaml (get_processed_root(cfg)).

        Gợi ý các bước:
          1. set_seed(cfg["seed"]); run_dir = make_run_dir(outputs_dir, "yolo", cfg["train"]["run_name"], cfg)
          2. model = load_model(cfg["model"]["pretrained"])
          3. model.train(data=..., epochs=cfg["train"]["epochs"], batch=cfg["train"]["batch_size"] (TỔNG, không chia),
                         imgsz=cfg["data"]["image_size"], seed=cfg["seed"], patience=cfg["train"]["early_stopping_patience"], amp=cfg["train"]["amp"],
                         device=[0, 1] nếu cfg["num_gpus"] == 2, project=run_dir.parent, name=run_dir.name,
                         + tham số từ build_online_transforms(cfg, "ultralytics"))
          4. copy best.pt sang cfg["model"]["weights"] để suy luận dùng
          5. nếu cfg["train"]["require_wandb"]: log lên W&B (cfg["project"]); nếu cfg["eval"]["auto_eval_test"]: self.evaluate(run_dir)
        Trả về run_dir.
        """
        cfg = self.cfg
        set_seed(cfg["seed"])

        # 1. Tạo thư mục output run_dir
        outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
        run_name = cfg["train"].get("run_name") or f"yolo_s{cfg['seed']}_ep{cfg['train']['epochs']}_bs{cfg['train']['batch_size']}"
        run_dir = make_run_dir(outputs_dir, "yolo", run_name, cfg)

        # 2. Chuẩn bị đường dẫn data.yaml (tạo bản runtime với path chuẩn xác cho Kaggle/Local)
        processed_root = get_processed_root(cfg)
        source_yaml = processed_root / "data.yaml"
        if not source_yaml.exists():
            raise FileNotFoundError(f"Không tìm thấy data.yaml tại {source_yaml}. Hãy kiểm tra đường dẫn dataset.")

        import yaml
        with open(source_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = processed_root.resolve().as_posix()
        runtime_yaml = run_dir / "data.yaml"
        with open(runtime_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        # 3. Nạp model ban đầu
        pretrained = cfg["model"].get("pretrained", "yolo26m.pt")
        model = load_model(pretrained)

        # 4. Tham số huấn luyện Ultralytics
        online_args = build_online_transforms(cfg, "ultralytics")
        train_args = {
            "data": str(runtime_yaml),
            "epochs": int(cfg["train"]["epochs"]),
            "batch": int(cfg["train"]["batch_size"]),
            "imgsz": int(cfg["data"]["image_size"]),
            "seed": int(cfg["seed"]),
            "patience": int(cfg["train"].get("early_stopping_patience", 10)),
            "amp": bool(cfg["train"].get("amp", True)),
            "lr0": float(cfg["train"].get("lr0", 0.01)),
            "optimizer": cfg["train"].get("optimizer", "auto"),
            "project": str(run_dir.parent),
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

        # 5. Sao chép best.pt sang đường dẫn weights được chỉ định
        best_pt = self.best_checkpoint(run_dir)
        target_weights = resolve_path(cfg["model"]["weights"])
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, target_weights)
        print(f"[YoloPipeline] Đã lưu checkpoint tốt nhất sang: {target_weights}")

        # 6. Tự động đánh giá trên tập Test nếu được kích hoạt
        if cfg.get("eval", {}).get("auto_eval_test", False):
            try:
                print(f"[YoloPipeline] Tự động đánh giá trên tập Test...")
                self.evaluate(run_dir)
            except Exception as e:
                print(f"[YoloPipeline] Chú ý: Đánh giá tập Test bị bỏ qua: {e}")

        return run_dir

    def load_detector(self, weights=None) -> YoloDetector:
        return YoloDetector(weights or self.default_weights(), self.cfg)
