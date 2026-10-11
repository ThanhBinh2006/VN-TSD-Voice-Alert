"""YoloPipeline: huấn luyện YOLO qua Ultralytics API.   Phụ trách: Thành viên 1"""
from __future__ import annotations

from pathlib import Path

from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.yolo.infer import YoloDetector


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
        # TODO: Thành viên 1
        raise NotImplementedError("TODO: Thành viên 1")

    def load_detector(self, weights=None) -> YoloDetector:
        return YoloDetector(weights or self.default_weights(), self.cfg)
