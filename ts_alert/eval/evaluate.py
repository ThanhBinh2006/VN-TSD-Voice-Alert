"""Luồng đánh giá một detector trên tập Test: độ chính xác (mAP...) + tốc độ (FPS)."""
from __future__ import annotations

from ts_alert.config.resolve import get_num_classes
from ts_alert.eval import benchmark, metrics
from ts_alert.runtime import video_io


def load_split_samples(cfg: dict, split: str = "test") -> list[dict]:
    """Phụ trách: Thành viên 2. Đọc <get_processed_root(cfg)>/{images,labels}/<split>.

    Trả [{"image_path": Path, "gt": [{"bbox": [x1,y1,x2,y2] (pixel), "class_id": int}, ...]}, ...]
    """
    # TODO: Thành viên 2
    raise NotImplementedError("TODO: Thành viên 2")


def evaluate_detector(detector, cfg: dict, split: str = "test") -> dict:
    samples = load_split_samples(cfg, split)

    frames, preds, gts = [], [], []
    for s in samples:
        frame = video_io.read_image(s["image_path"])
        frames.append(frame)
        preds.append(detector.predict(frame))
        gts.append(s["gt"])

    result = metrics.compute_metrics(preds, gts, get_num_classes(cfg))
    bench = cfg["eval"]["benchmark"]
    result.update(benchmark.measure_fps(detector, frames[:bench["num_frames"]], warmup=bench["warmup"]))
    return result
