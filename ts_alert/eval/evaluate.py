"""Luồng đánh giá một detector trên tập Test: độ chính xác (mAP...) + tốc độ (FPS)."""
from __future__ import annotations

from ts_alert.config.resolve import get_num_classes
from ts_alert.eval import benchmark, metrics
from ts_alert.runtime import video_io


def load_split_samples(cfg: dict, split: str = "test") -> list[dict]:
    """Đọc <get_processed_root(cfg)>/{images,labels}/<split>.

    Trả [{"image_path": Path, "gt": [{"bbox": [x1,y1,x2,y2] (pixel), "class_id": int}, ...]}, ...]
    """
    from PIL import Image
    from ts_alert.config.resolve import get_processed_root

    processed = get_processed_root(cfg)
    img_dir = processed / "images" / split
    lbl_dir = processed / "labels" / split
    samples = []
    if not img_dir.is_dir():
        return samples

    for img_p in sorted(img_dir.glob("*.*")):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        gt = []
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            h, w = 640, 640
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
            except Exception:
                pass
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            gt.append({"bbox": [x1, y1, x2, y2], "class_id": cid})
                        except ValueError:
                            continue
        samples.append({"image_path": img_p, "gt": gt})
    return samples


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
