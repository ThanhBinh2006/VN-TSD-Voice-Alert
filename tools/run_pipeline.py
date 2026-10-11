"""CLI xử lý một ảnh/video end-to-end (không cần web).

python -m tools.run_pipeline --input sample.mp4 --model yolo
"""
import argparse
import json

from ts_alert.config.resolve import parse_overrides, resolve_config
from ts_alert.pipeline import process_media
from ts_alert.types import MODEL_NAMES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--runtime", default="local")
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    cfg = resolve_config(args.model, args.runtime, parse_overrides(args.set))
    result = process_media(args.input, args.model, cfg=cfg, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
