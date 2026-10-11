"""CLI đánh giá checkpoint trên tập Test.   python -m tools.run_eval --pipeline yolo --run-dir outputs/yolo/<run_id>"""
import argparse
import json
from pathlib import Path

from tools.common import add_common_args, build_overrides
from ts_alert.config.resolve import resolve_config
from ts_alert.pipelines import get_pipeline
from ts_alert.types import MODEL_NAMES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, choices=MODEL_NAMES)
    p.add_argument("--run-dir", required=True)
    add_common_args(p)
    args = p.parse_args()
    cfg = resolve_config(args.pipeline, args.runtime, build_overrides(args))
    metrics = get_pipeline(args.pipeline, cfg).evaluate(Path(args.run_dir))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
