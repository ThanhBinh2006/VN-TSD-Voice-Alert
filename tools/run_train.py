"""CLI huấn luyện.

python -m tools.run_train --pipeline yolo|faster_rcnn --runtime configs/runtime/local.yaml
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml \
    --run-name <tên> --overrides outputs/_notebook/overrides_faster_rcnn.yaml --data-root <processed> --no-online
"""
import argparse

from tools.common import add_common_args, build_overrides
from ts_alert.config.resolve import resolve_config
from ts_alert.pipelines import get_pipeline
from ts_alert.types import MODEL_NAMES
from ts_alert.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, choices=MODEL_NAMES)
    add_common_args(p)
    args = p.parse_args()

    cfg = resolve_config(args.pipeline, args.runtime, build_overrides(args))
    set_seed(cfg["seed"])
    run_dir = get_pipeline(args.pipeline, cfg).fit()
    print(f"Huấn luyện xong. Kết quả: {run_dir}")


if __name__ == "__main__":
    main()
