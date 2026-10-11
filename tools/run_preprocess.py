"""CLI tiền xử lý offline.   python -m tools.run_preprocess --data-root <DATA_PATH>"""
import argparse
import json

from ts_alert.config.resolve import parse_overrides, resolve_config
from ts_alert.data.preprocess_offline import run_preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="thư mục dữ liệu gốc (mặc định paths.raw_dir)")
    p.add_argument("--runtime", default="local")
    p.add_argument("--set", nargs="*", default=[], help="ghi đè cấu hình, vd. image_size=512")
    args = p.parse_args()
    cfg = resolve_config(None, args.runtime, parse_overrides(args.set))
    print(json.dumps(run_preprocess(cfg, args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
