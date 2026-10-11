"""Tham số dòng lệnh dùng chung cho run_train / run_eval (khớp cách notebook gọi)."""
from __future__ import annotations

import argparse

from ts_alert.config.resolve import deep_merge, parse_overrides
from ts_alert.utils.io import read_yaml


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runtime", default="local", help="local | kaggle | đường dẫn file yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--overrides", default=None, help="file yaml ghi đè cấu hình (notebook sinh ra)")
    p.add_argument("--data-root", default=None, help="thư mục processed (ghi đè data.root)")
    p.add_argument("--online", action=argparse.BooleanOptionalAction, default=None,
                   help="bật/tắt online_preprocess.enabled")
    p.add_argument("--set", nargs="*", default=[], help="ghi đè nhanh, vd. train.epochs=5")


def build_overrides(args) -> dict:
    """Thứ tự ưu tiên (thấp -> cao): --overrides, --set, rồi các cờ riêng lẻ."""
    ov: dict = read_yaml(args.overrides) if args.overrides else {}
    ov = deep_merge(ov, parse_overrides(args.set))
    if args.data_root:
        ov = deep_merge(ov, {"data": {"root": args.data_root}})
    if args.online is not None:
        ov = deep_merge(ov, {"online_preprocess": {"enabled": args.online}})
    if args.run_name:
        ov = deep_merge(ov, {"train": {"run_name": args.run_name}})
    return ov
