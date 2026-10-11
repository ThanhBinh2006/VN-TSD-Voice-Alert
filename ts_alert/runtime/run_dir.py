"""Tạo thư mục output cho một lần chạy: outputs/<pipeline>/<tên>_<timestamp>_<hash>/"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from ts_alert.utils.io import write_yaml


def make_run_dir(output_root, pipeline: str, run_name: str | None = None, cfg: dict | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(json.dumps(cfg or {}, sort_keys=True, default=str).encode()).hexdigest()[:6]
    path = Path(output_root) / pipeline / f"{run_name or pipeline}_{stamp}_{digest}"
    path.mkdir(parents=True, exist_ok=True)
    if cfg:
        write_yaml(cfg, path / "config.yaml")
    return path
