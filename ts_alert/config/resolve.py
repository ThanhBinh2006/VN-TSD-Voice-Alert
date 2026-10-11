"""Gộp cấu hình nhiều tầng: shared.yaml -> pipelines/<name>.yaml (inherits) -> runtime/<env>.yaml -> CLI (--set)."""
from __future__ import annotations

from pathlib import Path

import yaml

from ts_alert.utils.io import load_classes, read_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_path(p) -> Path:
    """Đường dẫn tương đối -> tuyệt đối theo thư mục gốc dự án (đường dẫn tuyệt đối giữ nguyên)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(items) -> dict:
    """["train.epochs=5", "flag=true"] -> {"train": {"epochs": 5}, "flag": True}"""
    result: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def _load_with_inherits(path: Path) -> dict:
    cfg = read_yaml(path)
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = deep_merge(_load_with_inherits(resolve_path(parent)), cfg)
    return cfg


def _runtime_path(runtime: str) -> Path:
    cand = Path(runtime)
    if cand.suffix in (".yaml", ".yml"):
        return cand if cand.is_absolute() or cand.exists() else PROJECT_ROOT / cand
    return CONFIG_DIR / "runtime" / f"{runtime}.yaml"


def resolve_config(pipeline: str | None = None, runtime: str | None = "local", overrides: dict | None = None) -> dict:
    if pipeline:
        cfg = _load_with_inherits(CONFIG_DIR / "pipelines" / f"{pipeline}.yaml")
    else:
        cfg = read_yaml(CONFIG_DIR / "shared.yaml")
    if runtime:
        cfg = deep_merge(cfg, read_yaml(_runtime_path(runtime)))
    cfg = deep_merge(cfg, overrides or {})
    cfg["pipeline"] = pipeline
    return cfg


def get_processed_root(cfg: dict) -> Path:
    """Thư mục dữ liệu đã tiền xử lý để train/eval: ưu tiên data.root, nếu null dùng paths.processed_dir."""
    return resolve_path(cfg["data"].get("root") or cfg["paths"]["processed_dir"])


def get_num_classes(cfg: dict) -> int:
    """Số lớp biển báo (KHÔNG tính background). Ưu tiên data.num_classes, nếu null thì đếm configs/classes.csv."""
    n = cfg["data"].get("num_classes")
    return int(n) if n else len(load_classes(resolve_path(cfg["paths"]["classes_csv"])))
