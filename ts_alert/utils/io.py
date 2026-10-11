"""Đọc/ghi YAML, JSON, CSV an toàn UTF-8."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_classes(csv_path) -> dict:
    """Đọc configs/classes.csv -> {class_id: {"name": str, "tts_text": str}}."""
    classes = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            classes[int(row["id"])] = {"name": row["name"], "tts_text": row.get("tts_text", "")}
    return classes
