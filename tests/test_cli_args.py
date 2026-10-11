import argparse

from tools.common import add_common_args, build_overrides
from ts_alert.config.resolve import get_processed_root, resolve_config
from ts_alert.utils.io import write_yaml


def _parse(argv):
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p.parse_args(argv)


def test_build_overrides_priority(tmp_path):
    f = tmp_path / "ov.yaml"
    write_yaml({"train": {"epochs": 3, "lr": 0.1}, "seed": 1}, f)
    args = _parse(["--overrides", str(f), "--set", "train.epochs=7", "--data-root", "/x/processed",
                   "--no-online", "--run-name", "r1"])
    ov = build_overrides(args)
    assert ov["train"] == {"epochs": 7, "lr": 0.1, "run_name": "r1"}   # --set thắng file; run-name thêm vào
    assert ov["data"]["root"] == "/x/processed"
    assert ov["online_preprocess"]["enabled"] is False
    assert ov["seed"] == 1


def test_online_flag_default_is_none():
    assert "online_preprocess" not in build_overrides(_parse([]))


def test_processed_root_fallback_and_override():
    cfg = resolve_config("yolo", "local")
    assert get_processed_root(cfg).name == "processed"
    cfg2 = resolve_config("yolo", "local", {"data": {"root": "/kaggle/input/ds"}})
    assert str(get_processed_root(cfg2)) == "/kaggle/input/ds"


def test_shared_keys_present():
    cfg = resolve_config("faster_rcnn", "local")
    for key in ("early_stopping_patience", "amp", "primary_metric", "batch_size", "require_wandb", "run_name"):
        assert key in cfg["train"]
    assert cfg["eval"]["batch_size"] == 16 and cfg["eval"]["auto_eval_test"] is True
