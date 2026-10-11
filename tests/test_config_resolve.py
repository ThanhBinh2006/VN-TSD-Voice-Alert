from ts_alert.config.resolve import deep_merge, get_num_classes, parse_overrides, resolve_config


def test_deep_merge_nested():
    assert deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "b": 1}


def test_parse_overrides_types():
    assert parse_overrides(["train.epochs=5", "flag=true"]) == {"train": {"epochs": 5}, "flag": True}


def test_inherits_shared_and_pipeline_merge():
    cfg = resolve_config("yolo", "local")
    assert cfg["pipeline"] == "yolo"
    assert cfg["train"]["batch_size"] == 8        # từ shared.yaml (inherits)
    assert cfg["train"]["epochs"] == 50           # từ yolo.yaml, deep-merge không làm mất batch_size
    assert cfg["model"]["name"] == "yolo26m"
    assert cfg["device"] == "auto"                # từ runtime/local.yaml


def test_kaggle_runtime_and_cli_override():
    cfg = resolve_config("faster_rcnn", "configs/runtime/kaggle.yaml", {"train": {"epochs": 1}})
    assert cfg["num_gpus"] == 2
    assert cfg["paths"]["outputs_dir"].startswith("/kaggle/working")
    assert cfg["paths"]["classes_csv"] == "configs/classes.csv"   # path không ghi đè vẫn giữ
    assert cfg["train"]["epochs"] == 1


def test_get_num_classes_fallback_to_csv():
    cfg = resolve_config("yolo", "local")
    assert get_num_classes(cfg) >= 1
    cfg["data"]["num_classes"] = 52
    assert get_num_classes(cfg) == 52
