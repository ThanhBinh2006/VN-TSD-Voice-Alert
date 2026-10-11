from ts_alert.runtime.run_dir import make_run_dir


def test_make_run_dir(tmp_path):
    d = make_run_dir(tmp_path, "yolo", cfg={"seed": 1})
    assert d.is_dir() and d.parent.name == "yolo"
    assert (d / "config.yaml").exists()
