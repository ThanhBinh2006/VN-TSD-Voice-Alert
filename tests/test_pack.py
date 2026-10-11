import zipfile

from ts_alert.runtime.pack import pack_run_dir


def test_pack_run_dir(tmp_path):
    (tmp_path / "metrics.json").write_text("{}")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"x")
    out = pack_run_dir(tmp_path)
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["metrics.json", "weights/best.pt"]
