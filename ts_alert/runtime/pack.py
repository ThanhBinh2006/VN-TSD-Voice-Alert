"""Đóng gói thư mục kết quả một lần chạy (checkpoint, metrics, log) thành run.zip để tải về từ Kaggle."""
from __future__ import annotations

import zipfile
from pathlib import Path


def pack_run_dir(run_dir, out_zip=None) -> Path:
    run_dir = Path(run_dir)
    out_zip = Path(out_zip) if out_zip else run_dir / "run.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file() and f.resolve() != out_zip.resolve():
                zf.write(f, f.relative_to(run_dir))
    return out_zip
