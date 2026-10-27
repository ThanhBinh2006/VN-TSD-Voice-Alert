"""Công cụ đẩy notebook lên Kaggle và theo dõi tiến trình huấn luyện qua Kaggle CLI.

Sử dụng:
    # 1. Đẩy notebook và bắt đầu huấn luyện trên Kaggle GPU:
    python -m tools.kaggle_push --pipeline yolo

    # 2. Kiểm tra trạng thái thực thi trên Kaggle:
    python -m tools.kaggle_push --status

    # 3. Tải kết quả (run.zip, weights best.pt) sau khi huấn luyện xong:
    python -m tools.kaggle_push --pull
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "deploy" / "kaggle"


def _get_username() -> str:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("username", "zero001206")
        except Exception:
            pass
    return "zero001206"


def push_kernel(pipeline: str = "yolo") -> None:
    username = _get_username()
    pipe_dir = KAGGLE_DIR / pipeline
    pipe_dir.mkdir(parents=True, exist_ok=True)

    nb_src = PROJECT_ROOT / "notebooks" / f"04_train_{pipeline}.ipynb"
    if not nb_src.exists():
        nb_src = PROJECT_ROOT / "notebooks" / f"03_train_{pipeline}.ipynb"
    if not nb_src.exists():
        raise FileNotFoundError(f"Không tìm thấy notebook huấn luyện cho pipeline {pipeline} tại {nb_src}")

    nb_dest = pipe_dir / nb_src.name
    shutil.copy2(nb_src, nb_dest)

    kernel_slug = f"tsd-voice-alert-{pipeline}26m" if pipeline == "yolo" else f"tsd-voice-alert-{pipeline}"
    kernel_id = f"{username}/{kernel_slug}"

    meta = {
        "id": kernel_id,
        "title": kernel_slug,
        "code_file": nb_src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            f"{username}/tsd-voice-alert-preprocessed-data"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    meta_file = pipe_dir / "kernel-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f">> [Kaggle Push] Đang đẩy notebook {nb_src.name} lên Kaggle ({kernel_id})...")
    res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(pipe_dir)], capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    print(f"\n[OK] Đã đẩy lên Kaggle! Bạn có thể xem trực tiếp tại:")
    print(f"     https://www.kaggle.com/code/{kernel_id}")
    print(f"\nĐể kiểm tra trạng thái:")
    print(f"     python -m tools.kaggle_push --status --pipeline {pipeline}")


def check_status(pipeline: str = "yolo") -> None:
    username = _get_username()
    kernel_slug = f"tsd-voice-alert-{pipeline}26m" if pipeline == "yolo" else f"tsd-voice-alert-{pipeline}"
    kernel_id = f"{username}/{kernel_slug}"

    print(f">> [Kaggle Status] Kiểm tra trạng thái kernel: {kernel_id}...")
    res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kernel_id], capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)


def pull_output(pipeline: str = "yolo", out_dir: str | Path | None = None) -> None:
    username = _get_username()
    kernel_slug = f"tsd-voice-alert-{pipeline}26m" if pipeline == "yolo" else f"tsd-voice-alert-{pipeline}"
    kernel_id = f"{username}/{kernel_slug}"

    target_dir = Path(out_dir) if out_dir else (PROJECT_ROOT / "outputs" / f"kaggle_{pipeline}")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f">> [Kaggle Pull] Đang tải output từ {kernel_id} về {target_dir}...")
    res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", str(target_dir)], capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    # Nếu có best.pt trong output tải về, tự động sao chép sang models/yolo/best.pt
    for p in target_dir.rglob("best.pt"):
        model_dest = PROJECT_ROOT / "models" / pipeline / "best.pt"
        model_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, model_dest)
        print(f"[OK] Đã tự động cập nhật weights tốt nhất sang: {model_dest}")
        break


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Kernels Push & Manage Tool")
    parser.add_argument("--pipeline", default="yolo", choices=["yolo", "faster_rcnn"])
    parser.add_argument("--status", action="store_true", help="Kiểm tra trạng thái đang chạy trên Kaggle")
    parser.add_argument("--pull", action="store_true", help="Tải output / weights từ Kaggle về máy")
    parser.add_argument("--out-dir", default=None, help="Thư mục lưu output tải về")
    args = parser.parse_args()

    if args.status:
        check_status(args.pipeline)
    elif args.pull:
        pull_output(args.pipeline, args.out_dir)
    else:
        push_kernel(args.pipeline)


if __name__ == "__main__":
    main()
