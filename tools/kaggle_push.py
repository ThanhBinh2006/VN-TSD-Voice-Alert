"""Công cụ tổng quát đẩy notebook lên Kaggle và quản lý tiến trình huấn luyện qua Kaggle CLI.

Không phụ thuộc vào file JSON cố định: nhận thông tin xác thực và cấu hình qua
tham số dòng lệnh (CLI) hoặc biến môi trường (trong môi trường ảo venv / file .env).

Biến môi trường hỗ trợ:
    KAGGLE_USERNAME      Tên tài khoản Kaggle (bắt buộc nếu không truyền --username)
    KAGGLE_KEY           API Key của Kaggle (tùy chọn nếu đã cấu hình trong môi trường)
    KAGGLE_DATASET       Dataset gắn kèm (phân tách bằng dấu phẩy nếu có nhiều dataset)
    KAGGLE_KERNEL_SLUG   Slug mặc định của kernel trên Kaggle

Ví dụ sử dụng:
    # 1. Thiết lập biến môi trường trong venv (PowerShell):
    $env:KAGGLE_USERNAME = "your_username"
    $env:KAGGLE_KEY = "your_api_key"
    $env:KAGGLE_DATASET = "your_username/tsd-voice-alert-preprocessed-data"

    # 2. Đẩy notebook và bắt đầu huấn luyện trên Kaggle GPU:
    python -m tools.kaggle_push --pipeline yolo
    # Hoặc truyền trực tiếp qua tham số:
    python -m tools.kaggle_push --pipeline yolo --username your_username --dataset your_username/tsd-processed

    # 3. Kiểm tra trạng thái thực thi trên Kaggle:
    python -m tools.kaggle_push --status --pipeline yolo

    # 4. Tải kết quả (run.zip, weights best.pt / best.pth) sau khi huấn luyện xong:
    python -m tools.kaggle_push --pull --pipeline yolo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_NOTEBOOKS: dict[str, str] = {
    "yolo": "04_train_yolo.ipynb",
    "faster_rcnn": "03_train_faster_rcnn.ipynb",
    "preprocess": "02_preprocess_offline.ipynb",
    "eda": "01_data_analysis.ipynb",
    "demo": "05_demo_pipeline.ipynb",
}

PIPELINE_WEIGHTS: dict[str, str] = {
    "yolo": "best.pt",
    "faster_rcnn": "best.pth",
}


def load_dotenv(env_file: Path | None = None) -> None:
    """Nạp biến môi trường từ file .env (nếu có) vào os.environ mà không ghi đè biến đã có."""
    path = env_file or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key:
                os.environ.setdefault(key, val)
    except Exception:
        pass


def resolve_credentials(username: str | None = None, key: str | None = None) -> tuple[str, dict[str, str]]:
    """Lấy username và chuẩn bị biến môi trường thực thi cho Kaggle CLI."""
    load_dotenv()
    env = os.environ.copy()

    resolved_user = (username or env.get("KAGGLE_USERNAME") or "").strip()
    resolved_key = (key or env.get("KAGGLE_KEY") or env.get("KAGGLE_API_TOKEN") or "").strip()

    if not resolved_user:
        raise ValueError(
            "Chưa xác định được Kaggle Username.\n"
            "Hãy truyền tham số `--username <tên_user>` hoặc đặt biến môi trường trong venv / file `.env`:\n"
            "  PowerShell : $env:KAGGLE_USERNAME = 'your_username'\n"
            "  Bash/Zsh   : export KAGGLE_USERNAME='your_username'\n"
            "  File .env  : KAGGLE_USERNAME=your_username"
        )

    env["KAGGLE_USERNAME"] = resolved_user
    if resolved_key:
        env["KAGGLE_KEY"] = resolved_key

    return resolved_user, env


def resolve_kernel_id(
    pipeline: str,
    username: str,
    kernel_slug: str | None = None,
) -> tuple[str, str]:
    """Trả về (kernel_slug, kernel_id) theo dạng `<username>/<kernel_slug>`."""
    slug = (kernel_slug or os.environ.get("KAGGLE_KERNEL_SLUG") or "").strip()
    if not slug:
        default_suffix = f"{pipeline}26m" if pipeline == "yolo" else pipeline
        slug = f"tsd-voice-alert-{default_suffix}"

    if "/" in slug:
        owner, _, short_slug = slug.partition("/")
        return short_slug, f"{owner}/{short_slug}"
    return slug, f"{username}/{slug}"


def resolve_notebook_path(pipeline: str, notebook: str | Path | None = None) -> Path:
    """Tìm đường dẫn file notebook cần đẩy lên Kaggle."""
    if notebook:
        nb_path = Path(notebook)
        if not nb_path.is_Absolute() if hasattr(nb_path, "is_Absolute") else not nb_path.is_absolute():
            nb_path = PROJECT_ROOT / nb_path
        if not nb_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy notebook tại: {nb_path}")
        return nb_path.resolve()

    mapped = PIPELINE_NOTEBOOKS.get(pipeline)
    candidates = [
        PROJECT_ROOT / "notebooks" / mapped if mapped else None,
        PROJECT_ROOT / "notebooks" / f"04_train_{pipeline}.ipynb",
        PROJECT_ROOT / "notebooks" / f"03_train_{pipeline}.ipynb",
    ]
    for cand in candidates:
        if cand is not None and cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Không tìm thấy notebook mặc định cho pipeline '{pipeline}'. "
        "Hãy chỉ định rõ qua tham số `--notebook <đường_dẫn_ipynb>`."
    )


def resolve_datasets(
    username: str,
    datasets: list[str] | None = None,
    no_dataset: bool = False,
    default_dataset_name: str = "tsd-voice-alert-preprocessed-data",
) -> list[str]:
    """Chuẩn hóa danh sách dataset gắn vào kernel (`<owner>/<dataset-slug>`)."""
    if no_dataset:
        return []

    raw_items: list[str] = []
    if datasets:
        for item in datasets:
            raw_items.extend(part.strip() for part in item.split(",") if part.strip())
    elif os.environ.get("KAGGLE_DATASET"):
        raw_items.extend(
            part.strip() for part in os.environ["KAGGLE_DATASET"].split(",") if part.strip()
        )
    else:
        raw_items.append(default_dataset_name)

    normalized: list[str] = []
    for ds in raw_items:
        normalized.append(ds if "/" in ds else f"{username}/{ds}")
    return normalized


def _inject_wandb_env_if_present(nb_path: Path, env: dict[str, str]) -> None:
    """Chèn biến môi trường W&B từ .env cục bộ vào bản sao notebook tạm thời khi đẩy kernel private."""
    wandb_keys = ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_PROJECT", "WANDB_ENTITY")
    kv = {k: env[k].strip() for k in wandb_keys if env.get(k, "").strip()}
    if not kv:
        return
    try:
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        lines = ["import os\n"] + [
            f"os.environ.setdefault({json.dumps(k)}, {json.dumps(v)})\n" for k, v in kv.items()
        ]
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["source"] = lines + list(cell.get("source", []))
                break
        nb_path.write_text(json.dumps(nb_data, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def push_kernel(
    pipeline: str = "yolo",
    username: str | None = None,
    key: str | None = None,
    notebook: str | Path | None = None,
    kernel_slug: str | None = None,
    title: str | None = None,
    datasets: list[str] | None = None,
    no_dataset: bool = False,
    competitions: list[str] | None = None,
    kernels: list[str] | None = None,
    models: list[str] | None = None,
    enable_gpu: bool = True,
    enable_tpu: bool = False,
    enable_internet: bool = True,
    is_private: bool = True,
) -> None:
    resolved_user, env = resolve_credentials(username, key)
    nb_src = resolve_notebook_path(pipeline, notebook)
    slug, kernel_id = resolve_kernel_id(pipeline, resolved_user, kernel_slug)
    dataset_sources = resolve_datasets(resolved_user, datasets, no_dataset=no_dataset)

    meta = {
        "id": kernel_id,
        "title": title or slug,
        "code_file": nb_src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": str(bool(is_private)).lower(),
        "enable_gpu": str(bool(enable_gpu)).lower(),
        "enable_tpu": str(bool(enable_tpu)).lower(),
        "enable_internet": str(bool(enable_internet)).lower(),
        "dataset_sources": dataset_sources,
        "competition_sources": competitions or [],
        "kernel_sources": kernels or [],
        "model_sources": models or [],
    }

    # Tạo thư mục tạm thời để đẩy lên Kaggle rồi tự động dọn sạch, không lưu file JSON cố định
    with tempfile.TemporaryDirectory(prefix=f"kaggle_{pipeline}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        nb_dest = tmp_path / nb_src.name
        shutil.copy2(nb_src, nb_dest)
        if is_private:
            _inject_wandb_env_if_present(nb_dest, env)
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f">> [Kaggle Push] Đang đẩy notebook '{nb_src.name}' lên Kaggle ({kernel_id})...")
        print(f"   - Datasets : {dataset_sources or '(không gắn)'}")
        print(f"   - GPU      : {enable_gpu} | Internet: {enable_internet} | Private: {is_private}")

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )

    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)

    print(f"\n[OK] Đã đẩy lên Kaggle! Xem trực tiếp tại:")
    print(f"     https://www.kaggle.com/code/{kernel_id}")
    print(f"\nĐể kiểm tra trạng thái:")
    print(f"     python -m tools.kaggle_push --status --pipeline {pipeline}")


def check_status(
    pipeline: str = "yolo",
    username: str | None = None,
    key: str | None = None,
    kernel_slug: str | None = None,
) -> None:
    resolved_user, env = resolve_credentials(username, key)
    _, kernel_id = resolve_kernel_id(pipeline, resolved_user, kernel_slug)

    print(f">> [Kaggle Status] Kiểm tra trạng thái kernel: {kernel_id}...")
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", kernel_id],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)


def pull_output(
    pipeline: str = "yolo",
    out_dir: str | Path | None = None,
    username: str | None = None,
    key: str | None = None,
    kernel_slug: str | None = None,
    weights_name: str | None = None,
) -> None:
    resolved_user, env = resolve_credentials(username, key)
    _, kernel_id = resolve_kernel_id(pipeline, resolved_user, kernel_slug)

    target_dir = Path(out_dir) if out_dir else (PROJECT_ROOT / "outputs" / f"kaggle_{pipeline}")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f">> [Kaggle Pull] Đang tải output từ {kernel_id} về {target_dir}...")
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", str(target_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)

    expected_weight = weights_name or PIPELINE_WEIGHTS.get(pipeline)
    if expected_weight:
        for p in target_dir.rglob(expected_weight):
            model_dest = PROJECT_ROOT / "models" / pipeline / expected_weight
            model_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, model_dest)
            print(f"[OK] Đã tự động cập nhật weights tốt nhất sang: {model_dest}")
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Công cụ tổng quát đẩy notebook và quản lý huấn luyện trên Kaggle qua CLI / biến môi trường."
    )
    parser.add_argument(
        "--pipeline",
        default="yolo",
        help="Tên pipeline (yolo, faster_rcnn, preprocess, eda, demo). Mặc định: yolo",
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help="Đường dẫn tới file .ipynb cần đẩy (ghi đè notebook mặc định của pipeline)",
    )
    parser.add_argument(
        "-u",
        "--username",
        default=None,
        help="Kaggle username (hoặc đặt biến môi trường KAGGLE_USERNAME)",
    )
    parser.add_argument(
        "-k",
        "--key",
        default=None,
        help="Kaggle API key (hoặc đặt biến môi trường KAGGLE_KEY)",
    )
    parser.add_argument(
        "--kernel-slug",
        default=None,
        help="Tên định danh (slug) của notebook trên Kaggle (hoặc đặt KAGGLE_KERNEL_SLUG)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Tiêu đề hiển thị của notebook trên Kaggle (mặc định trùng với kernel-slug)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="Dataset gắn kèm (vd: user/dataset-slug hoặc chỉ dataset-slug). Có thể truyền nhiều lần hoặc đặt KAGGLE_DATASET",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Không gắn kèm dataset nào vào notebook",
    )
    parser.add_argument(
        "--gpu",
        dest="enable_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật/tắt GPU trên Kaggle (mặc định: --gpu)",
    )
    parser.add_argument(
        "--tpu",
        dest="enable_tpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bật/tắt TPU trên Kaggle (mặc định: --no-tpu)",
    )
    parser.add_argument(
        "--internet",
        dest="enable_internet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật/tắt Internet trên Kaggle (mặc định: --internet)",
    )
    parser.add_argument(
        "--private",
        dest="is_private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Đặt chế độ riêng tư/công khai cho notebook (mặc định: --private, dùng --no-private để public)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Kiểm tra trạng thái đang chạy trên Kaggle",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Tải output / weights từ Kaggle về máy",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Thư mục lưu output tải về khi dùng --pull",
    )
    parser.add_argument(
        "--weights-name",
        default=None,
        help="Tên file trọng số tự động sao chép vào models/<pipeline>/ khi dùng --pull (mặc định: best.pt hoặc best.pth)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.status:
            check_status(
                pipeline=args.pipeline,
                username=args.username,
                key=args.key,
                kernel_slug=args.kernel_slug,
            )
        elif args.pull:
            pull_output(
                pipeline=args.pipeline,
                out_dir=args.out_dir,
                username=args.username,
                key=args.key,
                kernel_slug=args.kernel_slug,
                weights_name=args.weights_name,
            )
        else:
            push_kernel(
                pipeline=args.pipeline,
                username=args.username,
                key=args.key,
                notebook=args.notebook,
                kernel_slug=args.kernel_slug,
                title=args.title,
                datasets=args.datasets,
                no_dataset=args.no_dataset,
                enable_gpu=args.enable_gpu,
                enable_tpu=args.enable_tpu,
                enable_internet=args.enable_internet,
                is_private=args.is_private,
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[Lỗi] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
