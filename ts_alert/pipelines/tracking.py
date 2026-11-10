"""Theo dõi huấn luyện, quản lý W&B và lưu thống kê theo epoch."""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import shlex
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ts_alert.config.resolve import PROJECT_ROOT

IMMUTABLE_SHARED_KEYS = (
    "seed",
    "data.image_size",
    "data.pad_value",
    "data.split",
    "data.num_classes",
    "train.batch_size",
    "train.epochs",
    "train.early_stopping_patience",
    "train.amp",
    "train.primary_metric",
    "eval.batch_size",
    "eval.split",
    "eval.metrics",
)

_TRAIN_KEYS = {
    "lr",
    "lr0",
    "optimizer",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
}
_ONLINE_KEYS = {"occlusion_p", "occlusion_ratio"}

_STANDARD_COLUMNS = (
    "epoch",
    "train/loss",
    "val/loss",
    "lr",
    "system/learning_rate",
    "val/mAP50-95",
    "val/mAP50",
    "val/mAP75",
    "val/AR100",
    "val/precision",
    "val/recall",
    "val/f1",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50",
    "metrics/mAP50-95",
    "best/epoch",
    "best/metric",
    "is_best",
)

_METRIC_ALIASES = {
    "mAP50-95": ("mAP50-95", "map", "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    "mAP50": ("mAP50", "map_50", "metrics/mAP50(B)", "metrics/mAP50"),
    "mAP75": ("mAP75", "map_75"),
    "AR100": ("AR100", "mar_100"),
    "precision": ("precision", "metrics/precision(B)"),
    "recall": ("recall", "metrics/recall(B)"),
    "f1": ("f1", "F1"),
}


def _is_main_process() -> bool:
    """Nhận diện rank chính mà không bắt buộc nhập Torch."""
    torch = sys.modules.get("torch")
    distributed = getattr(torch, "distributed", None)
    if distributed is not None:
        try:
            if distributed.is_available() and distributed.is_initialized():
                return distributed.get_rank() == 0
        except Exception:
            pass

    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("LOCAL_RANK", "0")
    try:
        return int(rank) in (-1, 0)
    except (TypeError, ValueError):
        return False


def _warn(message: str) -> None:
    """In cảnh báo tĩnh trên tiến trình chính."""
    if _is_main_process():
        print(f"Cảnh báo: {message}", file=sys.stderr)


def _number(value: Any, default: float = 0.0) -> float:
    """Chuyển giá trị vô hướng thành số hữu hạn."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return default
    return result if math.isfinite(result) else default


def _jsonable(value: Any) -> Any:
    """Chuyển dữ liệu Python, NumPy và Torch thành JSON hợp lệ."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]

    try:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach().cpu()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return _jsonable(tolist())
        item = getattr(value, "item", None)
        if callable(item):
            return _jsonable(item())
    except Exception:
        return None

    return str(value)


def _metric(values: Mapping[str, Any], name: str) -> float:
    """Đọc metric từ tên chuẩn hoặc tên tương đương."""
    candidates = (f"val/{name}",) + _METRIC_ALIASES.get(name, (name,))
    for key in candidates:
        if key in values and values[key] is not None:
            return _number(values[key])
    return 0.0


def _protected(key: str) -> bool:
    """Chặn khóa bất biến, khóa con và khóa cha bao phủ chúng."""
    return any(
        key == protected
        or key.startswith(f"{protected}.")
        or protected.startswith(f"{key}.")
        for protected in IMMUTABLE_SHARED_KEYS
    )


def load_wandb_env(cfg: dict) -> None:
    """Nạp URL và API key từ môi trường, .env hoặc Kaggle Secrets."""
    project = cfg.get("project") or {}
    base_url = project.get("wandb_base_url")
    if base_url:
        os.environ["WANDB_BASE_URL"] = str(base_url)

    if os.environ.get("WANDB_API_KEY"):
        return

    try:
        with (PROJECT_ROOT / ".env").open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    tokens = shlex.split(line, comments=True, posix=True)
                except ValueError:
                    continue

                if tokens and tokens[0] == "export":
                    tokens = tokens[1:]
                if len(tokens) != 1:
                    continue

                key, separator, value = tokens[0].partition("=")
                if (
                    separator
                    and key == "WANDB_API_KEY"
                    and value.strip()
                ):
                    os.environ["WANDB_API_KEY"] = value.strip()
                    break
    except Exception:
        pass

    if os.environ.get("WANDB_API_KEY"):
        return

    on_kaggle = (
        bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
        or bool(os.environ.get("KAGGLE_URL_BASE"))
        or Path("/kaggle/working").is_dir()
    )
    if not on_kaggle:
        return

    try:
        secrets = importlib.import_module("kaggle_secrets")
        api_key = secrets.UserSecretsClient().get_secret("WANDB_API_KEY")
        if isinstance(api_key, str) and api_key.strip():
            os.environ["WANDB_API_KEY"] = api_key.strip()
    except Exception:
        pass


def apply_sweep_overrides(cfg: dict, sweep_params: dict) -> dict:
    """Áp dụng tham số sweep lên bản sao và bảo vệ cấu hình chung."""
    result = deepcopy(cfg)

    for raw_key, value in sweep_params.items():
        if not isinstance(raw_key, str):
            continue

        key = raw_key.strip()
        if key in _TRAIN_KEYS:
            key = f"train.{key}"
        elif key in _ONLINE_KEYS:
            key = f"online_preprocess.{key}"

        parts = key.split(".")
        if not key or any(not part or part != part.strip() for part in parts):
            continue
        if _protected(key):
            continue

        node = result
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child

        node[parts[-1]] = deepcopy(value)
        if key in {"train.lr", "train.lr0"}:
            node["lr"] = deepcopy(value)
            node["lr0"] = deepcopy(value)

    return result


def _define_metrics(run: Any) -> None:
    """Đăng ký epoch làm trục bước cho các metric huấn luyện."""
    run.define_metric("epoch")
    for name in (
        "train/*",
        "val/*",
        "system/*",
        "best/*",
        "lr",
        "system/learning_rate",
    ):
        run.define_metric(name, step_metric="epoch")


def init_wandb_run(
    cfg: dict,
    run_dir: Path,
    *,
    job_type: str = "train",
    run_id: str | None = None,
    resume: str | None = None,
) -> tuple[Any | None, bool]:
    """Khởi tạo hoặc dùng lại W&B run trên tiến trình chính."""
    if not _is_main_process():
        return None, False

    load_wandb_env(cfg)

    if run_id is None:
        run_id = os.environ.get("TS_ALERT_WANDB_RUN_ID")
    if run_id and resume is None:
        resume = os.environ.get("TS_ALERT_WANDB_RESUME", "allow")

    try:
        wandb = importlib.import_module("wandb")
    except Exception:
        _warn("Không thể nạp W&B; tiếp tục ghi thống kê cục bộ.")
        return None, False

    existing = getattr(wandb, "run", None)
    if existing is not None:
        try:
            resolved = apply_sweep_overrides(cfg, dict(existing.config))
            cfg.clear()
            cfg.update(resolved)
            existing.config.update(_jsonable(cfg), allow_val_change=True)
            _define_metrics(existing)
            return existing, False
        except Exception:
            _warn("Không thể cấu hình W&B run hiện có.")
            return None, False

    train = cfg.get("train") or {}
    if not train.get("require_wandb", False):
        _warn("W&B đang tắt; tiếp tục ghi thống kê cục bộ.")
        return None, False
    if not os.environ.get("WANDB_API_KEY"):
        _warn("Thiếu WANDB_API_KEY; tiếp tục ghi thống kê cục bộ.")
        return None, False

    project = cfg.get("project") or {}
    options: dict[str, Any] = {
        "dir": str(run_dir),
        "job_type": job_type,
        "config": _jsonable(cfg),
    }
    for key, value in (
        ("project", project.get("wandb_project")),
        ("entity", project.get("wandb_entity")),
        ("name", train.get("run_name")),
        ("id", run_id),
        ("resume", resume),
    ):
        if value is not None:
            options[key] = value

    run = None
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        run = wandb.init(**options)
        if run is None:
            _warn("W&B không tạo được run; tiếp tục ghi thống kê cục bộ.")
            return None, False
        _define_metrics(run)
        return run, True
    except Exception:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        _warn("Khởi tạo W&B thất bại; tiếp tục ghi thống kê cục bộ.")
        return None, False


def standardize_epoch_payload(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict,
    lr: float,
    best_epoch: int,
    best_metric: float,
    is_best: bool,
    extra: dict | None = None,
) -> dict:
    """Chuẩn hóa thống kê epoch và bổ sung F1 khi cần."""
    scores = {
        name: _metric(val_metrics, name)
        for name in _METRIC_ALIASES
    }

    precision = scores["precision"]
    recall = scores["recall"]
    if scores["f1"] == 0.0 and precision + recall > 0.0:
        scores["f1"] = 2.0 * precision * recall / (precision + recall)

    learning_rate = _number(lr)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "train/loss": _number(train_loss),
        "val/loss": _number(val_loss),
        "lr": learning_rate,
        "system/learning_rate": learning_rate,
        **{f"val/{name}": value for name, value in scores.items()},
        "metrics/mAP50(B)": scores["mAP50"],
        "metrics/mAP50-95(B)": scores["mAP50-95"],
        "metrics/mAP50": scores["mAP50"],
        "metrics/mAP50-95": scores["mAP50-95"],
        "best/epoch": int(best_epoch),
        "best/metric": _number(best_metric),
        "is_best": int(bool(is_best)),
    }
    if extra:
        payload.update(_jsonable(extra))

    payload["epoch"] = int(epoch)
    payload["is_best"] = int(bool(payload.get("is_best")))
    return payload


def log_epoch_to_wandb(
    wandb_run: Any,
    payload: dict,
    primary_metric: str = "mAP50-95",
) -> None:
    """Ghi thống kê epoch và cập nhật kết quả tốt nhất."""
    if wandb_run is None or not _is_main_process():
        return

    try:
        wandb_run.log(_jsonable(payload), step=int(payload["epoch"]))
        if payload.get("is_best"):
            wandb_run.summary.update({
                "best_epoch": int(
                    payload.get("best/epoch", payload["epoch"])
                ),
                "best_metric": _number(payload.get("best/metric")),
                f"best/{primary_metric}": _number(
                    payload.get(
                        f"val/{primary_metric}",
                        payload.get("best/metric"),
                    )
                ),
                "best/precision": _number(payload.get("val/precision")),
                "best/recall": _number(payload.get("val/recall")),
                "best/f1": _number(payload.get("val/f1")),
            })
    except Exception:
        _warn("Không thể ghi thống kê epoch lên W&B.")


def _epoch_index(row: Mapping[str, Any]) -> int | None:
    """Đọc số epoch nguyên hợp lệ của một dòng thống kê."""
    value = row.get("epoch")
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _merge_history(*groups: list[dict]) -> list[dict]:
    """Gộp từng trường theo epoch; nguồn phía sau được ưu tiên."""
    merged: dict[int, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            if not isinstance(row, Mapping):
                continue
            epoch = _epoch_index(row)
            if epoch is None:
                continue
            current = merged.setdefault(epoch, {})
            current.update(_jsonable(row))
            current["epoch"] = epoch
    return [merged[epoch] for epoch in sorted(merged)]


def save_epoch_artifacts(
    run_dir: Path,
    history: list[dict],
    payload: dict | None = None,
) -> None:
    """Gộp thống kê theo epoch và lưu JSON cùng CSV trên rank chính."""
    if not _is_main_process():
        return

    directory = Path(run_dir)
    history_path = directory / "metrics_history.json"
    previous: list[dict] = []

    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, list):
                previous = loaded
        except (OSError, UnicodeError, ValueError):
            _warn("Không thể đọc lịch sử cũ; sử dụng lịch sử hiện tại.")

    rows = _merge_history(
        previous,
        history,
        [payload] if payload is not None else [],
    )
    if not rows:
        return

    directory.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    extra_columns = sorted(
        {key for row in rows for key in row}
        - set(_STANDARD_COLUMNS)
    )
    columns = [*_STANDARD_COLUMNS, *extra_columns]

    with (directory / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            })


def plot_training_curves(
    history: list[dict],
    out_path: Path | str,
    best_epoch: int | None = None,
    primary_metric: str = "mAP50-95",
) -> Path | None:
    """Vẽ bốn nhóm thống kê và đánh dấu epoch có metric tốt nhất."""
    if not _is_main_process():
        return None

    rows = _merge_history(history)
    if not rows:
        return None

    fig = None
    plt = None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in rows]
        marker = "o" if len(epochs) <= 5 else None
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 9), constrained_layout=True
        )

        groups = (
            (
                axes[0, 0],
                "Train vs Val Loss",
                ("train/loss", "val/loss"),
                "Loss",
            ),
            (
                axes[0, 1],
                "Validation Detection Metrics",
                (
                    "val/mAP50-95",
                    "val/mAP50",
                    "val/mAP75",
                    "val/AR100",
                ),
                "Score",
            ),
            (
                axes[1, 0],
                "Precision, Recall, F1",
                ("val/precision", "val/recall", "val/f1"),
                "Score",
            ),
            (
                axes[1, 1],
                "Learning Rate",
                ("system/learning_rate",),
                "Learning rate",
            ),
        )
        optional_metrics = {"val/mAP75", "val/AR100"}
        line_styles = ("-", "--", "-.", ":")

        for axis, title, keys, ylabel in groups:
            for index, key in enumerate(keys):
                values = [
                    _number(
                        (
                            row.get("lr")
                            if row.get("system/learning_rate") is None
                            else row.get("system/learning_rate")
                        )
                        if key == "system/learning_rate"
                        else row.get(key),
                        default=math.nan,
                    )
                    for row in rows
                ]
                if key in optional_metrics and not any(
                    math.isfinite(value) for value in values
                ):
                    continue

                axis.plot(
                    epochs,
                    values,
                    label=key,
                    marker=marker,
                    linestyle=line_styles[index % len(line_styles)],
                    linewidth=2.0 if key == "system/learning_rate" else 1.5,
                )

            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend()

        if best_epoch is not None:
            axis = axes[0, 1]
            best_index = int(best_epoch)
            axis.axvline(
                best_index, color="red", linestyle="--", alpha=0.7
            )
            best = next(
                (row for row in rows if row["epoch"] == best_index),
                None,
            )
            if best is not None:
                value = _number(
                    best.get(f"val/{primary_metric}"),
                    default=math.nan,
                )
                if math.isfinite(value):
                    axis.plot(
                        [best_index],
                        [value],
                        marker="*",
                        markersize=16,
                        color="red",
                        linestyle="None",
                        label=(
                            f"Best {primary_metric}: {value:.4f} "
                            f"(epoch {best_epoch})"
                        ),
                    )
            axis.legend()

        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
        return destination
    except Exception:
        _warn("Không thể tạo biểu đồ huấn luyện.")
        return None
    finally:
        if plt is not None and fig is not None:
            plt.close(fig)


def _class_ids(values: Any) -> list[int] | None:
    """Kiểm tra danh sách nhãn nguyên không âm."""
    if not isinstance(values, list):
        return None

    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
            if (
                not math.isfinite(number)
                or not number.is_integer()
                or number < 0
            ):
                return None
            result.append(int(number))
        except (TypeError, ValueError, OverflowError):
            return None
    return result


def _confusion_chart(wandb: Any, path: Path) -> Any | None:
    """Tạo confusion matrix từ nhãn đã ghép cặp trong JSON."""
    matrix = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(matrix, dict):
        return None

    truth = _class_ids(matrix.get("y_true"))
    if not truth:
        return None

    predictions = _class_ids(matrix.get("y_pred"))
    if predictions is None:
        predictions = _class_ids(matrix.get("preds"))
    if predictions is None or len(truth) != len(predictions):
        return None

    raw_names = matrix.get("labels")
    if not isinstance(raw_names, list) or not raw_names:
        raw_names = matrix.get("class_names")
    if not isinstance(raw_names, list):
        return None

    names = [
        name.strip()
        if isinstance(name, str) and name.strip()
        else f"Class_{index}"
        for index, name in enumerate(raw_names)
    ]

    maximum = max(max(truth), max(predictions))
    if maximum >= len(names):
        # Chỉ số ngay sau các lớp là background.
        names.append("background")
        while len(names) <= maximum:
            names.append(f"Class_{len(names)}")

    return wandb.plot.confusion_matrix(
        y_true=truth,
        preds=predictions,
        class_names=names,
    )


def log_final_to_wandb(
    wandb_run: Any,
    run_dir: Path,
    test_metrics: dict | None = None,
    cm_json_path: Path | None = None,
    extra_plots: list[Path] | None = None,
    val_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Ghi metric test, validation cuối cùng và biểu đồ lên W&B."""
    if wandb_run is None or not _is_main_process():
        return

    if test_metrics:
        metrics = {
            str(key) if str(key).startswith("test/") else f"test/{key}":
            _jsonable(value)
            for key, value in test_metrics.items()
        }
        try:
            wandb_run.summary.update(metrics)
            wandb_run.log(metrics)
        except Exception:
            _warn("Không thể ghi kết quả test lên W&B.")

    if val_metrics:
        final_metrics = {
            str(key)
            if str(key).startswith("val_final/")
            else f"val_final/{key}": _jsonable(value)
            for key, value in val_metrics.items()
        }
        final_summary = dict(final_metrics)
        if test_metrics is not None and "conf_threshold" in test_metrics:
            final_summary["eval/best_conf_threshold"] = _jsonable(
                test_metrics["conf_threshold"]
            )

        try:
            wandb_run.summary.update(final_summary)
        except Exception:
            _warn("Không thể cập nhật kết quả validation cuối cùng trên W&B.")

        try:
            wandb_run.log(final_metrics)
        except Exception:
            _warn("Không thể ghi kết quả validation cuối cùng lên W&B.")

    try:
        wandb = importlib.import_module("wandb")
    except Exception:
        _warn("Không thể nạp W&B để tải biểu đồ.")
        return

    directory = Path(run_dir)
    plots = [
        directory / "training_curves.png",
        directory / "pr_curve.png",
        directory / "eval_comparison.png",
        directory / "confusion_matrix.png",
        directory / "confusion_matrix_test.png",
        directory / "confusion_matrix_val.png",
        *(Path(path) for path in (extra_plots or [])),
    ]
    charts: dict[str, Any] = {}
    visited: set[Path] = set()

    for path in plots:
        if not path.is_file():
            continue
        try:
            identity = path.resolve()
            if identity in visited:
                continue
            visited.add(identity)
            charts[f"charts/{path.stem}"] = wandb.Image(str(path))
        except Exception:
            _warn("Không thể chuẩn bị một ảnh biểu đồ cho W&B.")

    matrix_path = (
        Path(cm_json_path)
        if cm_json_path is not None
        else directory / "confusion_matrix.json"
    )
    if matrix_path.is_file():
        try:
            chart = _confusion_chart(wandb, matrix_path)
            if chart is not None:
                charts["charts/confusion_matrix_interactive"] = chart
        except Exception:
            _warn("Không thể tạo confusion matrix tương tác.")

    if charts:
        try:
            wandb_run.log(charts)
        except Exception:
            _warn("Không thể tải biểu đồ lên W&B.")


def format_epoch_line(
    payload: dict,
    total_epochs: int | None = None,
    primary_metric: str = "mAP50-95",
) -> str:
    """Tạo dòng thống kê ngắn gọn cho một epoch."""
    epoch = int(payload["epoch"])
    progress = (
        f"{epoch}/{total_epochs}"
        if total_epochs is not None
        else str(epoch)
    )
    line = (
        f"Epoch {progress} | "
        f"Loss train/val "
        f"{_number(payload.get('train/loss')):.4f}/"
        f"{_number(payload.get('val/loss')):.4f} | "
        f"{primary_metric}="
        f"{_number(payload.get(f'val/{primary_metric}')):.4f} | "
        f"P={_number(payload.get('val/precision')):.4f} "
        f"R={_number(payload.get('val/recall')):.4f} "
        f"F1={_number(payload.get('val/f1')):.4f} | "
        f"LR={_number(payload.get('lr', payload.get('system/learning_rate'))):.6g}"
    )
    return line + (" * BEST" if payload.get("is_best") else "")


def print_best_banner(
    best_epoch: int,
    best_metric: float,
    primary_metric: str = "mAP50-95",
) -> None:
    """In banner epoch tốt nhất trên tiến trình chính."""
    if not _is_main_process():
        return

    message = (
        f"BEST epoch {int(best_epoch)} | "
        f"{primary_metric}={_number(best_metric):.4f}"
    )
    border = "=" * max(48, len(message) + 4)
    print(f"\n{border}\n  {message}\n{border}\n", flush=True)
