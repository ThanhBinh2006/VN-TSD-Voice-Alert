"""Chạy W&B Sweep cho YOLO và Faster R-CNN theo cấu hình dự án."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from typing import Any

from tools.common import add_common_args, build_overrides
from ts_alert.config.resolve import resolve_config, resolve_path
from ts_alert.types import MODEL_NAMES

IMMUTABLE_SHARED_KEYS: frozenset[str] = frozenset({
    "seed",
    "data",
    "eval",
    "train.batch_size",
    "train.epochs",
    "train.early_stopping_patience",
    "train.amp",
    "train.primary_metric",
})
IMMUTABLE_PREFIXES: tuple[str, ...] = (
    "data.",
    "eval.",
    "seed",
    "train.batch_size",
    "train.epochs",
    "train.early_stopping_patience",
    "train.amp",
    "train.primary_metric",
)


def build_sweep_spec(cfg: dict, method: str | None = None) -> dict:
    """Tạo đặc tả sweep và loại tham số ảnh hưởng cấu hình bất biến."""
    sweep_cfg = cfg.get("sweep") or {}
    selected_method = method or sweep_cfg.get("method", "bayes")
    if selected_method not in {"bayes", "random", "grid"}:
        raise ValueError("Phương pháp sweep phải là bayes, random hoặc grid.")

    parameters: dict[str, Any] = {}
    for key, definition in (sweep_cfg.get("parameters") or {}).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Tên tham số sweep phải là chuỗi không rỗng.")

        key = key.strip()
        # Chặn cả tham số cha có thể thay thế một nhánh bất biến.
        immutable = (
            key in IMMUTABLE_SHARED_KEYS
            or key.startswith(IMMUTABLE_PREFIXES)
            or any(
                protected.startswith(f"{key}.")
                for protected in IMMUTABLE_SHARED_KEYS
            )
        )
        if immutable:
            print(
                f"Cảnh báo: bỏ tham số bất biến {key!r}.",
                file=sys.stderr,
            )
            continue
        parameters[key] = deepcopy(definition)

    spec = {
        "method": selected_method,
        "metric": deepcopy(sweep_cfg.get(
            "metric", {"name": "best/metric", "goal": "maximize"}
        )),
        "parameters": parameters,
    }
    if selected_method != "grid" and "early_terminate" in sweep_cfg:
        spec["early_terminate"] = deepcopy(sweep_cfg["early_terminate"])
    return spec


def _run_distributed(
    pipeline: str,
    runtime: str,
    cfg: dict,
    run_id: str,
    num_gpus: int,
) -> None:
    """Chạy torchrun bằng Python hiện tại và luôn xóa YAML tạm."""
    import yaml

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("W&B run ID không hợp lệ để tạo tên file.")

    output_dir = resolve_path(cfg["paths"]["outputs_dir"]) / "_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    override_path = output_dir / f"{run_id}.yaml"

    env = os.environ.copy()
    env["TS_ALERT_WANDB_RUN_ID"] = run_id
    env["TS_ALERT_WANDB_RESUME"] = "allow"

    try:
        # Chế độ x tránh ghi đè file của một trial đang tồn tại.
        with override_path.open("x", encoding="utf-8") as stream:
            try:
                yaml.safe_dump(
                    cfg, stream, allow_unicode=True, sort_keys=False
                )
            except BaseException:
                override_path.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise SystemExit("File override của trial đã tồn tại.") from None

    try:
        # Tương đương torchrun, bảo đảm dùng đúng môi trường Python.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "-m",
                "tools.run_train",
                "--pipeline",
                pipeline,
                "--runtime",
                runtime,
                "--overrides",
                str(override_path),
            ],
            cwd=resolve_path("."),
            env=env,
            check=True,
        )
    finally:
        override_path.unlink(missing_ok=True)


def main() -> None:
    """Đọc CLI, xem trước đặc tả hoặc thực thi các trial W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pipeline", required=True, choices=MODEL_NAMES)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--method", choices=["bayes", "random", "grid"], default=None
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = resolve_config(
        args.pipeline, args.runtime, build_overrides(args)
    )
    spec = build_sweep_spec(base_cfg, args.method)
    sweep_cfg = base_cfg.get("sweep") or {}
    count = args.count if args.count is not None else sweep_cfg.get("count", 10)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        parser.error("--count hoặc sweep.count phải là số nguyên dương.")

    project_cfg = base_cfg.setdefault("project", {})
    project = (
        args.project
        if args.project is not None
        else project_cfg.get("wandb_project")
    )
    entity = (
        args.entity
        if args.entity is not None
        else project_cfg.get("wandb_entity")
    )
    project_cfg["wandb_project"] = project
    project_cfg["wandb_entity"] = entity

    if args.dry_run:
        print(json.dumps(
            {
                "pipeline": args.pipeline,
                "method": spec["method"],
                "count": count,
                "project": project,
                "metric": spec["metric"],
                "parameters": spec["parameters"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    # Import trễ để dry-run không phụ thuộc W&B, torch hay tracking.
    from ts_alert.pipelines.tracking import (
        apply_sweep_overrides,
        load_wandb_env,
    )

    load_wandb_env(base_cfg)
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise SystemExit(
            "Thiếu WANDB_API_KEY. Hãy đặt biến môi trường, khai báo trong "
            ".env hoặc thêm Kaggle Secret WANDB_API_KEY và bật quyền truy cập."
        )

    import wandb

    from ts_alert.pipelines import get_pipeline
    from ts_alert.utils.seed import set_seed

    if not wandb.login():
        raise SystemExit("Đăng nhập W&B thất bại; kiểm tra cấu hình xác thực.")

    sweep_id = args.sweep_id or wandb.sweep(
        spec, project=project, entity=entity
    )

    def _run_trial() -> None:
        """Huấn luyện một trial và luôn đóng phiên W&B sau khi kết thúc."""
        exit_code = 1
        try:
            run = wandb.init()
            if run is None:
                raise RuntimeError("W&B không khởi tạo được trial.")

            trial_cfg = apply_sweep_overrides(
                deepcopy(base_cfg), dict(run.config)
            )
            train_cfg = trial_cfg.setdefault("train", {})
            train_cfg["require_wandb"] = True
            train_cfg["run_name"] = f"{args.pipeline}_sweep_{run.id}"

            num_gpus = int(trial_cfg.get("num_gpus", 1))
            distributed = False
            if args.pipeline == "faster_rcnn" and num_gpus > 1:
                import torch

                available_gpus = torch.cuda.device_count()
                distributed = available_gpus > 1
                if distributed and num_gpus > available_gpus:
                    raise ValueError(
                        f"Yêu cầu {num_gpus} GPU nhưng chỉ có "
                        f"{available_gpus} GPU khả dụng."
                    )

            if distributed:
                run_id = run.id
                # Đóng writer hiện tại trước khi tiến trình con resume run.
                run.finish()
                _run_distributed(
                    args.pipeline, args.runtime, trial_cfg, run_id, num_gpus
                )
            else:
                set_seed(trial_cfg["seed"])
                get_pipeline(args.pipeline, trial_cfg).fit()
            exit_code = 0
        finally:
            wandb.finish(exit_code=exit_code)

    wandb.agent(
        sweep_id,
        function=_run_trial,
        count=count,
        project=project,
        entity=entity,
    )


if __name__ == "__main__":
    main()
