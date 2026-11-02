"""YoloPipeline: huấn luyện YOLO qua Ultralytics API kèm đẩy số liệu realtime lên W&B.
Phụ trách: Thành viên 1
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from ts_alert.config.resolve import get_processed_root, resolve_path
from ts_alert.data.preprocess_online import build_online_transforms
from ts_alert.pipelines.base import BasePipeline
from ts_alert.pipelines.yolo.infer import YoloDetector
from ts_alert.pipelines.yolo.model import load_model
from ts_alert.runtime.run_dir import make_run_dir
from ts_alert.utils.seed import set_seed


def _load_dotenv_if_exists() -> None:
    """Nạp biến môi trường từ file .env ở thư mục gốc dự án (nếu có)."""
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and v:
                os.environ.setdefault(k, v)
    except Exception:
        pass


def _resolve_wandb_api_key() -> str:
    """Tìm WANDB_API_KEY từ biến môi trường, file .env hoặc Kaggle Secrets."""
    _load_dotenv_if_exists()
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        secret_key = str(UserSecretsClient().get_secret("WANDB_API_KEY") or "").strip()
        if secret_key:
            os.environ["WANDB_API_KEY"] = secret_key
            return secret_key
    except Exception:
        pass
    return ""


class _RealtimeWandbLogger:
    """Đẩy số liệu (loss, mAP, precision, recall, lr) sau mỗi epoch lên W&B để W&B tự động dựng đồ thị realtime.

    Hoàn toàn tách rời khỏi `model.callbacks` của Ultralytics để không gây lỗi
    `TypeError: cannot pickle '_thread.lock' object` khi Ultralytics khởi tạo DDP 2 GPU bằng cloudpickle.
    """

    def __init__(self, cfg: dict[str, Any], run_name: str, run_dir: Path, outputs_dir: Path) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.run_dir = run_dir
        self.outputs_dir = outputs_dir
        self.run: Any = None
        self._lock = threading.Lock()
        self._logged_epochs: set[int] = set()
        self._stop_event = threading.Event()
        self._watcher_thread: threading.Thread | None = None

    def start(self) -> bool:
        api_key = _resolve_wandb_api_key()
        require_wandb = bool(self.cfg.get("train", {}).get("require_wandb", False))

        if not require_wandb and not api_key:
            return False

        proj = self.cfg.get("project") or {}
        base_url = proj.get("wandb_base_url") or os.environ.get("WANDB_BASE_URL")
        if base_url:
            os.environ["WANDB_BASE_URL"] = str(base_url).strip()

        try:
            import wandb

            if api_key:
                wandb.login(key=api_key, relogin=True)
            elif not wandb.api.api_key:
                print(
                    "[YoloPipeline][W&B] Cảnh báo: REQUIRE_WANDB=True nhưng chưa tìm thấy WANDB_API_KEY. Bỏ qua W&B."
                )
                return False

            project_name = (
                proj.get("wandb_project")
                or os.environ.get("WANDB_PROJECT")
                or "TSD-Voice-Alert"
            )
            entity_name = (
                proj.get("wandb_entity")
                or os.environ.get("WANDB_ENTITY")
                or None
            )

            self.run = wandb.init(
                project=project_name,
                entity=entity_name,
                name=self.run_name,
                config=self.cfg,
                dir=str(self.outputs_dir),
                job_type="train",
                reinit=True,
            )
            url = getattr(self.run, "url", None)
            print(f"[YoloPipeline][W&B] Đang truyền số liệu trực tiếp lên W&B tại: {url}")

            self._stop_event.clear()
            self._watcher_thread = threading.Thread(
                target=self._watch_results_csv,
                name="wandb-csv-watcher",
                daemon=True,
            )
            self._watcher_thread.start()
            return True
        except Exception as exc:
            print(f"[YoloPipeline][W&B] Không thể khởi tạo W&B ({exc}). Tiếp tục huấn luyện bình thường.")
            self.run = None
            return False

    def _watch_results_csv(self) -> None:
        while not self._stop_event.wait(2.0):
            self.sync_from_csv()

    def sync_from_csv(self) -> None:
        """Đọc số liệu từng epoch mới trong results.csv và bắn `run.log(...)` để W&B tự vẽ đồ thị."""
        if self.run is None:
            return
        results_csv = self.run_dir / "results.csv"
        if not results_csv.is_file():
            return

        with self._lock:
            try:
                with open(results_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v is not None}
                        if "epoch" not in clean_row:
                            continue
                        try:
                            epoch = int(float(clean_row["epoch"]))
                        except ValueError:
                            continue

                        if epoch in self._logged_epochs:
                            continue

                        payload: dict[str, float | int] = {"epoch": epoch}
                        for col, val_str in clean_row.items():
                            if col == "epoch":
                                continue
                            try:
                                payload[col] = float(val_str)
                            except ValueError:
                                continue

                        self.run.log(payload, step=epoch)
                        self._logged_epochs.add(epoch)
            except Exception:
                pass

    def log_test_metrics(self, test_metrics: dict[str, Any] | None = None) -> None:
        """Gửi số liệu đánh giá tập Test cuối cùng lên W&B."""
        if self.run is None:
            return
        self._stop_event.set()
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=3.0)

        self.sync_from_csv()

        if test_metrics:
            test_payload = {
                f"test/{k}": float(v)
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
            if test_payload:
                self.run.log(test_payload)
                for k, v in test_payload.items():
                    self.run.summary[k] = v

    def finish(self) -> None:
        self._stop_event.set()
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
            self.run = None


class YoloPipeline(BasePipeline):
    name = "yolo"

    def fit(self) -> Path:
        """Fine-tune YOLO trên <processed root>/data.yaml (get_processed_root(cfg)).

        Các bước:
          1. set_seed(cfg["seed"]); run_dir = make_run_dir(outputs_dir, "yolo", cfg["train"]["run_name"], cfg)
          2. Khởi tạo _RealtimeWandbLogger truyền số liệu mỗi epoch lên W&B (độc lập với trainer.callbacks)
          3. model = load_model(cfg["model"]["pretrained"])
          4. model.train(...)
          5. copy best.pt sang cfg["model"]["weights"]
          6. Tự động đánh giá tập Test (nếu auto_eval_test=True) và log số liệu Test lên W&B
        Trả về run_dir.
        """
        cfg = self.cfg
        set_seed(cfg["seed"])

        # 1. Tạo thư mục output run_dir
        outputs_dir = resolve_path(cfg["paths"]["outputs_dir"])
        run_name = (
            cfg["train"].get("run_name")
            or f"yolo_s{cfg['seed']}_ep{cfg['train']['epochs']}_bs{cfg['train']['batch_size']}"
        )
        run_dir = make_run_dir(outputs_dir, "yolo", run_name, cfg)

        # 2. Chuẩn bị đường dẫn data.yaml (tạo bản runtime với path chuẩn xác cho Kaggle/Local)
        processed_root = get_processed_root(cfg)
        source_yaml = processed_root / "data.yaml"
        if not source_yaml.exists():
            raise FileNotFoundError(
                f"Không tìm thấy data.yaml tại {source_yaml}. Hãy kiểm tra đường dẫn dataset."
            )

        import yaml

        with open(source_yaml, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = processed_root.resolve().as_posix()

        # Lưu runtime data.yaml tại outputs_dir để tránh bị DDP đa GPU xóa nhầm khi khởi tạo run_dir
        runtime_yaml = outputs_dir / f"data_runtime_{cfg['seed']}.yaml"
        with open(runtime_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        try:
            shutil.copy2(runtime_yaml, run_dir / "data.yaml")
        except Exception:
            pass

        # Tắt callback W&B tích hợp sẵn của Ultralytics TRƯỚC khi tạo model
        # và KHÔNG dùng model.add_callback để tránh cloudpickle lỗi `_thread.lock` khi chạy DDP 2 GPU
        try:
            from ultralytics import settings as ul_settings

            ul_settings.update({"wandb": False})
        except Exception:
            pass

        # 3. Khởi tạo bộ truyền số liệu realtime lên W&B (chạy luồng độc lập theo dõi results.csv)
        wb_logger = _RealtimeWandbLogger(cfg, run_name, run_dir, outputs_dir)
        wb_logger.start()

        # 4. Nạp model ban đầu (giữ nguyên callbacks mặc định thuần túy để DDP cloudpickle thành công)
        pretrained = cfg["model"].get("pretrained", "yolo26m.pt")
        model = load_model(pretrained)

        # 5. Tham số huấn luyện Ultralytics
        online_args = build_online_transforms(cfg, "ultralytics")
        train_args = {
            "data": runtime_yaml.resolve().as_posix(),
            "epochs": int(cfg["train"]["epochs"]),
            "batch": int(cfg["train"]["batch_size"]),
            "imgsz": int(cfg["data"]["image_size"]),
            "seed": int(cfg["seed"]),
            "patience": int(cfg["train"].get("early_stopping_patience", 10)),
            "amp": bool(cfg["train"].get("amp", True)),
            "lr0": float(cfg["train"].get("lr0", 0.01)),
            "optimizer": cfg["train"].get("optimizer", "auto"),
            "project": run_dir.parent.resolve().as_posix(),
            "name": run_dir.name,
            "exist_ok": True,
            **online_args,
        }

        num_gpus = int(cfg.get("num_gpus", 1))
        if num_gpus == 2:
            train_args["device"] = [0, 1]
        elif num_gpus == 1:
            train_args["device"] = 0
        else:
            train_args["device"] = "cpu"

        print(f"[YoloPipeline] Bắt đầu fine-tune YOLO26m tại: {run_dir}")
        try:
            model.train(**train_args)

            # 6. Sao chép best.pt sang đường dẫn weights được chỉ định
            best_pt = self.best_checkpoint(run_dir)
            target_weights = resolve_path(cfg["model"]["weights"])
            target_weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, target_weights)
            print(f"[YoloPipeline] Đã lưu checkpoint tốt nhất sang: {target_weights}")

            # 7. Tự động đánh giá trên tập Test nếu được kích hoạt
            test_metrics: dict[str, Any] | None = None
            if cfg.get("eval", {}).get("auto_eval_test", False):
                try:
                    print("[YoloPipeline] Tự động đánh giá trên tập Test...")
                    test_metrics = self.evaluate(run_dir)
                except Exception as e:
                    print(f"[YoloPipeline] Chú ý: Đánh giá tập Test bị bỏ qua: {e}")

            if test_metrics is None and (run_dir / "metrics.json").is_file():
                try:
                    test_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                except Exception:
                    pass

            wb_logger.log_test_metrics(test_metrics)
        finally:
            wb_logger.finish()

        return run_dir

    def load_detector(self, weights=None) -> YoloDetector:
        return YoloDetector(weights or self.default_weights(), self.cfg)
