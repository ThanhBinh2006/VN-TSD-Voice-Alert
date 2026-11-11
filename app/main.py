"""Web localhost, API xử lý media và cảnh báo realtime.

Chạy:
    python -m app.main
    python -m app.main --fake
    python -m app.main --store memory

Không truyền analysis_mode khi tạo job sẽ giữ nguyên pipeline legacy.
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import uuid
import weakref
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from werkzeug.exceptions import BadRequest, HTTPException
from werkzeug.utils import secure_filename

from app.jobs import JobRunner
from ts_alert.config.resolve import resolve_config, resolve_path
from ts_alert.runtime import video_io
from ts_alert.runtime.cpu_opt import CPU_PRESETS, probe_runtime_backends
from ts_alert.runtime.realtime_service import RealtimeSessionManager
from ts_alert.types import IMAGE_EXTS, MODEL_NAMES, VIDEO_EXTS

log = logging.getLogger(__name__)

_ANALYSIS_MODES = frozenset(
    {"legacy", "realtime_overlay", "realtime", "buffered"}
)
_PLAYBACK_MODES = frozenset({"realtime", "buffered"})
_BACKENDS = frozenset({"auto", "pytorch", "onnx", "openvino"})
_TERMINAL_CAMERA_STATES = frozenset(
    {"stopped", "stopping", "closed", "cancelled", "canceled", "failed", "done"}
)
_SSE_HEADERS = {
    "Cache-Control": "no-store, no-cache",
    "X-Accel-Buffering": "no",
}


def _make_store(kind: str, db_path: Path) -> Any:
    """Tạo SQLite store hoặc dùng RAM khi repository vẫn là stub."""
    from app.fake import is_stub
    from ts_alert.db.repository import Repository

    if kind == "auto":
        kind = "memory" if is_stub(Repository.__init__) else "sqlite"
    if kind == "memory":
        from app.store_memory import InMemoryStore

        log.warning("Lưu log trong RAM (InMemoryStore): dữ liệu mất khi tắt server.")
        return InMemoryStore()
    if kind == "sqlite":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return Repository(db_path)
    raise ValueError(f"store không hợp lệ: {kind!r}")


def _synthesize_runtime_audio(
    text: str,
    out_path: Path,
    cfg: dict[str, Any],
) -> Path:
    """Tra cứu synthesize lúc gọi để hỗ trợ fake và monkeypatch trong test."""
    from ts_alert.postprocess import tts

    return tts.synthesize(text, out_path, cfg)


def create_app(
    runtime: str = "local",
    fake: bool = False,
    store: Any = None,
    store_kind: str = "auto",
    sync: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Flask:
    """Khởi tạo Flask, worker legacy và trình quản lý phiên realtime."""
    base_cfg = resolve_config(None, runtime, overrides)
    paths = base_cfg["paths"]
    outputs_dir = resolve_path(paths["outputs_dir"]).resolve()
    uploads_dir = resolve_path(paths["uploads_dir"]).resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    cfg_cache: dict[str, dict[str, Any]] = {}
    cfg_lock = threading.Lock()

    def cfg_for(model_name: str) -> dict[str, Any]:
        """Lấy cấu hình model đã cache, an toàn khi nhiều worker truy cập."""
        with cfg_lock:
            if model_name not in cfg_cache:
                cfg_cache[model_name] = resolve_config(
                    model_name, runtime, overrides
                )
            return cfg_cache[model_name]

    fake_names: list[str] = []

    def restore_fakes() -> None:
        """Không cần khôi phục khi ứng dụng dùng implementation thật."""

    if fake:
        from app.fake import fake_detector_factory, install_fakes

        restore_fakes, fake_names = install_fakes()
        detector_factory = fake_detector_factory
    else:
        from ts_alert.pipelines import get_detector

        detector_factory = get_detector

    store = store if store is not None else _make_store(
        store_kind, resolve_path(paths["db_path"])
    )
    _fail_unfinished_jobs(store)
    runner = JobRunner(
        store, cfg_for, detector_factory, outputs_dir, sync=sync
    )
    try:
        realtime = RealtimeSessionManager(
            detector_provider=runner.get_detector,
            cfg_for=cfg_for,
            outputs_dir=outputs_dir,
            store=store,
            progress_tracker=runner.progress,
            synthesizer=_synthesize_runtime_audio,
        )
    except Exception:
        runner.shutdown(wait=False)
        raise

    shutdown_lock = threading.Lock()
    shutdown_started = False

    def shutdown() -> None:
        """Đóng cả hai bộ xử lý đúng một lần khi server hoặc interpreter dừng."""
        nonlocal shutdown_started
        with shutdown_lock:
            if shutdown_started:
                return
            shutdown_started = True
        try:
            realtime.shutdown(wait=False)
        finally:
            runner.shutdown(wait=False)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = (
        int(base_cfg["app"]["max_upload_mb"]) * 1024 * 1024
    )
    app.json.ensure_ascii = False
    app.extensions["tsd"] = {
        "store": store,
        "runner": runner,
        "realtime": realtime,
        "cfg": base_cfg,
        "cfg_for": cfg_for,
        "fake": fake,
        "fake_parts": fake_names,
        "restore_fakes": restore_fakes,
        "outputs_dir": outputs_dir,
        "uploads_dir": uploads_dir,
        "job_options": {},
        "job_lock": threading.RLock(),
        "shutdown": shutdown,
    }

    _register_routes(app)
    weakref.finalize(app, shutdown)
    return app


def _fail_unfinished_jobs(store: Any) -> None:
    """Đánh dấu các job bị bỏ dở trước khi server khởi động lại."""
    for status in ("queued", "processing"):
        while True:
            jobs = store.get_jobs(limit=1000, status=status)
            if not jobs:
                break
            for job in jobs:
                store.fail_job(
                    job["id"],
                    "Server đã tắt khi job đang chạy. Hãy gửi lại file.",
                )
                log.warning(
                    "Job %s bị bỏ dở từ lần chạy trước -> failed", job["id"]
                )


def _ctx() -> dict[str, Any]:
    """Lấy ngữ cảnh xử lý của ứng dụng hiện tại."""
    from flask import current_app

    return current_app.extensions["tsd"]


def _error(status: int, message: str) -> Response:
    """Tạo phản hồi lỗi JSON với HTTP status tương ứng."""
    response = jsonify({"error": message})
    response.status_code = status
    return response


def _job_or_404(job_id: int) -> dict[str, Any]:
    """Lấy job hoặc trả HTTP 404."""
    job = _ctx()["store"].get_job(job_id)
    if job is None:
        abort(404, description=f"Không có job {job_id}")
    return job


def _safe_media_path(
    path_str: str | Path | None,
    root: Path,
    missing_message: str,
) -> Path:
    """Chỉ cho phép phục vụ file thực nằm trong thư mục được chỉ định."""
    if not path_str:
        abort(404, description=missing_message)
    try:
        path = Path(path_str).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            abort(404, description=missing_message)
    except (OSError, ValueError, RuntimeError):
        abort(404, description=missing_message)
    return path


def _safe_output_path(path_str: str | Path | None) -> Path:
    """Kiểm tra file kết quả hoặc âm thanh thuộc outputs_dir."""
    return _safe_media_path(
        path_str,
        _ctx()["outputs_dir"],
        "Không tìm thấy file kết quả",
    )


def _safe_upload_path(path_str: str | Path | None) -> Path:
    """Kiểm tra file nguồn thuộc uploads_dir, kể cả sau khi resolve symlink."""
    return _safe_media_path(
        path_str,
        _ctx()["uploads_dir"],
        "Không tìm thấy file nguồn",
    )


def _choice(
    value: Any,
    field: str,
    choices: Any,
) -> str:
    """Kiểm tra một trường chuỗi thuộc tập giá trị hợp lệ."""
    if not isinstance(value, str):
        raise BadRequest(description=f"{field} phải là chuỗi")
    value = value.strip()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise BadRequest(
            description=f"{field} không hợp lệ. Chọn một trong: {allowed}"
        )
    return value


def _integer(
    value: Any,
    field: str,
    minimum: int,
    maximum: int | None = None,
    *,
    allow_string: bool = False,
) -> int:
    """Đọc số nguyên, từ chối boolean, số thực và chuỗi không phải ASCII."""
    if isinstance(value, bool):
        raise BadRequest(description=f"{field} phải là số nguyên")
    if isinstance(value, int):
        result = value
    elif allow_string and isinstance(value, str):
        raw = value.strip()
        if not raw or not raw.isascii() or not raw.isdecimal():
            raise BadRequest(description=f"{field} phải là số nguyên")
        try:
            result = int(raw)
        except ValueError as exc:
            raise BadRequest(
                description=f"{field} phải là số nguyên"
            ) from exc
    else:
        raise BadRequest(description=f"{field} phải là số nguyên")

    if result < minimum or (maximum is not None and result > maximum):
        interval = (
            f"[{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise BadRequest(description=f"{field} phải nằm trong {interval}")
    return result


def _number(
    value: Any,
    field: str,
    minimum: float = 0.0,
    *,
    strictly_positive: bool = False,
    allow_string: bool = False,
) -> float:
    """Đọc số hữu hạn và kiểm tra miền giá trị."""
    if isinstance(value, bool):
        raise BadRequest(description=f"{field} phải là số hữu hạn")
    if isinstance(value, str):
        if not allow_string or not value.strip():
            raise BadRequest(description=f"{field} phải là số hữu hạn")
    elif not isinstance(value, (int, float)):
        raise BadRequest(description=f"{field} phải là số hữu hạn")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BadRequest(
            description=f"{field} phải là số hữu hạn"
        ) from exc

    if not math.isfinite(result):
        raise BadRequest(description=f"{field} phải là số hữu hạn")
    if result < minimum or (strictly_positive and result <= minimum):
        operator = ">" if strictly_positive else ">="
        raise BadRequest(
            description=f"{field} phải {operator} {minimum:g}"
        )
    return result


def _json_object() -> dict[str, Any]:
    """Đọc JSON object; từ chối JSON lỗi, array, null và content type sai."""
    if not request.is_json:
        raise BadRequest(description="Request phải có JSON object")
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise BadRequest(description="Request phải có JSON object")
    return payload


def _nonempty(values: Mapping[str, Any], key: str, default: Any) -> Any:
    """Trả về giá trị của key nếu khác rỗng, ngược lại trả default."""
    raw = values.get(key)
    if raw is None:
        return default
    if isinstance(raw, str) and not raw.strip():
        return default
    return raw


def _runtime_selection(
    values: Mapping[str, Any],
    *,
    default_model: str | None = None,
) -> tuple[str, str, str]:
    """Kiểm tra model, CPU preset và backend trước khi tạo tài nguyên."""
    model = _choice(
        _nonempty(values, "model", default_model),
        "model",
        MODEL_NAMES,
    )
    preset = _choice(
        _nonempty(values, "cpu_preset", "fast_cpu"),
        "cpu_preset",
        CPU_PRESETS[model],
    )
    backend = _choice(
        _nonempty(values, "backend", "pytorch"),
        "backend",
        _BACKENDS,
    )
    return model, preset, backend


def _video_session_or_404(job_id: int) -> RealtimeSessionManager:
    """Kiểm tra job video và phiên realtime tương ứng."""
    job = _job_or_404(job_id)
    if job.get("input_type") != "video":
        abort(400, description="Endpoint chỉ hỗ trợ job video")
    realtime = _ctx()["realtime"]
    if not realtime.has_video_session(job_id):
        abort(404, description="Không có phiên realtime cho job này")
    return realtime


def _camera_session_or_404(
    session_id: str,
    *,
    require_active: bool = False,
) -> RealtimeSessionManager:
    """Kiểm tra phiên camera và trạng thái hoạt động khi nhận frame."""
    realtime = _ctx()["realtime"]
    if not realtime.has_camera_session(session_id):
        abort(404, description="Không có phiên camera")
    if require_active:
        state = realtime.get_camera_status(session_id)
        if state is None:
            abort(404, description="Không có phiên camera")
        if (
            state.get("status") in _TERMINAL_CAMERA_STATES
            or state.get("stage") in _TERMINAL_CAMERA_STATES
            or state.get("stopped") is True
            or state.get("active") is False
            or state.get("running") is False
        ):
            abort(404, description="Phiên camera đã dừng")
    return realtime


def _video_urls(job_id: int) -> dict[str, str]:
    """Tạo các URL phục vụ video gốc và điều khiển phân tích."""
    return {
        "source_url": url_for("job_source", job_id=job_id),
        "events_url": url_for("job_events", job_id=job_id),
        "playback_url": url_for("job_playback", job_id=job_id),
        "cancel_url": url_for("job_cancel", job_id=job_id),
        "export_url": url_for("job_export", job_id=job_id),
    }


def _progress_json(job: dict[str, Any]) -> dict[str, Any]:
    """Kết hợp trạng thái trong store với tiến độ hiện tại trong RAM."""
    tracked = _ctx()["runner"].progress.get(job["id"])
    if tracked is None:
        progress = {
            "stage": (
                job["status"]
                if job["status"] in ("queued", "done", "failed")
                else "processing"
            ),
            "percent": None,
            "frames_done": None,
            "frame_count": None,
            "fps": None,
            "eta_sec": None,
            "video_fps": None,
            "duration_sec": None,
            "current_timestamp_sec": None,
            "frame_stride": None,
            "elapsed_ms": None,
            "layer_time_ms": job.get("layer_time_ms"),
            "tts_time_ms": job.get("tts_time_ms"),
            "unique_sign_count": job.get("unique_sign_count"),
            "voice_play_count": job.get("voice_play_count"),
        }
    else:
        progress = dict(tracked)

    if job["status"] == "done":
        progress.update(stage="done", percent=100.0, eta_sec=None)
        for key in (
            "unique_sign_count",
            "voice_play_count",
            "layer_time_ms",
            "tts_time_ms",
        ):
            if job.get(key) is not None:
                progress[key] = job[key]
    elif job["status"] == "failed":
        progress.update(stage="failed", eta_sec=None)
    return progress


def _epoch(value: str | int | float | None) -> int | float | None:
    """Chuyển thời điểm SQLite UTC thành số giây kể từ epoch."""
    if value is None or isinstance(value, (int, float)):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError:
        return None


def _job_json(job: dict[str, Any]) -> dict[str, Any]:
    """Giữ dữ liệu legacy và bổ sung URL, trạng thái runtime cho video."""
    ctx = _ctx()
    data = dict(job)
    for key in ("created_at", "started_at", "finished_at"):
        if key in data:
            data[key] = _epoch(data[key])

    progress = _progress_json(job)
    data["progress"] = progress
    with ctx["job_lock"]:
        options = dict(ctx["job_options"].get(job["id"], {}))
    data["frame_stride"] = (
        progress.get("frame_stride")
        or options.get("frame_stride")
        or int((ctx["cfg"].get("inference") or {}).get("frame_stride", 1))
    )
    data["input_name"] = (
        Path(job["input_path"]).name if job.get("input_path") else None
    )

    if job.get("input_type") == "video":
        data["stream_url"] = url_for("job_stream", job_id=job["id"])
        data["frame_url"] = url_for("job_frame", job_id=job["id"])
        data.update(_video_urls(job["id"]))
        runtime = ctx["realtime"].get_video_status(job["id"])
        data["analysis_mode"] = options.get("analysis_mode") or (
            "realtime_overlay" if runtime is not None else "legacy"
        )
        if options.get("playback_mode"):
            data["playback_mode"] = options["playback_mode"]
        if (
            runtime is not None
            and isinstance(runtime.get("playback"), Mapping)
            and "playback_rate" in runtime["playback"]
        ):
            data["playback_rate"] = runtime["playback"]["playback_rate"]
        elif "playback_rate" in options:
            data["playback_rate"] = options["playback_rate"]
        if options.get("cpu_preset"):
            data["cpu_preset"] = options["cpu_preset"]
        if options.get("backend"):
            data["backend"] = options["backend"]
        if runtime is not None:
            data["runtime"] = runtime

    if job.get("status") == "done" and job.get("output_path"):
        data["result_url"] = url_for("job_result", job_id=job["id"])
        data["download_url"] = url_for(
            "job_result", job_id=job["id"], download=1
        )
    return data


def _sse_response(kind: str, session_key: int | str) -> Response:
    """Tạo phản hồi SSE không cache cho phiên video hoặc camera."""
    return Response(
        stream_with_context(
            _ctx()["realtime"].iter_sse_events(kind, session_key)
        ),
        mimetype="text/event-stream",
        headers=dict(_SSE_HEADERS),
    )


def _frame_metadata_value(field: str) -> Any:
    """Đọc metadata frame từ form, query string rồi đến HTTP header."""
    if field in request.form:
        return request.form[field]
    if field in request.args:
        return request.args[field]
    headers = {
        "timestamp_sec": ("X-Timestamp-Sec", "X-Timestamp", "Timestamp-Sec"),
        "frame_id": ("X-Frame-Id", "Frame-Id"),
    }
    for header in headers[field]:
        value = request.headers.get(header)
        if value is not None:
            return value
    return None


def _register_routes(app: Flask) -> None:
    """Đăng ký giao diện, API legacy và API realtime."""

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Response | HTTPException:
        """Trả lỗi HTTP dạng JSON cho các endpoint API."""
        if request.path.startswith("/api/"):
            return _error(exc.code or 500, exc.description or exc.name)
        return exc

    @app.get("/")
    def index() -> str:
        """Hiển thị giao diện chính."""
        return render_template("index.html")

    @app.get("/api/health")
    def health() -> Response:
        """Trả dữ liệu health legacy cùng thông tin CPU runtime."""
        ctx = _ctx()
        models = {}
        loaded_models = ctx["runner"].loaded_models()
        for name in MODEL_NAMES:
            weights = resolve_path(
                resolve_config(name, None)["model"]["weights"]
            )
            models[name] = {
                "weights": str(weights),
                "weights_exists": weights.exists(),
                "loaded": name in loaded_models,
            }
        return jsonify({
            "status": "ok",
            "fake": ctx["fake"],
            "fake_parts": ctx["fake_parts"],
            "store": type(ctx["store"]).__name__,
            "models": models,
            "max_upload_mb": ctx["cfg"]["app"]["max_upload_mb"],
            "accept": sorted(VIDEO_EXTS | IMAGE_EXTS),
            "frame_stride": int(
                (ctx["cfg"].get("inference") or {}).get("frame_stride", 1)
            ),
            "realtime": {
                "backends": probe_runtime_backends(),
                "presets": sorted(CPU_PRESETS["yolo"].keys()),
                "video_playback_rate": float(
                    (ctx["cfg"].get("realtime") or {}).get(
                        "video_playback_rate", 1.0
                    )
                ),
                "tts": {
                    "playback_rate": float(
                        (ctx["cfg"].get("realtime") or {}).get(
                            "tts_playback_rate", 1.4
                        )
                    ),
                    "compact": bool(
                        (ctx["cfg"].get("realtime") or {}).get(
                            "tts_compact", True
                        )
                    ),
                    "cutoff_on_pass": bool(
                        (ctx["cfg"].get("realtime") or {}).get(
                            "tts_cutoff_on_pass", True
                        )
                    ),
                    "cutoff_grace_sec": float(
                        (ctx["cfg"].get("realtime") or {}).get(
                            "tts_cutoff_grace_sec", 0.25
                        )
                    ),
                    "cache_enabled": bool(
                        (ctx["cfg"].get("realtime") or {}).get(
                            "tts_cache_enabled", True
                        )
                    ),
                },
            },
        })

    @app.post("/api/jobs")
    def create_job() -> Response:
        """Validate toàn bộ form trước khi lưu upload và khởi chạy job."""
        ctx = _ctx()
        file = request.files.get("file")
        model_name = (request.form.get("model") or "").strip()
        if file is None or not file.filename:
            return _error(400, "Thiếu file (field 'file')")
        if model_name not in MODEL_NAMES:
            return _error(
                400,
                f"Model không hợp lệ: {model_name!r}. "
                f"Chọn một trong {list(MODEL_NAMES)}",
            )

        ext = Path(file.filename).suffix.lower()
        try:
            input_type = video_io.media_type(file.filename)
        except ValueError:
            return _error(
                400,
                f"Định dạng không hỗ trợ: {ext or '(không có đuôi)'}. "
                f"Hỗ trợ: {', '.join(sorted(VIDEO_EXTS | IMAGE_EXTS))}",
            )

        raw_stride = (request.form.get("frame_stride") or "").strip()
        job_overrides: dict[str, Any] | None = None
        stride_val = int(
            (ctx["cfg"].get("inference") or {}).get("frame_stride", 1)
        )
        if raw_stride:
            try:
                stride_val = _integer(
                    raw_stride,
                    "frame_stride",
                    1,
                    300,
                    allow_string=True,
                )
            except BadRequest:
                return _error(
                    400,
                    "frame_stride phải là chuỗi số nguyên dương "
                    "trong khoảng [1, 300]",
                )
            job_overrides = {"inference": {"frame_stride": stride_val}}

        analysis_mode = _choice(
            request.form.get("analysis_mode", "").strip() or "legacy",
            "analysis_mode",
            _ANALYSIS_MODES,
        )
        playback_mode = _choice(
            _nonempty(
                request.form,
                "playback_mode",
                "buffered" if analysis_mode == "buffered" else "realtime",
            ),
            "playback_mode",
            _PLAYBACK_MODES,
        )
        _, cpu_preset, backend = _runtime_selection(request.form)
        default_buffer = str(
            (ctx["cfg"].get("realtime") or {}).get("buffer_ahead_sec", 2.0)
        )
        buffer_ahead_sec = _number(
            _nonempty(request.form, "buffer_ahead_sec", default_buffer),
            "buffer_ahead_sec",
            allow_string=True,
        )
        playback_rate = _number(
            _nonempty(
                request.form,
                "playback_rate",
                (ctx["cfg"].get("realtime") or {}).get(
                    "video_playback_rate", 1.0
                ),
            ),
            "playback_rate",
            minimum=0.0,
            strictly_positive=True,
            allow_string=True,
        )

        stem = secure_filename(Path(file.filename).stem) or "upload"
        saved = ctx["uploads_dir"] / f"{uuid.uuid4().hex[:8]}_{stem}{ext}"
        file.save(saved)

        store = ctx["store"]
        job_id = store.create_job(input_type, str(saved), model_name)
        store.set_status(job_id, "queued")
        options = {
            "frame_stride": stride_val,
            "analysis_mode": analysis_mode,
            "playback_mode": playback_mode,
            "playback_rate": playback_rate,
            "cpu_preset": cpu_preset,
            "backend": backend,
            "buffer_ahead_sec": buffer_ahead_sec,
            "overrides": job_overrides,
        }
        with ctx["job_lock"]:
            ctx["job_options"][job_id] = options

        session: dict[str, Any] | None = None
        use_realtime = input_type == "video" and analysis_mode != "legacy"
        if use_realtime:
            try:
                session = ctx["realtime"].start_video_session(
                    job_id,
                    saved,
                    model_name,
                    cpu_preset=cpu_preset,
                    backend=backend,
                    mode=playback_mode,
                    buffer_ahead_sec=buffer_ahead_sec,
                    frame_stride=stride_val if raw_stride else None,
                    log_detections=True,
                    playback_rate=playback_rate,
                )
            except (ValueError, TypeError) as exc:
                store.fail_job(job_id, str(exc))
                return _error(400, str(exc))
            except Exception as exc:
                store.fail_job(job_id, str(exc))
                raise
            if session and session.get("frame_stride"):
                stride_val = int(session["frame_stride"])
                with ctx["job_lock"]:
                    ctx["job_options"][job_id]["frame_stride"] = stride_val
        else:
            ctx["runner"].submit(
                job_id,
                saved,
                model_name,
                overrides=job_overrides,
            )

        job = _job_or_404(job_id)
        data: dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "status_url": url_for("get_job", job_id=job_id),
            "frame_stride": stride_val,
        }
        if input_type == "video":
            data["stream_url"] = url_for("job_stream", job_id=job_id)
        if use_realtime:
            data.update(_video_urls(job_id))
            data.update({
                "analysis_mode": analysis_mode,
                "playback_mode": playback_mode,
                "playback_rate": playback_rate,
                "cpu_preset": cpu_preset,
                "backend": backend,
                "session": session,
            })

        response = jsonify(data)
        response.status_code = 202
        return response

    @app.get("/api/jobs")
    def list_jobs() -> Response:
        """Liệt kê job theo phân trang."""
        limit = request.args.get("limit", default=50, type=int)
        offset = request.args.get("offset", default=0, type=int)
        jobs = _ctx()["store"].get_jobs(
            limit=max(1, min(limit, 500)),
            offset=max(0, offset),
        )
        return jsonify([_job_json(job) for job in jobs])

    @app.get("/api/jobs/<int:job_id>")
    def get_job(job_id: int) -> Response:
        """Trả chi tiết, tiến độ và trạng thái runtime của job."""
        return jsonify(_job_json(_job_or_404(job_id)))

    @app.get("/api/jobs/<int:job_id>/detections")
    def job_detections(job_id: int) -> Response:
        """Trả detection và chuẩn hóa bbox theo DATA_CONTRACT."""
        _job_or_404(job_id)
        detections = [
            dict(det) for det in _ctx()["store"].get_detections(job_id)
        ]
        for det in detections:
            if "bbox" not in det and "x1" in det:
                det["bbox"] = [
                    det["x1"], det["y1"], det["x2"], det["y2"]
                ]
        return jsonify(detections)

    @app.get("/api/jobs/<int:job_id>/voice-events")
    def job_voice_events(job_id: int) -> Response:
        """Trả sự kiện giọng nói cùng vị trí và URL âm thanh."""
        _job_or_404(job_id)
        store = _ctx()["store"]
        events = [dict(event) for event in store.get_voice_events(job_id)]
        position_of = (
            {
                detection["id"]: detection.get("position")
                for detection in store.get_detections(job_id)
            }
            if events
            else {}
        )
        for event in events:
            event["position"] = position_of.get(event.get("detection_id"))
            event["audio_url"] = url_for(
                "voice_audio",
                job_id=job_id,
                event_id=event["id"],
            )
        return jsonify(events)

    @app.get("/api/jobs/<int:job_id>/voice-events/<int:event_id>/audio")
    def voice_audio(job_id: int, event_id: int) -> Response:
        """Trả file âm thanh của sự kiện thuộc job."""
        _job_or_404(job_id)
        event = next(
            (
                item
                for item in _ctx()["store"].get_voice_events(job_id)
                if item["id"] == event_id
            ),
            None,
        )
        if event is None:
            abort(404, description=f"Không có voice event {event_id}")
        return send_file(_safe_output_path(event.get("audio_path")))

    @app.get("/api/jobs/<int:job_id>/result")
    def job_result(job_id: int) -> Response:
        """Hiển thị hoặc tải file kết quả khi job hoàn tất."""
        job = _job_or_404(job_id)
        if job["status"] != "done":
            return _error(
                409,
                f"Job đang ở trạng thái '{job['status']}', chưa có kết quả",
            )
        path = _safe_output_path(job.get("output_path"))
        download = request.args.get("download", default=0, type=int) == 1
        return send_file(
            path,
            as_attachment=download,
            download_name=path.name,
            conditional=True,
        )

    @app.get("/api/jobs/<int:job_id>/stream")
    def job_stream(job_id: int) -> Response:
        """Giữ nguyên luồng MJPEG của pipeline legacy."""
        job = _job_or_404(job_id)
        if job.get("input_type") != "video":
            return _error(400, "Luồng live chỉ hỗ trợ job video")

        runner = _ctx()["runner"]
        if not runner.live.has_slot(job_id):
            return _error(404, "Luồng live không khả dụng")

        return Response(
            stream_with_context(runner.live.iter_mjpeg(job_id)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs/<int:job_id>/frame")
    def job_frame(job_id: int) -> Response:
        """Trả JPEG mới nhất của job legacy."""
        _job_or_404(job_id)
        jpeg = _ctx()["runner"].live.latest_frame(job_id)
        if jpeg is None:
            abort(404, description="Chưa có khung hình")
        return Response(
            jpeg,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/jobs/<int:job_id>/source")
    def job_source(job_id: int) -> Response:
        """Phục vụ video upload gốc với hỗ trợ HTTP Range."""
        job = _job_or_404(job_id)
        if job.get("input_type") != "video":
            return _error(400, "File nguồn video chỉ dành cho job video")
        return send_file(
            _safe_upload_path(job.get("input_path")),
            conditional=True,
        )

    @app.get("/api/jobs/<int:job_id>/events")
    def job_events(job_id: int) -> Response:
        """Phát các sự kiện SSE của phiên phân tích video."""
        _video_session_or_404(job_id)
        return _sse_response("video", job_id)

    @app.post("/api/jobs/<int:job_id>/playback")
    def job_playback(job_id: int) -> Response:
        """Validate và cập nhật đồng hồ phát video của client."""
        realtime = _video_session_or_404(job_id)
        payload = _json_object()
        allowed = {
            "timestamp_sec",
            "playing",
            "playback_rate",
            "seeked",
            "mode",
            "buffer_ahead_sec",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise BadRequest(
                description=f"Trường không hỗ trợ: {', '.join(sorted(unknown))}"
            )

        options: dict[str, Any] = {}
        for field in ("timestamp_sec", "buffer_ahead_sec"):
            if field in payload:
                options[field] = _number(payload[field], field)
        if "playback_rate" in payload:
            options["playback_rate"] = _number(
                payload["playback_rate"],
                "playback_rate",
                strictly_positive=True,
            )
        for field in ("playing", "seeked"):
            if field in payload:
                if not isinstance(payload[field], bool):
                    raise BadRequest(
                        description=f"{field} phải là boolean"
                    )
                options[field] = payload[field]
        if "mode" in payload:
            options["mode"] = _choice(
                payload["mode"], "mode", _PLAYBACK_MODES
            )

        try:
            state = realtime.update_video_playback(job_id, **options)
        except KeyError:
            abort(404, description="Không có phiên realtime cho job này")
        except (ValueError, TypeError) as exc:
            return _error(400, str(exc))
        ctx = _ctx()
        playback = state.get("playback") if isinstance(state, Mapping) else None
        if isinstance(playback, Mapping):
            with ctx["job_lock"]:
                job_opts = ctx["job_options"].get(job_id)
                if job_opts is not None:
                    if "playback_rate" in playback:
                        job_opts["playback_rate"] = playback["playback_rate"]
                    if "mode" in playback:
                        job_opts["playback_mode"] = playback["mode"]
        return jsonify(state)

    @app.post("/api/jobs/<int:job_id>/cancel")
    def job_cancel(job_id: int) -> Response:
        """Hủy phiên realtime hoặc job legacy tương ứng."""
        ctx = _ctx()
        with ctx["job_lock"]:
            _job_or_404(job_id)
            if ctx["realtime"].has_video_session(job_id):
                cancelled = ctx["realtime"].cancel_video_session(job_id)
            else:
                cancelled = ctx["runner"].cancel(job_id)
        return jsonify({
            "job_id": job_id,
            "cancelled": bool(cancelled),
        })

    @app.post("/api/jobs/<int:job_id>/export")
    def job_export(job_id: int) -> Response:
        """Xuất video bằng pipeline legacy khi không còn phiên realtime."""
        ctx = _ctx()
        with ctx["job_lock"]:
            job = _job_or_404(job_id)
            if job.get("input_type") != "video":
                return _error(400, "Chỉ hỗ trợ xuất MP4 cho job video")
            runtime_state = ctx["realtime"].get_video_status(job_id)
            if (
                runtime_state is not None
                and runtime_state.get("status")
                not in {"done", "failed", "cancelled", "stopped"}
            ):
                return _error(
                    409,
                    "Phiên phân tích realtime đang chạy. Hãy hủy hoặc chờ hoàn tất trước khi xuất MP4.",
                )
            if job["status"] in ("queued", "processing"):
                return _error(409, "Job đang được xử lý")

            source_path = _safe_upload_path(job.get("input_path"))
            payload = _json_object() if request.is_json else {}
            if not request.is_json and request.get_data():
                raise BadRequest(description="Request phải có JSON object")

            unknown = set(payload) - {"frame_stride"}
            if unknown:
                raise BadRequest(
                    description=(
                        "Trường không hỗ trợ: "
                        f"{', '.join(sorted(unknown))}"
                    )
                )

            options = ctx["job_options"].get(job_id, {})
            job_overrides = options.get("overrides")
            if "frame_stride" in payload:
                frame_stride = _integer(
                    payload["frame_stride"], "frame_stride", 1, 300
                )
                job_overrides = {
                    "inference": {"frame_stride": frame_stride}
                }
            else:
                frame_stride = int(
                    options.get("frame_stride")
                    or (ctx["cfg"].get("inference") or {}).get(
                        "frame_stride", 1
                    )
                )

            ctx["job_options"][job_id] = {
                **options,
                "analysis_mode": "legacy",
                "frame_stride": frame_stride,
                "overrides": job_overrides,
            }
            ctx["store"].set_status(job_id, "queued")
            ctx["runner"].submit(
                job_id,
                source_path,
                job["model_name"],
                overrides=job_overrides,
                render_video=True,
            )
            current = _job_or_404(job_id)

        response = jsonify({
            "job_id": job_id,
            "status": current["status"],
            "status_url": url_for("get_job", job_id=job_id),
            "frame_stride": frame_stride,
            "analysis_mode": "legacy",
            "stream_url": url_for("job_stream", job_id=job_id),
        })
        response.status_code = 202
        return response

    @app.get("/api/jobs/<int:job_id>/runtime-audio/<alert_id>")
    def job_runtime_audio(job_id: int, alert_id: str) -> Response:
        """Trả âm thanh runtime nằm trong outputs_dir."""
        _job_or_404(job_id)
        path = _ctx()["realtime"].get_runtime_audio_path(
            "video", job_id, alert_id
        )
        return send_file(_safe_output_path(path), conditional=True)

    @app.get("/api/runtime/backends")
    def runtime_backends() -> Response:
        """Trả các backend khả dụng và đầy đủ cấu hình CPU preset."""
        return jsonify({
            "backends": probe_runtime_backends(),
            "presets": CPU_PRESETS,
        })

    @app.post("/api/camera/sessions")
    def camera_create_session() -> Response:
        """Tạo phiên camera từ JSON hoặc multipart form-data."""
        if request.is_json:
            values: Mapping[str, Any] = _json_object()
            allow_string = False
        elif request.mimetype in (
            "multipart/form-data",
            "application/x-www-form-urlencoded",
        ):
            values = request.form
            allow_string = True
        elif not request.get_data():
            values = {}
            allow_string = False
        else:
            raise BadRequest(
                description="Request phải là JSON hoặc form-data"
            )

        model, preset, backend = _runtime_selection(
            values, default_model="yolo"
        )
        frame_stride = _integer(
            values.get("frame_stride", 1),
            "frame_stride",
            1,
            300,
            allow_string=allow_string,
        )
        try:
            session = _ctx()["realtime"].create_camera_session(
                model,
                cpu_preset=preset,
                backend=backend,
                frame_stride=frame_stride,
                synthesizer=_synthesize_runtime_audio,
            )
        except (ValueError, TypeError) as exc:
            return _error(400, str(exc))

        data = dict(session)
        session_id = data.get("session_id") or data.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(
                "RealtimeSessionManager không trả session_id hợp lệ"
            )
        data["session_id"] = session_id
        data.update({
            "events_url": url_for(
                "camera_events", session_id=session_id
            ),
            "frame_url": url_for(
                "camera_frame", session_id=session_id
            ),
            "stop_url": url_for(
                "camera_stop_session", session_id=session_id
            ),
        })
        response = jsonify(data)
        response.status_code = 201
        return response

    @app.post("/api/camera/sessions/<session_id>/frame")
    def camera_frame(session_id: str) -> Response:
        """Nhận một frame và trả kết quả nhận diện cùng telemetry."""
        realtime = _camera_session_or_404(
            session_id, require_active=True
        )

        raw_timestamp = _frame_metadata_value("timestamp_sec")
        raw_frame_id = _frame_metadata_value("frame_id")
        timestamp_sec = (
            _number(
                raw_timestamp,
                "timestamp_sec",
                allow_string=True,
            )
            if raw_timestamp is not None
            else None
        )
        frame_id = (
            _integer(
                raw_frame_id,
                "frame_id",
                0,
                allow_string=True,
            )
            if raw_frame_id is not None
            else None
        )

        upload = request.files.get("file")
        if upload is None:
            upload = request.files.get("frame")
        if upload is not None:
            frame_bytes = upload.read()
        elif request.mimetype == "image/jpeg":
            frame_bytes = request.get_data()
        else:
            return _error(
                400,
                "Thiếu ảnh: dùng field 'file'/'frame' hoặc body image/jpeg",
            )
        if not frame_bytes:
            return _error(400, "Ảnh rỗng hoặc không hợp lệ")

        try:
            result = realtime.push_camera_frame(
                session_id,
                frame_bytes,
                timestamp_sec=timestamp_sec,
                frame_id=frame_id,
            )
        except KeyError:
            abort(404, description="Không có phiên camera")
        except RuntimeError:
            abort(404, description="Phiên camera đã dừng")
        except (ValueError, TypeError) as exc:
            return _error(400, str(exc))
        return jsonify(result)

    @app.get("/api/camera/sessions/<session_id>/events")
    def camera_events(session_id: str) -> Response:
        """Phát detection, cảnh báo và telemetry của camera qua SSE."""
        _camera_session_or_404(session_id)
        return _sse_response("camera", session_id)

    @app.get("/api/camera/sessions/<session_id>/audio/<alert_id>")
    def camera_audio(session_id: str, alert_id: str) -> Response:
        """Phục vụ âm thanh cảnh báo của phiên camera."""
        _camera_session_or_404(session_id)
        path = _ctx()["realtime"].get_runtime_audio_path(
            "camera", session_id, alert_id
        )
        return send_file(_safe_output_path(path), conditional=True)

    @app.post("/api/camera/sessions/<session_id>/stop")
    def camera_stop_session(session_id: str) -> Response:
        """Dừng phiên camera và trả trạng thái cuối."""
        realtime = _camera_session_or_404(session_id)
        try:
            state = realtime.stop_camera_session(session_id)
        except KeyError:
            abort(404, description="Không có phiên camera")
        return jsonify(state)


def main() -> None:
    """Khởi chạy server và đóng toàn bộ worker khi server dừng."""
    parser = argparse.ArgumentParser(
        description="Web TSD Voice Alert (localhost)"
    )
    parser.add_argument(
        "--runtime",
        default="local",
        help="local | kaggle | đường dẫn yaml",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="detector + hậu xử lý giả để demo",
    )
    parser.add_argument(
        "--store",
        default="auto",
        choices=["auto", "memory", "sqlite"],
    )
    parser.add_argument("--host", default=None, help="mặc định cfg app.host")
    parser.add_argument(
        "--port", type=int, default=None, help="mặc định cfg app.port"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(
        runtime=args.runtime,
        fake=args.fake,
        store_kind=args.store,
    )
    app_cfg = app.extensions["tsd"]["cfg"]["app"]
    try:
        app.run(
            host=args.host or app_cfg["host"],
            port=args.port or app_cfg["port"],
            debug=args.debug,
            use_reloader=False,
            threaded=True,
        )
    finally:
        app.extensions["tsd"]["shutdown"]()


if __name__ == "__main__":
    main()
