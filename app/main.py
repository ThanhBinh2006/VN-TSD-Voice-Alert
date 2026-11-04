"""Web localhost.   Phụ trách: Thành viên 3

Tự do chọn framework (Flask/FastAPI), cấu trúc thư mục và giao diện. Chỉ cần tuân thủ:
  1. Nhận file ảnh/video + lựa chọn model ("yolo" | "faster_rcnn").
  2. Gọi ts_alert.pipeline.process_media(input_path, model_name, ...) — NÊN chạy nền để không treo request,
     và NÊN giữ detector trong bộ nhớ (get_detector) để không nạp lại trọng số mỗi lần.
  3. Cho người dùng xem và tải video/ảnh kết quả, hiển thị các số liệu process_media trả về.
  4. Chạy được bằng:  python -m app.main   (host/port lấy từ cfg["app"] trong configs/shared.yaml)

Giao diện: mở http://127.0.0.1:5000 sau khi chạy.

Chạy:
    python -m app.main                 # dùng model thật (cần models/*/best.* và code của TV1, TV2, TV4)
    python -m app.main --fake          # detector giả + hậu xử lý giả cho phần còn là vỏ -> demo được ngay
    python -m app.main --store memory  # ép lưu log trong RAM (mặc định: SQLite nếu TV5 đã xong, không thì RAM)

API (JSON):
    GET  /api/health                              trạng thái server, model, nơi lưu log
    POST /api/jobs            form-data: file, model   -> 202 {"job_id", "status"}
    GET  /api/jobs[?limit=50&offset=0]            danh sách job (mới nhất trước)
    GET  /api/jobs/<id>                           chi tiết + số liệu + link kết quả + tiến độ ("progress")
    GET  /api/jobs/<id>/detections                các detection đã ghi
    GET  /api/jobs/<id>/voice-events              các lần phát giọng nói (+ vị trí, link file âm thanh)
    GET  /api/jobs/<id>/voice-events/<eid>/audio  file âm thanh của một lần phát
    GET  /api/jobs/<id>/result[?download=1]       xem / tải video hoặc ảnh kết quả

"progress" trong mỗi job (frontend gọi lại GET /api/jobs/<id> mỗi ~1 giây để vẽ thanh tiến độ):
    {"stage": "queued" | "loading_model" | "processing" | "finalizing" | "done" | "failed",
     "percent": 0-100 hoặc null (chưa biết tổng số khung), "frames_done", "frame_count",
     "fps": tốc độ xử lý (khung/giây), "eta_sec": ước lượng số giây còn lại hoặc null}
"""
from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from app.jobs import JobRunner
from ts_alert.config.resolve import resolve_config, resolve_path
from ts_alert.runtime import video_io
from ts_alert.types import IMAGE_EXTS, MODEL_NAMES, VIDEO_EXTS

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- khởi tạo

def _make_store(kind: str, db_path: Path):
    """kind: "auto" | "memory" | "sqlite". auto = SQLite nếu Repository của TV5 đã hiện thực, không thì RAM."""
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


def create_app(runtime: str = "local", fake: bool = False, store=None, store_kind: str = "auto",
               sync: bool = False, overrides: dict | None = None) -> Flask:
    """runtime: local | kaggle | đường dẫn yaml. fake: dùng detector + hậu xử lý giả.
    store: truyền sẵn ResultStore (test); None -> tạo theo store_kind. sync: chạy job ngay (test)."""
    base_cfg = resolve_config(None, runtime, overrides)
    paths = base_cfg["paths"]
    outputs_dir = resolve_path(paths["outputs_dir"])
    uploads_dir = resolve_path(paths["uploads_dir"])
    uploads_dir.mkdir(parents=True, exist_ok=True)

    cfg_cache: dict[str, dict] = {}

    def cfg_for(model_name: str) -> dict:
        if model_name not in cfg_cache:
            cfg_cache[model_name] = resolve_config(model_name, runtime, overrides)
        return cfg_cache[model_name]

    fake_names: list[str] = []
    restore_fakes = lambda: None  # noqa: E731
    if fake:
        from app.fake import fake_detector_factory, install_fakes
        restore_fakes, fake_names = install_fakes()
        detector_factory = fake_detector_factory
    else:
        from ts_alert.pipelines import get_detector
        detector_factory = get_detector

    store = store if store is not None else _make_store(store_kind, resolve_path(paths["db_path"]))
    _fail_unfinished_jobs(store)
    runner = JobRunner(store, cfg_for, detector_factory, outputs_dir, sync=sync)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(base_cfg["app"]["max_upload_mb"]) * 1024 * 1024
    app.json.ensure_ascii = False          # trả tiếng Việt có dấu thay vì \uXXXX
    app.extensions["tsd"] = {"store": store, "runner": runner, "cfg": base_cfg, "fake": fake,
                             "fake_parts": fake_names, "restore_fakes": restore_fakes,
                             "outputs_dir": outputs_dir, "uploads_dir": uploads_dir}

    _register_routes(app)
    return app


def _fail_unfinished_jobs(store) -> None:
    """SQLite giữ dữ liệu qua các lần chạy: job còn queued/processing lúc khởi động là job bị bỏ dở khi tắt
    server lần trước (không còn ai chạy tiếp). Đánh dấu lỗi để giao diện không chờ mãi."""
    for status in ("queued", "processing"):
        for job in store.get_jobs(limit=1000, status=status):
            store.fail_job(job["id"], "Server đã tắt khi job đang chạy. Hãy gửi lại file.")
            log.warning("Job %s bị bỏ dở từ lần chạy trước -> failed", job["id"])


# ----------------------------------------------------------------------------- tiện ích

def _ctx() -> dict:
    from flask import current_app
    return current_app.extensions["tsd"]


def _error(status: int, message: str):
    resp = jsonify({"error": message})
    resp.status_code = status
    return resp


def _job_or_404(job_id: int) -> dict:
    job = _ctx()["store"].get_job(job_id)
    if job is None:
        abort(404, description=f"Không có job {job_id}")
    return job


def _safe_output_path(path_str: str | None) -> Path:
    """Chỉ cho phép trả file nằm trong outputs_dir (chống đọc file tuỳ ý qua đường dẫn lưu trong DB)."""
    if not path_str:
        abort(404, description="Chưa có file kết quả")
    path = Path(path_str).resolve()
    if not path.is_relative_to(_ctx()["outputs_dir"].resolve()) or not path.is_file():
        abort(404, description="Không tìm thấy file kết quả")
    return path


def _progress_json(job: dict) -> dict:
    """Tiến độ của job. Trạng thái trong store là chuẩn; tracker (RAM) bổ sung số khung / % / fps / ETA.
    Job không còn trong tracker (vd. server đã khởi động lại) vẫn trả được stage theo status."""
    progress = _ctx()["runner"].progress.get(job["id"]) or {
        "stage": job["status"] if job["status"] in ("queued", "done", "failed") else "processing",
        "percent": None, "frames_done": None, "frame_count": None, "fps": None, "eta_sec": None}
    if job["status"] == "done":                      # store đã ghi xong -> chắc chắn 100%
        progress.update(stage="done", percent=100.0, eta_sec=None)
    elif job["status"] == "failed":
        progress.update(stage="failed", eta_sec=None)
    return progress


def _epoch(value):
    """Thời điểm -> số giây kể từ 1970 (giao diện cần dạng này). InMemoryStore lưu sẵn dạng số;
    SQLite (CURRENT_TIMESTAMP) lưu chuỗi "YYYY-MM-DD HH:MM:SS" theo giờ UTC."""
    if value is None or isinstance(value, (int, float)):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _job_json(job: dict) -> dict:
    data = dict(job)
    for key in ("created_at", "started_at", "finished_at"):
        if key in data:
            data[key] = _epoch(data[key])
    data["progress"] = _progress_json(job)
    data["input_name"] = Path(job["input_path"]).name if job.get("input_path") else None
    if job.get("status") == "done" and job.get("output_path"):
        data["result_url"] = url_for("job_result", job_id=job["id"])
        data["download_url"] = url_for("job_result", job_id=job["id"], download=1)
    return data


# ----------------------------------------------------------------------------- routes

def _register_routes(app: Flask) -> None:

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return _error(exc.code or 500, exc.description or exc.name)
        return exc

    @app.get("/")
    def index():
        # Giao diện: app/templates/index.html + app/static/app.css, app.js (gọi các API bên dưới).
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        ctx = _ctx()
        models = {}
        for name in MODEL_NAMES:
            weights = resolve_path(resolve_config(name, None)["model"]["weights"])
            models[name] = {"weights": str(weights), "weights_exists": weights.exists(),
                            "loaded": name in ctx["runner"].loaded_models()}
        return jsonify({"status": "ok", "fake": ctx["fake"], "fake_parts": ctx["fake_parts"],
                        "store": type(ctx["store"]).__name__, "models": models,
                        "max_upload_mb": ctx["cfg"]["app"]["max_upload_mb"],
                        "accept": sorted(VIDEO_EXTS | IMAGE_EXTS)})

    @app.post("/api/jobs")
    def create_job():
        ctx = _ctx()
        file = request.files.get("file")
        model_name = (request.form.get("model") or "").strip()
        if file is None or not file.filename:
            return _error(400, "Thiếu file (field 'file')")
        if model_name not in MODEL_NAMES:
            return _error(400, f"Model không hợp lệ: {model_name!r}. Chọn một trong {list(MODEL_NAMES)}")
        ext = Path(file.filename).suffix.lower()
        try:
            input_type = video_io.media_type(file.filename)
        except ValueError:
            return _error(400, f"Định dạng không hỗ trợ: {ext or '(không có đuôi)'}. "
                               f"Hỗ trợ: {', '.join(sorted(VIDEO_EXTS | IMAGE_EXTS))}")

        stem = secure_filename(Path(file.filename).stem) or "upload"   # bỏ dấu / ký tự lạ
        saved = ctx["uploads_dir"] / f"{uuid.uuid4().hex[:8]}_{stem}{ext}"
        file.save(saved)

        store = ctx["store"]
        job_id = store.create_job(input_type, str(saved), model_name)
        store.set_status(job_id, "queued")
        ctx["runner"].submit(job_id, saved, model_name)
        job = store.get_job(job_id)
        resp = jsonify({"job_id": job_id, "status": job["status"],
                        "status_url": url_for("get_job", job_id=job_id)})
        resp.status_code = 202
        return resp

    @app.get("/api/jobs")
    def list_jobs():
        limit = request.args.get("limit", default=50, type=int)
        offset = request.args.get("offset", default=0, type=int)
        jobs = _ctx()["store"].get_jobs(limit=max(1, min(limit, 500)), offset=max(0, offset))
        return jsonify([_job_json(j) for j in jobs])

    @app.get("/api/jobs/<int:job_id>")
    def get_job(job_id: int):
        return jsonify(_job_json(_job_or_404(job_id)))

    @app.get("/api/jobs/<int:job_id>/detections")
    def job_detections(job_id: int):
        _job_or_404(job_id)
        detections = _ctx()["store"].get_detections(job_id)
        for det in detections:              # SQLite lưu x1, y1, x2, y2 riêng -> trả thêm "bbox" như DATA_CONTRACT
            if "bbox" not in det and "x1" in det:
                det["bbox"] = [det["x1"], det["y1"], det["x2"], det["y2"]]
        return jsonify(detections)

    @app.get("/api/jobs/<int:job_id>/voice-events")
    def job_voice_events(job_id: int):
        _job_or_404(job_id)
        store = _ctx()["store"]
        events = store.get_voice_events(job_id)
        # Kèm vị trí (left/front/right) của detection đã kích hoạt cảnh báo để giao diện xếp đúng làn.
        position_of = {d["id"]: d.get("position") for d in store.get_detections(job_id)} if events else {}
        for ev in events:
            ev["position"] = position_of.get(ev.get("detection_id"))
            ev["audio_url"] = url_for("voice_audio", job_id=job_id, event_id=ev["id"])
        return jsonify(events)

    @app.get("/api/jobs/<int:job_id>/voice-events/<int:event_id>/audio")
    def voice_audio(job_id: int, event_id: int):
        _job_or_404(job_id)
        ev = next((e for e in _ctx()["store"].get_voice_events(job_id) if e["id"] == event_id), None)
        if ev is None:
            abort(404, description=f"Không có voice event {event_id}")
        return send_file(_safe_output_path(ev["audio_path"]))

    @app.get("/api/jobs/<int:job_id>/result")
    def job_result(job_id: int):
        job = _job_or_404(job_id)
        if job["status"] != "done":
            return _error(409, f"Job đang ở trạng thái '{job['status']}', chưa có kết quả")
        path = _safe_output_path(job.get("output_path"))
        download = request.args.get("download", default=0, type=int) == 1
        # conditional=True (mặc định) hỗ trợ Range request -> tua được video trong trình duyệt.
        return send_file(path, as_attachment=download, download_name=path.name)


# ----------------------------------------------------------------------------- chạy

def main() -> None:
    parser = argparse.ArgumentParser(description="Web TSD Voice Alert (localhost)")
    parser.add_argument("--runtime", default="local", help="local | kaggle | đường dẫn yaml")
    parser.add_argument("--fake", action="store_true", help="detector + hậu xử lý giả để demo")
    parser.add_argument("--store", default="auto", choices=["auto", "memory", "sqlite"])
    parser.add_argument("--host", default=None, help="mặc định cfg app.host")
    parser.add_argument("--port", type=int, default=None, help="mặc định cfg app.port")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(runtime=args.runtime, fake=args.fake, store_kind=args.store)
    app_cfg = app.extensions["tsd"]["cfg"]["app"]
    # use_reloader=False: reloader chạy app 2 lần -> 2 worker, nạp model 2 lần.
    app.run(host=args.host or app_cfg["host"], port=args.port or app_cfg["port"],
            debug=args.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
