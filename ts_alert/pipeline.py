"""LUỒNG XỬ LÝ TỔNG THỂ (đã code sẵn)

    ảnh/video -> detector.predict -> vị trí (trái/phải/trước mặt) -> ghi log
              -> chống lặp -> vẽ layer -> TTS -> ghép A/V -> ghi log (thời gian, số liệu)

Các bước cụ thể gọi sang module của từng thành viên; chỉ cần hiện thực các hàm TODO trong đó.
Nơi lưu log là `repo` (giao diện ResultStore trong ts_alert/db/interface.py).
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

from ts_alert.config.resolve import resolve_config, resolve_path
from ts_alert.db.interface import ResultStore
from ts_alert.pipelines import get_detector
from ts_alert.postprocess import draw_layer, merge_av, position, tts
from ts_alert.postprocess.dedup import Deduplicator
from ts_alert.runtime import video_io
from ts_alert.types import Detection, SpeechEvent
from ts_alert.utils.io import load_classes

log = logging.getLogger(__name__)


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


def _resolve_frame_stride(cfg: dict) -> int:
    """Đọc inference.frame_stride, mặc định 1; từ chối bool, số thực lẻ và giá trị < 1."""
    value = (cfg.get("inference") or {}).get("frame_stride", 1)
    if value is None:
        return 1
    if isinstance(value, bool):
        raise ValueError("inference.frame_stride không được là bool")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"inference.frame_stride phải là số nguyên >= 1, nhận: {value!r}")

    try:
        frame_stride = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "inference.frame_stride phải chuyển được thành số nguyên >= 1"
        ) from exc

    if frame_stride < 1:
        raise ValueError("inference.frame_stride phải >= 1")

    return frame_stride


def _build_speech_text(label: str, pos: str, classes: dict, cfg: dict) -> str:
    """Truyền cfg khi chữ ký hỗ trợ, tương thích với hàm chỉ nhận ba đối số."""
    try:
        parameters = tuple(
            inspect.signature(tts.build_speech_text).parameters.values()
        )
    except (ValueError, TypeError):
        return tts.build_speech_text(label, pos, classes)

    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    for parameter in parameters:
        if parameter.name == "cfg":
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                return tts.build_speech_text(label, pos, classes, cfg=cfg)
            if parameter.kind in positional_kinds:
                return tts.build_speech_text(label, pos, classes, cfg)

    if len(parameters) >= 4 and parameters[3].kind in positional_kinds:
        return tts.build_speech_text(label, pos, classes, cfg)

    return tts.build_speech_text(label, pos, classes)


def process_media(
    input_path: str | PathLike[str],
    model_name: str,
    cfg: dict | None = None,
    detector: Any = None,
    repo: ResultStore | None = None,
    job_id: int | None = None,
    out_dir: str | PathLike[str] | None = None,
    on_frame: Callable[..., Any] | None = None,
) -> dict:
    """Xử lý một ảnh hoặc video end-to-end và ghi log. Trả dict tóm tắt kết quả.

    unique_sign_count : số biển báo (đối tượng) sau chống lặp = số SpeechEvent
    voice_play_count  : số lần giọng nói THỰC SỰ được tạo (TTS thành công)
    """
    input_path = Path(input_path)
    cfg = cfg or resolve_config(model_name)
    paths = cfg["paths"]
    out_dir = Path(out_dir) if out_dir else resolve_path(paths["outputs_dir"])
    if repo is None:
        from ts_alert.db.repository import Repository

        repo = Repository(resolve_path(paths["db_path"]))

    input_type = video_io.media_type(input_path)
    if job_id is None:
        job_id = repo.create_job(input_type, str(input_path), model_name)
    repo.set_status(job_id, "processing")

    t_total = time.perf_counter()
    layer_ms = tts_ms = 0.0
    unique_signs = voice_plays = 0
    audio_events: list[dict] = []

    try:
        frame_stride = _resolve_frame_stride(cfg)
        detector = detector or get_detector(model_name, cfg)
        classes = load_classes(resolve_path(paths["classes_csv"]))
        deduper = Deduplicator(cfg)
        cached_detections: list[Detection] = []

        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        stem = f"job{job_id}_{input_path.stem}"

        writer = None
        try:
            if input_type == "video":
                info = video_io.get_video_info(input_path)
                silent_path = out_dir / "videos" / f"{stem}_silent.mp4"
                final_path = out_dir / "videos" / f"{stem}_result.mp4"
                silent_path.parent.mkdir(parents=True, exist_ok=True)
                writer = video_io.VideoWriter(
                    silent_path,
                    info["fps"],
                    (info["width"], info["height"]),
                )
                frames = video_io.iter_frames(input_path)
            else:
                final_path = out_dir / "images" / f"{stem}_result.jpg"
                final_path.parent.mkdir(parents=True, exist_ok=True)
                frames = iter([(0, video_io.read_image(input_path), 0.0)])

            for frame_index, frame, timestamp in frames:
                should_predict = (
                    input_type != "video" or frame_index % frame_stride == 0
                )
                frame_events: list[SpeechEvent] = []

                if should_predict:
                    detections = detector.predict(frame)
                    detections = position.add_positions(
                        detections, frame.shape[1], cfg
                    )
                    repo.add_detections(
                        job_id, frame_index, timestamp, detections
                    )

                    frame_events = deduper.update(
                        detections, frame_index, timestamp
                    )
                    unique_signs += len(frame_events)
                    cached_detections = [dict(d) for d in detections]
                    draw_dets = detections
                else:
                    draw_dets = cached_detections

                t0 = time.perf_counter()
                drawn_frame = draw_layer.draw_detections(frame, draw_dets)
                layer_ms += _ms(t0)

                for event in frame_events:
                    text = _build_speech_text(
                        event["label"], event["position"], classes, cfg
                    )
                    audio_path = (
                        audio_dir / f"{stem}_{len(audio_events):03d}.mp3"
                    )
                    t0 = time.perf_counter()
                    try:
                        tts.synthesize(text, audio_path, cfg)
                    except Exception:
                        log.exception("TTS lỗi cho câu: %s", text)
                        continue
                    finally:
                        tts_ms += _ms(t0)

                    voice_plays += 1
                    repo.add_voice_event(
                        job_id,
                        event["detection"].get("db_id"),
                        text,
                        event["timestamp_sec"],
                        str(audio_path),
                    )
                    audio_events.append(
                        {
                            "audio_path": audio_path,
                            "timestamp_sec": event["timestamp_sec"],
                            "text": text,
                        }
                    )

                if writer is not None:
                    writer.write(drawn_frame)
                else:
                    video_io.write_image(final_path, drawn_frame)

                if on_frame is not None:
                    on_frame(
                        frame_index=frame_index,
                        timestamp_sec=timestamp,
                        frame=drawn_frame,
                        predicted=should_predict,
                        layer_time_ms=layer_ms,
                        tts_time_ms=tts_ms,
                        detections=draw_dets,
                        events=frame_events,
                    )
        finally:
            if writer is not None:
                writer.close()

        if input_type == "video":
            if audio_events:
                merge_av.merge_audio_video(
                    silent_path, audio_events, final_path
                )
                silent_path.unlink(missing_ok=True)
            else:
                silent_path.replace(final_path)

        total_ms = _ms(t_total)
        repo.finish_job(
            job_id,
            str(final_path),
            unique_signs,
            voice_plays,
            layer_ms,
            tts_ms,
            total_ms,
        )
        repo.set_status(job_id, "done")
        return {
            "job_id": job_id,
            "status": "done",
            "output_path": str(final_path),
            "unique_sign_count": unique_signs,
            "voice_play_count": voice_plays,
            "layer_time_ms": layer_ms,
            "tts_time_ms": tts_ms,
            "total_time_ms": total_ms,
            "audio_paths": [str(e["audio_path"]) for e in audio_events],
        }
    except Exception as exc:
        repo.fail_job(job_id, f"{type(exc).__name__}: {exc}")
        raise
