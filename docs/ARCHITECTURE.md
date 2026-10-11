# Kiến trúc

Luồng tổng thể được code sẵn trong `ts_alert/pipeline.py::process_media`.

```text
Web UI (app/) ── upload ──► (chạy nền) ──► process_media
                                                                          │
   ┌──────────────────────────────────────────────────────────────────────┘
   ▼
 video_io.iter_frames ─► detector.predict ─► position.add_positions ─► repo.add_detections
                                                                          │
                                       Deduplicator.update ◄──────────────┘
                                              │ SpeechEvent
              ┌───────────────────────────────┤
              ▼                               ▼
    draw_layer.draw_detections        tts.build_speech_text ─► tts.synthesize ─► repo.add_voice_event
              │                                                   │
              ▼                                                   ▼
        VideoWriter.write ─────────► merge_av.merge_audio_video ─► repo.finish_job (thời gian, số liệu)
```

Hai mô hình được nạp qua `ts_alert/pipelines/__init__.py` (`get_detector`) nên phần còn lại của hệ thống
không phụ thuộc YOLO hay Faster R-CNN.

TODO: bổ sung sơ đồ chính thức (draw.io / Mermaid) vào `docs/images/`.
