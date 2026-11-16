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

## Sơ đồ Thiết kế Chi tiết (UML & Mermaid)

Hệ thống cung cấp đầy đủ 4 sơ đồ chuẩn tại [docs/SYSTEM_DIAGRAMS.md](SYSTEM_DIAGRAMS.md):
- **Use Case Diagram**: Tương tác người dùng với hệ thống nhận diện và cảnh báo.
- **Sequence Diagram**: Luồng khởi tạo đệm 5s, phát đồng bộ video/audio/box và luồng tua video (Seek).
- **Activity Diagram**: Hoạt động song song giữa luồng phần cứng phía client và luồng suy luận/TTS phía server.
- **Class Diagram**: Cấu trúc lớp hướng đối tượng (OOP) của cả Frontend và Backend.
