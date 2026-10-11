# Data Contract

Những điều nhóm **phải tuân thủ** để các phần ghép được với nhau. Ngoài các điều này, mỗi thành viên được tự do
thiết kế bên trong phần của mình (cấu trúc DB, giao diện web, thuật toán...). Kiểu khai báo: `ts_alert/types.py`.

## 1. Đầu ra `predict(frame)` của YOLO và Faster R-CNN
```python
[{"bbox": [x1, y1, x2, y2], "label": str, "class_id": int, "confidence": float}]
```
- `frame`: ndarray BGR (H, W, 3), **ảnh gốc** (không phải ảnh đã letterbox).
- `bbox`: pixel theo ảnh gốc. `class_id` theo `configs/classes.csv` (**không** tính background của Faster R-CNN).
- Không phát hiện gì -> trả `[]`.

## 2. `position.add_positions` (Thành viên 4)
Thêm khoá `"position"` ∈ {`"left"`, `"front"`, `"right"`}; ngưỡng ở `configs/shared.yaml` (`postprocess.position`).

## 3. `Deduplicator.update` (Thành viên 4)
Nhận detection của khung hiện tại, trả danh sách **SpeechEvent** cần đọc (chỉ biển mới xuất hiện):
```python
{"detection": <Detection>, "label": str, "position": str, "frame_index": int, "timestamp_sec": float}
```

## 4. Âm thanh (Thành viên 4)
- `tts.build_speech_text(label, position, classes) -> str`
- `tts.synthesize(text, out_path, cfg) -> Path`
- `merge_av.merge_audio_video(video_path, audio_events, out_path)` với
  `audio_events = [{"audio_path": Path, "timestamp_sec": float, "text": str}]`

## 5. Video I/O (Thành viên 3)
- `get_video_info(path) -> {"fps", "width", "height", "frame_count"}`
- `iter_frames(path)` yield `(frame_index, frame_bgr, timestamp_sec)`

## 6. Nơi lưu log — `ResultStore` (Thành viên 5)
Giao diện ở `ts_alert/db/interface.py`. Cấu trúc bảng tuỳ nhóm, nhưng `Repository` phải có đủ các hàm
`create_job`, `set_status`, `add_detections`, `add_voice_event`, `finish_job`, `fail_job` và lưu được đủ **dữ liệu cốt lõi**:
biển nào / khung hình nào / thời điểm nào / vị trí; các lần phát giọng nói (câu đọc, biển kích hoạt, thời điểm);
đường dẫn file âm thanh và video/ảnh kết quả; số biển duy nhất; số lần giọng nói thực sự phát;
thời gian vẽ layer, tạo giọng nói và tổng.
- Thời gian tính bằng **mili-giây**, timestamp tính bằng **giây**.
- `unique_sign_count` = số SpeechEvent (biển sau chống lặp); `voice_play_count` = số lần TTS tạo âm thanh thành công.

## 7. Web (Thành viên 3)
Tự do framework và giao diện. Tối thiểu: upload ảnh/video + chọn model → gọi `ts_alert.pipeline.process_media`
→ xem và tải kết quả. Chạy được bằng `python -m app.main`.

## 8. Quy tắc cấu hình
- `seed`, `data.*`, `train.batch_size`, `train.early_stopping_patience`, `train.amp`, `train.primary_metric`, `eval.*` chỉ đặt trong `configs/shared.yaml`.
- `train.batch_size` là **tổng** trên mọi GPU: Faster R-CNN (DDP) dùng `per_rank_batch`; YOLO truyền nguyên tổng.
- Hai mô hình dùng chung split trong `data/processed` và chung bộ chỉ số `mAP50-95, mAP50, mAP75, AR100`.
