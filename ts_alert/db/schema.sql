-- Owner: Thành viên 5 — TODO: thiết kế bảng của nhóm (SQLite).
--
-- Cần lưu được (xem ts_alert/db/interface.py):
--   * job xử lý: input, model, output, số biển báo duy nhất, số lần giọng nói phát, thời gian layer / TTS / tổng, trạng thái
--   * detection: biển nào, khung hình nào, timestamp, bbox, độ tin cậy, vị trí (left | right | front)
--   * voice event: câu đã đọc, detection kích hoạt, timestamp, đường dẫn file âm thanh
--
-- Gợi ý khởi đầu: 3 bảng processing_job, detection, voice_event (detection/voice_event có khoá ngoại job_id).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS processing_job (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_type TEXT NOT NULL CHECK (input_type IN ('image', 'video')),
    input_path TEXT NOT NULL,
    model_name TEXT NOT NULL CHECK (model_name IN ('yolo', 'faster_rcnn')),
    output_path TEXT,
    unique_sign_count INTEGER NOT NULL DEFAULT 0 CHECK (unique_sign_count >= 0),
    voice_play_count INTEGER NOT NULL DEFAULT 0 CHECK (voice_play_count >= 0),
    layer_time_ms REAL CHECK (layer_time_ms IS NULL OR layer_time_ms >= 0),
    tts_time_ms REAL CHECK (tts_time_ms IS NULL OR tts_time_ms >= 0),
    total_time_ms REAL CHECK (total_time_ms IS NULL OR total_time_ms >= 0),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS detection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
    timestamp_sec REAL NOT NULL CHECK (timestamp_sec >= 0),
    label TEXT NOT NULL,
    class_id INTEGER NOT NULL CHECK (class_id >= 0),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('left', 'front', 'right')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (x2 >= x1 AND y2 >= y1),
    FOREIGN KEY (job_id) REFERENCES processing_job(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voice_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    detection_id INTEGER,
    spoken_text TEXT NOT NULL,
    timestamp_sec REAL NOT NULL CHECK (timestamp_sec >= 0),
    audio_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES processing_job(id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detection(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_job_created ON processing_job(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_status ON processing_job(status);
CREATE INDEX IF NOT EXISTS idx_detection_job_frame ON detection(job_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_detection_label ON detection(label);
CREATE INDEX IF NOT EXISTS idx_detection_class_id ON detection(class_id);
CREATE INDEX IF NOT EXISTS idx_voice_job_time ON voice_event(job_id, timestamp_sec);
