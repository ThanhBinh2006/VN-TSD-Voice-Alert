# TSD-Voice-Alert — Nhận diện & Cảnh báo Biển báo Giao thông bằng Giọng nói

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![YOLO26m](https://img.shields.io/badge/Ultralytics-YOLO26m-00FFFF)](configs/pipelines/yolo.yaml)
[![Faster R-CNN](https://img.shields.io/badge/torchvision-Faster%20R--CNN%20R50-orange)](configs/pipelines/faster_rcnn.yaml)
[![W&B](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)](configs/shared.yaml)

Hệ thống mô phỏng hỗ trợ lái xe nâng cao (**ADAS**) cho phép phát hiện **90 loại biển báo giao thông Việt Nam** từ ảnh/video và **cảnh báo bằng giọng nói tiếng Việt** đúng thời điểm biển xuất hiện theo vị trí làn (`trái` / `trước mặt` / `phải`).

Dự án huấn luyện, tinh chỉnh siêu tham số và đối sánh công bằng hai họ mô hình phát hiện đối tượng trên **cùng một bộ dữ liệu chuẩn hóa không rò rỉ**:
- **YOLO26m** (One-stage detector, tối ưu tốc độ thời gian thực) — [`ultralytics`](ts_alert/pipelines/yolo/)
- **Faster R-CNN ResNet-50 FPN** (Two-stage detector, hỗ trợ DDP đa GPU) — [`torchvision`](ts_alert/pipelines/faster_rcnn/)

---

## Mục lục
1. [Tính năng chính & Kiến trúc hệ thống](#1-tính-năng-chính--kiến-trúc-hệ-thống)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Dữ liệu & Tiền xử lý](#3-dữ-liệu--tiền-xử-lý)
4. [Cài đặt & Thiết lập môi trường](#4-cài-đặt--thiết-lập-môi-trường)
5. [Hướng dẫn các cách chạy](#5-hướng-dẫn-các-cách-chạy)
6. [Hệ thống cấu hình phân tầng](#6-hệ-thống-cấu-hình-phân-tầng)
7. [Phiên bản công nghệ](#7-phiên-bản-công-nghệ)
8. [Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
9. [Thành viên, Phân công & Quy trình Git](#9-thành-viên-phân-công--quy-trình-git)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Tính năng chính & Kiến trúc hệ thống

- **So sánh hai pipeline trên chuẩn dữ liệu thống nhất**: Chia Train/Val/Test bất biến theo `configs/shared.yaml`, cùng kích thước ảnh `640×640`, cùng metric chính `mAP50-95`, cùng bộ đánh giá `evaluate_detector`.
- **Theo dõi thí nghiệm & Early Stopping đồng nhất**: Tự động đánh dấu mốc `* BEST`, vẽ đường cong huấn luyện (`training_curves.png`), ma trận nhầm lẫn (`confusion_matrix.png`), log chuẩn hóa (`results.csv`, `metrics_history.json`, `summary.json`, `metrics.json`) và đồng bộ lên **Weights & Biases** (bao gồm `system/learning_rate`, precision/recall/F1).
- **W&B Sweep với phân phối liên tục**: Hỗ trợ tìm kiếm siêu tham số (`bayes` / `random`) bằng phân phối `log_uniform_values` / `uniform` cho các tham số thực và tự động bảo vệ các khóa bất biến trong `configs/shared.yaml`.
- **Xử lý hậu kỳ & Cảnh báo giọng nói**: Xác định vị trí biển (`left` / `front` / `right`), chống cảnh báo lặp theo IoU + thời gian mất dấu (`Deduplicator`), vẽ bounding box, tổng hợp giọng nói (`gTTS` / `pyttsx3`), ghép âm thanh vào video qua `ffmpeg` và lưu lịch sử vào **SQLite**.
- **Web App & REST API (`localhost`)**: Giao diện Flask trực quan cho phép chọn model, tải lên ảnh/video, theo dõi tiến độ theo thời gian thực và xem/tải kết quả kèm lịch sử sự kiện giọng nói.

```text
iter_frames ─► detector.predict ─► add_positions ─► repo.add_detections (SQLite)
                                                          │
                                  Deduplicator.update ◄───┘
                                         │ SpeechEvent
                ┌────────────────────────┴──────────────┐
                ▼                                       ▼
        draw_detections                    build_speech_text ─► synthesize (TTS) ─► repo.add_voice_event
                │                                                   │
          VideoWriter.write ─────────► merge_audio_video ◄──────────┘ ─► repo.finish_job
```

Chi tiết kiến trúc và hợp đồng dữ liệu giữa các module xem tại [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) và [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

---

## 2. Cấu trúc thư mục

```text
TSD-Voice-Alert/
├── analysis/                        # Khám phá dữ liệu (EDA) & xuất biểu đồ
│   ├── eda.py                       # Router tự nhận diện raw / processed
│   ├── eda_raw.py                   # 8 biểu đồ phân tích tập thô
│   ├── eda_processed.py             # 9 biểu đồ phân tích & kiểm tra rò rỉ tập processed
│   ├── figures/                     # Ảnh biểu đồ & eda_summary.json đã xuất
│   └── helpers/                     # dHash 1024-bit, clustering, leakage, box_metrics, quality
├── app/                             # Ứng dụng Web Flask & REST API
│   ├── main.py                      # Entrypoint Web (python -m app.main)
│   ├── jobs.py                      # JobRunner chạy nền + theo dõi progress/ETA
│   ├── store_memory.py              # Fallback lưu log trên RAM (--store memory)
│   ├── fake.py                      # Chế độ giả lập demo nhanh (--fake)
│   ├── static/                      # CSS, JS, fonts
│   └── templates/index.html         # Giao diện người dùng
├── configs/
│   ├── classes.csv                  # 90 lớp biển báo Việt Nam (id, name, tts_text)
│   ├── shared.yaml                  # Cấu hình BẤT BIẾN dùng chung cho cả hai mô hình
│   ├── pipelines/
│   │   ├── yolo.yaml                # Kế thừa shared.yaml; cấu hình YOLO26m + Sweep
│   │   └── faster_rcnn.yaml         # Kế thừa shared.yaml; cấu hình Faster R-CNN + Sweep
│   └── runtime/
│       ├── local.yaml               # Thiết bị, số GPU, workers cho máy cá nhân
│       └── kaggle.yaml              # Cấu hình 2× T4 GPU & đường dẫn /kaggle/...
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_CONTRACT.md
│   └── DATA_ANALYSIS_AND_PREPROCESSING.md
├── models/
│   ├── yolo/                        # Trọng số YOLO tốt nhất (best.pt)
│   └── faster_rcnn/                 # Trọng số Faster R-CNN tốt nhất (best.pth)
├── notebooks/
│   ├── 00_kaggle_train.ipynb        # Pipeline tự động chạy toàn bộ trên Kaggle
│   ├── 01_data_analysis.ipynb       # EDA tương tác
│   ├── 02_preprocess_offline.ipynb  # Tiền xử lý offline & lưu Kaggle dataset cache
│   ├── 03_train_faster_rcnn.ipynb   # Train / Sweep / Eval Faster R-CNN
│   ├── 04_train_yolo.ipynb          # Train / Sweep / Eval YOLO26m
│   └── 05_demo_pipeline.ipynb       # Demo end-to-end & đối chiếu 2 mô hình
├── tests/                           # Bộ kiểm thử tự động (pytest)
├── tools/
│   ├── common.py                    # Parse tham số CLI dùng chung
│   ├── run_preprocess.py            # CLI tiền xử lý offline
│   ├── run_train.py                 # CLI huấn luyện
│   ├── run_sweep.py                 # CLI chạy W&B Sweep
│   ├── run_eval.py                  # CLI đánh giá tập Test
│   ├── run_pipeline.py              # CLI xử lý ảnh/video end-to-end
│   ├── pack_outputs.py              # Đóng gói thư mục run thành run.zip
│   └── kaggle_push.py               # Đẩy notebook, xem status, tải weights từ Kaggle
├── ts_alert/
│   ├── pipeline.py                  # Luồng xử lý end-to-end (process_media)
│   ├── types.py                     # Kiểu chuẩn Detection, SpeechEvent
│   ├── config/resolve.py            # Gộp cấu hình nhiều tầng
│   ├── data/
│   │   ├── preprocess_offline.py    # Tiền xử lý offline 6 bước
│   │   ├── preprocess_online.py     # Augment online & partial occlusion trên từng box
│   │   └── helpers/                 # Crop an toàn lớp thiểu số & split theo cụm
│   ├── db/                          # SQLite schema & Repository (ResultStore)
│   ├── eval/                        # compute_metrics, confusion matrix, evaluate_detector, FPS
│   ├── pipelines/                   # YoloPipeline/Detector, FasterRCNNPipeline/Detector, tracking
│   ├── postprocess/                 # position, draw_layer, dedup, tts, merge_av
│   ├── runtime/                     # batch (per_rank_batch), run_dir, pack, video_io
│   └── utils/                       # io, seed
├── .env.example                     # Mẫu biến môi trường (Kaggle CLI, W&B)
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Dữ liệu & Tiền xử lý

- **Nguồn dữ liệu**: Bộ dữ liệu biển báo giao thông Việt Nam gồm **90 lớp** theo chuẩn QCVN 41:2019/BGTVT trong [`configs/classes.csv`](configs/classes.csv).
- **Tiền xử lý Offline 6 bước** ([`ts_alert/data/preprocess_offline.py`](ts_alert/data/preprocess_offline.py), chi tiết tại [`docs/DATA_ANALYSIS_AND_PREPROCESSING.md`](docs/DATA_ANALYSIS_AND_PREPROCESSING.md)):
  1. Lọc bỏ bounding box quá nhỏ (`< 8px` trên canvas `640×640`).
  2. Gom cụm bối cảnh ảnh chuỗi video bằng **dHash 1024-bit** (`Hamming <= 50`).
  3. Chia cụm `Train / Val / Test` bảo đảm đầy đủ các lớp có mặt trong Train và cân bằng ảnh mờ/tối.
  4. Tăng cường có kiểm soát cho các lớp thiểu số trong Train (tối đa `10` ảnh sinh mới/lớp, từ chối ảnh nếu dHash trùng cụm Val/Test).
  5. Chuẩn hóa **letterbox `640×640`** (pad màu đen `pad_value: 0`) và cập nhật tọa độ nhãn YOLO.
  6. Đối soát rò rỉ sau resize & dồn tập triệt để -> **0.00% Data Leakage**.

| Tập dữ liệu | Số ảnh | Số bounding box | Lớp xuất hiện | Ghi chú |
|---|---:|---:|:---:|---|
| **Gốc (`data/raw`)** | 10.231 | 20.114 | 90 / 90 | Đầy đủ 90 lớp biển báo |
| **Processed Cache (`data/processed`)** | Đã tiền xử lý | Đã lọc tiny box | Bảo toàn trong Train | 0.00% rò rỉ bối cảnh video (dHash <= 50) |

- **Tiền xử lý Online** ([`ts_alert/data/preprocess_online.py`](ts_alert/data/preprocess_online.py)): Mặc định **TẮT** (`online_preprocess.enabled: false`) để so sánh công bằng hai mô hình trên tập dữ liệu offline chuẩn. Khi bật (`--online` hoặc `ONLINE = True`):
  - **Cả 2 pipeline**: hỗ trợ biến đổi màu HSV (`hsv_h`, `hsv_s`, `hsv_v`), lật ngang (`hflip`, mặc định `false` do biển báo nhạy hướng) và **che khuất một phần góc/cạnh box** (`occlusion_p`, `occlusion_ratio`).
  - **Riêng YOLO**: hỗ trợ thêm `mosaic`, `mixup`, `scale`, `translate`.

---

## 4. Cài đặt & Thiết lập môi trường

### 4.1. Cài đặt trên máy cá nhân (Local)

Yêu cầu **Python 3.10 – 3.13** (khuyên dùng **3.11**).

**Windows (PowerShell):**
```powershell
git clone https://github.com/ThanhBinh2006/VN-TSD-Voice-Alert.git
cd VN-TSD-Voice-Alert
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS (Bash):**
```bash
git clone https://github.com/ThanhBinh2006/VN-TSD-Voice-Alert.git
cd VN-TSD-Voice-Alert
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Ngoài ra có thể cài dạng package cho môi trường cloud (không ghi đè PyTorch sẵn có trên Kaggle/Colab):
> `pip install -e ".[detect,tracking]"`

### 4.2. Tạo file biến môi trường `.env` (tùy chọn, dùng cho W&B / Kaggle CLI)

Sao chép `.env.example` thành `.env` (file `.env` đã được `.gitignore`, **không** commit lên Git):

```bash
cp .env.example .env    # PowerShell: Copy-Item .env.example .env
```

Điền các khóa cần dùng trong `.env`:
- `WANDB_API_KEY`: API key Weights & Biases (dùng khi `require_wandb: true` hoặc `tools.run_sweep`).
- `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_DATASET`: thông tin tài khoản & dataset khi dùng `tools.kaggle_push`.
- `DATA_ROOT`: thư mục dataset mặc định khi chạy `analysis.eda`.

---

## 5. Hướng dẫn các cách chạy

### 5.1. Chạy Web App (`localhost`)

Đặt trọng số đã train vào `models/yolo/best.pt` và/hoặc `models/faster_rcnn/best.pth` (xem mục [5.7](#57-huấn-luyện--tải-weights-từ-kaggle-2x-gpu-t4)), sau đó khởi chạy Flask server:

```bash
# Chạy chế độ đầy đủ (model thật + tự động dùng SQLite nếu sẵn sàng)
python -m app.main

# Chạy chế độ demo nhanh không cần file weights thật
python -m app.main --fake

# Tuỳ chỉnh host, port hoặc nơi lưu log (auto | memory | sqlite)
python -m app.main --host 0.0.0.0 --port 5000 --store sqlite
```

Mở trình duyệt tại: `http://127.0.0.1:5000`

<details>
<summary>Các endpoint REST API tích hợp sẵn trong <code>app/main.py</code></summary>

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái server, model và store (`sqlite` / `memory`) |
| `POST` | `/api/jobs` | Upload ảnh/video (`form-data`: `file`, `model=yolo\|faster_rcnn`) |
| `GET` | `/api/jobs` | Danh sách các job xử lý (`?limit=50&offset=0`) |
| `GET` | `/api/jobs/<id>` | Trạng thái, `% tiến độ`, `fps`, `eta_sec`, thời gian & số liệu |
| `GET` | `/api/jobs/<id>/detections` | Danh sách bounding box + nhãn + vị trí làn |
| `GET` | `/api/jobs/<id>/voice-events` | Danh sách sự kiện phát giọng nói + URL file audio |
| `GET` | `/api/jobs/<id>/voice-events/<eid>/audio` | Nghe/tải file âm thanh cảnh báo của từng sự kiện |
| `GET` | `/api/jobs/<id>/result` | Xem video/ảnh kết quả (hỗ trợ `?download=1` để tải về) |

</details>

---

### 5.2. Chạy xử lý 1 ảnh / video trực tiếp bằng CLI (không cần Web)

Xử lý end-to-end từ ảnh/video đầu vào sang ảnh/video đã vẽ bounding box, ghép giọng nói cảnh báo và lưu log DB:

```bash
# Dùng YOLO26m
python -m tools.run_pipeline --input sample.mp4 --model yolo

# Dùng Faster R-CNN và chỉ định thư mục xuất + ghi đè ngưỡng tin cậy
python -m tools.run_pipeline --input road.jpg --model faster_rcnn --out-dir outputs/demo --set inference.conf_threshold=0.35
```

---

### 5.3. Chạy Phân tích Dữ liệu (EDA) & Tiền xử lý Offline

Tạo dữ liệu chuẩn hóa `data/processed/` từ dữ liệu thô `data/raw/` (tải từ Kaggle và giải nén vào `data/raw/`):

```bash
# 1. Phân tích tập dữ liệu thô -> xuất 8 biểu đồ vào analysis/figures/raw/
python -m analysis.eda --data-root data/raw

# 2. Chạy tiền xử lý offline 6 bước -> tạo data/processed/ (images, labels, data.yaml, meta.json)
python -m tools.run_preprocess --data-root data/raw
# (hoặc chạy trực tiếp module: python -m ts_alert.data.preprocess_offline --data-root data/raw)

# 3. Kiểm tra & phân tích lại tập sau tiền xử lý -> xuất 9 biểu đồ vào analysis/figures/processed/
python -m analysis.eda --data-root data/processed
```

---

### 5.4. Huấn luyện mô hình trên Local hoặc Multi-GPU (DDP)

> **Lưu ý DDP**:
> - **YOLO26m**: Ultralytics tự quản lý đa GPU qua `num_gpus` — luôn chạy trực tiếp bằng `python -m tools.run_train` (**không** bọc qua `torchrun`).
> - **Faster R-CNN**: Hỗ trợ cả 1 GPU/CPU (`python -m ...`) và đa GPU qua `torchrun`.
> - **Batch size**: `train.batch_size` và `eval.batch_size` trong `configs/shared.yaml` luôn là **tổng trên tất cả GPU** (Faster R-CNN tự chia bằng `per_rank_batch`).

```bash
# --- YOLO26m ---
# Train cơ bản (tắt online_preprocess theo cấu hình chuẩn)
python -m tools.run_train --pipeline yolo --runtime local --no-online

# Bật online_preprocess (mosaic, mixup, HSV, partial occlusion) và đặt tên run
python -m tools.run_train --pipeline yolo --runtime local --online --run-name yolo_online_exp

# --- Faster R-CNN ResNet-50 FPN ---
# Train 1 GPU / CPU trên local
python -m tools.run_train --pipeline faster_rcnn --runtime local --no-online

# Train DDP 2 GPU (ví dụ runtime kaggle.yaml hoặc server 2 GPU)
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --no-online

# --- Ghi đè nhanh bất kỳ tham số nào qua CLI ---
python -m tools.run_train --pipeline yolo --runtime local --data-root data/processed \
    --set train.epochs=50 train.lr=0.005 train.require_wandb=true
```

Mỗi lần chạy sẽ tạo thư mục `outputs/<pipeline>/<run_name>/` chứa:
- Checkpoint tốt nhất (`weights/best.pt` hoặc `best.pth`) chọn theo `mAP50-95` và đánh dấu `* BEST` trong log.
- Biểu đồ & log chuẩn hóa: `results.csv`, `metrics_history.json`, `summary.json`, `training_curves.png`, `confusion_matrix.png`, `confusion_matrix.json`, `metrics.json` (khi `eval.auto_eval_test: true`).

---

### 5.5. Tối ưu siêu tham số với W&B Sweep (`tools.run_sweep`)

Cần cấu hình `WANDB_API_KEY` trong `.env` (hoặc Kaggle Secrets). Các khóa dùng chung trong `configs/shared.yaml` (`seed`, `data.*`, `eval.*`, `train.batch_size`, `train.epochs`, `train.early_stopping_patience`, `train.amp`, `train.primary_metric`) được tự động khóa bảo vệ khỏi sweep.

```bash
# Xem trước JSON cấu hình sweep (không kết nối W&B)
python -m tools.run_sweep --pipeline yolo --dry-run
python -m tools.run_sweep --pipeline faster_rcnn --dry-run

# Chạy W&B Sweep thực tế (mặc định method: bayes, count: 10 từ shared.yaml)
python -m tools.run_sweep --pipeline yolo --runtime local --count 10
python -m tools.run_sweep --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --count 10

# Tiếp tục vào một sweep_id có sẵn
python -m tools.run_sweep --pipeline yolo --sweep-id <entity>/<project>/<sweep_id> --count 5
```

> Lưu ý: Các tham số thực trong không gian sweep (`train.lr`, `train.weight_decay`, `train.momentum`, `train.warmup_epochs`) dùng phân phối liên tục (`log_uniform_values` / `uniform`), nên chỉ dùng `--method bayes` (mặc định) hoặc `random` (không dùng `grid`).

---

### 5.6. Đánh giá trên tập Test & Đóng gói kết quả

```bash
# Đánh giá lại checkpoint tốt nhất trên tập Test (tính mAP50-95, mAP50, mAP75, AR100, P/R/F1, FPS)
python -m tools.run_eval --pipeline yolo --runtime local --run-dir outputs/yolo/<run_id>
python -m tools.run_eval --pipeline faster_rcnn --runtime local --run-dir outputs/faster_rcnn/<run_id>

# Nén toàn bộ log, biểu đồ và checkpoint của một run thành run.zip
python -m tools.pack_outputs --run-dir outputs/yolo/<run_id>
```

---

### 5.7. Huấn luyện & Tải weights từ Kaggle (2x GPU T4)

#### Cách A: Dùng CLI đẩy tự động từ máy cá nhân (`tools.kaggle_push`)
Sau khi cấu hình `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_DATASET` (và `WANDB_API_KEY` nếu bật W&B cho kernel private) trong `.env`:

```bash
# 1. Đẩy notebook lên Kaggle và kích hoạt chạy GPU (pipeline: eda | preprocess | faster_rcnn | yolo | demo)
python -m tools.kaggle_push --pipeline preprocess --dataset <user>/vietnamese-traffic-signs
python -m tools.kaggle_push --pipeline yolo --dataset <user>/tsd-voice-alert-preprocessed-data
python -m tools.kaggle_push --pipeline faster_rcnn --dataset <user>/tsd-voice-alert-preprocessed-data

# 2. Kiểm tra trạng thái chạy trên Kaggle
python -m tools.kaggle_push --status --pipeline yolo

# 3. Tải output về máy (tự động copy best.pt / best.pth vào models/<pipeline>/)
python -m tools.kaggle_push --pull --pipeline yolo
python -m tools.kaggle_push --pull --pipeline faster_rcnn
```

#### Cách B: Chạy trực tiếp trên giao diện Notebook Kaggle
Cài đặt Notebook: **Accelerator = GPU T4 ×2**, **Internet = On**, dùng theo thứ tự:
1. [`notebooks/01_data_analysis.ipynb`](notebooks/01_data_analysis.ipynb): phân tích EDA.
2. [`notebooks/02_preprocess_offline.ipynb`](notebooks/02_preprocess_offline.ipynb) (Add Data `maitam/vietnamese-traffic-signs`): tạo `data/processed/`, lưu Output thành Kaggle Dataset dùng chung.
3. [`notebooks/03_train_faster_rcnn.ipynb`](notebooks/03_train_faster_rcnn.ipynb) hoặc [`notebooks/04_train_yolo.ipynb`](notebooks/04_train_yolo.ipynb): Add Input dataset ở bước 2, chỉnh biến tại mục 2 (bao gồm `ONLINE` hoặc `SWEEP_ENABLED`) rồi chạy hết notebook.
4. Hoặc chạy nhanh cả quy trình trong [`notebooks/00_kaggle_train.ipynb`](notebooks/00_kaggle_train.ipynb) và thử nghiệm end-to-end bằng [`notebooks/05_demo_pipeline.ipynb`](notebooks/05_demo_pipeline.ipynb).
5. Tải `run.zip` từ tab Output, đặt `best.pt` vào `models/yolo/` và `best.pth` vào `models/faster_rcnn/` để dùng cho Web App / CLI.

---

### 5.8. Chạy bộ kiểm thử tự động (`pytest`)

```bash
# Chạy toàn bộ test suite (unit tests, smoke tests, contract & notebook syntax)
pytest
# hoặc ngắn gọn hơn:
pytest tests/ -q

# Chạy riêng một nhóm test (ví dụ tiền xử lý hoặc training smoke)
pytest tests/test_preprocess.py tests/test_training_smoke.py -v
```

---

## 6. Hệ thống cấu hình phân tầng

[`ts_alert/config/resolve.py`](ts_alert/config/resolve.py) hợp nhất cấu hình từ thấp đến cao theo đúng 6 tầng:

```text
1. configs/shared.yaml                    (Cấu hình BẤT BIẾN dùng chung)
2. configs/pipelines/<pipeline>.yaml      (Khai báo `inherits: configs/shared.yaml`)
3. configs/runtime/<local|kaggle>.yaml    (Thiết bị, num_gpus, workers, đường dẫn Kaggle)
4. File --overrides <path.yaml>           (Do notebook 03/04 hoặc sweep trial sinh ra)
5. Tham số --set key=value                (Ghi đè nhanh trên CLI)
6. Cờ CLI riêng lẻ                        (--data-root, --online/--no-online, --run-name)
```

Các khóa trong [`configs/shared.yaml`](configs/shared.yaml) (`seed`, `data.*`, `train.batch_size`, `train.epochs`, `train.early_stopping_patience`, `train.amp`, `train.primary_metric`, `eval.*`) là **chuẩn bất biến chung** giữa hai pipeline — không ghi đè trong `configs/pipelines/*.yaml` để bảo đảm công bằng thực nghiệm.

---

## 7. Phiên bản công nghệ

| Thành phần | Phiên bản (Local) | Ghi chú |
|---|---:|---|
| Python | 3.11.x (3.10 – 3.13) | Khai báo tại [`pyproject.toml`](pyproject.toml) |
| PyTorch / torchvision | 2.9.1 / 0.24.1 | Trên Kaggle dùng bản cài sẵn của image |
| Ultralytics | 8.4.175 | Hỗ trợ YOLO26m |
| torchmetrics / pycocotools | 1.9.0 / 2.0.11 | Tính mAP/AR chuẩn COCO |
| OpenCV (headless) | 4.12.0.88 | Xử lý ảnh & khung hình |
| NumPy / pandas / matplotlib | 2.2.6 / 2.3.3 / 3.10.9 | Phân tích & trực quan hóa |
| Flask | 3.1.3 | Web Server & REST API |
| gTTS / pyttsx3 | 2.5.4 / 2.99 | Tổng hợp giọng nói online / offline |
| imageio-ffmpeg | 0.6.0 | Binary `ffmpeg` ghép audio/video |
| PyYAML / pytest | 6.0.3 / 9.1.1 | Cấu hình & kiểm thử |
| Weights & Biases (`wandb`) | 0.26.1 | Tracking & Sweep (tùy chọn) |

Danh sách đầy đủ tại [`requirements.txt`](requirements.txt).

---

## 8. Kết quả thực nghiệm

> Bảng dưới đây sẽ được cập nhật đầy đủ sau đợt huấn luyện chính thức 100 epochs trên **Kaggle 2× GPU T4** (`configs/shared.yaml`).

| Chỉ số (tập Test — 371 ảnh, 829 box) | YOLO26m | Faster R-CNN R50-FPN |
|---| :---: | :---: |
| `mAP50-95` *(metric chính)* | *Chờ chạy Kaggle* | *Chờ chạy Kaggle* |
| `mAP50` | *Chờ chạy Kaggle* | *Chờ chạy Kaggle* |
| `mAP75` | *Chờ chạy Kaggle* | *Chờ chạy Kaggle* |
| `AR100` | *Chờ chạy Kaggle* | *Chờ chạy Kaggle* |
| `Precision / Recall / F1` | *Chờ chạy Kaggle* | *Chờ chạy Kaggle* |
| Tốc độ suy luận (FPS — GPU / CPU) | *Chờ đo đạc* | *Chờ đo đạc* |

---

## 9. Thành viên, Phân công & Quy trình Git

| Thành viên | Vai trò | Nhánh Git | Phạm vi phụ trách |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | Data & YOLO26m | `data-yolo` | `analysis/`, `ts_alert/data/`, `ts_alert/pipelines/yolo/`, `configs/pipelines/yolo.yaml`, `configs/shared.yaml` |
| **Thành viên 2** | Faster R-CNN & Đánh giá | `faster-rcnn` | `ts_alert/pipelines/faster_rcnn/`, `ts_alert/eval/`, `configs/pipelines/faster_rcnn.yaml` |
| **Thành viên 3** | Backend & Web UI | `backend-ui` | `app/`, `ts_alert/runtime/video_io.py` |
| **Thành viên 4** | Layer & Voice | `layer-voice` | `ts_alert/postprocess/` |
| **Thành viên 5** | DB, Docs & Tích hợp | `db-docs` | `ts_alert/db/`, `docs/`, báo cáo, slide |

Quy ước chi tiết xem tại [`CONTRIBUTING.md`](CONTRIBUTING.md). Tóm tắt thao tác chuẩn hàng ngày:

```bash
# 1. Đồng bộ nhánh chính trước khi làm việc
git checkout main && git pull origin main
git checkout <nhanh-cua-ban> && git merge main

# 2. Chỉnh sửa code, kiểm thử trước khi commit
pytest tests/ -q

# 3. Commit theo Conventional Commits (feat / fix / docs / refactor / test)
git status
git add ts_alert/ configs/   # tuyệt đối KHÔNG add data/, outputs/, models/*.pt, models/*.pth, *.db, .env
git commit -m "feat: mo ta ngan gon thay doi"
git push -u origin <nhanh-cua-ban>
```

> **Cảnh báo**: Không push trực tiếp lên `main`, không sửa phá vỡ [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md), và **không** commit dữ liệu/checkpoint nặng hay khóa `.env`. Khi chạy notebook trên Kaggle với repo private, lưu GitHub token vào **Kaggle Secrets (`GH_TOKEN`)** thay vì ghi trực tiếp vào code.

---

## 10. Tài liệu tham khảo

1. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
2. Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.
3. Ultralytics YOLO Documentation: https://docs.ultralytics.com
4. Bộ dữ liệu biển báo giao thông Việt Nam (Kaggle): https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs
