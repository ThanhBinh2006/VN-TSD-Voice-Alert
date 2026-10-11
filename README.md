# TSD-Voice-Alert — Nhận diện và Cảnh báo Biển báo Giao thông bằng Giọng nói

Dự án môn học Thị giác máy tính: hệ thống hỗ trợ lái xe thông minh (ADAS) mô phỏng, phát hiện biển báo giao thông Việt Nam từ ảnh/video và **cảnh báo bằng giọng nói** đúng thời điểm biển xuất hiện. Hai kiến trúc được huấn luyện và so sánh trên **cùng một tập dữ liệu**:

- **YOLO26m** (One-stage detector, tối ưu tốc độ) — qua Ultralytics
- **Faster R-CNN ResNet-50 FPN** (Two-stage detector, tối ưu độ chính xác) — qua torchvision

Huấn luyện chính thức trên **Kaggle (2× GPU T4)**; ứng dụng web chạy trên **localhost**: upload ảnh/video → tách khung hình → AI trích xuất bounding box + nhãn → xác định vị trí (trái / phải / trước mặt) → chống cảnh báo lặp → TTS → ghép thành video kết quả có giọng nói → lưu log.

---

## 1. Thành viên & phân công

| Thành viên | Vai trò | Nhánh Git | Phụ trách chính |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | Data & YOLO | `data-yolo` | `analysis/`, `ts_alert/data/`, `ts_alert/pipelines/yolo/`, `configs/pipelines/yolo.yaml`, `configs/shared.yaml` |
| **Thành viên 2** | Faster R-CNN & Đánh giá | `faster-rcnn` | `ts_alert/pipelines/faster_rcnn/`, `ts_alert/eval/`, `configs/pipelines/faster_rcnn.yaml` |
| **Thành viên 3** | Backend & Web UI | `backend-ui` | `app/`, `ts_alert/runtime/video_io.py` |
| **Thành viên 4** | Layer & Voice | `layer-voice` | `ts_alert/postprocess/` |
| **Thành viên 5** | Quản lý & Tích hợp | `db-docs` | `ts_alert/db/`, `docs/`, báo cáo, slide |

- **Thành viên 1** — Chuẩn hóa, chia Train/Val/Test dùng chung; fine-tune YOLO; viết `YoloDetector.predict`; tạo template git.
- **Thành viên 2** — Fine-tune Faster R-CNN trên đúng split của Thành viên 1 (DDP 2 GPU); viết `FasterRCNNDetector.predict`; chạy tập Test, tính mAP/FPS và so sánh; tạo template git.
- **Thành viên 3** — Web (upload + chọn model + xem/tải kết quả); tách video thành khung hình (`video_io`).
- **Thành viên 4** — Vị trí trái/phải/trước mặt; vẽ bbox + nhãn; TTS; chống lặp; ghép âm thanh vào video; hỗ trợ DB.
- **Thành viên 5** — Thiết kế DB SQLite; template báo cáo và slide; tổng hợp nhận xét; tích hợp.

## 2. Phiên bản công nghệ

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11.x (hỗ trợ 3.10 – 3.13) |
| PyTorch / torchvision | 2.9.1 / 0.24.1 (local) — trên Kaggle dùng bản cài sẵn, ghi lại bằng `pip list` |
| Ultralytics (YOLO26m) | 8.4.175 |
| torchmetrics / pycocotools | 1.9.0 / 2.0.11 |
| OpenCV (headless) | 4.12.0.88 |
| NumPy / pandas / matplotlib | 2.2.6 / 2.3.3 / 3.10.9 |
| Flask (gợi ý, Thành viên 3 có thể đổi) | 3.1.3 |
| gTTS / pyttsx3 | 2.5.4 / 2.99 |
| imageio-ffmpeg (ffmpeg) | 0.6.0 |
| PyYAML / pytest | 6.0.3 / 9.1.1 |
| wandb (tuỳ chọn, theo dõi thí nghiệm) | 0.30.0 |

Danh sách ghim đầy đủ: [`requirements.txt`](requirements.txt).

## 3. Phần đã có sẵn (cốt lõi) và phần cần hiện thực

**Đã code sẵn — mọi người tuân thủ, không cần sửa:**

| Phần | File |
|---|---|
| Luồng xử lý tổng thể ảnh/video → video có giọng nói | `ts_alert/pipeline.py` |
| Gộp cấu hình nhiều tầng (shared → pipeline → runtime → CLI) | `ts_alert/config/resolve.py`, `configs/` |
| Khuôn mẫu hai mô hình + đăng ký | `ts_alert/pipelines/base.py`, `__init__.py` |
| Luồng tiền xử lý / đánh giá / đo FPS | `ts_alert/data/preprocess_offline.py::run_preprocess`, `ts_alert/eval/evaluate.py`, `benchmark.py` |
| Chia batch cho DDP, đóng gói kết quả Kaggle | `ts_alert/runtime/batch.py`, `pack.py` |
| Hợp đồng lưu log (giao diện `ResultStore`) | `ts_alert/db/interface.py` |
| CLI | `tools/` |
| Test của phần đã code | `tests/` |
| Notebook có sẵn luồng chạy (chỉ chừa TODO nhận xét) | `notebooks/01`–`05` |
| CLI nhận `--overrides`, `--data-root`, `--online/--no-online`, `--run-name` | `tools/common.py`, `run_train.py`, `run_eval.py` |

**Cần hiện thực (đang là vỏ `raise NotImplementedError("TODO: Thành viên N")`):**

| Thành viên | Cần điền |
|---|---|
| 1 | `data/preprocess_offline.py`: `list_raw_samples`, `split_samples`, `letterbox_sample`, `write_data_yaml` · `data/preprocess_online.py` · `pipelines/yolo/`: `load_model`, `YoloPipeline.fit`, `YoloDetector.predict` · `analysis/eda.py` · điền `configs/classes.csv`, `data.num_classes` |
| 2 | `pipelines/faster_rcnn/`: `build_model`, `load_checkpoint`, `FasterRCNNPipeline.fit`, `FasterRCNNDetector.predict` · `eval/metrics.py::compute_metrics` · `eval/evaluate.py::load_split_samples` |
| 3 | `app/main.py` (toàn bộ web, tự do thiết kế) · `runtime/video_io.py`: `get_video_info`, `iter_frames`, `read_image`, `write_image`, `VideoWriter` |
| 4 | `postprocess/`: `get_position`, `draw_detections`, `Deduplicator.reset/update`, `build_speech_text`, `synthesize`, `merge_audio_video` |
| 5 | `db/schema.sql` + `db/repository.py` (tự thiết kế bảng, miễn thoả `ResultStore`) · báo cáo, slide |

Mỗi hàm vỏ có docstring ghi đầu vào, đầu ra và gợi ý cách làm. **Được tuỳ biến thoải mái** cấu trúc DB, giao diện web và bên trong mọi hàm, miễn tuân thủ [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Hiện thực xong thì xoá `raise NotImplementedError` và bỏ `@pytest.mark.skip` ở test tương ứng.

## 4. Cấu trúc thư mục

```text
TSD-Voice-Alert/
├── analysis/eda.py                  # [TV1]
├── app/main.py                      # [TV3] web (tự thiết kế)
├── configs/
│   ├── classes.csv                  # id, name, tts_text
│   ├── shared.yaml                  # cấu hình BẤT BIẾN dùng chung (seed, data, batch, metric, postprocess...)
│   ├── pipelines/{yolo,faster_rcnn}.yaml   # inherits shared.yaml; model, train, online_preprocess
│   └── runtime/{local,kaggle}.yaml  # thiết bị, số GPU, đường dẫn
├── data/                            # KHÔNG commit (raw/, processed/)
├── docs/{ARCHITECTURE,DATA_CONTRACT}.md
├── models/{yolo,faster_rcnn}/       # trọng số best.pt / best.pth (KHÔNG commit)
├── notebooks/                       # 01 EDA · 02 preprocess · 03 train Faster R-CNN · 04 train YOLO · 05 demo
├── outputs/                         # KHÔNG commit
├── tests/
├── tools/{common,run_preprocess,run_train,run_eval,run_pipeline,pack_outputs}.py
├── ts_alert/
│   ├── pipeline.py                  # *** LUỒNG TỔNG THỂ ***
│   ├── types.py                     # Detection, SpeechEvent
│   ├── config/resolve.py
│   ├── data/{preprocess_offline,preprocess_online}.py        # [TV1]
│   ├── eval/{metrics,benchmark,evaluate}.py                  # [TV2]
│   ├── pipelines/{base.py, yolo/, faster_rcnn/}              # [TV1] [TV2]
│   ├── postprocess/{position,draw_layer,dedup,tts,merge_av}.py   # [TV4]
│   ├── db/{interface.py, repository.py, schema.sql}          # [TV5]
│   ├── runtime/{run_dir,batch,pack,video_io}.py
│   └── utils/{io,seed}.py
├── CONTRIBUTING.md  pyproject.toml  requirements.txt  README.md
```

## 5. Cấu hình (`configs/`)

Cấu hình gộp theo thứ tự: `shared.yaml` → `pipelines/<tên>.yaml` (khai báo `inherits`) → `runtime/<môi trường>.yaml` → file `--overrides` → tham số `--set` → các cờ `--data-root`, `--online`, `--run-name`.

- `shared.yaml`: **bất biến** cho cả hai mô hình: `seed`, `data.*`, `train.batch_size`, `train.early_stopping_patience`, `train.amp`, `train.primary_metric`, `eval.*`. Không ghi đè các khoá này trong `pipelines/*.yaml`. Notebook 03/04 **đọc** các giá trị này từ `shared.yaml` nên hai notebook không bị lệch nhau; muốn đổi thì sửa `shared.yaml` và báo cả nhóm.
- `train.batch_size` và `eval.batch_size` là **tổng** trên mọi GPU. Faster R-CNN dùng `per_rank_batch` để chia cho từng GPU; YOLO truyền nguyên tổng để Ultralytics tự chia.
- `pipelines/*.yaml`: `model`, `train` (epochs, lr...), `online_preprocess` (chỉ chạy khi `enabled: true`).
- `data.root`: thư mục processed dùng để train/eval; `null` thì lấy `paths.processed_dir`. Trên Kaggle thường là dataset đã gắn bằng Add Input.
- `eval.auto_eval_test: true`: `fit()` tự đánh giá tập Test sau khi train xong và ghi `metrics.json`.
- `project.*`: tuỳ chọn, dùng khi bật `train.require_wandb` để log lên Weights & Biases (điền `wandb_entity`, và `wandb_base_url` nếu dùng server riêng).
- Faster R-CNN: box predictor = `data.num_classes + 1` (thêm background).
- Ghi đè nhanh: `--set train.epochs=5`.

## 6. Dataset

- Nguồn: [Vietnamese Traffic Signs – Kaggle](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs)
- Số lớp: TODO · Số ảnh: TODO
- Tiền xử lý offline dùng chung: letterbox 640×640 (pad 114), chia Train/Val/Test: TODO % / TODO % / TODO %

## 7. Cài đặt và chạy local

```bash
git clone TODO_URL && cd TSD-Voice-Alert
python3.11 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest                                                   # kiểm tra phần đã code
```

Có thể cài như một package: `pip install -e ".[detect]"` (kèm `.[tracking]` nếu dùng W&B). Extras `detect` **không** gồm torch/torchvision vì Kaggle đã cài sẵn; ở máy local cài torch qua `requirements.txt`.

## 8. Huấn luyện trên Kaggle

Mỗi notebook có ô **Bootstrap**: trên Kaggle tự clone repo, lấy đúng nhánh và cài thư viện (điền `REPO_URL` và `REPO_BRANCH` ở ô đó; ở máy local nó dùng luôn repo đang mở, không động vào git của bạn).

Cài đặt notebook: Accelerator **GPU T4 ×2**, Internet **On**. Quy trình:

1. **`notebooks/02_preprocess_offline.ipynb`** (Add Input `maitam/vietnamese-traffic-signs`): sinh `data/processed`. Lưu Output thành một Kaggle Dataset (ví dụ `tsd-processed`) để dùng chung.
2. **`notebooks/03_train_faster_rcnn.ipynb`** và **`notebooks/04_train_yolo.ipynb`**: Add Input dataset processed ở bước 1, điền đường dẫn vào `CANDIDATES`, chỉnh biến ở mục 2 rồi chạy. Notebook ghi file overrides rồi gọi `torchrun --nproc_per_node=2 -m tools.run_train ...`, sau đó đọc `metrics.json` (tập Test), xem dự đoán và nén `run.zip`.
3. Tải `run.zip` từ tab **Output**, đặt `best.pt` vào `models/yolo/` và `best.pth` vào `models/faster_rcnn/` để web app dùng.

Tương đương dòng lệnh:

```bash
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml \
    --run-name faster_rcnn_s42_ep20_bs8 --overrides outputs/_notebook/overrides_faster_rcnn.yaml \
    --data-root <PROCESSED_ROOT> --no-online

python -m tools.run_eval --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml \
    --run-dir /kaggle/working/outputs/faster_rcnn/<run_id> --data-root <PROCESSED_ROOT>

python -m tools.pack_outputs --run-dir /kaggle/working/outputs/faster_rcnn/<run_id>
```

> TODO (Thành viên 1): xác nhận Ultralytics chạy đa GPU đúng khi bọc trong `torchrun`; nếu không, đổi notebook 04 sang chạy `python` thường với `device=[0, 1]`.

Theo dõi thí nghiệm bằng W&B là **tuỳ chọn** (`REQUIRE_WANDB` trong notebook): lưu `WANDB_API_KEY` vào Kaggle Secrets và điền `project.*` trong `configs/shared.yaml`.

## 9. Các lệnh khác

```bash
python -m analysis.eda --data-root <DATA_PATH>                  # EDA
python -m tools.run_pipeline --input sample.mp4 --model yolo    # xử lý 1 video end-to-end, không cần web
python -m app.main                                              # chạy web (sau khi Thành viên 3 hiện thực)
```

## 10. Luồng xử lý

```text
iter_frames ─► detector.predict ─► add_positions ─► repo.add_detections
                                                          │
                                  Deduplicator.update ◄───┘
                                         │ SpeechEvent
                ┌────────────────────────┴──────────────┐
                ▼                                       ▼
        draw_detections                    build_speech_text ─► synthesize ─► repo.add_voice_event
                │                                                   │
          VideoWriter.write ─────────► merge_audio_video ◄──────────┘ ─► repo.finish_job
```

Đầu ra chung của hai mô hình:

```python
# predict(frame) -> list[dict]
[{"bbox": [x1, y1, x2, y2],   # pixel, theo ảnh gốc
  "label": "Cấm rẽ trái",
  "class_id": 12,
  "confidence": 0.93}]
```

## 11. Kết quả thực nghiệm

| Chỉ số (tập Test) | YOLO26m | Faster R-CNN |
|---|---|---|
| mAP50-95 | TODO | TODO |
| mAP50 | TODO | TODO |
| mAP75 | TODO | TODO |
| AR100 | TODO | TODO |
| FPS (GPU / CPU) | TODO | TODO |
| Kích thước model (MB) | TODO | TODO |

Nhận xét tốc độ vs. độ chính xác, lớp dễ nhầm, biển nhỏ/xa, ánh sáng yếu: TODO.

## 12. Quy ước làm việc nhóm & Hướng dẫn thao tác Git

Chi tiết quy định xem tại [`CONTRIBUTING.md`](CONTRIBUTING.md). Dưới đây là hướng dẫn chi tiết các thao tác Git chuẩn cho toàn bộ thành viên trong nhóm:

### 12.1. Khởi tạo & Liên kết Remote Repository
> *Dành cho Trưởng nhóm / Người thiết lập repository ban đầu lên GitHub/GitLab:*

```bash
# 1. Kiểm tra trạng thái và tạo commit khởi tạo
git status
git commit -m "feat: khoi tao template du an TSD-Voice-Alert"

# 2. Đổi tên nhánh chính thành main
git branch -M main

# 3. Thêm địa chỉ remote repository (thay URL bằng repo thực tế của nhóm)
git remote add origin https://github.com/<to-chuc-hoac-user>/TSD-Voice-Alert.git

# 4. Đẩy mã nguồn ban đầu lên remote
git push -u origin main
```

### 12.2. Quy trình làm việc hàng ngày của từng thành viên

Mỗi thành viên phụ trách và làm việc độc lập trên **nhánh riêng** (đã phân công tại [Mục 1](#1-thành-viên--phân-công)):
- Thành viên 1: `data-yolo`
- Thành viên 2: `faster-rcnn`
- Thành viên 3: `backend-ui`
- Thành viên 4: `layer-voice`
- Thành viên 5: `db-docs`

#### Bước 1: Clone repo và tạo nhánh tính năng
```bash
# Clone dự án về máy cá nhân
git clone https://github.com/<to-chuc-hoac-user>/TSD-Voice-Alert.git
cd TSD-Voice-Alert

# Tạo và chuyển sang nhánh được phân công (ví dụ: Thành viên 1)
git checkout -b data-yolo
```

#### Bước 2: Đồng bộ mã nguồn mới nhất từ `main` trước khi code
Để tránh xung đột code, hãy luôn cập nhật nhánh của bạn với `main` trước mỗi buổi làm việc:
```bash
# Chuyển về main và kéo code mới nhất
git checkout main
git pull origin main

# Chuyển lại về nhánh cá nhân và gộp cập nhật từ main
git checkout data-yolo
git merge main
```

#### Bước 3: Lưu trữ và Commit thay đổi
Quy ước commit message theo chuẩn Conventional Commits:
- `feat: ...`: Tính năng mới (ví dụ: `feat: hoan thien pipeline yolo predict`)
- `fix: ...`: Sửa lỗi (ví dụ: `fix: xu ly ngoai le bounding box ngoai anh`)
- `docs: ...`: Cập nhật tài liệu (ví dụ: `docs: bo sung huong dan git vao readme`)
- `test: ...`: Viết thêm / sửa unit test
- `refactor: ...`: Tối ưu cấu trúc mã nguồn không làm đổi logic

```bash
# 1. Kiểm tra các file đã chỉnh sửa
git status

# 2. Thêm file vào staging (CHỈ thêm file code/config, KHÔNG add thư mục data/ hay model nặng)
git add ts_alert/ configs/
# hoặc thêm file cụ thể:
git add ts_alert/postprocess/position.py

# 3. Commit với thông điệp rõ ràng
git commit -m "feat: cai dat thuat toan xac dinh vi tri bien bao"
```

#### Bước 4: Chạy kiểm thử và Đẩy nhánh lên Remote
```bash
# Chạy bộ test để đảm bảo không làm gãy luồng chung
pytest

# Đẩy nhánh lên GitHub (lần đầu tiên đẩy nhánh dùng thêm cờ -u)
git push -u origin data-yolo

# Những lần đẩy sau trên nhánh này chỉ cần:
git push
```

### 12.3. Quy trình Review & Gộp nhánh (Pull Request)
1. **Tạo Pull Request (PR):** Sau khi hoàn thành một phần việc, lên GitHub/GitLab tạo PR từ nhánh cá nhân (ví dụ `data-yolo`) vào nhánh `main`.
2. **Review chéo:** Gán ít nhất **1 thành viên khác** trong nhóm review code và kiểm tra logic.
3. **Merge vào main:** Sau khi được duyệt (Approve) và toàn bộ test tự động vượt qua:
   - Gộp nhánh vào `main` (khuyến khích chọn *Squash and merge* hoặc *Rebase and merge* để lịch sử git gọn gàng).
4. **Cập nhật máy cá nhân sau khi merge:**
   ```bash
   git checkout main
   git pull origin main
   git checkout <nhanh-cua-ban>
   git merge main
   ```

### 12.4. Xử lý xung đột mã nguồn (Merge Conflict)
Nếu gặp thông báo conflict khi `git merge main` hoặc khi tạo PR:
1. Mở các file bị conflict, Git đánh dấu vùng xung đột như sau:
   ```text
   <<<<<<< HEAD (code trên nhánh hiện tại của bạn)
   code_cua_ban()
   =======
   code_moi_tu_main()
   >>>>>>> main
   ```
2. Thảo luận với người viết đoạn code đó nếu cần, giữ lại phần code đúng nhất và xóa các dòng đánh dấu (`<<<<<<<`, `=======`, `>>>>>>>`).
3. Chạy lại test để chắc chắn hệ thống ổn định:
   ```bash
   pytest
   git add <file-da-fix-conflict>
   git commit -m "fix: giai quyet xung dot voi nhanh main"
   git push
   ```

### 12.5. Các lưu ý & Điều cấm kỵ (Best Practices)
- ❌ **TUYỆT ĐỐI KHÔNG commit file nặng:** `data/`, `outputs/`, `models/*.pt`, `models/*.pth`, file CSDL `*.db`, file zip. Dự án đã có `.gitignore`, nhưng hãy luôn chạy `git status` trước khi commit.
- ❌ **KHÔNG push trực tiếp lên `main`** và **KHÔNG dùng `git push --force`** trên các nhánh dùng chung.
- ❌ **KHÔNG tự ý đổi chữ ký hàm (signature)** hoặc giao diện đã cam kết trong [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) nếu chưa họp bàn và thống nhất với cả nhóm.
- ✔️ Luôn chạy `pytest` trước khi tạo PR hoặc push code lên.

### 12.6. Đồng bộ mã nguồn trên Kaggle
Các notebook `notebooks/02`–`05` đều có sẵn ô **Bootstrap** tự động clone và lấy đúng nhánh:
- **Repo public**: Điền URL repo vào biến `REPO_URL` và nhánh vào `REPO_BRANCH` trong ô Bootstrap:
  ```python
  REPO_URL = "https://github.com/<tai-khoan>/TSD-Voice-Alert.git"
  REPO_BRANCH = "main"   # hoặc tên nhánh của bạn nếu đang thử nghiệm (vd: data-yolo)
  ```
- **Repo private**: Tạo Personal Access Token trên GitHub (quyền `repo`), lưu vào Kaggle **Add-ons → Secrets** (tên `GH_TOKEN`), rồi đặt trong ô Bootstrap:
  ```python
  from kaggle_secrets import UserSecretsClient
  token = UserSecretsClient().get_secret("GH_TOKEN")
  REPO_URL = f"https://{token}@github.com/<tai-khoan>/TSD-Voice-Alert.git"
  REPO_BRANCH = "main"
  ```
- Không để token trong code hay commit lên repo. Kết quả huấn luyện lấy về bằng `run.zip` (mục 8), không push từ Kaggle.

## 13. Hạn chế & hướng phát triển

TODO (biển nhỏ/xa, độ trễ khi xử lý video dài, chưa xử lý camera real-time; hướng: tracking DeepSORT, ONNX/TensorRT, stream camera).

## 14. Tài liệu tham khảo

1. Redmon et al., *You Only Look Once*, CVPR 2016.
2. Ren et al., *Faster R-CNN*, NeurIPS 2015.
3. Dataset Kaggle: https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs
