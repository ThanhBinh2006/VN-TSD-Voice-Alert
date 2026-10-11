"""Web localhost.   Phụ trách: Thành viên 3

Tự do chọn framework (Flask/FastAPI), cấu trúc thư mục và giao diện. Chỉ cần tuân thủ:
  1. Nhận file ảnh/video + lựa chọn model ("yolo" | "faster_rcnn").
  2. Gọi ts_alert.pipeline.process_media(input_path, model_name, ...) — NÊN chạy nền để không treo request,
     và NÊN giữ detector trong bộ nhớ (get_detector) để không nạp lại trọng số mỗi lần.
  3. Cho người dùng xem và tải video/ảnh kết quả, hiển thị các số liệu process_media trả về.
  4. Chạy được bằng:  python -m app.main   (host/port lấy từ cfg["app"] trong configs/shared.yaml)
"""
from __future__ import annotations


def create_app():
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


def main() -> None:
    # TODO: Thành viên 3
    raise NotImplementedError("TODO: Thành viên 3")


if __name__ == "__main__":
    main()
