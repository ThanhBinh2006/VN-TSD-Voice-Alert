# Quy tắc đóng góp

- Nhánh chính `main`, chỉ merge qua Pull Request, cần ít nhất 1 người review.
- Nhánh cá nhân:

| Thành viên | Nhánh |
|---|---|
| Thành viên 1 (Data & YOLO) | `data-yolo` |
| Thành viên 2 (Faster R-CNN) | `faster-rcnn` |
| Thành viên 3 (Backend & Web UI) | `backend-ui` |
| Thành viên 4 (Layer & Voice) | `layer-voice` |
| Thành viên 5 (Quản lý & DB) | `db-docs` |

- Commit: `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`.
- Không commit `data/`, `outputs/`, `models/*.pt`, `models/*.pth`, `*.db`.
- Không đổi `docs/DATA_CONTRACT.md` hoặc chữ ký hàm (signature) trong các file vỏ mà không báo cả nhóm.
- Cấu trúc DB, giao diện web và bên trong từng hàm TODO được tuỳ biến thoải mái, miễn tuân thủ `docs/DATA_CONTRACT.md`.
- Hiện thực xong hàm nào thì xoá `raise NotImplementedError`, và bỏ `@pytest.mark.skip` ở test tương ứng.
- Chạy `pytest` trước khi tạo Pull Request.

Hướng dẫn thao tác Git chi tiết (tạo repo, nhánh, commit, PR, xử lý xung đột, dùng trên Kaggle): xem `README.md` mục 12.
