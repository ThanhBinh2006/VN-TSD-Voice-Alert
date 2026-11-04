"""Test ts_alert/runtime/video_io.py (Thành viên 3). Tự sinh video/ảnh tổng hợp, không cần dữ liệu thật."""
import numpy as np
import pytest

from ts_alert.runtime import video_io

W, H, FPS, N = 64, 48, 10.0, 12


def _frame(i, w=W, h=H):
    """Khung hình có mức xám thay đổi theo i để kiểm tra thứ tự khung."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = (i * 20) % 256
    return f


def _make_video(path, n=N, fps=FPS, w=W, h=H):
    with video_io.VideoWriter(path, fps, (w, h)) as wr:
        for i in range(n):
            wr.write(_frame(i, w, h))
    return path


def test_media_type():
    assert video_io.media_type("a.MP4") == "video"
    assert video_io.media_type("b.jpeg") == "image"
    with pytest.raises(ValueError):
        video_io.media_type("c.txt")


def test_write_then_info(tmp_path):
    p = _make_video(tmp_path / "out" / "clip.mp4")      # thư mục cha chưa có -> tự tạo
    info = video_io.get_video_info(p)
    assert info["fps"] == pytest.approx(FPS, rel=0.01)
    assert (info["width"], info["height"]) == (W, H)
    assert abs(info["frame_count"] - N) <= 1


def test_iter_frames_index_timestamp_and_order(tmp_path):
    p = _make_video(tmp_path / "clip.mp4")
    frames = list(video_io.iter_frames(p))
    assert len(frames) == N
    for k, (idx, frame, ts) in enumerate(frames):
        assert idx == k
        assert ts == pytest.approx(k / FPS)
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    means = [f.mean() for _, f, _ in frames]
    assert means == sorted(means)                        # mức xám tăng dần -> đúng thứ tự (nén có sai số nhỏ)


def test_writer_rejects_wrong_size(tmp_path):
    with video_io.VideoWriter(tmp_path / "clip.mp4", FPS, (W, H)) as wr:
        with pytest.raises(ValueError):
            wr.write(_frame(0, W + 2, H))
        wr.write(_frame(0))                              # vẫn ghi tiếp được sau lỗi
    assert wr.frames_written == 1


def test_writer_close_twice(tmp_path):
    wr = video_io.VideoWriter(tmp_path / "clip.mp4", FPS, (W, H))
    wr.write(_frame(0))
    wr.close()
    wr.close()
    with pytest.raises(RuntimeError):
        wr.write(_frame(1))


def test_writer_odd_size_and_bad_fps(tmp_path):
    p = _make_video(tmp_path / "odd.mp4", n=3, fps=0, w=63, h=47)   # fps=0 -> mặc định
    frames = list(video_io.iter_frames(p))
    assert len(frames) == 3
    assert frames[0][1].shape[0] >= 47 and frames[0][1].shape[1] >= 63


def test_missing_or_broken_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        video_io.get_video_info(tmp_path / "none.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        video_io.get_video_info(bad)
    with pytest.raises(ValueError):
        list(video_io.iter_frames(bad))


def test_image_roundtrip_unicode_path(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    p = tmp_path / "ảnh có dấu" / "biển báo.png"        # đường dẫn tiếng Việt
    video_io.write_image(p, img)
    assert np.array_equal(video_io.read_image(p), img)  # PNG không nén mất dữ liệu


def test_read_image_gray_and_alpha_to_bgr(tmp_path):
    gray = np.full((H, W), 128, np.uint8)
    bgra = np.zeros((H, W, 4), np.uint8)
    video_io.write_image(tmp_path / "g.png", gray)
    video_io.write_image(tmp_path / "a.png", bgra)
    assert video_io.read_image(tmp_path / "g.png").shape == (H, W, 3)
    assert video_io.read_image(tmp_path / "a.png").shape == (H, W, 3)


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        video_io.read_image(tmp_path / "none.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"xx")
    with pytest.raises(ValueError):
        video_io.read_image(bad)
    with pytest.raises(ValueError):
        video_io.write_image(tmp_path / "x.gif", np.zeros((4, 4, 3), np.uint8))
