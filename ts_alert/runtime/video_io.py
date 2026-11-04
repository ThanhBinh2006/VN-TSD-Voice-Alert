"""Đọc/ghi ảnh và video (OpenCV 4.12).   Phụ trách: Thành viên 3

Ghi chú thiết kế:
- Đọc/ghi ảnh qua np.fromfile + cv2.imdecode / cv2.imencode + tofile để chạy được với đường dẫn có
  dấu tiếng Việt trên Windows (cv2.imread / cv2.imwrite trả None hoặc ghi lỗi với đường dẫn Unicode).
- Mọi khung hình trả ra luôn là BGR 3 kênh (ảnh xám / có kênh alpha được chuyển về BGR).
- timestamp_sec = frame_index / fps (đúng DATA_CONTRACT). Video kết quả cũng được ghi với fps cố định
  này nên âm thanh cảnh báo khớp với hình trong video đầu ra.
- VideoWriter ưu tiên ghi H.264 (libx264, yuv420p) qua ffmpeg của imageio-ffmpeg để trình duyệt phát
  được trực tiếp trong thẻ <video>; nếu không có imageio-ffmpeg thì dùng cv2.VideoWriter codec mp4v.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ts_alert.types import IMAGE_EXTS, VIDEO_EXTS

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
MAX_REASONABLE_FPS = 240.0


def media_type(path) -> str:
    """"video" hoặc "image" theo đuôi file; ValueError nếu không hỗ trợ."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    raise ValueError(f"Định dạng không hỗ trợ: {ext}")


# ----------------------------------------------------------------------------- helpers

def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Đưa ảnh xám (H, W) / (H, W, 1) hoặc BGRA (H, W, 4) về BGR (H, W, 3)."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _sanitize_fps(fps: float) -> float:
    """Một số file (webm, video quay màn hình) báo fps = 0, NaN hoặc 1000 -> dùng DEFAULT_FPS."""
    if not fps or math.isnan(fps) or fps <= 0 or fps > MAX_REASONABLE_FPS:
        log.warning("FPS không hợp lệ (%s), dùng mặc định %.1f", fps, DEFAULT_FPS)
        return DEFAULT_FPS
    return float(fps)


def _open_capture(path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Không mở được video: {path}")
    return cap


# ----------------------------------------------------------------------------- video

def get_video_info(path) -> dict:
    """Trả {"fps": float, "width": int, "height": int, "frame_count": int}.

    width/height lấy từ khung hình đầu tiên đã giải mã (không lấy từ metadata) vì video quay dọc
    trên điện thoại có metadata xoay: OpenCV tự xoay khung hình nhưng CAP_PROP_FRAME_WIDTH/HEIGHT
    vẫn báo kích thước chưa xoay -> VideoWriter sẽ bị sai kích thước.
    frame_count chỉ là ước lượng từ metadata (có thể lệch vài khung), không dùng để lặp.
    """
    cap = _open_capture(path)
    try:
        fps = _sanitize_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Video không có khung hình nào đọc được: {path}")
        height, width = frame.shape[:2]
    finally:
        cap.release()
    return {"fps": fps, "width": int(width), "height": int(height), "frame_count": frame_count}


def iter_frames(path) -> Iterator[tuple]:
    """Yield (frame_index, frame_bgr, timestamp_sec) lần lượt từng khung hình của video.
    timestamp_sec = frame_index / fps."""
    cap = _open_capture(path)
    try:
        fps = _sanitize_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame_index, _to_bgr(frame), frame_index / fps
            frame_index += 1
    finally:
        cap.release()


# ----------------------------------------------------------------------------- image

def read_image(path):
    """Đọc ảnh -> ndarray BGR (H, W, 3); raise nếu không đọc được."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy ảnh: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if frame is None:
        raise ValueError(f"Không đọc được ảnh (file hỏng hoặc sai định dạng): {path}")
    if frame.dtype != np.uint8:      # PNG 16-bit -> 8-bit
        frame = cv2.convertScaleAbs(frame, alpha=255.0 / 65535.0)
    return _to_bgr(frame)


def write_image(path, frame) -> None:
    """Ghi ndarray BGR ra file ảnh (tạo thư mục cha nếu thiếu)."""
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Đuôi ảnh không hỗ trợ: {ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, frame)
    if not ok:
        raise ValueError(f"Không mã hoá được ảnh sang {ext}: {path}")
    buf.tofile(str(path))


# ----------------------------------------------------------------------------- writer

class VideoWriter:
    """Ghi video KHÔNG tiếng (mp4, codec mp4v hoặc h264 nếu có).

    Mặc định ghi H.264 qua ffmpeg (imageio-ffmpeg) để trình duyệt phát được; kích thước lẻ được
    pad thêm 1 pixel vì yuv420p yêu cầu chiều rộng/cao chẵn. Fallback: cv2.VideoWriter + mp4v.
    Dùng được với `with VideoWriter(...) as w:`; close() gọi nhiều lần không lỗi.
    """

    def __init__(self, path, fps: float, size: tuple):
        """size = (width, height)."""
        self.path = Path(path)
        self.fps = _sanitize_fps(fps)
        self.size = (int(size[0]), int(size[1]))
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"Kích thước video không hợp lệ: {size}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        self._closed = False
        self._ffmpeg = None
        self._cv_writer = None
        self.backend = self._open()

    def _open(self) -> str:
        try:
            import imageio_ffmpeg
        except ImportError:
            imageio_ffmpeg = None
        if imageio_ffmpeg is not None:
            self._ffmpeg = imageio_ffmpeg.write_frames(
                str(self.path), self.size, fps=self.fps, codec="libx264",
                pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1,
                output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"],
                ffmpeg_log_level="error",
            )
            self._ffmpeg.send(None)   # khởi động generator (chạy tiến trình ffmpeg)
            return "ffmpeg-h264"
        log.warning("Không có imageio-ffmpeg, ghi bằng OpenCV mp4v (trình duyệt có thể không phát được)")
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not writer.isOpened():
            raise RuntimeError(f"Không mở được VideoWriter cho {self.path}")
        self._cv_writer = writer
        return "opencv-mp4v"

    def write(self, frame) -> None:
        if self._closed:
            raise RuntimeError("VideoWriter đã đóng")
        frame = _to_bgr(frame)
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            # cv2.VideoWriter bỏ qua khung sai kích thước mà không báo lỗi -> chặn sớm ở đây.
            raise ValueError(f"Khung hình {w}x{h} khác kích thước video {self.size[0]}x{self.size[1]}")
        if self._ffmpeg is not None:
            self._ffmpeg.send(np.ascontiguousarray(frame))
        else:
            self._cv_writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ffmpeg is not None:
            self._ffmpeg.close()
        if self._cv_writer is not None:
            self._cv_writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
