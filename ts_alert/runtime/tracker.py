"""Theo dõi biển báo cho runtime bằng detector và Sparse Lucas–Kanade.

Không thay đổi raw detections dùng cho ML evaluation. Mọi kết quả trả về đều
là bản sao; trạng thái nội bộ được bảo vệ bằng threading.RLock.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from ts_alert.postprocess.position import get_position
from ts_alert.types import Detection, TrackedDetection

TrackState = Literal["tentative", "active", "lost"]


@dataclass(frozen=True)
class TrackerConfig:
    """Cấu hình thời gian theo timestamp của video, không theo thời gian xử lý."""

    ttl_sec: float = 1.0
    reid_ttl_sec: float = 2.0
    max_missed_detections: int = 2
    max_extrapolate_sec: float = 0.65
    max_staleness_sec: float = 0.85
    min_hits_to_confirm: int = 2
    single_hit_conf_threshold: float | None = 0.70
    iou_threshold: float = 0.20
    center_distance_weight: float = 0.25
    use_optical_flow: bool = True
    flow_max_width: int = 320
    velocity_smoothing: float = 0.45
    velocity_damping: float = 0.82

    def __post_init__(self) -> None:
        """Từ chối cấu hình không hữu hạn hoặc nằm ngoài miền hợp lệ."""
        for name in (
            "ttl_sec",
            "reid_ttl_sec",
            "max_extrapolate_sec",
            "max_staleness_sec",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} phải hữu hạn và >= 0")

        for name in (
            "max_missed_detections",
            "min_hits_to_confirm",
            "flow_max_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} phải là số nguyên >= 1")

        for name in ("iou_threshold", "velocity_smoothing", "velocity_damping"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} phải thuộc [0, 1]")

        threshold = self.single_hit_conf_threshold
        if threshold is not None:
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("single_hit_conf_threshold phải thuộc [0, 1]")

        if (
            not math.isfinite(self.center_distance_weight)
            or self.center_distance_weight < 0
        ):
            raise ValueError("center_distance_weight phải hữu hạn và >= 0")


@dataclass
class _Track:
    """Trạng thái riêng của một đối tượng; không giữ ảnh nguồn."""

    track_id: int
    bbox: list[float]
    class_id: int
    label: str
    confidence: float
    last_detect_frame_id: int
    last_detect_timestamp: float
    last_update_timestamp: float
    measurement_bbox: list[float]
    measurement_timestamp: float
    state: TrackState = "tentative"
    confirmed: bool = False
    hits: int = 1
    missed_detections: int = 0
    vx: float = 0.0
    vy: float = 0.0
    alert_emitted: bool = False
    class_votes: dict[int, float] = field(default_factory=dict)
    class_labels: dict[int, str] = field(default_factory=dict)
    last_source: str = "detector"
    last_flow_compensated: bool = False


@dataclass(frozen=True)
class _Observation:
    """Quan sát detector và kết quả bù optical flow của quan sát đó."""

    bbox: list[float]
    class_id: int
    label: str
    confidence: float
    flow_compensated: bool = False


def _uint8_image(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa ảnh thành uint8 liên tục trong bộ nhớ."""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    values = np.nan_to_num(
        image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0
    )
    if np.issubdtype(image.dtype, np.floating) and values.size:
        if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
            values *= 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def compute_flow_gray(
    frame: np.ndarray,
    max_width: int = 320,
) -> tuple[np.ndarray, float]:
    """Trả ảnh xám uint8 thu nhỏ và tỷ lệ pixel xám/pixel ảnh gốc."""
    if max_width < 1:
        raise ValueError("max_width phải >= 1")
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame phải là ndarray không rỗng")
    if frame.ndim not in (2, 3):
        raise ValueError("frame phải là ảnh grayscale, BGR hoặc BGRA")

    height, width = frame.shape[:2]
    target_width = min(width, int(max_width))
    scale = target_width / width
    target_height = max(1, round(height * scale))

    image = frame
    if target_width != width:
        image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    image = _uint8_image(image)

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Số kênh ảnh phải là 1, 3 hoặc 4")

    return np.ascontiguousarray(gray), scale


def _center(bbox: list[float]) -> tuple[float, float]:
    """Trả tọa độ tâm của box."""
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


def _area(bbox: list[float]) -> float:
    """Trả diện tích không âm của box."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _clip_box(
    bbox: list[float],
    width: float | None,
    height: float | None,
) -> list[float]:
    """Trả bản sao box được kẹp trong kích thước ảnh đã biết."""
    result = list(bbox)
    if width is not None and width > 0:
        result[0] = min(width, max(0.0, result[0]))
        result[2] = min(width, max(0.0, result[2]))
    if height is not None and height > 0:
        result[1] = min(height, max(0.0, result[1]))
        result[3] = min(height, max(0.0, result[3]))
    return result


def _translate(bbox: list[float], dx: float, dy: float) -> list[float]:
    """Trả box sau khi tịnh tiến theo hai trục."""
    return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]


def _iou(first: list[float], second: list[float]) -> float:
    """Tính tỷ lệ diện tích giao trên diện tích hợp của hai box."""
    intersection = max(
        0.0, min(first[2], second[2]) - max(first[0], second[0])
    ) * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(
    first: list[float],
    second: list[float],
) -> float:
    """Chuẩn hóa khoảng cách tâm theo kích thước hai box."""
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    diagonal = 0.5 * (
        math.hypot(first[2] - first[0], first[3] - first[1])
        + math.hypot(second[2] - second[0], second[3] - second[1])
    )
    return math.hypot(first_x - second_x, first_y - second_y) / max(
        diagonal, 1.0
    )


def _warp_box_lk(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    bbox: list[float],
    prev_scale: float,
    curr_scale: float,
) -> tuple[list[float], bool]:
    """Ước lượng tịnh tiến/scale bền vững bằng LK rồi kẹp theo ảnh hiện tại.

    Ít hơn ba điểm inlier hoặc flow lỗi sẽ trả box ban đầu và False.
    """
    original = [float(value) for value in bbox]
    if (
        prev_gray.ndim != 2
        or curr_gray.ndim != 2
        or prev_gray.size == 0
        or curr_gray.size == 0
        or not math.isfinite(prev_scale)
        or not math.isfinite(curr_scale)
        or prev_scale <= 0
        or curr_scale <= 0
    ):
        return original, False

    try:
        prev_height, prev_width = prev_gray.shape
        curr_height, curr_width = curr_gray.shape
        if min(prev_height, prev_width, curr_height, curr_width) < 5:
            return original, False

        prev = _uint8_image(prev_gray)
        curr = _uint8_image(curr_gray)

        # LK yêu cầu hai ảnh cùng kích thước.
        ratio_x = curr_width / prev_width
        ratio_y = curr_height / prev_height
        if prev.shape != curr.shape:
            prev = cv2.resize(
                prev, (curr_width, curr_height), interpolation=cv2.INTER_LINEAR
            )

        scaled = [
            original[0] * prev_scale * ratio_x,
            original[1] * prev_scale * ratio_y,
            original[2] * prev_scale * ratio_x,
            original[3] * prev_scale * ratio_y,
        ]
        x1 = max(0, int(math.floor(scaled[0])))
        y1 = max(0, int(math.floor(scaled[1])))
        x2 = min(curr_width, int(math.ceil(scaled[2])))
        y2 = min(curr_height, int(math.ceil(scaled[3])))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return original, False

        corners = cv2.goodFeaturesToTrack(
            prev[y1:y2, x1:x2],
            maxCorners=24,
            qualityLevel=0.01,
            minDistance=3.0,
            blockSize=3,
        )
        point_sets: list[np.ndarray] = []
        if corners is not None:
            point_sets.append(
                corners.reshape(-1, 2)
                + np.array([x1, y1], dtype=np.float32)
            )

        # Lưới dự phòng giúp box ít texture vẫn có điểm thử nghiệm.
        grid_x, grid_y = np.meshgrid(
            np.linspace(x1 + 0.5, x2 - 1.5, 4, dtype=np.float32),
            np.linspace(y1 + 0.5, y2 - 1.5, 4, dtype=np.float32),
        )
        point_sets.append(np.column_stack((grid_x.ravel(), grid_y.ravel())))
        points = np.concatenate(point_sets).astype(np.float32).reshape(-1, 1, 2)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev,
            curr,
            points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                15,
                0.03,
            ),
            minEigThreshold=1e-4,
        )
        if next_points is None or status is None:
            return original, False

        previous = points.reshape(-1, 2)
        following = next_points.reshape(-1, 2)
        valid = (
            (status.reshape(-1) == 1)
            & np.isfinite(following).all(axis=1)
            & (following[:, 0] >= 0)
            & (following[:, 0] < curr_width)
            & (following[:, 1] >= 0)
            & (following[:, 1] < curr_height)
        )
        previous, following = previous[valid], following[valid]
        if len(previous) < 3:
            return original, False

        displacement = following - previous
        median_displacement = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median_displacement, axis=1)
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        inliers = residual <= max(1.5, median_residual + 3.0 * 1.4826 * mad)
        previous, following = previous[inliers], following[inliers]
        if len(previous) < 3:
            return original, False

        magnification = 1.0
        if len(previous) >= 6:
            old_radius = np.linalg.norm(
                previous - np.median(previous, axis=0), axis=1
            )
            new_radius = np.linalg.norm(
                following - np.median(following, axis=0), axis=1
            )
            usable = old_radius >= 2.0
            if int(usable.sum()) >= 4:
                magnification = float(
                    np.clip(
                        np.median(new_radius[usable] / old_radius[usable]),
                        0.92,
                        1.08,
                    )
                )

        offset = np.median(following - magnification * previous, axis=0)
        warped = [
            float((magnification * scaled[0] + offset[0]) / curr_scale),
            float((magnification * scaled[1] + offset[1]) / curr_scale),
            float((magnification * scaled[2] + offset[0]) / curr_scale),
            float((magnification * scaled[3] + offset[1]) / curr_scale),
        ]
        if not all(math.isfinite(value) for value in warped):
            return original, False

        return _clip_box(
            warped, curr_width / curr_scale, curr_height / curr_scale
        ), True
    except (cv2.error, ValueError, FloatingPointError):
        return original, False


class TrafficSignTracker:
    """Tracker CPU cho một luồng video; reset khi đổi video hoặc seek.

    Optical flow chỉ dùng ảnh trước đó đã thu nhỏ. Ảnh của detector có thể
    được truyền qua source_gray nhưng không được giữ trong tracker.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        config: TrackerConfig | None = None,
        classes: dict | None = None,
    ) -> None:
        """Khởi tạo cấu hình, nhãn lớp và trạng thái tracker."""
        self._lock = threading.RLock()
        self.cfg = cfg or {}

        if config is None:
            runtime = self.cfg.get("runtime", {})
            settings = (
                runtime.get("tracker", self.cfg.get("tracker", {}))
                if isinstance(runtime, Mapping)
                else self.cfg.get("tracker", {})
            )
            allowed = {item.name for item in fields(TrackerConfig)}
            settings = settings if isinstance(settings, Mapping) else {}
            config = TrackerConfig(
                **{key: value for key, value in settings.items() if key in allowed}
            )
        self.config = config
        self._class_labels = self._parse_class_labels(classes or {})
        self.reset()

    @staticmethod
    def _parse_class_labels(classes: Mapping[Any, Any]) -> dict[int, str]:
        """Hỗ trợ class_id -> label hoặc class_id -> bản ghi lớp."""
        result: dict[int, str] = {}
        for key, value in classes.items():
            try:
                class_id = int(key)
            except (ValueError, TypeError, OverflowError):
                continue
            if isinstance(value, str):
                result[class_id] = value
            elif isinstance(value, Mapping):
                label = value.get("label") or value.get("name")
                if label is not None:
                    result[class_id] = str(label)
        return result

    def reset(self) -> None:
        """Xóa tracks, lịch sử cảnh báo, thống kê và ảnh xám trước đó."""
        with self._lock:
            self._tracks: dict[int, _Track] = {}
            self._lost_events: list[dict[str, Any]] = []
            self._next_track_id = 1
            self._prev_gray: np.ndarray | None = None
            self._prev_scale = 1.0
            self._last_step_timestamp: float | None = None
            self._last_frame_id: int | None = None
            self._last_detection_frame_id: int | None = None
            self._width: int | None = None
            self._height: int | None = None
            self._step_count = 0
            self._step_total_ms = 0.0
            self.stats: dict[str, Any] = {
                "total_tracks_created": 0,
                "compensated_detections": 0,
                "dropped_stale_detections": 0,
                "dropped_stale_batches": 0,
                "dropped_out_of_order_batches": 0,
                "expired_tracks": 0,
                "reidentified_tracks": 0,
            }

    @staticmethod
    def _timestamp(value: float) -> float:
        """Chuyển timestamp thành float hữu hạn."""
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Timestamp phải hữu hạn")
        return result

    def _set_dimensions(
        self,
        frame: np.ndarray | None,
        width: int | None,
        height: int | None,
    ) -> None:
        """Cập nhật kích thước ảnh từ frame hoặc các tham số tường minh."""
        if frame is not None:
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                raise ValueError("current_frame phải là ndarray không rỗng")
            if frame.ndim not in (2, 3):
                raise ValueError("current_frame phải là ảnh 2D hoặc 3D")
            height, width = frame.shape[:2]
        if width is not None:
            if width <= 0:
                raise ValueError("frame_width phải > 0")
            self._width = int(width)
        if height is not None:
            if height <= 0:
                raise ValueError("frame_height phải > 0")
            self._height = int(height)

    def _lose(
        self,
        track: _Track,
        *,
        timestamp: float | None = None,
        reason: str = "lost",
    ) -> None:
        """Ghi sự kiện một lần mỗi lần mất dấu; caller phải giữ lock."""
        if track.state != "lost":
            lost_ts = timestamp
            if lost_ts is None:
                lost_ts = (
                    self._last_step_timestamp
                    if self._last_step_timestamp is not None
                    else track.last_update_timestamp
                )
            track.state = "lost"
            self.stats["expired_tracks"] += 1
            self._lost_events.append(
                {
                    "track_id": int(track.track_id),
                    "class_id": int(track.class_id),
                    "label": str(track.label),
                    "lost_timestamp": float(lost_ts),
                    "reason": str(reason),
                    "confirmed": bool(track.confirmed),
                    "alert_emitted": bool(track.alert_emitted),
                }
            )

    def _cleanup(self, timestamp: float) -> None:
        """TTL là giới hạn cứng, kể cả khi optical flow vẫn thành công."""
        for track_id, track in list(self._tracks.items()):
            age = timestamp - track.last_detect_timestamp
            if age > self.config.reid_ttl_sec:
                self._lose(track, timestamp=timestamp, reason="expired")
                del self._tracks[track_id]
            elif age > self.config.ttl_sec:
                self._lose(track, timestamp=timestamp, reason="expired")

    def _visible_box(self, bbox: list[float]) -> tuple[list[float], bool]:
        """Kẹp box và kiểm tra phần diện tích còn nhìn thấy."""
        clipped = _clip_box(bbox, self._width, self._height)
        original_area = _area(bbox)
        visible_area = _area(clipped)
        ratio = visible_area / original_area if original_area > 0 else 0.0
        return clipped, visible_area > 4.0 and ratio >= 0.25

    def _update_velocity(
        self,
        track: _Track,
        old_box: list[float],
        new_box: list[float],
        elapsed: float,
    ) -> None:
        """Cập nhật vận tốc tâm box bằng trung bình trượt hàm mũ."""
        if elapsed <= 1e-6:
            return
        old_x, old_y = _center(old_box)
        new_x, new_y = _center(new_box)
        alpha = self.config.velocity_smoothing
        track.vx = (1.0 - alpha) * track.vx + alpha * (
            (new_x - old_x) / elapsed
        )
        track.vy = (1.0 - alpha) * track.vy + alpha * (
            (new_y - old_y) / elapsed
        )

    def _advance(
        self,
        frame: np.ndarray | None,
        frame_id: int,
        timestamp: float,
        width: int | None,
        height: int | None,
    ) -> None:
        """Tiến toàn bộ tracks đến khung hiện tại; caller phải giữ lock."""
        if (
            self._last_step_timestamp is not None
            and timestamp - self._last_step_timestamp < -0.1
        ):
            self.reset()

        self._set_dimensions(frame, width, height)
        previous_timestamp = self._last_step_timestamp
        elapsed = (
            max(0.0, timestamp - previous_timestamp)
            if previous_timestamp is not None
            else 0.0
        )
        gray: np.ndarray | None = None
        scale = 1.0
        if frame is not None and self.config.use_optical_flow:
            gray, scale = compute_flow_gray(frame, self.config.flow_max_width)

        self._cleanup(timestamp)
        for track in self._tracks.values():
            if track.state == "lost":
                continue

            track_dt = max(0.0, timestamp - track.last_update_timestamp)
            if track_dt <= 0.0:
                continue

            old_box = track.bbox
            new_box = old_box
            flow_ok = False
            if (
                elapsed > 0
                and gray is not None
                and self._prev_gray is not None
                and previous_timestamp is not None
                and abs(track.last_update_timestamp - previous_timestamp) < 1e-6
            ):
                new_box, flow_ok = _warp_box_lk(
                    self._prev_gray,
                    gray,
                    old_box,
                    self._prev_scale,
                    scale,
                )

            if not flow_ok:
                if (
                    timestamp - track.last_detect_timestamp
                    > self.config.max_extrapolate_sec
                ):
                    self._lose(track, timestamp=timestamp, reason="expired")
                    continue
                motion_dt = min(track_dt, self.config.max_extrapolate_sec)
                damping = self.config.velocity_damping
                new_box = _translate(
                    old_box,
                    track.vx * motion_dt * damping,
                    track.vy * motion_dt * damping,
                )
                # Giảm chấn theo thời gian để không phụ thuộc FPS hiển thị.
                decay = damping**track_dt
                track.vx *= decay
                track.vy *= decay
                track.last_source = "motion_pred"
            else:
                self._update_velocity(track, old_box, new_box, track_dt)
                track.last_source = "optical_flow"

            track.last_flow_compensated = flow_ok
            track.bbox, visible = self._visible_box(new_box)
            track.last_update_timestamp = timestamp
            if not visible:
                self._lose(track, timestamp=timestamp, reason="exited_frame")

        # Không tái dùng ảnh cũ sau một bước ngoại suy không có frame.
        if gray is not None:
            self._prev_gray = gray
            self._prev_scale = scale
        elif elapsed > 0 or frame is not None:
            self._prev_gray = None

        self._last_step_timestamp = (
            timestamp
            if previous_timestamp is None
            else max(timestamp, previous_timestamp)
        )
        self._last_frame_id = frame_id
        self._cleanup(timestamp)

    def _position_config(self, override: dict | None) -> dict[str, Any]:
        """Chuẩn hóa cấu hình phân vùng trái, trước và phải."""
        settings: Any = self.cfg if override is None else override
        if isinstance(settings, Mapping):
            settings = settings.get("postprocess", settings)
        if isinstance(settings, Mapping):
            settings = settings.get("position", settings)

        left_max, right_min = 0.33, 0.66
        if isinstance(settings, Mapping):
            try:
                left = float(settings.get("left_max", left_max))
                right = float(settings.get("right_min", right_min))
                if 0 <= left <= right <= 1:
                    left_max, right_min = left, right
            except (ValueError, TypeError, OverflowError):
                pass

        return {
            "postprocess": {
                "position": {"left_max": left_max, "right_min": right_min}
            }
        }

    def _snapshot(
        self,
        include_tentative: bool = True,
        position_cfg: dict | None = None,
    ) -> list[TrackedDetection]:
        """Trả bản sao các track hiện diện theo đầy đủ hợp đồng dữ liệu."""
        settings = self._position_config(position_cfg)
        result: list[TrackedDetection] = []
        for track in self._tracks.values():
            if track.state == "lost":
                continue
            if not include_tentative and not track.confirmed:
                continue

            current_ref_ts = float(
                self._last_step_timestamp
                if self._last_step_timestamp is not None
                else track.last_detect_timestamp
            )
            position = (
                get_position(track.bbox, self._width, settings)
                if self._width is not None
                else "front"
            )
            result.append(
                TrackedDetection(
                    frame_id=int(
                        self._last_frame_id
                        if self._last_frame_id is not None
                        else track.last_detect_frame_id
                    ),
                    source_timestamp=current_ref_ts,
                    timestamp_sec=current_ref_ts,
                    class_id=track.class_id,
                    label=track.label,
                    confidence=track.confidence,
                    bbox=list(track.bbox),
                    position=position,
                    track_id=track.track_id,
                    vx=float(track.vx),
                    vy=float(track.vy),
                    state=track.state,
                    confirmed=bool(track.confirmed),
                    hits=int(track.hits),
                    missed_detections=int(track.missed_detections),
                    age_sec=max(
                        0.0,
                        float(current_ref_ts - track.last_detect_timestamp),
                    ),
                    ttl_sec=float(self.config.ttl_sec),
                    lag_ms=float(self.stats.get("last_wall_lag_ms", 0.0)),
                    source=track.last_source,
                    last_detect_frame_id=track.last_detect_frame_id,
                    last_detect_timestamp=track.last_detect_timestamp,
                    alert_emitted=track.alert_emitted,
                    class_votes=dict(track.class_votes),
                )
            )
        return result

    def _parse_observation(self, raw: Mapping[str, Any]) -> _Observation | None:
        """Chuẩn hóa một detection hợp lệ mà không sửa dữ liệu đầu vào."""
        try:
            bbox = [float(value) for value in raw["bbox"]]
            confidence = float(raw["confidence"])
            class_id = int(raw["class_id"])
            if (
                len(bbox) != 4
                or not all(math.isfinite(value) for value in bbox)
                or not math.isfinite(confidence)
                or _area(bbox) <= 0
            ):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

        label = str(
            self._class_labels.get(class_id, raw.get("label", str(class_id)))
        )
        return _Observation(
            bbox, class_id, label, min(1.0, max(0.0, confidence))
        )

    def _association_cost(
        self,
        observation: _Observation,
        track: _Track,
        bbox: list[float] | None = None,
    ) -> float | None:
        """Tính chi phí ghép; trả None nếu hình học hoặc lớp không phù hợp."""
        candidate = track.bbox if bbox is None else bbox
        overlap = _iou(observation.bbox, candidate)
        distance = _normalized_center_distance(observation.bbox, candidate)
        same_class = observation.class_id == track.class_id

        if not same_class:
            # Chỉ cho đổi nhãn khi hình học cung cấp bằng chứng mạnh.
            if overlap < max(0.65, self.config.iou_threshold):
                return None
        elif overlap < self.config.iou_threshold:
            area_ratio = _area(observation.bbox) / max(_area(candidate), 1.0)
            if distance > 0.70 or not 0.25 <= area_ratio <= 4.0:
                return None

        return (
            1.0
            - overlap
            + self.config.center_distance_weight * distance
            + (0.0 if same_class else 0.25)
            + (0.03 if track.state == "lost" else 0.0)
        )

    def _source_gray(
        self,
        source: tuple[np.ndarray, float] | np.ndarray | None,
    ) -> tuple[np.ndarray, float] | None:
        """Chuẩn hóa ảnh xám nguồn và tỷ lệ thu nhỏ phục vụ bù độ trễ."""
        if source is None:
            return None
        if isinstance(source, tuple):
            if len(source) != 2:
                return None
            gray, scale = source
        else:
            gray = source
            scale = (
                gray.shape[1] / self._width
                if isinstance(gray, np.ndarray)
                and gray.ndim == 2
                and self._width
                else 1.0
            )
        try:
            scale = float(scale)
            if (
                not isinstance(gray, np.ndarray)
                or gray.ndim != 2
                or gray.size == 0
                or not math.isfinite(scale)
                or scale <= 0
            ):
                return None
            gray = _uint8_image(gray)
            if gray.shape[1] > self.config.flow_max_width:
                gray, extra_scale = compute_flow_gray(
                    gray, self.config.flow_max_width
                )
                scale *= extra_scale
            return gray, scale
        except (TypeError, ValueError, OverflowError):
            return None

    def _compensate(
        self,
        observation: _Observation,
        elapsed: float,
        source: tuple[np.ndarray, float] | None,
        timestamp: float,
    ) -> _Observation:
        """Bù độ trễ bằng LK hoặc vận tốc và giữ kết quả thành công của LK."""
        bbox = observation.bbox
        flow_ok = False
        if (
            self.config.use_optical_flow
            and source is not None
            and self._prev_gray is not None
        ):
            bbox, flow_ok = _warp_box_lk(
                source[0],
                self._prev_gray,
                bbox,
                source[1],
                self._prev_scale,
            )

        if not flow_ok:
            best_track: _Track | None = None
            best_cost = math.inf
            motion_dt = min(elapsed, self.config.max_extrapolate_sec)
            for track in self._tracks.values():
                if timestamp - track.last_detect_timestamp > self.config.reid_ttl_sec:
                    continue
                historical_box = _translate(
                    track.bbox,
                    -track.vx * motion_dt * self.config.velocity_damping,
                    -track.vy * motion_dt * self.config.velocity_damping,
                )
                cost = self._association_cost(observation, track, historical_box)
                if cost is not None and cost < best_cost:
                    best_cost, best_track = cost, track

            if best_track is not None:
                bbox = _translate(
                    bbox,
                    best_track.vx * motion_dt * self.config.velocity_damping,
                    best_track.vy * motion_dt * self.config.velocity_damping,
                )
            # Chưa có bằng chứng chuyển động: dùng prior vận tốc bằng 0.

        self.stats["compensated_detections"] += 1
        return _Observation(
            bbox,
            observation.class_id,
            observation.label,
            observation.confidence,
            flow_compensated=flow_ok,
        )

    def _confirm(self, track: _Track) -> None:
        """Xác nhận track theo số hits hoặc ngưỡng confidence một lần thấy."""
        threshold = self.config.single_hit_conf_threshold
        track.confirmed = (
            track.confirmed
            or track.hits >= self.config.min_hits_to_confirm
            or (threshold is not None and track.confidence >= threshold)
        )
        track.state = "active" if track.confirmed else "tentative"

    def _record_vote(self, track: _Track, observation: _Observation) -> None:
        """Cộng phiếu có trọng số confidence và chọn nhãn lớp ổn định."""
        class_id = observation.class_id
        track.class_votes[class_id] = (
            track.class_votes.get(class_id, 0.0)
            + max(observation.confidence, 1e-6)
        )
        track.class_labels[class_id] = observation.label
        winner = max(
            track.class_votes,
            key=lambda candidate: (
                track.class_votes[candidate],
                candidate == track.class_id,
            ),
        )
        track.class_id = winner
        track.label = self._class_labels.get(
            winner, track.class_labels.get(winner, str(winner))
        )

    def update_from_detection(
        self,
        raw_detections: list[Detection] | list[dict],
        *,
        detection_frame_id: int,
        detection_timestamp: float,
        current_frame_id: int | None = None,
        current_timestamp: float | None = None,
        current_frame: np.ndarray | None = None,
        source_gray: tuple[np.ndarray, float] | np.ndarray | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        wall_lag_ms: float | None = None,
        position_cfg: dict | None = None,
        include_tentative: bool = True,
    ) -> list[TrackedDetection]:
        """Nhận một chu kỳ detector hoàn tất và ghép vào khung đang hiển thị.

        wall_lag_ms chỉ phục vụ chẩn đoán; bù chuyển động và TTL luôn sử dụng
        timestamp của video. Batch trùng hoặc đảo thứ tự không tăng hits/misses.
        """
        with self._lock:
            detection_ts = self._timestamp(detection_timestamp)
            current_ts = self._timestamp(
                detection_ts if current_timestamp is None else current_timestamp
            )
            frame_id = (
                detection_frame_id
                if current_frame_id is None
                else current_frame_id
            )
            self._advance(
                current_frame, frame_id, current_ts, frame_width, frame_height
            )
            if wall_lag_ms is not None:
                lag = float(wall_lag_ms)
                if math.isfinite(lag):
                    self.stats["last_wall_lag_ms"] = max(0.0, lag)

            elapsed = current_ts - detection_ts
            if elapsed < -0.05 or elapsed > self.config.max_staleness_sec:
                self.stats["dropped_stale_detections"] += len(raw_detections)
                self.stats["dropped_stale_batches"] += 1
                return self._snapshot(include_tentative, position_cfg)

            if (
                self._last_detection_frame_id is not None
                and detection_frame_id <= self._last_detection_frame_id
            ):
                self.stats["dropped_out_of_order_batches"] += 1
                return self._snapshot(include_tentative, position_cfg)
            self._last_detection_frame_id = detection_frame_id

            source = self._source_gray(source_gray) if elapsed > 0 else None
            observations: list[_Observation] = []
            for raw in raw_detections:
                if not isinstance(raw, Mapping):
                    continue
                observation = self._parse_observation(raw)
                if observation is None:
                    continue
                if elapsed > 0:
                    observation = self._compensate(
                        observation, elapsed, source, current_ts
                    )
                bbox, visible = self._visible_box(observation.bbox)
                if visible:
                    observations.append(
                        _Observation(
                            bbox,
                            observation.class_id,
                            observation.label,
                            observation.confidence,
                            flow_compensated=observation.flow_compensated,
                        )
                    )

            # Ma trận chi phí thưa; greedy toàn cục, không phụ thuộc SciPy.
            candidates = list(self._tracks.values())
            edges: list[tuple[float, int, int]] = []
            for observation_index, observation in enumerate(observations):
                for track in candidates:
                    cost = self._association_cost(observation, track)
                    if cost is not None:
                        edges.append((cost, observation_index, track.track_id))
            edges.sort()

            matched_observations: set[int] = set()
            matched_tracks: set[int] = set()
            for _, observation_index, track_id in edges:
                if (
                    observation_index in matched_observations
                    or track_id in matched_tracks
                ):
                    continue
                observation = observations[observation_index]
                track = self._tracks[track_id]
                was_lost = track.state == "lost"

                self._update_velocity(
                    track,
                    track.measurement_bbox,
                    observation.bbox,
                    current_ts - track.measurement_timestamp,
                )
                track.bbox = list(observation.bbox)
                track.measurement_bbox = list(observation.bbox)
                track.measurement_timestamp = current_ts
                track.last_update_timestamp = current_ts
                track.last_detect_timestamp = detection_ts
                track.last_detect_frame_id = detection_frame_id
                track.confidence = observation.confidence
                track.last_flow_compensated = observation.flow_compensated
                track.last_source = (
                    "optical_flow"
                    if observation.flow_compensated
                    else "motion_pred" if elapsed > 0 else "detector"
                )
                track.hits += 1
                track.missed_detections = 0
                self._record_vote(track, observation)
                self._confirm(track)
                if was_lost:
                    self.stats["reidentified_tracks"] += 1
                    self._lost_events = [
                        event
                        for event in self._lost_events
                        if event["track_id"] != track_id
                    ]
                matched_observations.add(observation_index)
                matched_tracks.add(track_id)

            for track in candidates:
                if track.track_id in matched_tracks or track.state == "lost":
                    continue
                if track.last_detect_frame_id < detection_frame_id or elapsed >= 0:
                    track.missed_detections += 1
                    if (
                        track.missed_detections
                        >= self.config.max_missed_detections
                    ):
                        self._lose(
                            track, timestamp=current_ts, reason="missed"
                        )

            for index, observation in enumerate(observations):
                if index in matched_observations:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(
                    track_id=track_id,
                    bbox=list(observation.bbox),
                    class_id=observation.class_id,
                    label=observation.label,
                    confidence=observation.confidence,
                    last_detect_frame_id=detection_frame_id,
                    last_detect_timestamp=detection_ts,
                    last_update_timestamp=current_ts,
                    measurement_bbox=list(observation.bbox),
                    measurement_timestamp=current_ts,
                    last_source=(
                        "optical_flow"
                        if observation.flow_compensated
                        else "motion_pred" if elapsed > 0 else "detector"
                    ),
                    last_flow_compensated=observation.flow_compensated,
                )
                self._record_vote(track, observation)
                self._confirm(track)
                self._tracks[track_id] = track
                self.stats["total_tracks_created"] += 1

            self._cleanup(current_ts)
            # Detection quá tuổi ngoại suy chỉ được vẽ nếu vừa warp bằng flow.
            if (
                elapsed > self.config.max_extrapolate_sec
                and not (
                    self.config.use_optical_flow
                    and source is not None
                    and self._prev_gray is not None
                )
            ):
                for track in self._tracks.values():
                    if track.last_detect_frame_id == detection_frame_id:
                        self._lose(track, timestamp=current_ts, reason="stale")

            return self._snapshot(include_tentative, position_cfg)

    def step_frame(
        self,
        current_frame: np.ndarray | None,
        *,
        frame_id: int,
        timestamp_sec: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
        include_tentative: bool = True,
        position_cfg: dict | None = None,
    ) -> list[TrackedDetection]:
        """Tiến tracker tại khung hiển thị; tua ngược > 0.1 giây sẽ reset."""
        started = time.perf_counter()
        with self._lock:
            timestamp = self._timestamp(timestamp_sec)
            seeking_back = (
                self._last_step_timestamp is not None
                and timestamp - self._last_step_timestamp < -0.1
            )
            self._advance(
                current_frame,
                frame_id,
                timestamp,
                frame_width,
                frame_height,
            )
            result = (
                []
                if seeking_back
                else self._snapshot(include_tentative, position_cfg)
            )
            self._step_count += 1
            self._step_total_ms += (time.perf_counter() - started) * 1000.0
            return result

    def pop_lost_tracks(self) -> list[dict[str, Any]]:
        """Trả bản sao các sự kiện mất dấu chưa đọc và xóa hàng đợi."""
        with self._lock:
            events = [dict(event) for event in self._lost_events]
            self._lost_events.clear()
            return events

    def get_confirmed_alerts(self) -> list[TrackedDetection]:
        """Trả track đang hiện diện, đã xác nhận và chưa phát cảnh báo."""
        with self._lock:
            return [
                detection
                for detection in self._snapshot(include_tentative=False)
                if not detection["alert_emitted"]
            ]

    def mark_alert_emitted(self, track_id: int) -> None:
        """Đánh dấu sau khi cảnh báo được chấp nhận; giữ cờ qua re-ID."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track is not None:
                track.alert_emitted = True

    def get_stats(self) -> dict[str, Any]:
        """Trả bản sao thống kê, không để lộ cấu trúc trạng thái nội bộ."""
        with self._lock:
            result = dict(self.stats)
            result.update(
                active_count=sum(
                    track.state == "active" for track in self._tracks.values()
                ),
                tentative_count=sum(
                    track.state == "tentative" for track in self._tracks.values()
                ),
                lost_count=sum(
                    track.state == "lost" for track in self._tracks.values()
                ),
                avg_step_ms=(
                    self._step_total_ms / self._step_count
                    if self._step_count
                    else 0.0
                ),
            )
            return result
