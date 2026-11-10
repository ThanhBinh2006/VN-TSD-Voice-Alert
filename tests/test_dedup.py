from ts_alert.postprocess.dedup import Deduplicator


def make_cfg(iou_match=0.3, lost_frames=15):
    return {
        "postprocess": {
            "dedup": {
                "iou_match": iou_match,
                "lost_frames": lost_frames,
            }
        }
    }


def make_detection(
    label="stop",
    bbox=None,
    position="front",
    confidence=0.95,
):
    return {
        "bbox": bbox or [100, 100, 200, 200],
        "label": label,
        "class_id": 0,
        "confidence": confidence,
        "position": position,
    }


def test_dedup():
    """Deduplicator: cùng biển xuất hiện 30 khung liên tiếp chỉ sinh 1 SpeechEvent; vắng > lost_frames rồi xuất hiện lại -> sự kiện mới."""
    deduper = Deduplicator(make_cfg())

    detection = make_detection()

    events = []

    for frame_index in range(30):
        frame_events = deduper.update(
            [detection],
            frame_index=frame_index,
            timestamp_sec=frame_index / 30,
        )
        events.extend(frame_events)

    assert len(events) == 1
    assert events[0]["label"] == "stop"
    assert events[0]["position"] == "front"
    assert events[0]["frame_index"] == 0


def test_dedup_new_event_after_lost_frames():
    deduper = Deduplicator(make_cfg(lost_frames=15))

    detection = make_detection()

    first_events = deduper.update(
        [detection],
        frame_index=0,
        timestamp_sec=0.0,
    )

    assert len(first_events) == 1

    # 16 - 0 = 16 > lost_frames (15)
    # => track cũ đã hết hạn.
    second_events = deduper.update(
        [detection],
        frame_index=16,
        timestamp_sec=16 / 30,
    )

    assert len(second_events) == 1
    assert second_events[0]["frame_index"] == 16


def test_dedup_different_label_is_new_event():
    deduper = Deduplicator(make_cfg())

    first = make_detection(label="stop")

    second = make_detection(label="no_left")

    assert len(
        deduper.update(first and [first], 0, 0.0)
    ) == 1

    assert len(
        deduper.update([second], 1, 1 / 30)
    ) == 1


def test_dedup_respects_frame_stride_larger_than_lost_frames() -> None:
    """Giữ track qua mỗi stride 30 frame và tạo sự kiện mới khi vắng quá 30 frame."""
    cfg = make_cfg(lost_frames=15)
    cfg["inference"] = {"frame_stride": 30}
    deduper = Deduplicator(cfg)
    detection = make_detection()

    first_events = deduper.update(
        [detection],
        frame_index=0,
        timestamp_sec=0.0,
    )

    assert len(first_events) == 1
    assert first_events[0]["frame_index"] == 0

    for frame_index in (30, 60):
        assert deduper.update(
            [detection],
            frame_index=frame_index,
            timestamp_sec=frame_index / 30,
        ) == []

    assert deduper.update(
        [],
        frame_index=90,
        timestamp_sec=90 / 30,
    ) == []

    # 120 - 60 = 60 > max(lost_frames, frame_stride) = 30.
    second_events = deduper.update(
        [detection],
        frame_index=120,
        timestamp_sec=120 / 30,
    )

    assert len(second_events) == 1
    assert second_events[0]["frame_index"] == 120
