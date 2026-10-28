from ts_alert.postprocess.position import add_positions, get_position


def make_cfg(left_max=0.33, right_min=0.66):
    return {
        "postprocess": {
            "position": {
                "left_max": left_max,
                "right_min": right_min,
            }
        }
    }


def make_bbox(center_x, width=20):
    return [
        center_x - width / 2,
        100,
        center_x + width / 2,
        200,
    ]


def test_position():
    """get_position: tâm bbox ở 0-33% -> left, 33-66% -> front, 66-100% -> right"""
    cfg = make_cfg()
    frame_width = 1000

    assert get_position(make_bbox(100), frame_width, cfg) == "left"
    assert get_position(make_bbox(500), frame_width, cfg) == "front"
    assert get_position(make_bbox(800), frame_width, cfg) == "right"


def test_position_boundary():
    cfg = make_cfg()
    frame_width = 1000

    assert get_position(make_bbox(329), frame_width, cfg) == "left"
    assert get_position(make_bbox(330), frame_width, cfg) == "front"
    assert get_position(make_bbox(659), frame_width, cfg) == "front"
    assert get_position(make_bbox(660), frame_width, cfg) == "right"


def test_add_positions():
    cfg = make_cfg()
    frame_width = 1000

    detections = [
        {
            "bbox": make_bbox(100),
            "label": "stop",
            "class_id": 0,
            "confidence": 0.95,
        },
        {
            "bbox": make_bbox(500),
            "label": "speed_limit",
            "class_id": 1,
            "confidence": 0.90,
        },
        {
            "bbox": make_bbox(800),
            "label": "turn_right",
            "class_id": 2,
            "confidence": 0.88,
        },
    ]

    result = add_positions(detections, frame_width, cfg)

    assert result[0]["position"] == "left"
    assert result[1]["position"] == "front"
    assert result[2]["position"] == "right"
