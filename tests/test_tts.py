"""Kiểm thử câu cảnh báo TTS và tích hợp với cấu hình, CSV, pipeline."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from ts_alert import pipeline
from ts_alert.db.interface import ResultStore
from ts_alert.postprocess.tts import build_speech_text
from ts_alert.types import Detection
from ts_alert.utils.io import load_classes, read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = PROJECT_ROOT / "configs" / "classes.csv"
SHARED_CONFIG_PATH = PROJECT_ROOT / "configs" / "shared.yaml"

ClassMap = dict[int, dict[str, str]]
Config = dict[str, Any]


@pytest.fixture
def real_classes() -> ClassMap:
    """Nạp dữ liệu lớp thật, độc lập với thư mục chạy pytest."""
    return load_classes(CLASSES_PATH)


@pytest.fixture
def shared_cfg() -> Config:
    """Nạp mới cấu hình chung cho mỗi ca kiểm thử."""
    return deepcopy(read_yaml(SHARED_CONFIG_PATH))


@pytest.mark.parametrize(
    ("position", "position_text"),
    [
        ("left", "bên trái"),
        ("front", "phía trước"),
        ("right", "bên phải"),
    ],
)
@pytest.mark.parametrize("tts_text", ["", "TODO_cau_doc"])
def test_build_speech_text_positions_when_tts_text_empty(
    position: str,
    position_text: str,
    tts_text: str,
) -> None:
    """Dùng tên biển và vị trí mặc định khi tên đọc rỗng hoặc chưa hoàn thiện."""
    classes = {0: {"name": "cấm rẽ trái", "tts_text": tts_text}}

    text = build_speech_text("cấm rẽ trái", position, classes)

    assert text == f"Chú ý, {position_text} có biển cấm rẽ trái"


@pytest.mark.parametrize(
    ("label", "tts_text"),
    [
        (
            "Giới hạn tốc độ (50km/h)",
            "giới hạn tốc độ năm mươi ki lô mét một giờ",
        ),
        ("Cấm rẽ trái", "cấm rẽ trái"),
    ],
)
@pytest.mark.parametrize(
    ("position", "position_text"),
    [
        ("left", "bên trái"),
        ("front", "phía trước"),
        ("right", "bên phải"),
    ],
)
def test_build_speech_text_uses_tts_text_and_includes_position(
    label: str,
    tts_text: str,
    position: str,
    position_text: str,
) -> None:
    """Giữ tên đọc tự nhiên trong tts_text và bổ sung vị trí."""
    classes = {0: {"name": label, "tts_text": tts_text}}

    text = build_speech_text(label, position, classes)

    assert text == f"Chú ý, {position_text} có biển {tts_text}"


def test_build_speech_text_avoids_duplicate_bien_prefix() -> None:
    """Không lặp từ biển khi cả template và tên đọc đều có tiền tố này."""
    label = "Biển gộp làn đường theo phương tiện"
    classes = {
        12: {
            "name": label,
            "tts_text": "biển gộp làn đường theo phương tiện",
        }
    }

    text = build_speech_text(label, "right", classes)

    assert text == "Chú ý, bên phải có biển gộp làn đường theo phương tiện"
    assert "biển biển" not in text.casefold()


@pytest.mark.parametrize(
    ("position", "position_text"),
    [
        ("left", "bên trái"),
        ("front", "phía trước"),
        ("right", "bên phải"),
    ],
)
def test_build_speech_text_all_real_classes_include_position(
    real_classes: ClassMap,
    shared_cfg: Config,
    position: str,
    position_text: str,
) -> None:
    """Kiểm tra toàn bộ 52 lớp thật ở cả ba vị trí với cấu hình YAML thật."""
    assert len(real_classes) == 52
    assert set(real_classes) == set(range(52))

    for class_id, info in real_classes.items():
        text = build_speech_text(
            info["name"],
            position,
            real_classes,
            cfg=shared_cfg,
        )
        context = f"class_id={class_id}, position={position}, text={text!r}"

        assert position_text in text, context
        assert "biển biển" not in text.casefold(), context
        assert "TODO_" not in text, context
        assert "{" not in text, context
        assert "}" not in text, context
        assert text == text.strip(), context
        assert re.search(r"\s{2,}", text) is None, context


@pytest.mark.parametrize(
    ("label", "position", "expected"),
    [
        (
            "  cấm rẽ trái  ",
            "left",
            "Chú ý, bên trái có biển không được rẽ trái",
        ),
        (
            "Biển thử nghiệm chưa có",
            "right",
            "Chú ý, bên phải có biển thử nghiệm chưa có",
        ),
        (
            "Cấm rẽ trái",
            "",
            "Chú ý, có biển không được rẽ trái",
        ),
    ],
)
def test_build_speech_text_case_insensitive_unknown_and_empty_position(
    label: str,
    position: str,
    expected: str,
) -> None:
    """Khớp nhãn linh hoạt, fallback nhãn lạ và chuẩn hóa khoảng trắng."""
    classes = {
        7: {
            "name": "Cấm rẽ trái",
            # Khác tên lớp để chứng minh đã khớp đúng bản ghi.
            "tts_text": "không được rẽ trái",
        }
    }

    text = build_speech_text(label, position, classes)

    assert text == expected
    assert text == text.strip()
    assert "  " not in text
    assert " ," not in text


@pytest.mark.parametrize(
    ("tts_text", "expected"),
    [
        (
            "cấm rẽ trái",
            "Cảnh báo: ở bên trái đường xuất hiện cấm rẽ trái.",
        ),
        (
            "Lưu ý, {position_text} có {name}.",
            "Lưu ý, ở bên trái đường có Cấm rẽ trái.",
        ),
        (
            "Lưu ý, {position} có {name}.",
            "Lưu ý, ở bên trái đường có Cấm rẽ trái.",
        ),
    ],
)
def test_build_speech_text_custom_cfg_and_placeholder_tts_text(
    tts_text: str,
    expected: str,
) -> None:
    """Áp dụng template cấu hình và template riêng chứa vị trí trong tts_text."""
    classes = {7: {"name": "Cấm rẽ trái", "tts_text": tts_text}}
    cfg: Config = {
        "postprocess": {
            "tts": {
                "template": "Cảnh báo: {position_text} xuất hiện {name}.",
                "position_text": {"left": "ở bên trái đường"},
            }
        }
    }

    text = build_speech_text("Cấm rẽ trái", "left", classes, cfg=cfg)

    assert text == expected
    assert "{" not in text
    assert "}" not in text


@pytest.mark.parametrize(
    ("custom_tts_cfg", "expected"),
    [
        (
            False,
            "Chú ý, bên trái có biển "
            "giới hạn tốc độ năm mươi ki lô mét một giờ",
        ),
        (
            True,
            "Cảnh báo: bên trái đường xuất hiện "
            "giới hạn tốc độ năm mươi ki lô mét một giờ.",
        ),
    ],
    ids=["shared-yaml", "custom-tts-config"],
)
def test_pipeline_passes_cfg_and_speaks_position_for_real_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_classes: ClassMap,
    shared_cfg: Config,
    custom_tts_cfg: bool,
    expected: str,
) -> None:
    """Chạy pipeline ảnh thật với TTS text, phân vị trí và chống lặp thật.

    Nhánh cấu hình tùy chỉnh phát hiện lỗi pipeline bỏ qua cfg khi gọi
    build_speech_text.
    """
    cfg = shared_cfg
    cfg["paths"]["classes_csv"] = str(CLASSES_PATH)

    if custom_tts_cfg:
        cfg["postprocess"]["tts"].update(
            {
                "template": "Cảnh báo: {position_text} xuất hiện {name}.",
                "position_text": {
                    "left": "bên trái đường",
                    "front": "phía trước đường",
                    "right": "bên phải đường",
                },
            }
        )

    label = "Giới hạn tốc độ (50km/h)"
    class_id = next(
        class_id
        for class_id, info in real_classes.items()
        if info["name"] == label
    )
    assert real_classes[class_id]["tts_text"] == (
        "giới hạn tốc độ năm mươi ki lô mét một giờ"
    )

    input_path = tmp_path / "traffic_sign.png"
    frame = np.zeros((240, 300, 3), dtype=np.uint8)
    pipeline.video_io.write_image(input_path, frame)
    assert input_path.is_file()

    detection: Detection = {
        "bbox": [10, 40, 50, 100],
        "label": label,
        "class_id": class_id,
        "confidence": 0.99,
    }
    detector = Mock()
    detector.predict.return_value = [detection]

    repo = Mock(spec=ResultStore)
    job_id = 123
    repo.create_job.return_value = job_id

    synthesize_mock = Mock(
        side_effect=lambda text, out_path, cfg: Path(out_path)
    )
    merge_mock = Mock()
    monkeypatch.setattr(pipeline.tts, "synthesize", synthesize_mock)
    monkeypatch.setattr(pipeline.merge_av, "merge_audio_video", merge_mock)

    result = pipeline.process_media(
        input_path=input_path,
        model_name="yolo",
        cfg=cfg,
        detector=detector,
        repo=repo,
        out_dir=tmp_path / "outputs",
    )

    assert result["status"] == "done"
    assert result["job_id"] == job_id
    assert result["unique_sign_count"] == 1
    assert result["voice_play_count"] == 1
    assert Path(result["output_path"]).is_file()
    assert len(result["audio_paths"]) == 1

    detector.predict.assert_called_once()
    synthesize_mock.assert_called_once()
    repo.add_voice_event.assert_called_once()

    spoken_text, audio_path, received_cfg = synthesize_mock.call_args.args
    assert spoken_text == expected
    assert "bên trái" in spoken_text
    assert "giới hạn tốc độ năm mươi ki lô mét một giờ" in spoken_text
    assert received_cfg is cfg

    repo.add_voice_event.assert_called_once_with(
        job_id,
        detection.get("db_id"),
        expected,
        0.0,
        str(audio_path),
    )
    assert result["audio_paths"] == [str(audio_path)]

    repo.set_status.assert_any_call(job_id, "processing")
    repo.set_status.assert_any_call(job_id, "done")
    repo.finish_job.assert_called_once()
    repo.fail_job.assert_not_called()
    merge_mock.assert_not_called()
