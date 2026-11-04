from ts_alert.postprocess.tts import build_speech_text


def test_build_speech_text_left():
    classes = {
        0: {
            "name": "cấm rẽ trái",
            "tts_text": "",
        }
    }

    text = build_speech_text(
        "cấm rẽ trái",
        "left",
        classes,
    )

    assert text == "Chú ý, bên trái có biển cấm rẽ trái"


def test_build_speech_text_front():
    classes = {
        0: {
            "name": "dừng lại",
            "tts_text": "",
        }
    }

    text = build_speech_text(
        "dừng lại",
        "front",
        classes,
    )

    assert text == "Chú ý, phía trước có biển dừng lại"


def test_build_speech_text_custom_tts():
    classes = {
        0: {
            "name": "cấm rẽ trái",
            "tts_text": "Xin chú ý, có biển cấm rẽ trái",
        }
    }

    text = build_speech_text(
        "cấm rẽ trái",
        "left",
        classes,
    )

    assert text == "Xin chú ý, có biển cấm rẽ trái"