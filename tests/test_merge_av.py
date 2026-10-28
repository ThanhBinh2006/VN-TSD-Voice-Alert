from pathlib import Path

import ts_alert.postprocess.merge_av as merge_av


def test_merge_audio_video_without_audio(tmp_path):
    video_path = tmp_path / "input.mp4"
    out_path = tmp_path / "output.mp4"

    video_path.write_bytes(b"fake video")

    result = merge_av.merge_audio_video(
        video_path,
        [],
        out_path,
    )

    assert result == out_path
    assert out_path.read_bytes() == b"fake video"


def test_merge_audio_video_builds_delayed_audio_command(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "input.mp4"
    audio_path = tmp_path / "alert.mp3"
    out_path = tmp_path / "output.mp4"

    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")

    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(
        merge_av.subprocess,
        "run",
        fake_run,
    )

    monkeypatch.setattr(
        merge_av.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: "fake-ffmpeg",
    )

    result = merge_av.merge_audio_video(
        video_path,
        [
            {
                "audio_path": audio_path,
                "timestamp_sec": 2.5,
                "text": "cảnh báo",
            }
        ],
        out_path,
    )

    command = captured["command"]

    assert result == out_path
    assert captured["check"] is True

    assert command[0] == "fake-ffmpeg"
    assert "-i" in command
    assert str(video_path) in command
    assert str(audio_path) in command

    filter_complex = command[
        command.index("-filter_complex") + 1
    ]

    assert "[1:a]adelay=2500:all=1[a0]" in filter_complex
    assert "amix=inputs=1" in filter_complex

    assert "0:v:0" in command
    assert "[aout]" in command
    assert "-c:v" in command
    assert "copy" in command
    assert "-c:a" in command
    assert "aac" in command