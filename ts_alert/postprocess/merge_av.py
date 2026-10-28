"""Ghép âm thanh vào video theo timestamp.   Phụ trách: Thành viên 4"""
from __future__ import annotations

from pathlib import Path
import subprocess

import imageio_ffmpeg

def merge_audio_video(video_path: Path, audio_events: list, out_path: Path) -> Path:
    """audio_events: [{"audio_path": Path, "timestamp_sec": float, "text": str}, ...].

    Đặt từng đoạn audio bắt đầu tại timestamp_sec rồi mix thành một track gắn vào video_path -> out_path.
    Gợi ý: ffmpeg (binary lấy từ imageio-ffmpeg 0.6.0) với filter adelay + amix. Trả out_path.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    video_path = Path(video_path)
    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_events:
        out_path.write_bytes(video_path.read_bytes())
        return out_path

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
    ]

    for event in audio_events:
        command.extend([
            "-i",
            str(event["audio_path"]),
        ])

    filters = []
    delayed_audio = []

    for index, event in enumerate(audio_events):
        delay_ms = max(
            0,
            int(round(event["timestamp_sec"] * 1000)),
        )

        input_index = index + 1
        label = f"a{index}"

        filters.append(
            f"[{input_index}:a]"
            f"adelay={delay_ms}:all=1"
            f"[{label}]"
        )

        delayed_audio.append(f"[{label}]")

    filters.append(
        "".join(delayed_audio)
        + f"amix=inputs={len(delayed_audio)}:"
          "duration=longest:"
          "dropout_transition=0"
        "[aout]"
    )

    command.extend([
        "-filter_complex",
        ";".join(filters),
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ])

    subprocess.run(
        command,
        check=True,
    )

    return out_path
