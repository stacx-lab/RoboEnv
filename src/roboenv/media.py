"""Simulator-independent video editing with source-frame provenance."""

import json
import math
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from roboenv.recording import file_digest, write_json


def decode(path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                    "-f", "null", "-"], check=True)
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(path),
    ], text=True))


def edit(replay, output, title, *, speed=1, fps=None):
    if (type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= 100
            or fps is not None and (type(fps) is not int or not 0 < fps <= 120)):
        raise ValueError("Invalid playback speed or frame rate")
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing video; choose a fresh output")
    playback = json.loads((replay / "replay.json").read_text())
    if playback.get("passed") is not True:
        raise ValueError("Edit only verified source media")
    media = playback["media"]
    names = [camera["file"] for camera in media["cameras"]]
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("Invalid camera manifest")
    if not {"frames.json", *names} <= media["files"].keys():
        raise ValueError("Source media hashes are incomplete")
    for name, expected in media["files"].items():
        if Path(name).name != name or file_digest(replay / name) != expected:
            raise ValueError("Source media hash mismatch")
    source_frames = json.loads((replay / "frames.json").read_text())
    source_rate = Fraction(decode(replay / names[0])["streams"][0]["r_frame_rate"])
    output_rate = Fraction(fps) if fps is not None else source_rate * Fraction(str(speed))
    if not 0 < output_rate <= 120:
        raise ValueError("Output frame rate exceeds 120 fps; select an explicit fps or lower speed")
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = [part for name in names for part in ("-i", str(replay / name))]
    stack = ("".join(f"[{i}:v]" for i in range(len(names))) + f"hstack=inputs={len(names)},"
             if len(names) > 1 else "[0:v]")
    filters = (stack + "pad=iw:ih+128:0:128:color=0x101827,"
               "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
               "textfile=title.txt:expansion=none:fontsize=25:fontcolor=white:x=28:y=22,"
               f"setpts=PTS/{speed},tpad=start_duration=1:start_mode=clone:stop_duration=2:stop_mode=clone[v]")
    with tempfile.TemporaryDirectory(prefix="media-", dir=output.parent) as temporary:
        (Path(temporary) / "title.txt").write_text(title)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", *inputs,
                        "-filter_complex", filters, "-map", "[v]", "-an", "-r", str(output_rate),
                        "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
                       cwd=temporary, check=True)
    source_fps = float(source_rate)
    mapping = {
        "speed": speed, "fps": float(output_rate), "source_fps": source_fps,
        "frame_rate_policy": "source-cadence" if fps is None else "explicit",
        "source_cadence_preserved": output_rate >= source_rate * Fraction(str(speed)),
        "title_hold_seconds": 1, "final_hold_seconds": 2,
        "waiting_policy": media["waiting_policy"],
        "playback_kind": playback["kind"], "validation_scope": playback["validation_scope"],
        "frames": [{**frame, "edited_seconds": 1 + i / (source_fps * speed)}
                   for i, frame in enumerate(source_frames)],
    }
    write_json(output.with_suffix(".frames.json"), mapping)
    receipt = {**decode(output), "sha256": file_digest(output)}
    write_json(output.with_suffix(".media.json"), receipt)
    return receipt
