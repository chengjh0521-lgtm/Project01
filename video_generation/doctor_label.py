"""Independent fixed doctor-label burn-in step."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path


DEFAULT_DOCTOR_LABEL_FILE = Path(__file__).with_name("label.png")
DOCTOR_LABEL_WIDTH_PIXELS = 67


def apply_doctor_label(video_path: str | Path, label_path: str | Path | None = None) -> str:
    """Burn the doctor label above every other existing video layer."""
    source = Path(video_path).expanduser().resolve()
    label = Path(label_path).expanduser().resolve() if label_path else DEFAULT_DOCTOR_LABEL_FILE
    if not source.is_file():
        raise FileNotFoundError("Video for doctor label is missing: {}".format(source))
    if not label.is_file():
        raise FileNotFoundError("Doctor label is missing: {}".format(label))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable; cannot burn the fixed doctor label.")

    output = source.with_name("{}_label{}".format(source.stem, source.suffix))
    filter_graph = (
        "[1:v]format=rgba,scale={}:-1:flags=lanczos,setsar=1[label];"
        "[0:v][label]overlay=x=20:y=20:eof_action=pass:repeatlast=1:format=auto:alpha=straight[outv]"
    ).format(DOCTOR_LABEL_WIDTH_PIXELS)
    command = [
        ffmpeg, "-y", "-i", str(source), "-loop", "1", "-framerate", "30", "-i", str(label),
        "-filter_complex", filter_graph,
        "-map", "[outv]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output),
    ]
    logging.warning(
        "Doctor-label module started: source=%s, label=%s, output=%s, filter=%s",
        source, label, output, filter_graph,
    )
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        error = completed.stderr[-1000:]
        logging.error("Doctor-label module failed: %s", error)
        raise RuntimeError("Doctor-label overlay failed: {}".format(error))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Doctor-label overlay produced no usable output: {}".format(output))
    logging.warning("Doctor-label module completed: %s", output)
    return str(output)
