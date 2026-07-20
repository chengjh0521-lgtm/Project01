"""Create a short static question card with an Edge TTS voice-over."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_QUESTION_BACKGROUND = Path(__file__).with_name("question_intro_background.png")
DEFAULT_TTS_VOICE = "zh-CN-YunxiNeural"
DEFAULT_TTS_RATE = "+35%"
FAST_TTS_RATE = "+60%"
MAX_QUESTION_INTRO_SECONDS = 3.0


def question_intro_background_path() -> Path:
    configured = os.environ.get("FUNCLIP_QUESTION_INTRO_BACKGROUND")
    return Path(configured).expanduser() if configured else DEFAULT_QUESTION_BACKGROUND


def _synthesize_question_audio(question: str, audio_path: Path, voice: str, rate: str) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed. Install requirements-web.txt first.") from exc

    async def save_audio() -> None:
        communicator = edge_tts.Communicate(question, voice, rate=rate)
        await communicator.save(str(audio_path))

    asyncio.run(save_audio())


def _audio_duration_seconds(audio_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is unavailable; install FFmpeg before creating a question intro.")
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError("Could not read question audio duration: {}".format(completed.stderr[-500:]))
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Question audio has no usable duration.") from exc


def create_question_intro(
        question: str,
        *,
        background_path: str | Path | None = None,
        output_path: str | Path | None = None,
        width: int = 1080,
        height: int = 1920,
        voice: str = DEFAULT_TTS_VOICE,
) -> str:
    """Create an H.264/AAC static question intro no longer than three seconds."""
    text = "".join(str(question or "").split())
    if not text:
        raise ValueError("Question text is required.")
    if width < 2 or height < 2:
        raise ValueError("Question intro width and height must be positive.")

    background = Path(background_path).expanduser() if background_path else question_intro_background_path()
    background = background.resolve()
    if not background.is_file():
        raise FileNotFoundError("Question intro background is missing: {}".format(background))

    destination = Path(output_path).expanduser() if output_path else background.with_name("question_intro.mp4")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio_path = destination.with_suffix(".mp3")

    _synthesize_question_audio(text, audio_path, voice, DEFAULT_TTS_RATE)
    duration = _audio_duration_seconds(audio_path)
    if duration > MAX_QUESTION_INTRO_SECONDS:
        _synthesize_question_audio(text, audio_path, voice, FAST_TTS_RATE)
        duration = _audio_duration_seconds(audio_path)
    if duration > MAX_QUESTION_INTRO_SECONDS:
        raise ValueError(
            "Question voice is {:.2f}s, above the {:.1f}s limit. Use a shorter question.".format(
                duration, MAX_QUESTION_INTRO_SECONDS
            )
        )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable; cannot create a question intro.")
    visual_filter = (
        "scale={}:{}:force_original_aspect_ratio=decrease,"
        "pad={}:{}:(ow-iw)/2:(oh-ih):color=black,format=yuv420p"
    ).format(width, height, width, height)
    command = [
        ffmpeg, "-y", "-loop", "1", "-framerate", "30", "-i", str(background), "-i", str(audio_path),
        "-filter:v", visual_filter, "-map", "0:v:0", "-map", "1:a:0",
        "-t", "{:.3f}".format(duration), "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(destination),
    ]
    logging.warning(
        "Question-intro prototype started: background=%s, duration=%.3fs, output=%s",
        background, duration, destination,
    )
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError("Question intro FFmpeg render failed: {}".format(completed.stderr[-1000:]))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Question intro render produced no usable output: {}".format(destination))
    logging.warning("Question-intro prototype completed: %s", destination)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a static question video with Edge TTS narration.")
    parser.add_argument("--question", required=True, help="Question to narrate. Keep it brief enough for three seconds.")
    parser.add_argument("--background", default=None, help="Background image; defaults to FUNCLIP_QUESTION_INTRO_BACKGROUND.")
    parser.add_argument("--output", default=None, help="Output MP4 path.")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--voice", default=DEFAULT_TTS_VOICE)
    args = parser.parse_args()
    print(create_question_intro(
        args.question,
        background_path=args.background,
        output_path=args.output,
        width=args.width,
        height=args.height,
        voice=args.voice,
    ))


if __name__ == "__main__":
    main()
