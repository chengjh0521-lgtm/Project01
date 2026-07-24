"""Create a short static question card with an Edge TTS voice-over."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from video_generation.font_config import subtitle_fonts_directory, unified_font_family


DEFAULT_QUESTION_BACKGROUND = Path(__file__).with_name("question_intro_background.png")
DEFAULT_COVER_BACKGROUND = Path(__file__).with_name("cover_background.png")
DEFAULT_TTS_VOICE = "zh-CN-YunxiNeural"
DEFAULT_TTS_RATE = "+35%"
FAST_TTS_RATE = "+60%"
MAX_QUESTION_INTRO_SECONDS = 3.0
COVER_FRAME_SECONDS = 1 / 30
# The intro question uses a deliberately large font. Seven Chinese characters
# fit the approved layout; trailing punctuation is handled separately so it
# cannot become an orphaned line on narrow portrait videos.
MAX_QUESTION_LINE_CHARACTERS = 7
_QUESTION_TRAILING_PUNCTUATION = "？?！!。"
QUESTION_TEXT_ASS_COLOR = "&H0000FFFF"


def question_intro_background_path() -> Path:
    configured = os.environ.get("FUNCLIP_QUESTION_INTRO_BACKGROUND")
    return Path(configured).expanduser() if configured else DEFAULT_QUESTION_BACKGROUND


def cover_background_path() -> Path:
    """Return the reserved cover image path without requiring it to exist yet."""
    configured = os.environ.get("FUNCLIP_COVER_BACKGROUND")
    return Path(configured).expanduser() if configured else DEFAULT_COVER_BACKGROUND


def _escape_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _wrap_question_text(question: str) -> str:
    lines, remaining = [], "".join(str(question or "").split())
    while len(remaining) > MAX_QUESTION_LINE_CHARACTERS:
        candidates = [
            index + 1 for index, char in enumerate(remaining[:MAX_QUESTION_LINE_CHARACTERS])
            if char in "，、；：,;:"
        ]
        split_at = candidates[-1] if candidates else MAX_QUESTION_LINE_CHARACTERS
        # Never allow a trailing question mark or other terminal punctuation
        # to become an orphaned third line.
        if len(remaining) > split_at and remaining[split_at] in _QUESTION_TRAILING_PUNCTUATION:
            split_at += 1
        lines.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        if lines and len(remaining) == 1 and remaining in _QUESTION_TRAILING_PUNCTUATION:
            lines[-1] += remaining
        else:
            lines.append(remaining)
    return "\\N".join(lines)


def _display_question_text(question: str, question_lines: list[str] | None = None) -> str:
    lines = [
        "".join(str(line or "").split())
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        for line in (question_lines or [])
    ]
    if len(lines) == 2 and all(lines) and all(len(line) <= MAX_QUESTION_LINE_CHARACTERS for line in lines):
        return "\\N".join(lines)
    return _wrap_question_text(question)


def _write_question_ass(
        question: str, ass_path: Path, width: int, height: int,
        question_lines: list[str] | None = None) -> None:
    base_font_size = max(44, min(80, round(min(width, height) * 0.07)))
    font_size = base_font_size * 2
    top_margin = round(height * 0.70)
    escaped = question.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Question,{font_family},{font_size},{text_color},&H00000000,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,2,1,8,80,80,{top_margin},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.00,0:00:03.00,Question,,0,0,0,,{text}
""".format(
        width=width,
        height=height,
        font_family=unified_font_family(),
        font_size=font_size,
        text_color=QUESTION_TEXT_ASS_COLOR,
        top_margin=top_margin,
        text=_display_question_text(escaped, question_lines),
    )
    ass_path.write_text(header, encoding="utf-8")


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


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is unavailable; install FFmpeg before adding a question intro.")
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(video_path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError("Could not read source video dimensions: {}".format(completed.stderr[-500:]))
    try:
        width, height = (int(value) for value in completed.stdout.strip().split("x", 1))
    except ValueError as exc:
        raise RuntimeError("Source video has no usable dimensions.") from exc
    return width, height


def create_question_intro(
        question: str,
        *,
        question_lines: list[str] | None = None,
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
    subtitle_path = destination.with_suffix(".ass")
    _write_question_ass(text, subtitle_path, width, height, question_lines)

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
    subtitle_filter = "subtitles=filename={}:charenc=UTF-8".format(_escape_filter_path(subtitle_path))
    font_dir = subtitle_fonts_directory()
    if font_dir:
        subtitle_filter += ":fontsdir={}".format(_escape_filter_path(font_dir))
    command = [
        ffmpeg, "-y", "-loop", "1", "-framerate", "30", "-i", str(background), "-i", str(audio_path),
        "-filter:v", "{},{}".format(visual_filter, subtitle_filter), "-map", "0:v:0", "-map", "1:a:0",
        "-t", "{:.3f}".format(duration), "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
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


def create_title_cover_frame(
        question: str,
        *,
        question_lines: list[str] | None = None,
        background_path: str | Path | None = None,
        output_path: str | Path | None = None,
        width: int = 1080,
        height: int = 1920,
) -> str:
    """Create a one-frame silent title cover before the narrated question card.

    ``video_generation/cover_background.png`` is intentionally the default
    slot for the future cover artwork. Until that file is provided, a neutral
    canvas keeps the four-part composition renderable.
    """
    text = "".join(str(question or "").split())
    if not text:
        raise ValueError("Cover title is required.")
    if width < 2 or height < 2:
        raise ValueError("Cover frame dimensions must be positive.")

    background = Path(background_path).expanduser() if background_path else cover_background_path()
    destination = Path(output_path).expanduser() if output_path else background.with_name("title_cover.mp4")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = destination.with_suffix(".ass")
    _write_question_ass(text, subtitle_path, width, height, question_lines)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable; cannot create a title cover.")
    subtitle_filter = "subtitles=filename={}:charenc=UTF-8".format(_escape_filter_path(subtitle_path))
    font_dir = subtitle_fonts_directory()
    if font_dir:
        subtitle_filter += ":fontsdir={}".format(_escape_filter_path(font_dir))
    visual_filter = (
        "scale={}:{}:force_original_aspect_ratio=decrease,"
        "pad={}:{}:(ow-iw)/2:(oh-ih):color=0xf3f3f3,format=yuv420p,{}"
    ).format(width, height, width, height, subtitle_filter)
    if background.is_file():
        inputs = ["-loop", "1", "-framerate", "30", "-i", str(background.resolve())]
    else:
        logging.warning("Cover background is not installed yet; using a neutral placeholder: %s", background)
        inputs = ["-f", "lavfi", "-i", "color=c=0xf3f3f3:s={}x{}:r=30".format(width, height)]
    command = [
        ffmpeg, "-y", *inputs,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter:v", visual_filter, "-map", "0:v:0", "-map", "1:a:0",
        "-frames:v", "1", "-t", "{:.6f}".format(COVER_FRAME_SECONDS), "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError("Title cover FFmpeg render failed: {}".format(completed.stderr[-1000:]))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Title cover render produced no usable output: {}".format(destination))
    logging.warning("Title cover frame completed: %s", destination)
    return str(destination)


def concat_video_segments(video_paths: list[str | Path], output_path: str | Path) -> str:
    """Normalize and concatenate ordered H.264/AAC sections without SAR drift."""
    sources = [Path(path).expanduser().resolve() for path in video_paths]
    if len(sources) < 2:
        raise ValueError("At least two video sections are required for concatenation.")
    if any(not source.is_file() for source in sources):
        missing = next(source for source in sources if not source.is_file())
        raise FileNotFoundError("Video section is missing: {}".format(missing))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable; cannot concatenate video sections.")

    width, height = _video_dimensions(sources[0])
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    video_filter = "fps=30,scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih):color=black,format=yuv420p,setsar=1"
    audio_filter = "aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo"
    filters, concat_inputs = [], []
    for index in range(len(sources)):
        filters.append("[{}:v]{}[v{}]".format(index, video_filter.format(width, height, width, height), index))
        filters.append("[{}:a]{}[a{}]".format(index, audio_filter, index))
        concat_inputs.append("[v{}][a{}]".format(index, index))
    filters.append("{}concat=n={}:v=1:a=1[outv][outa]".format("".join(concat_inputs), len(sources)))
    command = [ffmpeg, "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
    ])
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError("Video section concat failed: {}".format(completed.stderr[-1000:]))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Video section concat produced no usable output: {}".format(destination))
    return str(destination)


def prepend_question_intro(
        video_path: str | Path, question: str, question_lines: list[str] | None = None) -> str:
    """Create and prepend a narrated question card, keeping the main video's dimensions."""
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Video for question intro is missing: {}".format(source))
    width, height = _video_dimensions(source)
    intro = create_question_intro(
        question,
        question_lines=question_lines,
        output_path=source.with_name("{}_question_intro.mp4".format(source.stem)),
        width=width,
        height=height,
    )
    output = source.with_name("{}_with_question_intro{}".format(source.stem, source.suffix))
    logging.warning("Prepending question intro: source=%s, question=%s, output=%s", source, question, output)
    concat_video_segments([intro, source], output)
    logging.warning("Question intro prepended: %s", output)
    return str(output)


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
