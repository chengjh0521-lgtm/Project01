"""Video rendering entry point backed by the upstream FunClip callback."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from funclip_loader import get_launch


def _escape_filter_path(path: Path) -> str:
    """Escape an absolute path for FFmpeg's subtitles filter."""
    return path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _burn_srt_with_ffmpeg(video_path: str | Path, clip_srt: str) -> str:
    if not str(clip_srt or "").strip():
        raise RuntimeError("高光剪辑没有生成可烧录的字幕 SRT。")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法烧录字幕。")

    source = Path(video_path).resolve()
    subtitle_file = source.with_name("{}_captions.srt".format(source.stem))
    output = source.with_name("{}_captioned{}".format(source.stem, source.suffix))
    subtitle_file.write_text(str(clip_srt), encoding="utf-8")

    filter_parts = ["subtitles=filename={}".format(_escape_filter_path(subtitle_file)), "charenc=UTF-8"]
    launch_dir = os.environ.get("FUNCLIP_LAUNCH_DIR")
    if launch_dir:
        font_dir = Path(launch_dir).resolve().parent / "font"
        if font_dir.is_dir():
            filter_parts.append("fontsdir={}".format(_escape_filter_path(font_dir)))
    subtitle_filter = ":".join(filter_parts)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    logging.warning("使用 FFmpeg 烧录字幕：%s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        raise RuntimeError("FFmpeg 字幕烧录失败：{}".format(completed.stderr[-2000:]))
    logging.warning("FFmpeg 字幕烧录完成：%s", output)
    return str(output)


def render_highlight_video(
        llm_result: str,
        video_state,
        output_dir: str | Path = "",
        burn_subtitles: bool = True,
        start_offset_ms: int = 0,
        end_offset_ms: int = 100):
    """Render LLM timestamp ranges. Returns video, audio, message, and clip SRT."""
    launch = get_launch()
    if burn_subtitles:
        # Use the upstream clip callback for timestamps/concatenation, then burn
        # its returned SRT with FFmpeg. This preserves aspect ratio and avoids
        # ImageMagick/TextClip font-path failures in imported launch.py.
        video, audio, message, clip_srt = launch.AI_clip(
            llm_result, "", "", start_offset_ms, end_offset_ms,
            video_state, None, str(output_dir),
        )
        if video is None:
            return video, audio, message, clip_srt
        captioned_video = _burn_srt_with_ffmpeg(video, clip_srt)
        return captioned_video, audio, "{}; burned subtitles via FFmpeg".format(message), clip_srt
    return launch.AI_clip(
        llm_result, "", "", start_offset_ms, end_offset_ms,
        video_state, None, str(output_dir),
    )
