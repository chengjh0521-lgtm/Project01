"""Video rendering entry point backed by the upstream FunClip callback."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from funclip_loader import get_launch


def _escape_filter_path(path: Path) -> str:
    """Escape an absolute path for FFmpeg's subtitles filter."""
    return path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


_SRT_TIME_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)


def _ass_timecode(value: str) -> str:
    hours, minutes, seconds = value.strip().replace(".", ",").split(":")
    second, millis = seconds.split(",")
    return "{}:{:02d}:{:02d}.{:02d}".format(
        int(hours), int(minutes), int(second), int(millis.ljust(3, "0")[:3]) // 10
    )


def _parse_srt_cues(srt_text: str) -> list[tuple[str, str, str]]:
    lines = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    timestamps = [index for index, line in enumerate(lines) if _SRT_TIME_RE.match(line)]
    cues = []
    for position, time_index in enumerate(timestamps):
        match = _SRT_TIME_RE.match(lines[time_index])
        assert match is not None
        end_index = timestamps[position + 1] if position + 1 < len(timestamps) else len(lines)
        if end_index > time_index + 1 and re.match(r"^\s*\d+\s*$", lines[end_index - 1]):
            end_index -= 1
        text = "\n".join(line.strip() for line in lines[time_index + 1 : end_index] if line.strip())
        if text:
            cues.append((match.group("start"), match.group("end"), text))
    return cues


def _parse_keywords(keywords: str | None) -> list[str]:
    values = []
    for value in re.split(r"[\n,，;；、]+", str(keywords or "")):
        keyword = re.sub(r"^\s*\d+\s*[.、)]\s*", "", value).strip(" \t\"'“”")
        if keyword and keyword not in values:
            values.append(keyword)
    return values


def _highlight_ass_text(text: str, keywords: list[str]) -> str:
    escaped = (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )
    if not keywords:
        return escaped
    pattern = re.compile("|".join(re.escape(keyword) for keyword in sorted(keywords, key=len, reverse=True)))
    return pattern.sub(lambda match: r"{\c&H0000FFFF&}" + match.group(0) + r"{\c&H00FFFFFF&}", escaped)


def _write_ass_subtitles(clip_srt: str, ass_file: Path, keywords: list[str]) -> int:
    cues = _parse_srt_cues(clip_srt)
    if not cues:
        raise RuntimeError("高光剪辑没有生成可烧录的字幕 SRT。")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,STHeiti,48,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,48,48,64,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = [
        "Dialogue: 0,{},{},Default,,0,0,0,,{}".format(
            _ass_timecode(start), _ass_timecode(end), _highlight_ass_text(text, keywords)
        )
        for start, end, text in cues
    ]
    ass_file.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(cues)


def _burn_srt_with_ffmpeg(video_path: str | Path, clip_srt: str, keywords: str | None = None) -> str:
    if not str(clip_srt or "").strip():
        raise RuntimeError("高光剪辑没有生成可烧录的字幕 SRT。")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法烧录字幕。")

    source = Path(video_path).resolve()
    subtitle_file = source.with_name("{}_captions.ass".format(source.stem))
    output = source.with_name("{}_captioned{}".format(source.stem, source.suffix))
    highlight_keywords = _parse_keywords(keywords)
    cue_count = _write_ass_subtitles(clip_srt, subtitle_file, highlight_keywords)

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
    logging.warning(
        "使用 FFmpeg 烧录字幕：字幕=%d 条，黄色关键词=%d 个，命令=%s",
        cue_count,
        len(highlight_keywords),
        " ".join(command),
    )
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
        keywords: str | None = None,
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
        captioned_video = _burn_srt_with_ffmpeg(video, clip_srt, keywords)
        return captioned_video, audio, "{}; burned subtitles via FFmpeg".format(message), clip_srt
    return launch.AI_clip(
        llm_result, "", "", start_offset_ms, end_offset_ms,
        video_state, None, str(output_dir),
    )
