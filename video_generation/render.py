"""Video rendering entry point backed by the upstream FunClip callback."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from funclip_loader import get_launch
from subtitle_processing.sound_effect_binding import resolve_sound_effect_file
from subtitle_processing.visual_asset_binding import get_visual_asset_definition, resolve_visual_asset_file


_DOCTOR_LABEL_FILE = Path(__file__).with_name("label.png")
_DOCTOR_LABEL_WIDTH_PIXELS = 200


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


def _time_to_ms(value: str) -> int:
    hours, minutes, seconds = value.strip().replace(".", ",").split(":")
    second, millis = seconds.split(",")
    return (int(hours) * 3_600_000 + int(minutes) * 60_000 + int(second) * 1_000 + int(millis.ljust(3, "0")[:3]))


def _parse_sound_bindings(value: str | None) -> list[dict[str, str]]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return []
    cues = payload.get("cues", []) if isinstance(payload, dict) else []
    return [item for item in cues if isinstance(item, dict) and isinstance(item.get("sound_id"), str)]


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(max(0, int(milliseconds)), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, millis)


def _sound_effect_events(clip_srt: str, sound_bindings: str | None) -> list[dict]:
    bindings = _parse_sound_bindings(sound_bindings)
    events = []
    for start, end, subtitle_text in _parse_srt_cues(clip_srt):
        start_ms, end_ms = _time_to_ms(start), _time_to_ms(end)
        compact = subtitle_text.replace("\n", "")
        for cue in bindings:
            target_word, effect_name = str(cue.get("text", "")), cue["sound_id"]
            index = compact.find(target_word)
            effect_file = resolve_sound_effect_file(effect_name)
            if index < 0 or effect_file is None:
                continue
            offset = start_ms + round((end_ms - start_ms) * index / max(1, len(compact)))
            events.append({
                "offset_ms": offset,
                "timestamp": _format_timestamp(offset),
                "sound_id": effect_name,
                "sound_file": effect_file.name,
                "target_word": target_word,
                "subtitle": subtitle_text,
                "reason": str(cue.get("reason", "")),
                "sentence_id": cue.get("sentence_id"),
            })
    return events


def describe_sound_effect_events(clip_srt: str, sound_bindings: str | None) -> list[dict]:
    """Return the exact serializable sound-effect placements used by FFmpeg."""
    return _sound_effect_events(clip_srt, sound_bindings)


def _parse_visual_bindings(value: str | None) -> list[dict]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return []
    placements = payload.get("placements", []) if isinstance(payload, dict) else []
    return [item for item in placements if isinstance(item, dict) and isinstance(item.get("asset_id"), str)]


def _visual_asset_duration_seconds(definition: dict, requested: object) -> float:
    """Preserve each asset's configured minimum while honoring the LLM's semantic duration."""
    default = 3.0 if definition.get("media_type") == "animated_gif" else 0.2
    try:
        minimum = float(definition.get("duration_seconds", default))
    except (TypeError, ValueError):
        minimum = default
    try:
        duration = float(requested)
    except (TypeError, ValueError):
        duration = minimum
    return max(max(0.04, minimum), min(12.0, duration))


def _visual_asset_events(clip_srt: str, visual_bindings: str | None) -> list[dict]:
    bindings = _parse_visual_bindings(visual_bindings)
    events = []
    for sentence_id, (start, end, subtitle_text) in enumerate(_parse_srt_cues(clip_srt), start=1):
        start_ms, end_ms = _time_to_ms(start), _time_to_ms(end)
        compact = subtitle_text.replace("\n", "")
        for placement in bindings:
            bound_sentence_id = placement.get("sentence_id")
            if isinstance(bound_sentence_id, str) and bound_sentence_id.isdigit():
                bound_sentence_id = int(bound_sentence_id)
            if bound_sentence_id != sentence_id:
                continue
            target_word, asset_id = str(placement.get("target_word", "")), placement["asset_id"]
            word_index = compact.find(target_word)
            asset_file, definition = resolve_visual_asset_file(asset_id), get_visual_asset_definition(asset_id)
            if word_index < 0 or asset_file is None or not definition:
                continue
            duration = _visual_asset_duration_seconds(definition, placement.get("duration_seconds"))
            offset = start_ms + round((end_ms - start_ms) * word_index / max(1, len(compact)))
            technical = definition.get("technical_metadata") if isinstance(definition.get("technical_metadata"), dict) else {}
            events.append({
                "offset_ms": offset,
                "timestamp": _format_timestamp(offset),
                "duration_seconds": duration,
                "asset_id": asset_id,
                "asset_file": asset_file.name,
                "media_type": definition.get("media_type", "image"),
                "requires_chroma_key": bool(technical.get("requires_chroma_key")),
                "position": placement.get("position", "upper_right"),
                "target_word": target_word,
                "subtitle": subtitle_text,
                "reason": str(placement.get("reason", "")),
                "sentence_id": sentence_id,
            })
    return events


def describe_visual_asset_events(clip_srt: str, visual_bindings: str | None) -> list[dict]:
    """Return serializable GIF/PNG placements using the same timing as rendering."""
    return _visual_asset_events(clip_srt, visual_bindings)


def _visual_position(position: str) -> tuple[str, str]:
    positions = {
        "upper_left": ("36", "72"),
        "upper_right": ("W-w-36", "72"),
        "top_center": ("(W-w)/2", "72"),
        "middle_left": ("36", "H*0.40-h/2"),
        "middle_right": ("W-w-36", "H*0.40-h/2"),
    }
    return positions.get(position, positions["upper_right"])


def _overlay_visual_assets(video_path: str | Path, clip_srt: str, visual_bindings: str | None) -> tuple[str, int]:
    events = _visual_asset_events(clip_srt, visual_bindings)
    if not events:
        logging.warning(
            "Visual overlay skipped: placements=%d, matched render events=0.",
            len(_parse_visual_bindings(visual_bindings)),
        )
        return str(video_path), 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logging.warning("FFmpeg is unavailable; skipping %d visual assets.", len(events))
        return str(video_path), 0
    source = Path(video_path).resolve()
    output = source.with_name("{}_visual{}".format(source.stem, source.suffix))
    command = [ffmpeg, "-y", "-i", str(source)]
    filters, previous = [], "0:v"
    for index, event in enumerate(events, start=1):
        asset_file = resolve_visual_asset_file(str(event["asset_id"]))
        if asset_file is None:
            continue
        duration = float(event["duration_seconds"])
        if event["media_type"] == "animated_gif":
            command.extend(["-stream_loop", "-1", "-i", str(asset_file)])
        else:
            command.extend(["-loop", "1", "-t", "{:.3f}".format(duration), "-i", str(asset_file)])
        asset_label, output_label = "asset{}".format(index), "visual{}".format(index)
        asset_filter = "[{}:v]fps=15,scale=260:-1,format=rgba".format(index)
        if event["requires_chroma_key"]:
            asset_filter += ",chromakey=0x00FF00:0.16:0.08"
        asset_filter += ",trim=duration={:.3f},setpts=PTS-STARTPTS+{:.3f}/TB[{}]".format(
            duration, int(event["offset_ms"]) / 1000, asset_label
        )
        x, y = _visual_position(str(event["position"]))
        filters.extend([
            asset_filter,
            "[{}][{}]overlay=x={}:y={}:eof_action=pass:shortest=0[{}]".format(
                previous, asset_label, x, y, output_label
            ),
        ])
        previous = output_label
    if previous == "0:v":
        return str(video_path), 0
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[{}]".format(previous), "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy",
        "-movflags", "+faststart", str(output),
    ])
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        logging.warning("Visual-asset overlay failed; returning captioned video: %s", completed.stderr[-1000:])
        return str(video_path), 0
    logging.warning("Overlay %d GIF/PNG visual assets into %s", len(events), output)
    return str(output), len(events)


def _overlay_doctor_label(video_path: str | Path) -> tuple[str, bool]:
    """Keep the doctor label visible for the full rendered clip."""
    if not _DOCTOR_LABEL_FILE.is_file():
        raise FileNotFoundError("Doctor label is missing: {}".format(_DOCTOR_LABEL_FILE))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is unavailable; cannot burn the fixed doctor label.")

    source = Path(video_path).resolve()
    output = source.with_name("{}_label{}".format(source.stem, source.suffix))
    # The label is deliberately the final video filter. The looped image input
    # keeps it present for every frame of the rendered clip.
    filter_graph = (
        "[1:v]scale={}:-1,format=rgba[label];"
        "[0:v][label]overlay=x=20:y=20:eof_action=pass:repeatlast=1[outv]"
    ).format(_DOCTOR_LABEL_WIDTH_PIXELS)
    command = [
        ffmpeg, "-y", "-i", str(source), "-loop", "1", "-framerate", "30", "-i", str(_DOCTOR_LABEL_FILE),
        "-filter_complex", filter_graph,
        "-map", "[outv]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(output),
    ]
    logging.warning(
        "Burning fixed doctor label: label=%s, width=%dpx, output=%s, filter=%s",
        _DOCTOR_LABEL_FILE, _DOCTOR_LABEL_WIDTH_PIXELS, output, filter_graph,
    )
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        error = completed.stderr[-1000:]
        logging.error("Doctor-label overlay failed: %s", error)
        raise RuntimeError("Doctor-label overlay failed: {}".format(error))
    logging.warning("Fixed doctor label overlaid for the full clip: %s", output)
    return str(output), True


def _mix_sound_effects(video_path: str | Path, clip_srt: str, sound_bindings: str | None) -> tuple[str, int]:
    events = _sound_effect_events(clip_srt, sound_bindings)
    if not events:
        return str(video_path), 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logging.warning("FFmpeg is unavailable; skipping %d sound effects.", len(events))
        return str(video_path), 0
    source = Path(video_path).resolve()
    output = source.with_name("{}_sfx{}".format(source.stem, source.suffix))
    command = [ffmpeg, "-y", "-i", str(source)]
    filters, mix_inputs = [], ["[0:a]"]
    for index, event in enumerate(events, start=1):
        effect_file = resolve_sound_effect_file(str(event["sound_id"]))
        if effect_file is None:
            continue
        command.extend(["-i", str(effect_file)])
        label = "sfx{}".format(index)
        filters.append("[{}:a]adelay={}:all=1,volume=0.80[{}]".format(index, max(0, int(event["offset_ms"])), label))
        mix_inputs.append("[{}]".format(label))
    filters.append("{}amix=inputs={}:duration=first:normalize=0[aout]".format("".join(mix_inputs), len(mix_inputs)))
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ])
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        logging.warning("Sound-effect mixing failed; returning captioned video: %s", completed.stderr[-1000:])
        return str(video_path), 0
    logging.warning("Mixed %d sound effects into %s", len(events), output)
    return str(output), len(events)


def _caption_font_size(text: str) -> int:
    longest_line = max((len(line) for line in text.split("\n")), default=0)
    if longest_line > 30:
        return 28
    if longest_line > 24:
        return 34
    if longest_line > 20:
        return 40
    return 48


def _wrap_caption_two_lines(text: str) -> str:
    """Split a long cue into two balanced visual lines without changing timing."""
    compact = "".join(part.strip() for part in text.splitlines())
    if len(compact) <= 20:
        return compact
    midpoint = len(compact) // 2
    candidates = [
        index for index, char in enumerate(compact)
        if char in "，。！？；：、,.!?;: " and 6 <= index <= len(compact) - 6
    ]
    if candidates:
        split_at = min(candidates, key=lambda index: abs(index - midpoint)) + 1
    else:
        split_at = midpoint
    return "{}\n{}".format(compact[:split_at].rstrip(), compact[split_at:].lstrip())


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
        "Dialogue: 0,{},{},Default,,0,0,0,,{{\\fs{}}}{}".format(
            _ass_timecode(start),
            _ass_timecode(end),
            _caption_font_size(_wrap_caption_two_lines(text)),
            _highlight_ass_text(_wrap_caption_two_lines(text), keywords),
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
        sound_bindings: str | None = None,
        visual_bindings: str | None = None,
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
        visual_video, visual_count = _overlay_visual_assets(captioned_video, clip_srt, visual_bindings)
        mixed_video, sound_count = _mix_sound_effects(visual_video, clip_srt, sound_bindings)
        final_video, label_applied = _overlay_doctor_label(mixed_video)
        return final_video, audio, "{}; burned subtitles via FFmpeg; {} GIF/PNG assets overlaid; fixed doctor label={}; {} sound effects mixed".format(
            message, visual_count, label_applied, sound_count
        ), clip_srt
    return launch.AI_clip(
        llm_result, "", "", start_offset_ms, end_offset_ms,
        video_state, None, str(output_dir),
    )
