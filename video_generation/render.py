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
from video_generation.doctor_label import apply_doctor_label
from video_generation.font_config import subtitle_fonts_directory, unified_font_family
from video_generation.question_intro import prepend_question_intro
from video_generation.reference_layout import (
    CAPTION_CENTER,
    CAPTION_FONT_SIZE,
    DISCLAIMER_CENTER,
    DISCLAIMER_FONT_SIZE,
    TITLE_BACKGROUND_COLOR,
    TITLE_BACKGROUND_BORDER_COLOR,
    TITLE_BACKGROUND_HEIGHT,
    TITLE_BACKGROUND_TOP,
    TITLE_FONT_SIZE,
    TITLE_LINE_ONE,
    TITLE_LINE_TWO,
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    SINGLE_LINE_TITLE_FONT_SIZE,
    VIDEO_BRIGHTNESS,
    VIDEO_CONTRAST,
    VIDEO_SATURATION,
    scaled_font_size,
    scaled_position,
)


def _escape_filter_path(path: Path) -> str:
    """Escape an absolute path for FFmpeg's subtitles filter."""
    return path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


_SRT_TIME_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)
_HIGHLIGHT_RANGE_RE = re.compile(
    r"\[?\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*"
    r"(?:-|-->|~|\u2013|\u2014)\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]?"
)
_MAX_CAPTION_LINE_CHARACTERS = 15
_CAPTION_SHADOW_SIZE = 4
_IMPACT_CAPTION_FONT_SIZE = TITLE_FONT_SIZE + (TITLE_FONT_SIZE - CAPTION_FONT_SIZE)
_CAPTION_CONNECTORS = ("但是", "所以", "因为", "如果", "而且", "或者", "并且", "然后", "以及", "同时", "不过", "而是", "还是")
_CAPTION_PUNCTUATION = "，。！？；：、,.!?;:"


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


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("服务器未安装 ffprobe，无法按参考布局烧录字幕。")
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
        raise RuntimeError("无法读取视频尺寸：{}".format(completed.stderr[-500:]))
    try:
        width, height = (int(value) for value in completed.stdout.strip().split("x", 1))
    except ValueError as exc:
        raise RuntimeError("视频没有可用的画面尺寸。") from exc
    return width, height


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


def _highlight_ranges(llm_result: str) -> list[tuple[int, int]]:
    ranges = []
    for match in _HIGHLIGHT_RANGE_RE.finditer(str(llm_result or "")):
        start_ms, end_ms = _time_to_ms(match.group("start")), _time_to_ms(match.group("end"))
        if end_ms > start_ms:
            ranges.append((start_ms, end_ms))
    return ranges


def _rebase_caption_srt(
        source_srt: str,
        llm_result: str,
        start_offset_ms: int = 0,
        end_offset_ms: int = 100) -> str:
    """Map source-video caption timestamps onto the concatenated clip timeline."""
    ranges = _highlight_ranges(llm_result)
    if not ranges:
        return ""

    rebased, clip_offset = [], 0
    for range_start, range_end in ranges:
        for cue_start, cue_end, text in _parse_srt_cues(source_srt):
            cue_start_ms, cue_end_ms = _time_to_ms(cue_start), _time_to_ms(cue_end)
            visible_start, visible_end = max(cue_start_ms, range_start), min(cue_end_ms, range_end)
            if visible_end <= visible_start:
                continue
            rebased.append({
                "start": _format_timestamp(clip_offset + visible_start - range_start),
                "end": _format_timestamp(clip_offset + visible_end - range_start),
                "text": text,
            })
        clip_offset += max(1, range_end + int(end_offset_ms) - range_start - int(start_offset_ms))

    return "\n\n".join(
        "{}\n{} --> {}\n{}".format(index, cue["start"], cue["end"], cue["text"])
        for index, cue in enumerate(rebased, start=1)
    ) + ("\n" if rebased else "")


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


def _visual_asset_duration_seconds(cue_end_ms: int, asset_start_ms: int) -> float:
    """Keep an asset on screen only for the remaining lifetime of its subtitle."""
    return max(0.001, (cue_end_ms - asset_start_ms) / 1000)


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
            offset = start_ms + round((end_ms - start_ms) * word_index / max(1, len(compact)))
            duration = _visual_asset_duration_seconds(end_ms, offset)
            technical = definition.get("technical_metadata") if isinstance(definition.get("technical_metadata"), dict) else {}
            events.append({
                "offset_ms": offset,
                "timestamp": _format_timestamp(offset),
                "duration_seconds": duration,
                "asset_id": asset_id,
                "asset_file": asset_file.name,
                "media_type": definition.get("media_type", "image"),
                "requires_chroma_key": bool(technical.get("requires_chroma_key")),
                # Visual assets are intentionally kept below and left of the
                # current caption. Ignore legacy saved placement values so a
                # prior LLM response cannot send assets back to the top edge.
                "position": "caption_lower_left",
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
        # Caption centre is y=1250 on the 1920px reference canvas. Start the
        # asset below it and keep clear of the left doctor-label strip.
        "caption_lower_left": ("W*0.22", "H*0.70"),
    }
    return positions.get(position, positions["caption_lower_left"])


def _asset_pixel_format(asset_file: Path) -> str:
    """Best-effort alpha diagnostic for the exact file selected by the LLM."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "unknown (ffprobe unavailable)"
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=pix_fmt", "-of", "default=noprint_wrappers=1:nokey=1",
            str(asset_file),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "unknown"


def _pixel_format_has_alpha(pixel_format: str) -> bool:
    value = str(pixel_format or "").strip().lower()
    return value in {"rgba", "bgra", "argb", "abgr"} or value.startswith(("yuva", "gbrap"))


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
    # Composite in RGBA explicitly. Letting overlay auto-negotiate from a YUV
    # main video can flatten alpha-only PNG/GIF pixels to white on some builds.
    filters, previous = ["[0:v]format=rgba[base_rgba]"], "base_rgba"
    for index, event in enumerate(events, start=1):
        asset_file = resolve_visual_asset_file(str(event["asset_id"]))
        if asset_file is None:
            continue
        pixel_format = _asset_pixel_format(asset_file)
        has_alpha = _pixel_format_has_alpha(pixel_format)
        apply_chroma_key = bool(event["requires_chroma_key"]) and not has_alpha
        logging.warning(
            "Visual asset alpha probe: id=%s file=%s pix_fmt=%s alpha=%s chroma_key=%s.",
            event["asset_id"], asset_file, pixel_format, has_alpha, apply_chroma_key,
        )
        duration = float(event["duration_seconds"])
        if event["media_type"] == "animated_gif":
            command.extend(["-stream_loop", "-1", "-i", str(asset_file)])
        else:
            command.extend(["-loop", "1", "-framerate", "30", "-t", "{:.3f}".format(duration), "-i", str(asset_file)])
        asset_label, output_label = "asset{}".format(index), "visual{}".format(index)
        # Extract and re-merge alpha explicitly. Palette GIFs can advertise
        # BGRA yet lose their transparent index in an implicit scale/overlay
        # conversion, which leaves a white rectangle in the rendered video.
        asset_filter = (
            "[{0}:v]format=rgba,split=2[asset_rgb_src{0}][asset_alpha_src{0}];"
            "[asset_rgb_src{0}]format=rgb24,scale=260:-1:flags=lanczos[asset_rgb{0}];"
            "[asset_alpha_src{0}]alphaextract,scale=260:-1:flags=lanczos[asset_alpha{0}];"
            "[asset_rgb{0}][asset_alpha{0}]alphamerge,format=rgba,setsar=1"
        ).format(index)
        if apply_chroma_key:
            asset_filter += ",chromakey=0x00FF00:0.16:0.08"
        asset_filter += ",trim=duration={:.3f},setpts=PTS-STARTPTS+{:.3f}/TB[{}]".format(
            duration, int(event["offset_ms"]) / 1000, asset_label
        )
        x, y = _visual_position(str(event["position"]))
        filters.extend([
            asset_filter,
            "[{}][{}]overlay=x={}:y={}:eof_action=pass:shortest=0:format=rgb:alpha=straight[{}]".format(
                previous, asset_label, x, y, output_label
            ),
        ])
        previous = output_label
    if previous == "base_rgba":
        return str(video_path), 0
    final_label = "visual_yuv"
    filters.append("[{}]format=yuv420p[{}]".format(previous, final_label))
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[{}]".format(final_label), "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", str(output),
    ])
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        logging.warning("Visual-asset overlay failed; returning captioned video: %s", completed.stderr[-1000:])
        return str(video_path), 0
    logging.warning("Overlay %d GIF/PNG visual assets into %s", len(events), output)
    return str(output), len(events)


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
    return CAPTION_FONT_SIZE


def _caption_split_at(text: str, maximum: int) -> int:
    """Find a readable one-line break point, preferring punctuation near the end."""
    compact = "".join(part.strip() for part in text.splitlines())
    if len(compact) <= maximum:
        return len(compact)
    midpoint = min(maximum, len(compact) // 2)
    candidates: list[tuple[int, int]] = []
    for index, char in enumerate(compact, start=1):
        if char in "，。！？；：、,.!?;: " and 5 <= index <= maximum:
            candidates.append((index, 0))
    for connector in _CAPTION_CONNECTORS:
        offset = compact.find(connector)
        while offset >= 5:
            if offset <= maximum:
                candidates.append((offset, 1))
            offset = compact.find(connector, offset + len(connector))
    if candidates:
        split_at, _ = min(
            candidates,
            key=lambda item: (
                abs(item[0] - midpoint),
                item[1],
            ),
        )
    else:
        split_at = min(maximum, max(1, midpoint))
    return split_at


def _split_caption_one_line(text: str) -> list[str]:
    """Break a cue into sequential, readable one-line subtitle events."""
    remaining = "".join(part.strip() for part in str(text or "").splitlines())
    lines = []
    while remaining:
        split_at = _caption_split_at(remaining, _MAX_CAPTION_LINE_CHARACTERS)
        line, remaining = remaining[:split_at].rstrip(), remaining[split_at:].lstrip()
        if line:
            lines.append(line)
        elif remaining:
            remaining = remaining[1:]
    return lines


def _wrap_caption_two_lines(text: str) -> str:
    """Compatibility helper for callers that still expect the old text wrapper."""
    return "\n".join(_split_caption_one_line(text))


def _strip_caption_fillers(text: str) -> str:
    """Remove spoken filler particles only from the burned display caption.

    This runs after the LLM sound-effect decision has already been saved, so
    removing a filler cannot change sound-effect keyword matching or timing.
    """
    compact = "".join(part.strip() for part in str(text or "").splitlines())
    compact = re.sub(r"[嗯呃啊呐呢]+", "", compact)
    compact = re.sub(r"[，、,]{2,}", "，", compact)
    return compact.strip(" \t，、,。！？；：")


def _ass_timecode_from_ms(value: int) -> str:
    centiseconds = max(0, int(round(value / 10)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, hundredths = divmod(remainder, 100)
    return "{}:{:02d}:{:02d}.{:02d}".format(hours, minutes, seconds, hundredths)


def _caption_display_events(start: str, end: str, text: str) -> list[tuple[str, str, str]]:
    """Assign each one-line display segment a consecutive portion of its cue."""
    lines = _split_caption_one_line(text)
    if not lines:
        return []
    start_ms, end_ms = _time_to_ms(start), _time_to_ms(end)
    duration = max(1, end_ms - start_ms)
    total_weight = max(1, sum(len(line) for line in lines))
    events, cursor, consumed = [], start_ms, 0
    for index, line in enumerate(lines):
        consumed += len(line)
        next_cursor = end_ms if index == len(lines) - 1 else start_ms + round(duration * consumed / total_weight)
        events.append((_ass_timecode_from_ms(cursor), _ass_timecode_from_ms(max(cursor + 1, next_cursor)), line))
        cursor = next_cursor
    return events


def _impact_caption_display_events(
        start: str, end: str, text: str, impact_keywords: list[str], used_keywords: set[str],
) -> list[tuple[str, str, str, bool]]:
    """Split the first appearance of a high-impact keyword into a title-size event."""
    events = []
    for event_start, event_end, line in _caption_display_events(start, end, text):
        candidates = [
            (line.find(keyword), keyword)
            for keyword in impact_keywords
            if keyword not in used_keywords and keyword and keyword in line
        ]
        if not candidates:
            events.append((event_start, event_end, line, False))
            continue
        word_index, keyword = min(candidates, key=lambda item: (item[0], -len(item[1])))
        before, after = line[:word_index], line[word_index + len(keyword):]
        pieces = [(before, False), (keyword, True), (after, False)]
        pieces = [(value, is_impact) for value, is_impact in pieces if value]
        start_ms, end_ms = _time_to_ms(event_start.replace(".", ",")), _time_to_ms(event_end.replace(".", ","))
        duration = max(1, end_ms - start_ms)
        total_weight = max(1, sum(len(value) for value, _ in pieces))
        cursor, consumed = start_ms, 0
        for index, (value, is_impact) in enumerate(pieces):
            consumed += len(value)
            next_cursor = end_ms if index == len(pieces) - 1 else start_ms + round(duration * consumed / total_weight)
            events.append((_ass_timecode_from_ms(cursor), _ass_timecode_from_ms(max(cursor + 1, next_cursor)), value, is_impact))
            cursor = next_cursor
        used_keywords.add(keyword)
    return events


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


def _escape_ass_text(text: str) -> str:
    return str(text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _title_lines(title: str, explicit_lines: list[str] | None = None) -> tuple[str, str]:
    clean_lines = ["".join(str(line or "").split()) for line in (explicit_lines or [])]
    if len(clean_lines) == 2 and all(clean_lines):
        return clean_lines[0], clean_lines[1]
    compact = "".join(str(title or "").split())
    if len(compact) <= 8:
        return compact, ""
    midpoint = len(compact) // 2
    split_at = midpoint
    for punctuation in "，、；：,;:":
        position = compact.rfind(punctuation, 0, midpoint + 1)
        if position >= 3:
            split_at = position + 1
            break
    return compact[:split_at], compact[split_at:]


def _write_reference_layout_ass(
        title: str, ass_file: Path, width: int, height: int,
        title_lines: list[str] | None = None) -> None:
    """Write the top title and medical disclaimer using the approved reference layout."""
    title_one, title_two = _title_lines(title, title_lines)
    title_one_x, title_one_y = scaled_position(TITLE_LINE_ONE, width, height)
    title_two_x, title_two_y = scaled_position(TITLE_LINE_TWO, width, height)
    disclaimer_x, disclaimer_y = scaled_position(DISCLAIMER_CENTER, width, height)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Title,{font_family},{title_size},&H00FFFFFF,&H00000000,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,5,5,48,48,0,1
Style: Disclaimer,{font_family},{disclaimer_size},&H00FFFFFF,&H00000000,&H00101010,&H90000000,0,0,0,0,100,100,0,0,1,2,2,5,48,48,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
""".format(
        width=width,
        height=height,
        font_family=unified_font_family(),
        title_size=scaled_font_size(TITLE_FONT_SIZE, width, height),
        disclaimer_size=scaled_font_size(DISCLAIMER_FONT_SIZE, width, height),
    )
    events = []
    if title_one:
        title_size = TITLE_FONT_SIZE if title_two else SINGLE_LINE_TITLE_FONT_SIZE
        title_y = title_one_y if title_two else round((TITLE_LINE_ONE[1] + TITLE_LINE_TWO[1]) / 2 * height / REFERENCE_HEIGHT)
        events.append(
            "Dialogue: 0,0:00:00.00,9:59:59.00,Title,,0,0,0,,{{\\pos({},{})\\fs{}\\c&H00FFFFFF&}}{}".format(
                title_one_x,
                title_y,
                scaled_font_size(title_size, width, height),
                _escape_ass_text(title_one),
            )
        )
    if title_two:
        events.append(
            "Dialogue: 0,0:00:00.00,9:59:59.00,Title,,0,0,0,,{{\\pos({},{})\\c&H006AF2FF&}}{}".format(
                title_two_x, title_two_y, _escape_ass_text(title_two)
            )
        )
    disclaimer = "科学科普 仅供参考\\N身体如有不适请线下就医"
    events.append(
        "Dialogue: 0,0:00:00.00,9:59:59.00,Disclaimer,,0,0,0,,{{\\pos({},{})}}{}".format(
            disclaimer_x, disclaimer_y, disclaimer
        )
    )
    ass_file.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _burn_reference_layout(
        video_path: str | Path, title: str | None, title_lines: list[str] | None = None) -> str:
    source = Path(video_path).resolve()
    width, height = _video_dimensions(source)
    ass_file = source.with_name("{}_reference_layout.ass".format(source.stem))
    output = source.with_name("{}_reference_layout{}".format(source.stem, source.suffix))
    _write_reference_layout_ass(str(title or ""), ass_file, width, height, title_lines)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法烧录参考版标题布局。")
    title_band_top = round(TITLE_BACKGROUND_TOP * height / REFERENCE_HEIGHT)
    title_band_height = round(TITLE_BACKGROUND_HEIGHT * height / REFERENCE_HEIGHT)
    subtitle_filter = (
        "drawbox=x=0:y={}:w=iw:h={}:color={}:t=fill,"
        "drawbox=x=0:y={}:w=iw:h=1:color={}:t=fill,"
        "drawbox=x=0:y={}:w=iw:h=1:color={}:t=fill,"
        "subtitles=filename={}:charenc=UTF-8"
    ).format(
        title_band_top,
        title_band_height,
        TITLE_BACKGROUND_COLOR,
        title_band_top,
        TITLE_BACKGROUND_BORDER_COLOR,
        title_band_top + title_band_height - 1,
        TITLE_BACKGROUND_BORDER_COLOR,
        _escape_filter_path(ass_file),
    )
    font_dir = subtitle_fonts_directory()
    if font_dir:
        subtitle_filter += ":fontsdir={}".format(_escape_filter_path(font_dir))
    command = [
        ffmpeg, "-y", "-i", str(source), "-vf", subtitle_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError("参考版标题布局烧录失败：{}".format(completed.stderr[-1000:]))
    return str(output)


def _write_ass_subtitles(
        clip_srt: str,
        ass_file: Path,
        keywords: list[str],
        width: int = REFERENCE_WIDTH,
        height: int = REFERENCE_HEIGHT,
        impact_keywords: list[str] | None = None,
) -> int:
    cues = _parse_srt_cues(clip_srt)
    if not cues:
        raise RuntimeError("高光剪辑没有生成可烧录的字幕 SRT。")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,{font_family},{font_size},&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2,{shadow_size},5,48,48,0,1
Style: Impact,{font_family},{impact_font_size},&H0000FFFF,&H0000FFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,5,5,48,48,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
""".format(
        width=width,
        height=height,
        font_family=unified_font_family(),
        font_size=scaled_font_size(CAPTION_FONT_SIZE, width, height),
        impact_font_size=scaled_font_size(_IMPACT_CAPTION_FONT_SIZE, width, height),
        shadow_size=_CAPTION_SHADOW_SIZE,
    )
    caption_x, caption_y = scaled_position(CAPTION_CENTER, width, height)
    events, used_impact_keywords = [], set()
    for start, end, text in cues:
        display_text = _strip_caption_fillers(text)
        if not display_text:
            continue
        for event_start, event_end, line, is_impact in _impact_caption_display_events(
                start, end, display_text, impact_keywords or [], used_impact_keywords):
            style = "Impact" if is_impact else "Default"
            ass_text = _escape_ass_text(line) if is_impact else _highlight_ass_text(line, keywords)
            events.append("Dialogue: 0,{},{},{},,0,0,0,,{{\\pos({},{})\\fs{}}}{}".format(
                event_start,
                event_end,
                style,
                caption_x,
                caption_y,
                scaled_font_size(_IMPACT_CAPTION_FONT_SIZE if is_impact else _caption_font_size(line), width, height),
                ass_text,
            ))
    ass_file.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)


def _burn_srt_with_ffmpeg(
        video_path: str | Path, clip_srt: str, keywords: str | None = None,
        impact_keywords: list[str] | None = None) -> str:
    if not str(clip_srt or "").strip():
        raise RuntimeError("高光剪辑没有生成可烧录的字幕 SRT。")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法烧录字幕。")

    source = Path(video_path).resolve()
    width, height = _video_dimensions(source)
    subtitle_file = source.with_name("{}_captions.ass".format(source.stem))
    output = source.with_name("{}_captioned{}".format(source.stem, source.suffix))
    highlight_keywords = _parse_keywords(keywords)
    cue_count = _write_ass_subtitles(
        clip_srt, subtitle_file, highlight_keywords, width, height, impact_keywords=impact_keywords,
    )

    color_filter = "eq=brightness={}:contrast={}:saturation={}".format(
        VIDEO_BRIGHTNESS,
        VIDEO_CONTRAST,
        VIDEO_SATURATION,
    )
    subtitle_parts = [
        "subtitles=filename={}".format(_escape_filter_path(subtitle_file)),
        "charenc=UTF-8",
    ]
    font_dir = subtitle_fonts_directory()
    if font_dir:
        subtitle_parts.append("fontsdir={}".format(_escape_filter_path(font_dir)))
    subtitle_filter = "{},{}".format(color_filter, ":".join(subtitle_parts))
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
        "-pix_fmt",
        "yuv420p",
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
        impact_keywords: list[str] | None = None,
        sound_bindings: str | None = None,
        visual_bindings: str | None = None,
        question: str | None = None,
        question_lines: list[str] | None = None,
        caption_srt: str | None = None,
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
        source_caption_srt = str(caption_srt or "")
        effective_srt = _rebase_caption_srt(
            source_caption_srt, llm_result, start_offset_ms, end_offset_ms
        ) if source_caption_srt else clip_srt
        if source_caption_srt and not effective_srt.strip():
            logging.warning("语义字幕无法映射到剪辑时间轴，回退为上游剪辑字幕。")
            effective_srt = clip_srt
        captioned_video = _burn_srt_with_ffmpeg(video, effective_srt, keywords, impact_keywords)
        visual_video, visual_count = _overlay_visual_assets(captioned_video, effective_srt, visual_bindings)
        mixed_video, sound_count = _mix_sound_effects(visual_video, effective_srt, sound_bindings)
        layout_video = _burn_reference_layout(mixed_video, question, question_lines)
        # The expert label belongs to the main consultation footage only. Burn
        # it before concatenation so the three-second question card stays clean.
        labelled_main_video = apply_doctor_label(layout_video)
        final_video = (
            prepend_question_intro(labelled_main_video, question, question_lines)
            if str(question or "").strip()
            else labelled_main_video
        )
        return final_video, audio, "{}; burned subtitles via FFmpeg; reference layout=True; question intro={}; {} GIF/PNG assets overlaid; fixed doctor label=True; {} sound effects mixed".format(
            message, bool(str(question or "").strip()), visual_count, sound_count
        ), effective_srt
    return launch.AI_clip(
        llm_result, "", "", start_offset_ms, end_offset_ms,
        video_state, None, str(output_dir),
    )
