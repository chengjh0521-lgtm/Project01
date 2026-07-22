"""Use an LLM to turn selected highlight cues into short semantic captions."""

from __future__ import annotations

import json
import re
from typing import Callable


MAX_CAPTION_CHARACTERS = 14

SYSTEM_PROMPT = """
你是医学短视频字幕语义分段器。输入是已经选中的高光字幕，每条包含 id 和 text。

请将每条字幕按自然语义拆成适合烧录的短弹幕。每个 segments 元素去掉空白后不超过 14 个字符。只能做断句，绝不能改写、增删、替换或调换原文任何字符；同一条字幕的所有 segments 去掉空白后按顺序拼接，必须与输入 text 去掉空白后完全一致。

每个输入 id 必须原样返回一次，顺序必须相同。segments 可以为一条或多条；输出字幕条数不需要与输入条数一致。不得输出时间戳、解释或 Markdown。

只返回合法 JSON：
{"entries":[{"id":"1","segments":["糖尿病患者","需要控制血糖"]}]}
""".strip()


class SemanticCaptionError(ValueError):
    """The semantic caption response cannot safely replace the source captions."""


def _compact(text: object) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _format_timecode(value: int) -> str:
    value = max(0, int(value))
    hours, value = divmod(value, 3_600_000)
    minutes, value = divmod(value, 60_000)
    seconds, millis = divmod(value, 1_000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, seconds, millis)


def _time_to_ms(value: str) -> int:
    hours, minutes, seconds = str(value).replace(".", ",").split(":")
    second, millis = seconds.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(second) * 1_000
        + int(millis.ljust(3, "0")[:3])
    )


def _load_json_object(text: str) -> dict:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for offset, char in enumerate(value):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(value[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise SemanticCaptionError("语义分段模型没有返回合法 JSON。")


def _validate_entries(raw: str, cues: list[dict[str, str]]) -> list[list[str]]:
    payload = _load_json_object(raw)
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(cues):
        raise SemanticCaptionError("语义分段返回的字幕条目数量与高光字幕不一致。")

    segmented: list[list[str]] = []
    for index, (entry, cue) in enumerate(zip(entries, cues), start=1):
        expected_id = str(index)
        if not isinstance(entry, dict) or str(entry.get("id", "")) != expected_id:
            raise SemanticCaptionError("语义分段返回的字幕 id 或顺序不正确。")
        segments = entry.get("segments")
        if not isinstance(segments, list) or not segments or not all(isinstance(item, str) for item in segments):
            raise SemanticCaptionError("语义分段返回了空字幕。")
        compact_segments = [_compact(item) for item in segments]
        if any(not item or len(item) > MAX_CAPTION_CHARACTERS for item in compact_segments):
            raise SemanticCaptionError("语义分段中存在空字幕或超过 14 个字的字幕。")
        if "".join(compact_segments) != _compact(cue["text"]):
            raise SemanticCaptionError("语义分段修改了高光字幕原文。")
        segmented.append(compact_segments)
    return segmented


def _split_cue(cue: dict[str, str], segments: list[str]) -> list[dict[str, str]]:
    if len(segments) == 1:
        return [{"start": cue["start"], "end": cue["end"], "text": segments[0]}]

    start_ms, end_ms = _time_to_ms(cue["start"]), _time_to_ms(cue["end"])
    duration = max(len(segments), end_ms - start_ms)
    total_weight = sum(len(item) for item in segments)
    cursor, consumed, result = start_ms, 0, []
    for index, segment in enumerate(segments):
        consumed += len(segment)
        next_cursor = end_ms if index == len(segments) - 1 else start_ms + round(duration * consumed / total_weight)
        next_cursor = max(cursor + 1, next_cursor)
        result.append({
            "start": _format_timecode(cursor),
            "end": _format_timecode(next_cursor),
            "text": segment,
        })
        cursor = next_cursor
    return result


def segment_highlight_cues(
        cues: list[dict[str, str]], call_llm: Callable[[str, str], str]) -> list[dict[str, str]]:
    """Return semantic sub-cues while preserving every source cue's timeline and text."""
    if not cues:
        raise SemanticCaptionError("没有可供语义分段的高光字幕。")
    source = {"entries": [{"id": str(index), "text": cue["text"]} for index, cue in enumerate(cues, start=1)]}
    raw = call_llm(
        SYSTEM_PROMPT,
        "请按系统要求处理以下 JSON，并只返回所需 JSON 对象：\n{}".format(
            json.dumps(source, ensure_ascii=False)
        ),
    )
    segments_by_cue = _validate_entries(raw, cues)
    return [piece for cue, segments in zip(cues, segments_by_cue) for piece in _split_cue(cue, segments)]
