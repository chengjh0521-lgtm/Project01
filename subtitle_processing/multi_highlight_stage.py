"""Stage 2: iteratively select low-overlap, publishable highlight clips."""

from __future__ import annotations

import re
import logging
import json
from typing import Callable


TIME_RE = re.compile(
    r"\[?\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*(?:-|-->|~|\u2013|\u2014)\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]?"
)
QUESTION_LINE_COUNT = 2
MAX_QUESTION_LINE_CHARACTERS = 7

SYSTEM_PROMPT = """
你是医学科普短视频选题与剪辑分析器。请从完整 SRT 中选择一段适合传播的独立内容。

每条成片都必须能够完整、直接地回答一个明确的普通观众问题。先在心中确定问题，再选择足以回答该问题的字幕时间段。问题必须具体、自然、可由观众主动提出，必须针对某类特定人群，例如“糖尿病患者能不能喝酒？”，而不能使用“胰岛素多久能停？”；不得使用“医生说了什么”“这段讲了什么”这类泛泛问题，也不得在问题中直接写出答案。

所选字幕必须只围绕该问题，包含必要结论、原因或建议，使观众不依赖原视频上下文也能理解答案。删除寒暄、病史确认、重复、无意义停顿和与该问题无关的内容。成片总时长必须为 40 到 90 秒之间，成片总时长必须为 40 到 90 秒之间，可由多个不连续片段组成；优先选择结论明确、对普通观众有价值、逻辑完整的医生解释。

只返回一个合法 JSON 对象，不得输出 Markdown 或其他解释：
{"question_lines":["糖尿病患者","能不能喝酒？"],"ranges":[{"start":"00:00:01,000","end":"00:00:12,500"}],"doctor_answer_ranges":[{"start":"00:00:05,000","end":"00:00:08,000"}],"reason":"该片段给出明确结论、风险原因和可执行建议"}

question_lines 必须恰好包含两条非空字符串。每条去除空白后最多 7 个字符（包括标点），两条按顺序拼接后必须是一个自然、完整、可由观众提出的问题。不得将问号单独作为一行；不得输出 question 字段。

doctor_answer_ranges 必须包含 1 或 2 条医生原话字幕。每条必须使用输入 SRT 中一条完整字幕的 start 和 end，不得改写、不概括、不拼接，并且必须落在 ranges 所选高光内。这一到两条原话将单独烧录在黄金三秒之后，用来直接回答 question_lines 所组成的问题；优先选择结论最直接、最有力的医生回答。

所有 start 和 end 必须完全来自输入 SRT 时间轴。无法选择一个能完整回答明确问题的片段时，返回：
{"question_lines":[],"ranges":[],"doctor_answer_ranges":[],"reason":""}
""".strip()


def _normalize(value: str) -> str:
    h, m, seconds = value.replace(".", ",").split(":")
    s, ms = seconds.split(",")
    return "{:02d}:{:02d}:{:02d},{:03d}".format(int(h), int(m), int(s), int(ms.ljust(3, "0")[:3]))


def _ms(value: str) -> int:
    h, m, seconds = _normalize(value).split(":")
    s, ms = seconds.split(",")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)


def _load_json_object(text: str) -> dict | None:
    """Accept a JSON object even when a provider wraps it in a code fence."""
    value = text.strip()
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
    return None


def _normalize_question_lines(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [re.sub(r"\s+", "", str(line or "")) for line in value]


def _valid_question_lines(lines: list[str]) -> bool:
    return (
        len(lines) == QUESTION_LINE_COUNT
        and all(0 < len(line) <= MAX_QUESTION_LINE_CHARACTERS for line in lines)
        and any(char not in "？?！!。" for char in lines[-1])
    )


def _parse_range_items(value: object) -> list[tuple[str, str]]:
    """Normalize a JSON list of timestamp ranges and remove duplicates."""
    if not isinstance(value, list):
        return []
    seen, values = set(), []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start, end = _normalize(str(item.get("start", ""))), _normalize(str(item.get("end", "")))
        except ValueError:
            continue
        if _ms(end) > _ms(start) and (start, end) not in seen:
            values.append((start, end))
            seen.add((start, end))
    return values


def parse_highlight_selection(text: str) -> dict:
    """Parse one LLM highlight candidate and retain its answerable question."""
    seen, values = set(), []
    question_lines, doctor_answer_ranges, reason = [], [], ""
    try:
        payload = _load_json_object(text)
        items = payload.get("ranges", []) if isinstance(payload, dict) else []
        if isinstance(payload, dict):
            question_lines = _normalize_question_lines(payload.get("question_lines"))
            doctor_answer_ranges = _parse_range_items(payload.get("doctor_answer_ranges"))
            reason = str(payload.get("reason", "")).strip()
        for item in items:
            if not isinstance(item, dict):
                continue
            start, end = _normalize(str(item.get("start", ""))), _normalize(str(item.get("end", "")))
            if _ms(end) > _ms(start) and (start, end) not in seen:
                values.append((start, end))
                seen.add((start, end))
        if values or isinstance(payload, dict):
            return {
                "ranges": values,
                "question": "".join(question_lines),
                "question_lines": question_lines,
                "doctor_answer_ranges": doctor_answer_ranges,
                "reason": reason,
            }
    except ValueError:
        pass
    for match in TIME_RE.finditer(text):
        start, end = _normalize(match.group("start")), _normalize(match.group("end"))
        if _ms(end) > _ms(start) and (start, end) not in seen:
            values.append((start, end))
            seen.add((start, end))
    return {
        "ranges": values,
        "question": "",
        "question_lines": [],
        "doctor_answer_ranges": [],
        "reason": reason,
    }


def parse_ranges(text: str) -> list[tuple[str, str]]:
    """Compatibility helper for callers that only need the timestamp ranges."""
    return parse_highlight_selection(text)["ranges"]


def duration_ms(ranges: list[tuple[str, str]]) -> int:
    return sum(_ms(end) - _ms(start) for start, end in ranges)


def overlap_ratio(candidate: list[tuple[str, str]], selected: list[tuple[str, str]]) -> float:
    total = duration_ms(candidate)
    if not total:
        return 1.0
    overlap = sum(
        max(0, min(_ms(end), _ms(other_end)) - max(_ms(start), _ms(other_start)))
        for start, end in candidate
        for other_start, other_end in selected
    )
    return min(1.0, overlap / total)


def _cue_ranges(srt_text: str) -> set[tuple[str, str]]:
    return {
        (_normalize(match.group("start")), _normalize(match.group("end")))
        for match in re.finditer(
            r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
            r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$",
            str(srt_text or ""),
            flags=re.MULTILINE,
        )
    }


def _valid_doctor_answer_ranges(
        answer_ranges: list[tuple[str, str]], highlight_ranges: list[tuple[str, str]],
        cue_ranges: set[tuple[str, str]]) -> bool:
    if not 1 <= len(answer_ranges) <= 2:
        return False
    for answer_start, answer_end in answer_ranges:
        if (answer_start, answer_end) not in cue_ranges:
            return False
        if not any(
            _ms(highlight_start) <= _ms(answer_start) and _ms(answer_end) <= _ms(highlight_end)
            for highlight_start, highlight_end in highlight_ranges
        ):
            return False
    return True


def select_multiple(
    corrected_srt: str,
    max_clips: int,
    call_llm: Callable[[str, str], str],
    *,
    max_overlap: float = 0.30,
    report: Callable[[str], None] | None = None,
) -> list[dict]:
    selected: list[dict] = []
    selected_ranges: list[tuple[str, str]] = []
    source_cue_ranges = _cue_ranges(corrected_srt)
    for number in range(1, max(1, max_clips) + 1):
        previous = "\n".join(
            "素材{}：问题：{}；时间段：{}".format(
                index,
                clip.get("question", ""),
                ", ".join("[{}-{}]".format(*item) for item in clip["ranges"]),
            )
            for index, clip in enumerate(selected, start=1)
        ) or "无"
        user_prompt = (
            "完整素材如下：\n{}\n\n已选素材如下：\n{}\n\n"
            "请提取一个新的、能完整回答明确问题的主题或不同角度。新问题不能与已选问题重复；"
            "新素材与所有已选素材的重合时长不得超过新素材总时长的 30%。"
            "如果无法提取合规的新素材，返回 {{\"question_lines\":[],\"ranges\":[],\"doctor_answer_ranges\":[],\"reason\":\"\"}}。"
        ).format(corrected_srt, previous)
        if report:
            report("阶段 2/5：正在提取第 {} 条低重合高光候选。".format(number))
        raw, ranges, question, question_lines, doctor_answer_ranges, reason = "", [], "", [], [], ""
        for attempt in range(1, 4):
            raw = call_llm(
                SYSTEM_PROMPT,
                user_prompt + "\n\nThis is attempt {} of 3. Return the exact JSON object only.".format(attempt),
            )
            logging.warning("Multi-highlight candidate %d attempt %d raw response:\n%s", number, attempt, raw)
            selection = parse_highlight_selection(raw)
            ranges = selection["ranges"]
            question = selection["question"]
            question_lines = selection["question_lines"]
            doctor_answer_ranges = selection["doctor_answer_ranges"]
            reason = selection["reason"]
            overlap = overlap_ratio(ranges, selected_ranges) if ranges else 1.0
            if (
                ranges
                and _valid_question_lines(question_lines)
                and _valid_doctor_answer_ranges(doctor_answer_ranges, ranges, source_cue_ranges)
                and overlap <= max_overlap
            ):
                break
            logging.warning(
                "Multi-highlight candidate %d attempt %d rejected: valid_question_lines=%s, valid_doctor_answer_ranges=%s, ranges=%d, overlap=%.1f%%.",
                number,
                attempt,
                _valid_question_lines(question_lines),
                _valid_doctor_answer_ranges(doctor_answer_ranges, ranges, source_cue_ranges),
                len(ranges),
                overlap * 100,
            )
        if (
            not ranges
            or not _valid_question_lines(question_lines)
            or not _valid_doctor_answer_ranges(doctor_answer_ranges, ranges, source_cue_ranges)
            or overlap_ratio(ranges, selected_ranges) > max_overlap
        ):
            break
        selected.append({
            "id": "clip_{:02d}".format(number),
            "question": question,
            "question_lines": question_lines,
            "highlight_reason": reason,
            "ranges": ranges,
            "doctor_answer_ranges": doctor_answer_ranges,
            "raw_result": raw,
        })
        selected_ranges.extend(ranges)
    return selected
