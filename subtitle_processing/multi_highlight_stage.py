"""Stage 2: iteratively select low-overlap, publishable highlight clips."""

from __future__ import annotations

import re
import logging
from typing import Callable


TIME_RE = re.compile(
    r"\[?\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*(?:-|-->|~)\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]?"
)

SYSTEM_PROMPT = """
你是医学科普短视频选题与剪辑分析器。请从完整 SRT 中选择一个适合传播的独立知识主题，
每条成片时长 40 到 90 秒，可由多个不连续片段组成。优先选择结论明确、对普通观众有价值、
逻辑完整的医生解释；删除寒暄、重复、病史确认和无意义停顿。

只输出时间段，每行格式严格为：
[开始时间-结束时间] 对应字幕原文
不得输出标题、分析或其他文字。
""".strip()


def _normalize(value: str) -> str:
    h, m, seconds = value.replace(".", ",").split(":")
    s, ms = seconds.split(",")
    return "{:02d}:{:02d}:{:02d},{:03d}".format(int(h), int(m), int(s), int(ms.ljust(3, "0")[:3]))


def _ms(value: str) -> int:
    h, m, seconds = _normalize(value).split(":")
    s, ms = seconds.split(",")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)


def parse_ranges(text: str) -> list[tuple[str, str]]:
    seen, values = set(), []
    for match in TIME_RE.finditer(text):
        start, end = _normalize(match.group("start")), _normalize(match.group("end"))
        if _ms(end) > _ms(start) and (start, end) not in seen:
            values.append((start, end))
            seen.add((start, end))
    return values


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
    for number in range(1, max(1, max_clips) + 1):
        previous = "\n".join(
            "素材{}：{}".format(index, ", ".join("[{}-{}]".format(*item) for item in clip["ranges"]))
            for index, clip in enumerate(selected, start=1)
        ) or "无"
        user_prompt = (
            "完整素材如下：\n{}\n\n已选素材如下：\n{}\n\n"
            "请提取一个新的主题或不同角度。新素材与所有已选素材的重合时长不得超过新素材总时长的 30%。"
            "如果无法提取合规的新素材，只输出 NONE。"
        ).format(corrected_srt, previous)
        if report:
            report("阶段 2/4：正在提取第 {} 条低重合高光候选。".format(number))
        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        logging.warning("Multi-highlight candidate %d raw response:\n%s", number, raw)
        if raw.strip().upper() == "NONE":
            break
        ranges = parse_ranges(raw)
        overlap = overlap_ratio(ranges, selected_ranges) if ranges else 1.0
        if not ranges:
            logging.warning("Multi-highlight candidate %d had no parseable timestamps.", number)
            break
        if overlap > max_overlap:
            logging.warning("Multi-highlight candidate %d rejected: overlap %.1f%% exceeds %.1f%%.", number, overlap * 100, max_overlap * 100)
            break
        selected.append({"id": "clip_{:02d}".format(number), "ranges": ranges, "raw_result": raw})
        selected_ranges.extend(ranges)
    return selected
