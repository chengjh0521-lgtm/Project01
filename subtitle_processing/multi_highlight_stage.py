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

SYSTEM_PROMPT = """
你是医学科普短视频选题与剪辑分析器。请从完整 SRT 中选择一段适合传播的独立内容。

每条成片都必须能够完整、直接地回答一个明确的普通观众问题。先在心中确定问题，再选择足以回答该问题的字幕时间段。问题必须具体、自然、可由观众主动提出，例如“糖尿病患者能不能喝酒？”；不得使用“医生说了什么”“这段讲了什么”这类泛泛问题，也不得在问题中直接写出答案。

所选字幕必须只围绕该问题，包含必要结论、原因或建议，使观众不依赖原视频上下文也能理解答案。删除寒暄、病史确认、重复、无意义停顿和与该问题无关的内容。成片总时长通常为 40 到 90 秒，可由多个不连续片段组成；优先选择结论明确、对普通观众有价值、逻辑完整的医生解释。

只返回一个合法 JSON 对象，不得输出 Markdown 或其他解释：
{"question":"糖尿病患者能不能喝酒？","ranges":[{"start":"00:00:01,000","end":"00:00:12,500"}],"reason":"该片段给出明确结论、风险原因和可执行建议"}

question 必须是非空字符串；start 和 end 必须完全来自输入 SRT 时间轴。无法选择一个能完整回答明确问题的片段时，返回：
{"question":"","ranges":[],"reason":""}
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


def parse_highlight_selection(text: str) -> dict:
    """Parse one LLM highlight candidate and retain its answerable question."""
    seen, values = set(), []
    question, reason = "", ""
    try:
        payload = _load_json_object(text)
        items = payload.get("ranges", []) if isinstance(payload, dict) else []
        if isinstance(payload, dict):
            question = str(payload.get("question", "")).strip()
            reason = str(payload.get("reason", "")).strip()
        for item in items:
            if not isinstance(item, dict):
                continue
            start, end = _normalize(str(item.get("start", ""))), _normalize(str(item.get("end", "")))
            if _ms(end) > _ms(start) and (start, end) not in seen:
                values.append((start, end))
                seen.add((start, end))
        if values or isinstance(payload, dict):
            return {"ranges": values, "question": question, "reason": reason}
    except ValueError:
        pass
    for match in TIME_RE.finditer(text):
        start, end = _normalize(match.group("start")), _normalize(match.group("end"))
        if _ms(end) > _ms(start) and (start, end) not in seen:
            values.append((start, end))
            seen.add((start, end))
    return {"ranges": values, "question": question, "reason": reason}


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
            "如果无法提取合规的新素材，返回 {{\"question\":\"\",\"ranges\":[],\"reason\":\"\"}}。"
        ).format(corrected_srt, previous)
        if report:
            report("阶段 2/5：正在提取第 {} 条低重合高光候选。".format(number))
        raw, ranges, question, reason = "", [], "", ""
        for attempt in range(1, 4):
            raw = call_llm(
                SYSTEM_PROMPT,
                user_prompt + "\n\nThis is attempt {} of 3. Return the exact JSON object only.".format(attempt),
            )
            logging.warning("Multi-highlight candidate %d attempt %d raw response:\n%s", number, attempt, raw)
            selection = parse_highlight_selection(raw)
            ranges = selection["ranges"]
            question = selection["question"]
            reason = selection["reason"]
            overlap = overlap_ratio(ranges, selected_ranges) if ranges else 1.0
            if ranges and question and overlap <= max_overlap:
                break
            logging.warning(
                "Multi-highlight candidate %d attempt %d rejected: question=%s, ranges=%d, overlap=%.1f%%.",
                number, attempt, bool(question), len(ranges), overlap * 100,
            )
        if not ranges or not question or overlap_ratio(ranges, selected_ranges) > max_overlap:
            break
        selected.append({
            "id": "clip_{:02d}".format(number),
            "question": question,
            "highlight_reason": reason,
            "ranges": ranges,
            "raw_result": raw,
        })
        selected_ranges.extend(ranges)
    return selected
