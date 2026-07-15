"""Stage 3: select display keywords for one highlight SRT."""

from __future__ import annotations

import re
from typing import Callable

SYSTEM_PROMPT = """你是医疗短视频字幕关键词分析器。只选择真正影响理解或传播的疾病、症状、药物、检查、风险、
结论、否定词、数字、百分比、时间或剂量。关键词必须完全出现在输入字幕中；每条字幕最多 8 个；
只输出关键词，以逗号或换行分隔，不输出解释。""".strip()


def select_keywords(highlight_srt: str, count: int, call_llm: Callable[[str, str], str]) -> str:
    raw = call_llm(SYSTEM_PROMPT, "高光字幕如下：\n{}".format(highlight_srt))
    values = []
    for part in re.split(r"[\n,，;；]+", raw):
        value = re.sub(r"^\s*\d+\s*[.、]\s*", "", part).strip(" \t\"'")
        if value and value in highlight_srt and value not in values:
            values.append(value)
    return "\n".join(values[: max(1, int(count))])
