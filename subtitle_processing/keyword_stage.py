"""Stage 3: select display keywords for one highlight SRT."""

from __future__ import annotations

import re
import json
from typing import Callable

SYSTEM_PROMPT = """你是医疗短视频字幕关键词分析器。只选择真正影响理解或传播的疾病、症状、药物、检查、风险、
结论、否定词、数字、百分比、时间或剂量。关键词必须完全出现在输入字幕中；每条字幕最多 8 个。

另请谨慎标记 impact：只有首次出现时值得被拆成独立字幕、放大到标题字号的词才设为 true。通常仅限明确疾病诊断、严重症状、严重并发症、不可逆后果或极强风险结论；每条高光最多标记 2 个 impact。普通医学词、数字、一般建议和普通高亮词必须为 false。

只输出合法 JSON，不输出 Markdown 或额外文字：
{"keywords":[{"word":"糖尿病","reason":"明确疾病诊断，首次出现适合做重点卡点","impact":true},{"word":"控制血糖","reason":"核心建议，适合普通黄色高亮","impact":false}]}
word 必须完全出现在输入字幕中；reason 仅用一句话说明其传播或理解价值。""".strip()


def _selection(raw: str, highlight_srt: str, count: int) -> tuple[str, list[dict[str, object]]]:
    values, details = [], []
    try:
        payload = json.loads(raw.strip())
        items = payload.get("keywords", []) if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("word", "")).strip(" \t\"'“”")
        if value and value in highlight_srt and value not in values:
            values.append(value)
            impact_value = item.get("impact", False)
            impact = impact_value is True or str(impact_value).strip().lower() == "true"
            details.append({
                "word": value,
                "reason": str(item.get("reason", "")).strip()[:200],
                "impact": impact,
            })
    if values:
        return "\n".join(values[: max(1, int(count))]), details[: max(1, int(count))]
    for part in re.split(r"[\n,，;；]+", raw):
        value = re.sub(r"^\s*\d+\s*[.、]\s*", "", part).strip(" \t\"'")
        if value and value in highlight_srt and value not in values:
            values.append(value)
            details.append({"word": value, "reason": "", "impact": False})
    limit = max(1, int(count))
    return "\n".join(values[:limit]), details[:limit]


def select_keywords(
        highlight_srt: str,
        count: int,
        call_llm: Callable[[str, str], str],
        *,
        include_reasons: bool = False,
) -> str | tuple[str, list[dict[str, object]]]:
    raw = call_llm(SYSTEM_PROMPT, "高光字幕如下：\n{}".format(highlight_srt))
    selected, details = _selection(raw, highlight_srt, count)
    return (selected, details) if include_reasons else selected
