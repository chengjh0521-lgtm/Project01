"""Generate publishable questions and strict 40-90 second subtitle answers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from subtitle_processing.pipeline import (
    _call_deepseek,
    _normalize_timecode,
    _time_to_ms,
    build_highlight_srt,
    parse_srt,
)


QUESTION_COUNT = 10
MIN_ANSWER_DURATION_MS = 40_000
MAX_ANSWER_DURATION_MS = 90_000
MAX_ANSWER_ATTEMPTS = 2
MAX_QUESTION_ATTEMPTS = 2
DEFAULT_TEXT_ANSWER_WORKERS = 8

QUESTION_SYSTEM_PROMPT = """
你是一名医学科普短视频选题策划。输入是一份已经清洗完成的完整 SRT 字幕。

请提出恰好 10 个能够抓住普通观众注意力、且能被本字幕完整回答的问题。问题应具体、自然、具有传播力，可以围绕误区、风险、症状、疾病、用药、饮食或医生的明确结论；不要写成“这段说了什么”之类的泛问题，也不要把答案直接写进问题。

最重要的前提：每个问题都必须能从输入字幕中找到总时长严格在 40 到 90 秒之间的独立回答素材。10 个问题可以有角度相近之处，但不能重复。

只返回合法 JSON，不输出 Markdown 或额外文字：
{"questions":[{"id":"q01","question":"糖尿病患者最容易忽视的风险是什么？","reason":"风险明确且具有反常识吸引力"}]}

questions 必须恰好 10 条，id 必须为 q01 到 q10，question 为中文问题，reason 为简短选题理由。
""".strip()

ANSWER_SYSTEM_PROMPT = """
你是一名医学短视频字幕剪辑分析器。输入包含一个问题和完整 SRT 字幕。

请只选择能够直接、完整回答该问题的字幕时间段，删除寒暄、重复、病史确认和无关内容。可选择多个不连续时间段，但所有时间段时长相加必须严格大于等于 40 秒且小于等于 90 秒。每个 start 必须完全等于某条输入字幕的开始时间，每个 end 必须完全等于某条输入字幕的结束时间；不得创造、修改或估算时间戳。

`ranges` 数组的顺序就是最终短视频和 Markdown 的叙事顺序，绝对不要因为时间戳先后而自动排序。允许跨原视频时间线重组不连续片段，且应采用“结论优先”：优先将能够直接回答问题、给出明确结论、关键风险或行动建议的字幕放在第一个 range；其次才放解释、证据、例外条件或能间接回答问题的重点字幕；最后才补充必要的患者提问、背景或过渡。若原视频后段有更直接的结论，必须把后段时间戳先输出，再输出前段的背景。只有确实需要按原顺序才能理解时，才保持时间顺序。

输出的 ranges 必须已经是你最终决定的叙事顺序。系统会严格保留该顺序，不会替你按时间戳重排。

只返回合法 JSON，不输出 Markdown 或额外文字：
{"ranges":[{"start":"00:01:02,000","end":"00:01:35,000"}],"answer_summary":"一句话概述医生给出的答案","reason":"该段包含结论、原因和建议"}

如果找不到严格符合 40 到 90 秒要求的完整回答，只返回：
{"ranges":[],"answer_summary":"","reason":""}
""".strip()


class TextQuestionTestError(ValueError):
    """The LLM result cannot satisfy the strict text-test contract."""


def _json_object(raw: str) -> dict:
    value = str(raw or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise TextQuestionTestError("大模型没有返回合法 JSON 对象。")


def _questions(raw: str) -> list[dict[str, str]]:
    payload = _json_object(raw)
    items = payload.get("questions")
    if not isinstance(items, list):
        raise TextQuestionTestError("问题生成结果缺少 questions 数组。")
    result, seen = [], set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        question = re.sub(r"\s+", "", str(item.get("question") or ""))
        if not question or question in seen:
            continue
        seen.add(question)
        result.append({
            "id": "q{:02d}".format(index),
            "question": question,
            "reason": str(item.get("reason") or "").strip()[:200],
        })
    if len(result) != QUESTION_COUNT:
        raise TextQuestionTestError("大模型必须返回恰好 10 个不重复的问题，当前得到 {} 个。".format(len(result)))
    return result


def _ranges(raw: str, valid_starts: set[str], valid_ends: set[str]) -> tuple[list[tuple[str, str]], str, str]:
    payload = _json_object(raw)
    items = payload.get("ranges")
    if not isinstance(items, list):
        raise TextQuestionTestError("回答结果缺少 ranges 数组。")
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = _normalize_timecode(str(item.get("start") or ""))
            end = _normalize_timecode(str(item.get("end") or ""))
        except ValueError:
            continue
        if start not in valid_starts or end not in valid_ends or _time_to_ms(end) <= _time_to_ms(start):
            continue
        pair = (start, end)
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    duration = sum(_time_to_ms(end) - _time_to_ms(start) for start, end in result)
    ordered_ranges = sorted(result, key=lambda item: _time_to_ms(item[0]))
    if any(
        _time_to_ms(previous[1]) > _time_to_ms(current[0])
        for previous, current in zip(ordered_ranges, ordered_ranges[1:])
    ):
        raise TextQuestionTestError("回答时间段之间不得重叠。")
    if not result or not MIN_ANSWER_DURATION_MS <= duration <= MAX_ANSWER_DURATION_MS:
        raise TextQuestionTestError(
            "回答时间段总时长必须严格处于 40-90 秒，当前为 {:.2f} 秒。".format(duration / 1000)
        )
    return (
        result,
        str(payload.get("answer_summary") or "").strip()[:300],
        str(payload.get("reason") or "").strip()[:300],
    )


def _answer_request(question: str, srt_text: str) -> str:
    return "问题：{}\n\n完整 SRT 字幕：\n{}".format(question, srt_text)


def _write_markdown(source_name: str, questions: list[dict], output_dir: str | Path) -> tuple[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 字幕问题与答案测试报告",
        "",
        "- 来源字幕：`{}`".format(source_name),
        "- 生成时间：{}".format(datetime.now().astimezone().isoformat(timespec="seconds")),
        "- 问题数量：{}".format(len(questions)),
        "",
    ]
    for index, item in enumerate(questions, start=1):
        lines.extend([
            "## {}. {}".format(index, item["question"]),
            "",
            "- 选题理由：{}".format(item.get("reason") or "模型未返回理由。"),
            "- 回答片段总时长：{:.2f} 秒".format(item["duration_ms"] / 1000),
            "- 回答选择理由：{}".format(item.get("answer_reason") or "模型未返回理由。"),
            "- 答案概述：{}".format(item.get("answer_summary") or "模型未返回概述。"),
            "",
            "### 时间戳",
            "",
            *["- [{}-{}]".format(start, end) for start, end in item["ranges"]],
            "",
            "### 对应字幕",
            "",
            "```srt",
            item["answer_srt"].rstrip(),
            "```",
            "",
        ])
    markdown = "\n".join(lines).rstrip() + "\n"
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:10]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem) or "subtitle"
    path = destination / "text_questions_{}_{}_{}.md".format(
        safe_stem, datetime.now().strftime("%Y%m%d_%H%M%S"), digest
    )
    path.write_text(markdown, encoding="utf-8")
    return str(path), markdown


def run_text_question_test(
        srt_text: str,
        source_name: str,
        api_key: str,
        model: str,
        output_dir: str | Path,
        report: Callable[[str, int | None], None] | None = None,
) -> tuple[str, str]:
    """Generate ten questions, then validate one strict answer clip for each."""
    cues = parse_srt(srt_text)
    valid_starts = {cue["start"] for cue in cues}
    valid_ends = {cue["end"] for cue in cues}
    last_question_error = ""
    for attempt in range(1, MAX_QUESTION_ATTEMPTS + 1):
        if report:
            report(
                "阶段 1/2：正在从已清洗字幕生成 10 个精彩问题（第 {}/{} 次）。".format(
                    attempt, MAX_QUESTION_ATTEMPTS
                ),
                5,
            )
        retry_note = "" if not last_question_error else "上一轮不合规：{}。请重新生成完整结果。".format(last_question_error)
        raw_questions = _call_deepseek(
            QUESTION_SYSTEM_PROMPT,
            "请只返回问题 JSON。" + retry_note,
            srt_text,
            api_key,
            model,
            "text-question attempt {}/{}".format(attempt, MAX_QUESTION_ATTEMPTS),
            json_response=True,
        )
        try:
            questions = _questions(raw_questions)
            break
        except TextQuestionTestError as exc:
            last_question_error = str(exc)
    else:
        raise TextQuestionTestError("未能生成恰好 10 个合规问题：{}".format(last_question_error))
    if report:
        report("阶段 1/2：已生成 10 个问题，正在逐题寻找 40-90 秒回答素材。", 15)

    try:
        configured_workers = int(os.environ.get("FUNCLIP_TEXT_ANSWER_WORKERS", DEFAULT_TEXT_ANSWER_WORKERS))
    except ValueError:
        configured_workers = DEFAULT_TEXT_ANSWER_WORKERS
    worker_count = max(1, min(len(questions), max(1, configured_workers)))
    if report:
        report(
            "阶段 2/2：正在并行寻找 10 个问题的 40-90 秒回答（最多 {} 路）。".format(worker_count),
            15,
        )

    def answer_question(index: int, item: dict) -> dict:
        last_error = ""
        for attempt in range(1, MAX_ANSWER_ATTEMPTS + 1):
            if report:
                report(
                    "阶段 2/2：正在为第 {}/10 个问题寻找合规回答（第 {}/{} 次）。".format(
                        index, attempt, MAX_ANSWER_ATTEMPTS
                    ),
                    15,
                )
            retry_note = "" if not last_error else "\n上一轮不合规：{}。请重新选择。".format(last_error)
            raw_answer = _call_deepseek(
                ANSWER_SYSTEM_PROMPT,
                "请严格返回该问题的回答时间段 JSON。" + retry_note,
                _answer_request(item["question"], srt_text),
                api_key,
                model,
                "text-answer {} attempt {}/{}".format(index, attempt, MAX_ANSWER_ATTEMPTS),
                json_response=True,
            )
            try:
                ranges, summary, answer_reason = _ranges(raw_answer, valid_starts, valid_ends)
                return {
                    **item,
                    "ranges": ranges,
                    "duration_ms": sum(_time_to_ms(end) - _time_to_ms(start) for start, end in ranges),
                    "answer_summary": summary,
                    "answer_reason": answer_reason,
                    "answer_srt": build_highlight_srt(srt_text, ranges),
                }
            except TextQuestionTestError as exc:
                last_error = str(exc)
        raise TextQuestionTestError("第 {} 个问题未找到严格合规的 40-90 秒回答：{}".format(index, last_error))

    completed: list[dict | None] = [None] * len(questions)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="text-answer") as executor:
        futures = {
            executor.submit(answer_question, index, item): index - 1
            for index, item in enumerate(questions, start=1)
        }
        for future in as_completed(futures):
            question_index = futures[future]
            completed[question_index] = future.result()
            completed_count += 1
            if report:
                report(
                    "阶段 2/2：已完成 {}/10 个问题的合规回答。".format(completed_count),
                    15 + round(80 * completed_count / QUESTION_COUNT),
                )

    if report:
        report("阶段 2/2：10 个问题与答案均已完成，正在写入 Markdown。", 96)
    path, markdown = _write_markdown(source_name, [item for item in completed if item is not None], output_dir)
    if report:
        report("文本测试完成，Markdown 已生成。", 100)
    return path, markdown
