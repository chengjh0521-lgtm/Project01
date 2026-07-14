"""Three-stage DeepSeek subtitle processing built on upstream FunClip calls."""

from __future__ import annotations

import os
import re
import json

from funclip_loader import get_launch


_SRT_TIME_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)
_RANGE_RE = re.compile(
    r"\[?\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*"
    r"(?:-|–|—|-->)\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]?"
)
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+\s*[.、)]\s*)?(?P<text>.+?)\s*$")

CORRECTION_SYSTEM_PROMPT = """
你是一名资深中文医疗访谈 ASR 字幕校对员，熟悉临床医学术语、医生口语表达，以及四川话造成的同音、近音误识别。

输入来自自动语音识别，默认可能存在错误。你必须逐条、逐字检查 target_entries，并结合 context_entries 的前后文主动发现错误。

必须严格遵守：
1. 只校对 target_entries 中的字幕正文。
2. 不得修改、合并、删除、新增或重新排序任何字幕条目。
3. 每个目标条目的 id 必须原样返回，返回数量和顺序必须与 target_ids 完全一致。
4. context_entries 仅用于理解上下文，不得出现在输出中。
5. 优先修正可由语境或医学常识明确判断的同音字、近音字、错别字、四川话及其他口音造成的误识别、漏字、多字、重复识别、明显错误的断句或标点、医学术语、疾病名称、药物名称、检查项目、治疗方式、解剖部位、剂量、数值、百分比、时间和单位。
6. 不要因为整句话表面通顺就默认正确；对疑似词必须结合医学语境重新判断。
7. 能够明确判断的错误应直接修正，不能因为过度保守而保留明显错误。
8. 确实无法根据上下文确定时，保留原文，不得臆造。
9. 只做校对，不做改写：保留原有表达顺序、医生和患者的口语风格、原本的重复、停顿词和语气；不总结、不扩写、不解释、不润色成书面语、不擅自改变医学观点。
10. text 中只能包含校对后的字幕正文，不得包含时间戳、字幕编号、spk0、spk1 等说话人标签、Markdown、批注、解释或修改说明。
11. 原文没有错误时，必须原样返回，不要为了显示修改而改变文字。

只返回一个合法 JSON 对象，格式必须为：
{
  "entries": [
    {"id": "1", "text": "校对后的字幕正文"},
    {"id": "2", "text": "校对后的字幕正文"}
  ]
}
""".strip()
CORRECTION_USER_PROMPT = "输入 JSON 中的 target_entries 是待校对字幕，context_entries 是完整上下文。请严格按系统要求返回 JSON。"
KEYWORD_SYSTEM_PROMPT = """
你是一个短视频字幕高亮分析器。

输入是一段已经筛选好的视频字幕，这段字幕本身已经具有完整的知识价值。你的任务是从字幕中挑选最值得在视频中进行视觉强调（高亮、变色、放大、音效）的关键词。

请遵循以下规则：
1. 只选择真正影响观众理解或传播效果的关键词，不要为了数量而选择。
2. 优先选择疾病名称、症状、药物名称、检查项目、治疗方式、生活习惯、行为动作、风险因素、医学结论、否定词、数字、百分比、时间、剂量、身体部位。例如：糖尿病、高血压、痛风、脂肪肝、抽烟、吸烟、喝酒、饮酒、熬夜、减肥、运动、吃药、胰岛素、二甲双胍、头晕、胸痛、失眠、血糖、血压、CT、核磁、7%、120、3个月、每天两次、不能、必须、一定、千万不要、建议、禁止、可以、不可以、需要、推荐、不推荐。
3. 不要选择助词、连词、语气词、人称代词、医生、患者等无实际传播价值的词，或重复出现且没有强调意义的普通词汇。
4. 输出的关键词必须完全出现在输入字幕中，不允许修改文字，不允许生成同义词。
5. 每个关键词长度一般控制在 1~6 个汉字。
6. 每段字幕最多输出 8 个关键词，宁缺毋滥。
将每个关键词以逗号或者换行隔开，除关键词外不要输出其他内容。
""".strip()
HIGHLIGHT_SYSTEM_PROMPT = """
你是一个专业的医学科普视频剪辑分析器。

输入为一份完整的视频 SRT 字幕，内容为医患之间的真实对话，其中包含大量寒暄、重复表达、病史询问、无意义停顿、情绪交流等无关内容。

你的任务是从整份字幕中提炼出适合制作短视频的医学知识片段。主角（医生）所说的语言是四川话，可能被 ASR 错误识别为其他内容；你需要联系上下文，尽量还原医生本意，适当理解字幕中的错误翻译。

要求如下：
1. 以知识点为核心，而不是按视频顺序。自动识别具有独立知识价值的医学科普主题。
2. 允许跨时间段组合。可以从视频不同位置提取同一知识点，不要求保持视频原始出现顺序，应按照最容易理解、逻辑最完整的顺序重新组织。
3. 删除所有无关内容，包括问候语、病史确认、患者重复提问、医生口头禅、重复解释、闲聊、无意义停顿和情绪表达。
4. 每个输出片段必须围绕一个完整主题，不要混杂多个主题。
5. 每个输出片段总时长保持在 40~90 秒；不足 40 秒可补充同主题内容，超过 90 秒保留最重要内容。
6. 输出组成片段的所有连续字幕段；连续字幕合并为一条 [开始时间-结束时间] 文本；不连续位置分别输出，并保持重组后的知识逻辑顺序。
7. 优先选择结论明确、通俗易懂、医学知识完整、对普通观众有价值且适合短视频传播的内容。
8. 最多输出 1 个知识主题，且优先选择与烟酒有关的主题。

输出格式严格如下：
1. [开始时间-结束时间] 文本
2. [开始时间-结束时间] 文本
3. [开始时间-结束时间] 文本

除上述内容外，不输出任何解释、分析、总结或额外文字。
""".strip()
HIGHLIGHT_USER_PROMPT = "以下是完整的校对后 SRT 字幕，请按系统要求提取高光。"


class SubtitlePipelineError(ValueError):
    pass


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:srt|text|plaintext)?\s*([\s\S]*?)\s*```", text, re.I)
    return match.group(1).strip() if match else text.strip()


def _normalize_timecode(value: str) -> str:
    hours, minutes, seconds = value.strip().replace(".", ",").split(":")
    second, millis = seconds.split(",")
    return "{:02d}:{:02d}:{:02d},{:03d}".format(
        int(hours), int(minutes), int(second), int(millis.ljust(3, "0")[:3])
    )


def _time_to_ms(value: str) -> int:
    normalized = _normalize_timecode(value)
    hours, minutes, second_part = normalized.split(":")
    seconds, millis = second_part.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


def parse_srt(srt_text: str) -> list[dict[str, str]]:
    lines = _strip_code_fence(str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n")).splitlines()
    timestamps = [index for index, line in enumerate(lines) if _SRT_TIME_RE.match(line)]
    if not timestamps:
        raise SubtitlePipelineError("未识别到有效 SRT 时间轴。")

    cues = []
    for position, time_index in enumerate(timestamps):
        match = _SRT_TIME_RE.match(lines[time_index])
        assert match is not None
        end_index = timestamps[position + 1] if position + 1 < len(timestamps) else len(lines)
        if end_index > time_index + 1 and re.match(r"^\s*\d+(?:\s+spk\S+)?\s*$", lines[end_index - 1]):
            end_index -= 1
        text = " ".join(line.strip() for line in lines[time_index + 1 : end_index] if line.strip())
        if not text:
            raise SubtitlePipelineError("存在没有文字内容的 SRT 条目。")
        cues.append(
            {
                "start": _normalize_timecode(match.group("start")),
                "end": _normalize_timecode(match.group("end")),
                "text": text,
            }
        )
    return cues


def render_srt(cues: list[dict[str, str]]) -> str:
    return "\n\n".join(
        "{}\n{} --> {}\n{}".format(index, cue["start"], cue["end"], cue["text"])
        for index, cue in enumerate(cues, start=1)
    ) + "\n"


def _call_deepseek(system_prompt: str, user_prompt: str, content: str, api_key: str, model: str) -> str:
    if not api_key:
        raise SubtitlePipelineError("请填写 DeepSeek API Key，或设置服务器的 DEEPSEEK_API_KEY。")
    launch = get_launch()
    result = launch.llm_inference(system_prompt, user_prompt, content, model, api_key)
    result = str(result or "").strip()
    if not result or result.lower().startswith("llm inference failed"):
        raise SubtitlePipelineError(result or "DeepSeek 没有返回内容。")
    return result


def _parse_correction_response(response: str, expected_ids: list[str]) -> list[str]:
    text = _strip_code_fence(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("{")
        if start < 0:
            raise SubtitlePipelineError("字幕洗稿未返回合法 JSON。")
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise SubtitlePipelineError("字幕洗稿未返回合法 JSON。") from exc
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or len(entries) != len(expected_ids):
        raise SubtitlePipelineError("字幕洗稿返回的 entries 数量与原字幕不一致。")
    corrected_texts = []
    for expected_id, entry in zip(expected_ids, entries):
        if not isinstance(entry, dict) or str(entry.get("id")) != expected_id:
            raise SubtitlePipelineError("字幕洗稿返回的 id 与原字幕顺序不一致。")
        text_value = entry.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise SubtitlePipelineError("字幕洗稿返回了空字幕正文。")
        corrected_texts.append(text_value.strip())
    return corrected_texts


def correct_srt(srt_text: str, api_key: str, model: str) -> str:
    original = parse_srt(srt_text)
    target_entries = [
        {"id": str(index), "text": cue["text"]}
        for index, cue in enumerate(original, start=1)
    ]
    request = {
        "target_ids": [entry["id"] for entry in target_entries],
        "target_entries": target_entries,
        # The entire transcript is deliberately supplied once more as context.
        # The model may use it for disambiguation but is forbidden to output it.
        "context_entries": target_entries,
    }
    response = _call_deepseek(
        CORRECTION_SYSTEM_PROMPT,
        CORRECTION_USER_PROMPT,
        json.dumps(request, ensure_ascii=False),
        api_key,
        model,
    )
    corrected_texts = _parse_correction_response(response, request["target_ids"])
    # LLM owns text only. The ASR time axis remains the sole video time source.
    for cue, corrected_text in zip(original, corrected_texts):
        cue["text"] = corrected_text
    return render_srt(original)


def extract_highlight_ranges(llm_result: str) -> list[tuple[str, str]]:
    seen = set()
    ranges = []
    for match in _RANGE_RE.finditer(llm_result):
        start, end = _normalize_timecode(match.group("start")), _normalize_timecode(match.group("end"))
        if _time_to_ms(end) <= _time_to_ms(start) or (start, end) in seen:
            continue
        seen.add((start, end))
        ranges.append((start, end))
    if not ranges:
        raise SubtitlePipelineError("高光提取未返回可用时间戳。")
    return ranges


def build_highlight_srt(srt_text: str, ranges: list[tuple[str, str]]) -> str:
    cues = parse_srt(srt_text)
    selected = []
    used_indexes = set()
    for range_start, range_end in ranges:
        start_ms, end_ms = _time_to_ms(range_start), _time_to_ms(range_end)
        for index, cue in enumerate(cues):
            if index in used_indexes:
                continue
            if _time_to_ms(cue["end"]) > start_ms and _time_to_ms(cue["start"]) < end_ms:
                used_indexes.add(index)
                selected.append(cue)
    if not selected:
        raise SubtitlePipelineError("高光时间戳没有匹配到校对后的字幕。")
    return render_srt(selected)


def build_corrected_video_state(video_state, corrected_srt: str):
    if video_state is None:
        return None
    state = dict(video_state)
    sentences, timestamps, raw_tokens = [], [], []
    for cue in parse_srt(corrected_srt):
        start_ms, end_ms = _time_to_ms(cue["start"]), _time_to_ms(cue["end"])
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_-]+", cue["text"]) or [cue["text"]]
        duration = max(len(tokens), end_ms - start_ms)
        token_timestamps = []
        for index in range(len(tokens)):
            token_start = start_ms + round(duration * index / len(tokens))
            token_end = start_ms + round(duration * (index + 1) / len(tokens))
            token_timestamps.append([token_start, max(token_start + 1, token_end)])
        token_timestamps[-1][1] = max(token_timestamps[-1][0] + 1, end_ms)
        sentences.append({"text": cue["text"], "timestamp": token_timestamps})
        timestamps.extend(token_timestamps)
        raw_tokens.extend(tokens)
    state["sentences"] = sentences
    state["timestamp"] = timestamps
    state["recog_res_raw"] = " ".join(raw_tokens)
    return state


def select_keywords(highlight_srt: str, api_key: str, model: str, keyword_count: int) -> str:
    user_prompt = (
        "以下是高光字幕。请按系统要求选择不超过 {} 个高价值关键词或短语。"
        "每行或每个逗号后只输出一个关键词，不得输出序号、解释、标点或其他文字。".format(keyword_count)
    )
    response = _call_deepseek(KEYWORD_SYSTEM_PROMPT, user_prompt, highlight_srt, api_key, model)
    keywords = []
    for part in re.split(r"[\n,，;；、]+", _strip_code_fence(response)):
        match = _NUMBERED_LINE_RE.match(part)
        keyword = match.group("text").strip(" \t\"'“”") if match else ""
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    if not keywords:
        raise SubtitlePipelineError("关键词识别没有返回有效内容。")
    return "\n".join(keywords[:keyword_count])


def process_subtitles(srt_text: str, api_key: str, keyword_count: int, video_state=None, model: str | None = None):
    """Run correction, highlight extraction, and keyword selection sequentially."""
    selected_model = model or os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-chat")
    corrected_srt = correct_srt(srt_text, api_key, selected_model)
    raw_highlights = _call_deepseek(
        HIGHLIGHT_SYSTEM_PROMPT, HIGHLIGHT_USER_PROMPT, corrected_srt, api_key, selected_model
    )
    ranges = extract_highlight_ranges(raw_highlights)
    canonical_ranges = "\n".join("[{}-{}]".format(start, end) for start, end in ranges)
    highlight_srt = build_highlight_srt(corrected_srt, ranges)
    keywords = select_keywords(highlight_srt, api_key, selected_model, max(1, int(keyword_count)))
    range_summary = "\n".join("[{}-{}]".format(start, end) for start, end in ranges)
    highlight_display = "高光时间戳：\n{}\n\n字幕3：\n{}".format(range_summary, highlight_srt)
    return (
        corrected_srt,
        highlight_display,
        keywords,
        canonical_ranges,
        build_corrected_video_state(video_state, corrected_srt),
    )
