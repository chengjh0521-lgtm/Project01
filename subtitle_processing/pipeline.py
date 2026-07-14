"""Three-stage DeepSeek subtitle processing built on upstream FunClip calls."""

from __future__ import annotations

import os
import re

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

CORRECTION_SYSTEM_PROMPT = """你是一名医疗视频字幕校对员，熟悉医生口语、医学术语和四川话常见同音误识别。
结合相邻字幕修正可明确判断的同音字、错别字、漏字、标点和医学术语。保留原意、语气和表达顺序；
不总结、不扩写、不删除内容，不改动任何字幕的数量、序号和时间轴。"""
CORRECTION_USER_PROMPT = """以下是 SRT 字幕。请逐条校对，严格输出完整 SRT：每条均为“序号 + 时间轴 + 修正后的文本”，
时间轴与条目数量必须和输入完全一致。除 SRT 内容外不要输出标题、解释、分析、Markdown 或代码块。"""
HIGHLIGHT_SYSTEM_PROMPT = """你是一名中文医疗短视频剪辑师。根据给定的校对后 SRT，选择主题完整、有信息价值且吸引人的高光片段。
避免只截取半句话；优先保留医生的结论、反差、类比、提醒和情绪转折。"""
HIGHLIGHT_USER_PROMPT = """从以下 SRT 中选择高光片段。每行只输出一个时间范围，格式必须严格为：
[HH:MM:SS,mmm-HH:MM:SS,mmm]
不要输出任何文字说明、编号、破折号、分析或总结。"""
KEYWORD_SYSTEM_PROMPT = """你负责为医疗短视频识别适合字幕强调的关键词或短语。关键词必须来自输入字幕，
优先选择疾病、指标、风险、结论、药物、剂量或有传播力的短语。"""


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


def correct_srt(srt_text: str, api_key: str, model: str) -> str:
    original = parse_srt(srt_text)
    response = _call_deepseek(CORRECTION_SYSTEM_PROMPT, CORRECTION_USER_PROMPT, srt_text, api_key, model)
    try:
        corrected = parse_srt(response)
        corrected_texts = [cue["text"] for cue in corrected]
    except SubtitlePipelineError:
        corrected_texts = []
        for line in _strip_code_fence(response).splitlines():
            match = _NUMBERED_LINE_RE.match(line)
            if not match:
                continue
            text = _RANGE_RE.sub("", match.group("text"), count=1).strip()
            if text:
                corrected_texts.append(text)
    if len(corrected_texts) != len(original):
        raise SubtitlePipelineError(
            "字幕洗稿返回 {} 条，原字幕为 {} 条；为保证时间轴准确，未使用该结果。".format(
                len(corrected_texts), len(original)
            )
        )
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
        "从以下高光字幕中选择恰好 {} 个关键词或短语。每行只输出一个关键词或短语，"
        "不得输出序号、解释、标点或其他文字。".format(keyword_count)
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
