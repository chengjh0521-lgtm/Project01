"""Three-stage DeepSeek subtitle processing built on upstream FunClip calls."""

from __future__ import annotations

import os
import re
import json
import logging
import time
import urllib.error
import urllib.request

from subtitle_processing.local_correction_engine import correct_srt as correct_srt_like_local_script
from subtitle_processing.sound_effect_binding import select_sound_cues
from subtitle_processing.visual_asset_binding import select_visual_assets
from subtitle_processing.multi_highlight_stage import select_multiple
from subtitle_processing.keyword_stage import select_keywords as select_keywords_for_clip
from subtitle_processing.correction_stage import run as run_correction_stage

_SRT_TIME_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)
_RANGE_RE = re.compile(
    r"\[?\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*"
    r"(?:-|–|—|-->)\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]?"
)
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+\s*[.、)]\s*)?(?P<text>.+?)\s*$")

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
CORRECTION_BATCH_SIZE = 30
CORRECTION_CONTEXT_SIZE = 15
DEEPSEEK_TIMEOUT_SECONDS = 180
DEEPSEEK_MAX_RETRIES = 6

CORRECTION_SYSTEM_PROMPT = """
你是一名资深中文医疗访谈 ASR 字幕校对员，熟悉临床医学术语、医生口语表达，以及四川话造成的同音、近音误识别。

输入来自自动语音识别，默认可能存在错误。你必须逐条、逐字检查 target_entries，并结合 context_entries 的前后文主动发现错误。

必须严格遵守：

1. 只校对 target_entries 中的字幕正文。
2. 不得修改、合并、删除、新增或重新排序任何字幕条目。
3. 每个目标条目的 id 必须原样返回，返回数量和顺序必须与 target_ids 完全一致。
4. context_entries 仅用于理解上下文，不得出现在输出中。
5. 优先修正可由语境或医学常识明确判断的：
   - 同音字、近音字、错别字；
   - 四川话及其他口音造成的误识别；
   - 漏字、多字、重复识别；
   - 明显错误的断句或标点；
   - 医学术语、疾病名称、药物名称；
   - 检查项目、治疗方式、解剖部位；
   - 剂量、数值、百分比、时间和单位。
6. 不要因为整句话表面通顺就默认正确；对疑似词必须结合医学语境重新判断。
7. 能够明确判断的错误应直接修正，不能因为过度保守而保留明显错误。
8. 确实无法根据上下文确定时，保留原文，不得臆造。
9. 只做校对，不做改写：
   - 保留原有表达顺序；
   - 保留医生和患者的口语风格；
   - 保留原本的重复、停顿词和语气；
   - 不总结、不扩写、不解释；
   - 不润色成书面语；
   - 不擅自改变医学观点。
10. text 中只能包含校对后的字幕正文：
   - 不得包含时间戳；
   - 不得包含字幕编号；
   - 不得包含 spk0、spk1 等说话人标签；
   - 不得包含 Markdown、批注、解释或修改说明。
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
                "header": (
                    lines[time_index - 1].strip()
                    if time_index > 0 and re.match(r"^\s*\d+(?:\s+.*)?$", lines[time_index - 1])
                    else str(len(cues) + 1)
                ),
                "start": _normalize_timecode(match.group("start")),
                "end": _normalize_timecode(match.group("end")),
                "text": text,
            }
        )
    return cues


def render_srt(cues: list[dict[str, str]]) -> str:
    return "\n\n".join(
        "{}\n{} --> {}\n{}".format(cue.get("header", index), cue["start"], cue["end"], cue["text"])
        for index, cue in enumerate(cues, start=1)
    ) + "\n"


def _call_deepseek(
        system_prompt: str,
        user_prompt: str,
        content: str,
        api_key: str,
        model: str,
        stage_label: str,
        json_response: bool = False):
    """Make one direct DeepSeek API request with deterministic JSON support."""
    if not api_key:
        raise SubtitlePipelineError("请填写 DeepSeek API Key，或设置服务器的 DEEPSEEK_API_KEY。")
    request_content = "{}\n\n{}".format(user_prompt, content)
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request_content},
        ],
        "temperature": 0,
        "stream": False,
    }
    if json_response:
        request_body["response_format"] = {"type": "json_object"}
    encoded_body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    logging.warning(
        "DeepSeek API %s：已发送，模型=%s，system=%d 字符，输入=%d 字符，JSON=%s。",
        stage_label,
        model,
        len(system_prompt),
        len(content),
        json_response,
    )
    last_error = None
    for attempt in range(DEEPSEEK_MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                DEEPSEEK_API_URL,
                data=encoded_body,
                method="POST",
                headers={
                    "Authorization": "Bearer {}".format(api_key),
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "FunClip-Module-Gateway/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=DEEPSEEK_TIMEOUT_SECONDS) as response:
                api_response = json.loads(response.read().decode("utf-8"))
            choices = api_response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("DeepSeek 返回中没有 choices。")
            result = choices[0].get("message", {}).get("content")
            if not isinstance(result, str) or not result.strip():
                raise ValueError("DeepSeek 返回了空 message.content。")
            result = result.strip()
            logging.warning("DeepSeek API %s：已收到，返回=%d 字符。", stage_label, len(result))
            return result
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = "HTTP {}: {}".format(exc.code, body or exc.reason)
            if exc.code in (400, 401, 402, 403, 404, 422):
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if attempt < DEEPSEEK_MAX_RETRIES:
            wait_seconds = min(2 ** attempt * 2, 45)
            logging.warning(
                "DeepSeek API %s：第 %d 次失败：%s；%d 秒后重试。",
                stage_label,
                attempt + 1,
                last_error,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise SubtitlePipelineError("DeepSeek API {} 最终失败：{}".format(stage_label, last_error))


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


def correct_srt(srt_text: str, api_key: str, model: str, status_callback=None) -> str:
    original = parse_srt(srt_text)
    total_batches = (len(original) + CORRECTION_BATCH_SIZE - 1) // CORRECTION_BATCH_SIZE
    for batch_number, start in enumerate(range(0, len(original), CORRECTION_BATCH_SIZE), start=1):
        end = min(start + CORRECTION_BATCH_SIZE, len(original))
        context_start = max(0, start - CORRECTION_CONTEXT_SIZE)
        context_end = min(len(original), end + CORRECTION_CONTEXT_SIZE)
        target_entries = [
            {"id": str(index + 1), "text": original[index]["text"]}
            for index in range(start, end)
        ]
        request = {
            "instruction": "只校对 target_entries。context_entries 仅供理解语境。严格按 target_ids 的数量、ID 和顺序返回 JSON。",
            "target_ids": [entry["id"] for entry in target_entries],
            "context_entries": [
                {
                    "id": str(index + 1),
                    "speaker": original[index].get("header", str(index + 1)),
                    "text": original[index]["text"],
                }
                for index in range(context_start, context_end)
            ],
            "target_entries": target_entries,
        }
        logging.warning(
            "字幕处理阶段 1/3：第 %d/%d 批发送 %d 条（%d-%d），剩余 %d 条未发送。",
            batch_number,
            total_batches,
            end - start,
            start + 1,
            end,
            len(original) - end,
        )
        if status_callback:
            status_callback(
                "阶段 1/3 洗稿：第 {}/{} 批，发送第 {}-{} 条字幕。".format(
                    batch_number, total_batches, start + 1, end
                )
            )
        response = _call_deepseek(
            CORRECTION_SYSTEM_PROMPT,
            CORRECTION_USER_PROMPT,
            json.dumps(request, ensure_ascii=False),
            api_key,
            model,
            "阶段 1/3 洗稿 第 {}/{} 批".format(batch_number, total_batches),
            json_response=True,
        )
        corrected_texts = _parse_correction_response(response, request["target_ids"])
        for cue, corrected_text in zip(original[start:end], corrected_texts):
            cue["text"] = corrected_text
        logging.warning(
            "字幕处理阶段 1/3：第 %d/%d 批已收到 %d 条，累计完成 %d/%d 条。",
            batch_number,
            total_batches,
            len(corrected_texts),
            end,
            len(original),
        )
        if status_callback:
            status_callback(
                "阶段 1/3 洗稿：第 {}/{} 批完成，累计 {} / {} 条。".format(
                    batch_number, total_batches, end, len(original)
                )
            )
    return render_srt(original)


def correct_srt(srt_text: str, api_key: str, model: str, status_callback=None) -> str:
    """Correct subtitles with the standalone script's parser and batch protocol."""
    source_total = len(parse_srt(srt_text))

    def call_api(payload: dict, batch_number: int, total_batches: int) -> str:
        sent = len(payload["target_entries"])
        remaining = max(0, source_total - batch_number * CORRECTION_BATCH_SIZE)
        logging.warning(
            "Stage 1/3 correction: batch %d/%d sent %d cues; %d cues remain.",
            batch_number, total_batches, sent, remaining,
        )
        if status_callback:
            status_callback(
                "Stage 1/3 correction: batch {}/{}, sent {} cues; {} remain.".format(
                    batch_number, total_batches, sent, remaining
                )
            )
        return _call_deepseek(
            CORRECTION_SYSTEM_PROMPT,
            "Please correct the following JSON. Return only the required JSON object:",
            json.dumps(payload, ensure_ascii=False),
            api_key,
            model,
            "stage 1/3 correction batch {}/{}".format(batch_number, total_batches),
            json_response=True,
        )

    def progress(batch_number: int, total_batches: int, completed: int, total: int) -> None:
        logging.warning(
            "Stage 1/3 correction: batch %d/%d received; %d/%d cues complete.",
            batch_number, total_batches, completed, total,
        )
        if status_callback:
            status_callback(
                "Stage 1/3 correction: batch {}/{} received; {} / {} cues complete.".format(
                    batch_number, total_batches, completed, total
                )
            )

    return correct_srt_like_local_script(
        srt_text,
        call_api,
        batch_size=CORRECTION_BATCH_SIZE,
        context_size=CORRECTION_CONTEXT_SIZE,
        progress=progress,
    )


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
        raise SubtitlePipelineError("高光提取未返回可用时间戳；请查看后端日志中的原始模型返回。")
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
    response = _call_deepseek(
        KEYWORD_SYSTEM_PROMPT,
        user_prompt,
        highlight_srt,
        api_key,
        model,
        "阶段 3/3 关键词",
    )
    logging.warning("字幕处理阶段 3/3：关键词原始返回（%d 字符）：\n%s", len(response), response)
    keywords = []
    for part in re.split(r"[\n,，;；、]+", _strip_code_fence(response)):
        match = _NUMBERED_LINE_RE.match(part)
        keyword = match.group("text").strip(" \t\"'“”") if match else ""
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    if not keywords:
        raise SubtitlePipelineError("关键词识别没有返回有效内容。")
    return "\n".join(keywords[:keyword_count])


def process_subtitles(
        srt_text: str,
        api_key: str,
        keyword_count: int,
        video_state=None,
        model: str | None = None,
        status_callback=None):
    """Run correction, highlight extraction, and keyword selection sequentially."""
    selected_model = model or os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-chat")
    source_count = len(parse_srt(srt_text))
    logging.warning(
        "字幕处理开始：模型=%s，原始字幕=%d 条，关键词目标=%d 个。",
        selected_model,
        source_count,
        keyword_count,
    )

    logging.warning("字幕处理阶段 1/3：正在调用 DeepSeek 洗稿。")
    if status_callback:
        status_callback("阶段 1/3：开始 DeepSeek 分批洗稿。")
    try:
        corrected_srt = correct_srt(srt_text, api_key, selected_model, status_callback)
    except Exception as exc:
        logging.exception("字幕处理阶段 1/3：洗稿失败。")
        raise SubtitlePipelineError("阶段 1/3 洗稿失败：{}".format(exc)) from exc
    corrected_count = len(parse_srt(corrected_srt))
    logging.warning("字幕处理阶段 1/3：洗稿完成，字幕2=%d 条。", corrected_count)
    if status_callback:
        status_callback("阶段 1/3：洗稿完成，得到字幕2。")

    logging.warning("字幕处理阶段 2/3：正在调用 DeepSeek 提取高光片段。")
    if status_callback:
        status_callback("阶段 2/3：正在调用 DeepSeek 提取高光片段。")
    try:
        raw_highlights = _call_deepseek(
            HIGHLIGHT_SYSTEM_PROMPT,
            HIGHLIGHT_USER_PROMPT,
            corrected_srt,
            api_key,
            selected_model,
            "阶段 2/3 高光",
        )
        logging.warning(
            "字幕处理阶段 2/3：高光原始返回（%d 字符）：\n%s",
            len(raw_highlights),
            raw_highlights,
        )
        ranges = extract_highlight_ranges(raw_highlights)
    except Exception as exc:
        logging.exception("字幕处理阶段 2/3：高光提取失败。")
        raise SubtitlePipelineError("阶段 2/3 高光提取失败：{}".format(exc)) from exc
    logging.warning("字幕处理阶段 2/3：高光提取完成，识别到 %d 个时间段。", len(ranges))
    if status_callback:
        status_callback("阶段 2/3：高光提取完成，识别到 {} 个时间段。".format(len(ranges)))
    canonical_ranges = "\n".join("[{}-{}]".format(start, end) for start, end in ranges)
    highlight_srt = build_highlight_srt(corrected_srt, ranges)

    logging.warning("字幕处理阶段 3/3：正在调用 DeepSeek 提取关键词。")
    if status_callback:
        status_callback("阶段 3/3：正在调用 DeepSeek 提取关键词。")
    try:
        keywords = select_keywords(highlight_srt, api_key, selected_model, max(1, int(keyword_count)))
    except Exception as exc:
        logging.exception("字幕处理阶段 3/3：关键词提取失败。")
        raise SubtitlePipelineError("阶段 3/3 关键词提取失败：{}".format(exc)) from exc
    logging.warning("字幕处理阶段 3/3：关键词提取完成，返回 %d 个关键词。", len(keywords.splitlines()))
    if status_callback:
        status_callback("阶段 3/3：关键词提取完成，返回 {} 个关键词。".format(len(keywords.splitlines())))

    logging.warning("字幕处理阶段 4/4：正在按历史规则和 DeepSeek 绑定音效。")
    if status_callback:
        status_callback("阶段 4/4：正在为关键词绑定音效。")
    try:
        sound_bindings = select_sound_cues(
            highlight_srt,
            keywords,
            api_key,
            selected_model,
            lambda system, user, content, key, selected: _call_deepseek(
                system, user, content, key, selected, "stage 4/4 sound effects", json_response=True
            ),
        )
    except Exception as exc:
        logging.exception("字幕处理阶段 4/4：音效绑定失败。")
        sound_bindings = '{"bindings": []}'
        logging.warning("字幕处理阶段 4/4：忽略音效绑定错误：%s", exc)
    if status_callback:
        status_callback("阶段 4/4：音效绑定完成。")
    range_summary = "\n".join("[{}-{}]".format(start, end) for start, end in ranges)
    highlight_display = "高光时间戳：\n{}\n\n字幕3：\n{}".format(range_summary, highlight_srt)
    return (
        corrected_srt,
        highlight_display,
        keywords,
        sound_bindings,
        canonical_ranges,
        build_corrected_video_state(video_state, corrected_srt),
    )


def _process_from_corrected_subtitles(
        corrected_srt: str, api_key: str, keyword_count: int, clip_count: int,
        video_state=None, model: str | None = None, status_callback=None):
    """Run stages 2-5 from an already corrected SRT subtitle file."""
    selected_model = model or os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash")
    def call_stage(system, user):
        return _call_deepseek(
            system, "Return the required JSON object only.", user, api_key, selected_model,
            "multi-highlight stage", json_response=True,
        )

    candidates = select_multiple(
        corrected_srt, max(1, int(clip_count)), call_stage, report=status_callback
    )
    if not candidates:
        raise SubtitlePipelineError("未提取到满足重合度限制的高光素材。")
    for index, candidate in enumerate(candidates, start=1):
        highlight_srt = build_highlight_srt(corrected_srt, candidate["ranges"])
        keywords = select_keywords_for_clip(
            highlight_srt, keyword_count,
            lambda system, user: _call_deepseek(system, user, "", api_key, selected_model, "keyword stage"),
        )
        sound_bindings = select_sound_cues(
            highlight_srt, keywords, api_key, selected_model,
            lambda system, user, content, key, chosen_model: _call_deepseek(
                system, user, content, key, chosen_model, "sound-effect stage", json_response=True
            ),
        )
        if status_callback:
            status_callback("阶段 5/5：正在为第 {} / {} 条素材选择 GIF/PNG。".format(index, len(candidates)))
        try:
            visual_bindings = select_visual_assets(
                highlight_srt, keywords, api_key, selected_model,
                lambda system, user, content, key, chosen_model: _call_deepseek(
                    system, user, content, key, chosen_model, "visual-asset stage", json_response=True
                ),
            )
        except Exception as exc:
            logging.exception("字幕处理阶段 5/5：视觉素材选择失败。")
            visual_bindings = '{"placements": []}'
            logging.warning("字幕处理阶段 5/5：忽略视觉素材选择错误：%s", exc)
        candidate.update({
            "highlight_srt": highlight_srt,
            "keywords": keywords,
            "sound_bindings": sound_bindings,
            "visual_bindings": visual_bindings,
        })
        if status_callback:
            status_callback("阶段 3-5/5：已完成第 {} / {} 条素材的关键词、音效与 GIF/PNG 选择。".format(index, len(candidates)))
    display = "\n\n".join(
        "素材 {}\n可回答的问题：{}\n\n高光时间戳：\n{}\n\n字幕3：\n{}\n\n选择理由：{}".format(
            index,
            candidate.get("question", "未返回问题"),
            "\n".join("[{}-{}]".format(start, end) for start, end in candidate["ranges"]),
            candidate.get("highlight_srt", ""),
            candidate.get("highlight_reason", "未返回理由"),
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    # Keep subtitle 2 with the render plan so the web layer can rebuild the
    # upstream video state even after Gradio has dropped its hidden State.
    plan = {"clips": candidates, "corrected_srt": corrected_srt}
    return corrected_srt, display, plan, build_corrected_video_state(video_state, corrected_srt)


def process_multiple_subtitles(
        srt_text: str, api_key: str, keyword_count: int, clip_count: int,
        video_state=None, model: str | None = None, status_callback=None, on_corrected=None):
    """Run all five stages beginning with DeepSeek subtitle correction."""
    selected_model = model or os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash")
    if status_callback:
        status_callback("阶段 1/5：正在校对字幕。")
    corrected_srt = run_correction_stage(
        srt_text, lambda source: correct_srt(source, api_key, selected_model, status_callback)
    )
    if on_corrected:
        on_corrected(corrected_srt)
    return _process_from_corrected_subtitles(
        corrected_srt, api_key, keyword_count, clip_count, video_state, selected_model, status_callback
    )


def process_from_corrected_subtitles(
        corrected_srt: str, api_key: str, keyword_count: int, clip_count: int,
        video_state=None, model: str | None = None, status_callback=None):
    """Resume processing from a persisted subtitle 2 file without another correction call."""
    try:
        parse_srt(corrected_srt)
    except Exception as exc:
        raise SubtitlePipelineError("保存的字幕2不是有效 SRT：{}".format(exc)) from exc
    if status_callback:
        status_callback("阶段 1/5：已载入保存的字幕2，跳过 ASR 与洗稿。")
    return _process_from_corrected_subtitles(
        corrected_srt, api_key, keyword_count, clip_count, video_state, model, status_callback
    )
