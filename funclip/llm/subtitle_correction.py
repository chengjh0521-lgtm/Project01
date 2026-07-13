import re


DEFAULT_SUBTITLE_CORRECTION_PROMPT = (
    "你是一名医疗视频字幕校对员。请结合上下文修正 ASR 造成的同音字、错别字、漏标点和医学术语错误，"
    "尤其留意四川话口音可能导致的误识别。保持医生原本的语气和含义，不扩写、不总结、不删除有效信息。"
)

_TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*"
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}(?:\s+.*)?$"
)
_TIMESTAMP_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_CORRECTION_LINE_RE = re.compile(
    r"^\s*(?:\d+[.、)]\s*)?\[\s*"
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*\]\s*"
    r"(?P<text>.+?)\s*$"
)


class SubtitleCorrectionError(ValueError):
    pass


def parse_srt_entries(srt_text):
    normalized = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SubtitleCorrectionError("SRT subtitles are empty. Run ASR first.")

    # FunClip's historical speaker-diarization SRT output omits blank lines
    # between cues. Parse from timestamp lines so both standard and continuous
    # SRT variants are accepted.
    lines = normalized.splitlines()
    timestamp_indices = [
        index for index, line in enumerate(lines) if _TIMESTAMP_RE.match(line)
    ]
    if not timestamp_indices:
        raise SubtitleCorrectionError("No valid SRT subtitle blocks were found.")

    entries = []
    for current_index, timestamp_index in enumerate(timestamp_indices):
        next_timestamp_index = (
            timestamp_indices[current_index + 1]
            if current_index + 1 < len(timestamp_indices)
            else len(lines)
        )
        prefix_index = timestamp_index - 1
        has_prefix = prefix_index >= 0 and re.match(
            r"^\s*\d+(?:\s+spk\S+)?\s*$", lines[prefix_index]
        )
        text_end = next_timestamp_index
        if current_index + 1 < len(timestamp_indices):
            next_prefix_index = next_timestamp_index - 1
            if re.match(r"^\s*\d+(?:\s+spk\S+)?\s*$", lines[next_prefix_index]):
                text_end = next_prefix_index
        text = "\n".join(lines[timestamp_index + 1 : text_end]).strip()
        if not text:
            raise SubtitleCorrectionError("Invalid SRT block: subtitle text is empty.")
        entries.append(
            {
                "prefix": [lines[prefix_index]] if has_prefix else [],
                "timestamp": lines[timestamp_index],
                "text": text,
            }
        )
    return entries


def render_srt_entries(entries):
    blocks = []
    for entry in entries:
        lines = list(entry.get("prefix") or [])
        lines.extend([entry["timestamp"], entry["text"]])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _normalize_timestamp(time_text):
    return str(time_text).strip().replace(".", ",")


def _timestamp_to_millis(time_text):
    hours, minutes, seconds_and_millis = _normalize_timestamp(time_text).split(":")
    seconds, millis = seconds_and_millis.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis.ljust(3, "0")[:3])
    )


def _entry_timestamp_range(entry):
    match = _TIMESTAMP_RANGE_RE.match(entry["timestamp"])
    if not match:
        raise SubtitleCorrectionError("Invalid SRT timestamp line.")
    return _normalize_timestamp(match.group("start")), _normalize_timestamp(match.group("end"))


def _entry_timestamp_millis_range(entry):
    start_time, end_time = _entry_timestamp_range(entry)
    return _timestamp_to_millis(start_time), _timestamp_to_millis(end_time)


def _sentence_timestamp_millis_range(sentence):
    timestamps = sentence.get("timestamp") if isinstance(sentence, dict) else None
    if not isinstance(timestamps, list) or not timestamps:
        return None
    try:
        return int(timestamps[0][0]), int(timestamps[-1][1])
    except (IndexError, TypeError, ValueError):
        return None


def _parse_correction_lines(response_text):
    text = str(response_text or "").strip()
    if not text:
        raise SubtitleCorrectionError("DeepSeek returned an empty response.")
    if (
        text.lower().startswith("llm inference failed:")
        or text.lower().startswith("api key is required")
    ):
        raise SubtitleCorrectionError(text)

    fenced = re.search(r"```(?:text)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    corrections = []
    for line in text.splitlines():
        match = _CORRECTION_LINE_RE.match(line)
        if not match:
            continue
        corrected_text = match.group("text").strip()
        if corrected_text:
            corrections.append(
                (
                    _normalize_timestamp(match.group("start")),
                    _normalize_timestamp(match.group("end")),
                    corrected_text,
                )
            )
    if not corrections:
        raise SubtitleCorrectionError(
            "DeepSeek did not return subtitle lines in the required timestamp format."
        )
    return corrections


def _build_chunks(entries, max_entries=80, max_chars=12000):
    chunks = []
    current = []
    current_chars = 0
    for item in entries:
        item_chars = len(item["text"])
        if current and (
            len(current) >= max_entries or current_chars + item_chars > max_chars
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        chunks.append(current)
    return chunks


def correct_srt_with_llm(srt_text, correction_prompt, call_model):
    entries = parse_srt_entries(srt_text)
    system_content = (
        (correction_prompt or DEFAULT_SUBTITLE_CORRECTION_PROMPT).strip()
        + "\n\n必须遵守以下规则：只修正字幕文字；不得增删、合并、拆分或重新排序字幕；"
        "每一行必须原样保留输入中的开始时间和结束时间，且顺序完全一致。"
        "输出格式严格如下：\n"
        "1. [开始时间-结束时间] 文本\n"
        "2. [开始时间-结束时间] 文本\n"
        "3. [开始时间-结束时间] 文本\n"
        "除上述内容外，不输出任何解释、分析、总结或额外文字。"
    )

    corrected_by_range = {}
    for entry in entries:
        corrected_by_range.setdefault(_entry_timestamp_range(entry), []).append(entry)

    matched_count = 0
    for chunk in _build_chunks(entries):
        source_lines = []
        for index, entry in enumerate(chunk, start=1):
            start_time, end_time = _entry_timestamp_range(entry)
            source_text = " ".join(entry["text"].splitlines()).strip()
            source_lines.append(f"{index}. [{start_time}-{end_time}] {source_text}")
        user_content = (
            "请校对下面这组连续字幕。只返回指定格式。\n"
            + "\n".join(source_lines)
        )
        response = call_model(user_content, system_content)
        for start_time, end_time, corrected_text in _parse_correction_lines(response):
            matching_entries = corrected_by_range.get((start_time, end_time))
            if not matching_entries:
                continue
            entry = matching_entries.pop(0)
            entry["corrected_text"] = corrected_text
            matched_count += 1

    if not matched_count:
        raise SubtitleCorrectionError(
            "No DeepSeek subtitle timestamps matched the original SRT."
        )

    changed_count = 0
    for entry in entries:
        corrected_text = entry.pop("corrected_text", entry["text"])
        if corrected_text != entry["text"]:
            changed_count += 1
        entry["text"] = corrected_text
    return render_srt_entries(entries), changed_count, len(entries), matched_count


def update_state_subtitles(state, corrected_srt):
    if state is None:
        return None, 0

    entries = parse_srt_entries(corrected_srt)
    texts_by_range = {
        _entry_timestamp_millis_range(entry): entry["text"]
        for entry in entries
    }
    # VideoFileClip carries thread locks and cannot be deep-copied. Only subtitle
    # records are mutable here; video/audio handles intentionally keep identity.
    updated_state = dict(state)
    updated_state["subtitle_text_overrides"] = {
        f"{start_ms}-{end_ms}": text
        for (start_ms, end_ms), text in texts_by_range.items()
    }
    copied_sentence_lists = {}

    def update_sentences(key):
        original_sentences = state.get(key)
        sentences = updated_state.get(key)
        if not isinstance(sentences, list):
            return 0
        list_key = id(original_sentences)
        if list_key not in copied_sentence_lists:
            copied_sentence_lists[list_key] = [
                dict(sentence) if isinstance(sentence, dict) else sentence
                for sentence in original_sentences
            ]
        updated_state[key] = copied_sentence_lists[list_key]
        synced = 0
        for sentence in updated_state[key]:
            timestamp_range = _sentence_timestamp_millis_range(sentence)
            corrected_text = texts_by_range.get(timestamp_range)
            if corrected_text is not None:
                sentence["text"] = corrected_text
                synced += 1
        return synced

    sentence_sync_count = update_sentences("sentences")
    speaker_sync_count = update_sentences("sd_sentences")
    return updated_state, max(sentence_sync_count, speaker_sync_count)
