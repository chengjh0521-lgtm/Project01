from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re


DEFAULT_SUBTITLE_CORRECTION_PROMPT = (
    "你是一名医疗视频字幕校对员。请结合上下文修正 ASR 造成的同音字、错别字、漏标点和医学术语错误，"
    "尤其留意四川话口音可能导致的误识别。保持医生原本的语气和含义，不扩写、不总结、不删除有效信息。"
)

_TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*"
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}(?:\s+.*)?$"
)


class SubtitleCorrectionError(ValueError):
    pass


def parse_srt_entries(srt_text):
    normalized = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SubtitleCorrectionError("SRT subtitles are empty. Run ASR first.")

    entries = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = block.splitlines()
        timestamp_index = next(
            (index for index, line in enumerate(lines) if _TIMESTAMP_RE.match(line)),
            None,
        )
        if timestamp_index is None:
            raise SubtitleCorrectionError(
                "Invalid SRT block: a subtitle timestamp line is missing."
            )
        text = "\n".join(lines[timestamp_index + 1 :]).strip()
        if not text:
            raise SubtitleCorrectionError("Invalid SRT block: subtitle text is empty.")
        entries.append(
            {
                "prefix": lines[:timestamp_index],
                "timestamp": lines[timestamp_index],
                "text": text,
            }
        )

    if not entries:
        raise SubtitleCorrectionError("No valid SRT subtitle blocks were found.")
    return entries


def render_srt_entries(entries):
    blocks = []
    for entry in entries:
        lines = list(entry.get("prefix") or [])
        lines.extend([entry["timestamp"], entry["text"]])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _decode_json_response(response_text):
    text = str(response_text or "").strip()
    if not text:
        raise SubtitleCorrectionError("DeepSeek returned an empty response.")
    if (
        text.lower().startswith("llm inference failed:")
        or text.lower().startswith("api key is required")
    ):
        raise SubtitleCorrectionError(text)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    candidates = []
    object_start, object_end = text.find("{"), text.rfind("}")
    array_start, array_end = text.find("["), text.rfind("]")
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])
    if array_start >= 0 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])
    candidates.append(text)

    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        raise SubtitleCorrectionError(
            "DeepSeek did not return valid JSON. The original subtitles were kept."
        )
    return payload


def _extract_corrected_text(response_text):
    payload = _decode_json_response(response_text)
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise SubtitleCorrectionError(
                "DeepSeek returned more than one corrected subtitle. The original subtitles were kept."
            )
        payload = payload[0]
    if not isinstance(payload, dict):
        raise SubtitleCorrectionError(
            "DeepSeek correction JSON must contain one text field. The original subtitles were kept."
        )
    text = str(
        payload.get("text")
        or payload.get("corrected_text")
        or payload.get("corrected")
        or ""
    ).strip()
    if not text:
        raise SubtitleCorrectionError(
            "DeepSeek returned an empty corrected subtitle. The original subtitles were kept."
        )
    return text


def _subtitle_context(entries, index):
    return {
        "previous_subtitle": entries[index - 1]["text"] if index > 0 else "",
        "current_subtitle": entries[index]["text"],
        "next_subtitle": (
            entries[index + 1]["text"] if index + 1 < len(entries) else ""
        ),
    }


def correct_srt_with_llm(srt_text, correction_prompt, call_model, max_workers=3):
    entries = parse_srt_entries(srt_text)
    system_content = (
        (correction_prompt or DEFAULT_SUBTITLE_CORRECTION_PROMPT).strip()
        + "\n\n本次请求只校对 current_subtitle 一条字幕。previous_subtitle 和 next_subtitle "
        "仅用于理解上下文，绝不能复述或修改它们。只修正当前字幕的文字，不扩写、不总结。"
        "不要输出编号、时间戳或解释。只返回严格 JSON：{\"text\":\"修正后的当前字幕\"}。"
    )

    def correct_one(index):
        user_content = json.dumps(_subtitle_context(entries, index), ensure_ascii=False)
        response = call_model(user_content, system_content)
        return index, _extract_corrected_text(response)

    corrected_by_index = {}
    worker_count = max(1, min(int(max_workers or 1), len(entries)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(correct_one, index) for index in range(len(entries))]
        for future in as_completed(futures):
            index, corrected_text = future.result()
            corrected_by_index[index] = corrected_text

    changed_count = 0
    for index, entry in enumerate(entries):
        corrected_text = corrected_by_index[index]
        if corrected_text != entry["text"]:
            changed_count += 1
        entry["text"] = corrected_text
    return render_srt_entries(entries), changed_count, len(entries)


def update_state_subtitles(state, corrected_srt):
    if state is None:
        return None

    entries = parse_srt_entries(corrected_srt)
    texts = [entry["text"] for entry in entries]
    # VideoFileClip carries thread locks and cannot be deep-copied. Only subtitle
    # records are mutable here; video/audio handles intentionally keep identity.
    updated_state = dict(state)
    copied_sentence_lists = {}

    def update_sentences(key, required):
        sentences = updated_state.get(key)
        if not isinstance(sentences, list):
            if required:
                raise SubtitleCorrectionError(
                    "The recognition state has no subtitle sentence list."
                )
            return
        eligible = [sentence for sentence in sentences if sentence.get("timestamp")]
        if len(eligible) != len(texts):
            if required:
                raise SubtitleCorrectionError(
                    "Corrected subtitle count does not match the recognition state. "
                    "The original subtitles were kept."
                )
            return
        original_sentences = state.get(key)
        list_key = id(original_sentences)
        if list_key not in copied_sentence_lists:
            copied_sentence_lists[list_key] = [
                dict(sentence) if isinstance(sentence, dict) else sentence
                for sentence in original_sentences
            ]
        updated_state[key] = copied_sentence_lists[list_key]
        copied_eligible = [
            sentence for sentence in updated_state[key]
            if isinstance(sentence, dict) and sentence.get("timestamp")
        ]
        for sentence, text in zip(copied_eligible, texts):
            sentence["text"] = text

    update_sentences("sentences", required=True)
    update_sentences("sd_sentences", required=False)
    return updated_state
