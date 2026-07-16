"""Stage 5: select configured PNG/GIF visual assets for highlight subtitles."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable


_ROOT = Path(__file__).resolve().parent.parent / "visual_assets"
_CONFIG_FILE = _ROOT / "picture_for_video_asset_index.json"
_SRT_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)

POSITION_PRESETS = ("upper_left", "upper_right", "top_center", "middle_left", "middle_right")

# This prompt is intentionally kept independent from audio-effect selection.
VISUAL_ASSET_PROMPT_READY = True
VISUAL_ASSET_SYSTEM_PROMPT = """
你是一名专业的医学科普短视频视觉导演（Visual Asset Director）。你的职责不是为关键词分类，而是根据字幕的完整语义，决定是否需要在画面中加入视觉素材，以帮助观众更快理解当前内容。

你必须依据提供的素材索引（picture_for_video_asset_index.json）完成所有判断，不允许凭空创造素材或修改素材定义。

输入包含 asset_index 和 sentences。asset_index 中每个素材包含 id、file_name、media_type、description_and_main_content、recommended_scenes、forbidden_scenes 和 technical_metadata。description_and_main_content 表示素材真正表达的内容；recommended_scenes 是推荐使用语义；forbidden_scenes 为强约束，必须严格遵守；technical_metadata 仅用于后续烧录，不参与素材优先级判断。

sentences 中每项包含 sentence_id、text 和 keywords。keywords 仅用于定位素材出现的位置，真正的判断依据始终是整句话。请逐句分析：判断是否值得加入视觉素材；若需要，从 asset_index 中选择一个最合适的素材；并从该句 keywords 中选择一个关键词作为绑定位置。若没有合适素材，则不使用素材。

先理解句子，再选择素材。素材应该帮助观众理解句子的核心信息，而不是仅仅对应某个名词。例如“糖尿病患者最好少吃油炸食品”应展示油炸食品素材，而不是糖尿病素材。只有当素材能明显提升理解效率时才使用：明确实物、人体器官或疾病示意图、生活方式、容易视觉化的医学概念，或需要重点提醒的结论。普通连接句、过渡句、寒暄、抽象推理、没有明确视觉对应物的内容通常不用素材。宁可不用，也不要强行选择。

选择素材时必须综合参考 description_and_main_content、recommended_scenes 和 forbidden_scenes。若违反 forbidden_scenes，即使内容相似也不得选择。连续几句话讨论同一知识点时，原则上只在最值得展示素材的一句使用，避免连续重复展示。

每个 sentence_id 必须且只能输出一次，顺序与输入一致。每句话最多一个素材、一个 target_word。target_word 必须来自该句 keywords 的 word，asset_id 必须来自 asset_index。不得新增、修改或删除素材、关键词或句子。当没有明确合适素材时，use_asset=false，asset_id 和 target_word 为 null。

仅输出合法 JSON，不输出 Markdown 或额外文字：
{"results":[{"sentence_id":15,"use_asset":true,"asset_id":"asset_041_avoid_fried_food","target_word":"油炸食品","confidence":0.98,"reason":"素材能够直接帮助观众理解应减少油炸食品摄入。"},{"sentence_id":16,"use_asset":false,"asset_id":null,"target_word":null,"confidence":0.99,"reason":"没有能够明显提升理解的视觉素材。"}]}
""".strip()


def visual_asset_directory() -> Path:
    configured = os.environ.get("FUNCLIP_VISUAL_ASSET_DIR")
    return Path(configured).expanduser() if configured else _ROOT / "files"


def _asset_config() -> dict:
    path = Path(os.environ.get("FUNCLIP_VISUAL_ASSET_CONFIG", _CONFIG_FILE))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("assets"), list) else {}


def _configured_assets() -> list[dict]:
    return [item for item in _asset_config().get("assets", []) if isinstance(item, dict) and item.get("id")]


def resolve_visual_asset_file(asset_id: str) -> Path | None:
    if not asset_id:
        return None
    item = next((candidate for candidate in _configured_assets() if candidate.get("id") == asset_id), None)
    if not item:
        return None
    file_name = str(item.get("file_name", ""))
    if not file_name or Path(file_name).name != file_name:
        return None
    path = visual_asset_directory() / file_name
    return path if path.is_file() else None


def get_visual_asset_definition(asset_id: str) -> dict:
    """Return the configured technical metadata needed by the FFmpeg renderer."""
    item = next((candidate for candidate in _configured_assets() if candidate.get("id") == asset_id), {})
    return dict(item) if isinstance(item, dict) else {}


def _available_assets() -> list[dict]:
    assets = []
    for item in _configured_assets():
        if resolve_visual_asset_file(str(item["id"])) is None:
            continue
        metadata = item.get("technical_metadata") if isinstance(item.get("technical_metadata"), dict) else {}
        assets.append({
            "id": item["id"],
            "file_name": item.get("file_name", ""),
            "media_type": item.get("media_type", "image"),
            "description_and_main_content": item.get("description_and_main_content", ""),
            "recommended_scenes": item.get("recommended_scenes", ""),
            "forbidden_scenes": item.get("forbidden_scenes", ""),
            "technical_metadata": {
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "frame_count": metadata.get("frame_count"),
                "has_transparency": bool(metadata.get("has_transparency")),
                "requires_chroma_key": bool(metadata.get("requires_chroma_key")),
            },
        })
    return assets


def _keywords(keywords_text: str) -> list[str]:
    values = []
    for part in re.split(r"[\n,，;；、]+", str(keywords_text or "")):
        value = re.sub(r"^\s*\d+\s*[.、)]\s*", "", part).strip(" \t\"'“”")
        if value and value not in values:
            values.append(value)
    return values


def _time_to_ms(value: str) -> int:
    hours, minutes, seconds = value.replace(".", ",").split(":")
    second, millis = seconds.split(",")
    return int(hours) * 3_600_000 + int(minutes) * 60_000 + int(second) * 1000 + int(millis.ljust(3, "0")[:3])


def _sentence_keywords(text: str, values: list[str], start: str, end: str) -> list[dict]:
    start_ms, end_ms = _time_to_ms(start), _time_to_ms(end)
    keyword_items = []
    for word in values:
        index = text.find(word)
        if index < 0:
            continue
        word_start = start_ms + round((end_ms - start_ms) * index / max(1, len(text)))
        word_end = start_ms + round((end_ms - start_ms) * (index + len(word)) / max(1, len(text)))
        keyword_items.append({"word": word, "start": round(word_start / 1000, 3), "end": round(word_end / 1000, 3)})
    return keyword_items


def build_visual_sentences(highlight_srt: str, keywords_text: str) -> list[dict]:
    """Create the stable sentence IDs shared by visual selection and rendering."""
    lines = str(highlight_srt or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    timestamp_indexes = [index for index, line in enumerate(lines) if _SRT_RANGE_RE.match(line)]
    keywords = _keywords(keywords_text)
    sentences = []
    for position, time_index in enumerate(timestamp_indexes, start=1):
        match = _SRT_RANGE_RE.match(lines[time_index])
        assert match is not None
        end_index = timestamp_indexes[position] if position < len(timestamp_indexes) else len(lines)
        if end_index > time_index + 1 and re.match(r"^\s*\d+(?:\s+spk\S+)?\s*$", lines[end_index - 1]):
            end_index -= 1
        text = " ".join(line.strip() for line in lines[time_index + 1:end_index] if line.strip())
        if not text:
            continue
        start, end = match.group("start").replace(".", ","), match.group("end").replace(".", ",")
        sentences.append({
            "sentence_id": position,
            "start": start,
            "end": end,
            "text": text,
            "keywords": _sentence_keywords(text, keywords, start, end),
        })
    return sentences


def select_visual_assets(
        highlight_srt: str, keywords_text: str, api_key: str, model: str,
        llm_call: Callable[[str, str, str, str, str], str]) -> str:
    """Run the placeholder visual-director stage and validate its placements."""
    assets, sentences = _available_assets(), build_visual_sentences(highlight_srt, keywords_text)
    if not VISUAL_ASSET_PROMPT_READY:
        return '{"placements": []}'
    if not assets or not sentences or not api_key:
        return '{"placements": []}'
    config = _asset_config()
    request = {
        "asset_index": {
            "schema_version": config.get("schema_version", ""),
            "purpose": config.get("purpose", ""),
            "selection_rules": config.get("selection_rules", {}),
            "assets": assets,
        },
        "sentences": [
            {"sentence_id": item["sentence_id"], "text": item["text"], "keywords": item["keywords"]}
            for item in sentences
        ],
    }
    raw = llm_call(
        VISUAL_ASSET_SYSTEM_PROMPT,
        "请严格按系统要求逐句选择视觉素材，只返回 JSON。",
        json.dumps(request, ensure_ascii=False),
        api_key,
        model,
    )
    try:
        payload = json.loads(raw.strip())
        results = payload.get("results", []) if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        results = []
    sentence_by_id = {item["sentence_id"]: item for item in sentences}
    valid_assets = {item["id"] for item in assets}
    clean, used_sentences = [], set()
    if not isinstance(results, list) or len(results) != len(sentences):
        logging.warning("Visual-asset stage rejected: expected %d sentence results, received %d.", len(sentences), len(results) if isinstance(results, list) else 0)
        return '{"placements": []}'
    for expected_sentence_id, result in zip((item["sentence_id"] for item in sentences), results):
        if not isinstance(result, dict):
            return '{"placements": []}'
        sentence_id = result.get("sentence_id")
        if isinstance(sentence_id, str) and sentence_id.isdigit():
            sentence_id = int(sentence_id)
        sentence = sentence_by_id.get(sentence_id)
        if sentence_id != expected_sentence_id or not sentence or sentence_id in used_sentences:
            return '{"placements": []}'
        used_sentences.add(sentence_id)
        if not result.get("use_asset"):
            continue
        asset_id, target_word = result.get("asset_id"), result.get("target_word")
        sentence_words = {item["word"] for item in sentence["keywords"]}
        if (
            asset_id not in valid_assets or not isinstance(target_word, str) or target_word not in sentence_words
        ):
            return '{"placements": []}'
        definition = get_visual_asset_definition(asset_id)
        media_type = definition.get("media_type", "image")
        clean.append({
            "sentence_id": sentence_id,
            "asset_id": asset_id,
            "target_word": target_word,
            "position": "upper_right",
            "duration_seconds": 1.8 if media_type == "animated_gif" else 2.5,
            "reason": str(result.get("reason", ""))[:160],
        })
    return json.dumps({"placements": clean}, ensure_ascii=False, indent=2)
