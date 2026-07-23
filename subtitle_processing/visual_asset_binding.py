"""Stage 5: select configured PNG/GIF visual assets for highlight subtitles."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable


_ROOT = Path(__file__).resolve().parent.parent / "visual_assets"
_DEFAULT_CONFIG_FILE = _ROOT / "picture_assets_index.json"
_STATIC_IMAGE_MIN_DURATION_SECONDS = 0.2
_SRT_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)

# This prompt is intentionally kept independent from audio-effect selection.
VISUAL_ASSET_PROMPT_READY = True
VISUAL_ASSET_SYSTEM_PROMPT = """
你是一名专业的医学科普短视频视觉导演（Visual Asset Director）。你的职责不是为关键词分类，而是根据字幕的完整语义，决定是否需要在画面中加入视觉素材，以帮助观众更快理解当前内容。

你必须依据提供的素材索引（picture_assets_index.json）完成所有判断，不允许凭空创造素材或修改素材定义。

输入包含 asset_index 和 sentences。asset_index 中每个素材包含 id、file_name、description、recommended_scenes、size 和 media_type。description 表示素材真正表达的内容；recommended_scenes 表示推荐使用语义。

sentences 中每项包含 sentence_id、start、end、text 和 keywords。素材会从 target_word 出现时开始展示，并在该条字幕结束时立即消失；展示时长由后端按此规则精确计算，不能由你指定。keywords 仅用于定位素材出现的位置，真正的判断依据始终是整句话。请逐句分析：判断是否值得加入视觉素材；若需要，从 asset_index 中选择一个最合适的素材；并从该句 keywords 中选择一个关键词作为绑定位置。若没有合适素材，则不使用素材。

先理解句子，再选择素材。素材应该帮助观众理解句子的核心信息，而不是仅仅对应某个名词。例如“糖尿病患者最好少吃油炸食品”应展示油炸食品素材，而不是糖尿病素材。只有当素材能明显提升理解效率时才使用：明确实物、人体器官或疾病示意图、生活方式、容易视觉化的医学概念，或需要重点提醒的结论。普通连接句、过渡句、寒暄、抽象推理、没有明确视觉对应物的内容通常不用素材。宁可不用，也不要强行选择。

选择素材时必须综合参考 description 和 recommended_scenes，并以整句语义判断是否真的有助于理解。连续几句话讨论同一知识点时，原则上只在最值得展示素材的一句使用，避免连续重复展示。此外每个视频必须至少使用3个插画，在素材相关性不高时，可以选择小贴士、划重点这类宽泛的素材。

每个 sentence_id 必须且只能输出一次，顺序与输入一致。即使某句没有可用 keywords，也必须返回该句并令 use_asset=false。每句话最多一个素材、一个 target_word。target_word 必须来自该句 keywords 的 word，asset_id 必须来自 asset_index。不得新增、修改或删除素材、关键词或句子。当没有明确合适素材时，use_asset=false，asset_id、target_word 为 null。

仅输出合法 JSON，不输出 Markdown 或额外文字：
{"results":[{"sentence_id":15,"use_asset":true,"asset_id":"asset_041_avoid_fried_food","target_word":"油炸食品","confidence":0.98,"reason":"素材能够直接帮助观众理解应减少油炸食品摄入。"},{"sentence_id":16,"use_asset":false,"asset_id":null,"target_word":null,"confidence":0.99,"reason":"没有能够明显提升理解的视觉素材。"}]}
""".strip()


def visual_asset_directory() -> Path:
    configured = os.environ.get("FUNCLIP_VISUAL_ASSET_DIR")
    return Path(configured).expanduser() if configured else _ROOT / "files"


def _decode_file_name(value: str) -> str:
    """Decode the #U4E2D-style file names used by the updated asset index."""
    return re.sub(r"#U([0-9A-Fa-f]{4})", lambda match: chr(int(match.group(1), 16)), str(value or ""))


def _config_paths() -> list[Path]:
    """Read every JSON index beside the configured index, in a stable order."""
    configured = os.environ.get("FUNCLIP_VISUAL_ASSET_CONFIG")
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            paths = sorted(candidate for candidate in path.glob("*.json") if candidate.is_file())
            if paths:
                return paths
            logging.warning("Configured visual asset index directory has no JSON files: %s", path)
        elif path.is_file():
            siblings = sorted(
                candidate for candidate in path.parent.glob("*.json")
                if candidate.is_file() and candidate != path
            )
            return [path, *siblings]
        else:
            logging.warning("Configured visual asset index is missing, falling back: %s", path)

    paths = sorted(candidate for candidate in _ROOT.glob("*.json") if candidate.is_file())
    if _DEFAULT_CONFIG_FILE in paths:
        paths.remove(_DEFAULT_CONFIG_FILE)
        paths.insert(0, _DEFAULT_CONFIG_FILE)
    return paths


def _load_asset_index(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logging.warning("Visual asset index is unreadable: %s", path)
        return None
    if isinstance(data, list):
        return {
            "schema_version": "array-index",
            "purpose": "Medical short-video visual asset index",
            "selection_rules": {},
            "assets": data,
        }
    if isinstance(data, dict) and isinstance(data.get("assets"), list):
        return data
    logging.warning("Visual asset index has no supported assets array: %s", path)
    return None


def _default_selection_rules() -> dict:
    return {
        "allow_no_asset": True,
        "max_assets_per_sentence": 1,
        "display_window": "From the bound keyword to the end of its subtitle cue.",
    }


def _asset_config() -> dict:
    """Merge compatible asset-index files without allowing duplicate asset IDs."""
    assets, source_files, seen_ids = [], [], set()
    purpose = "Medical short-video visual asset index"
    selection_rules = _default_selection_rules()
    for path in _config_paths():
        loaded = _load_asset_index(path)
        if loaded is None:
            continue
        source_files.append(str(path))
        purpose = str(loaded.get("purpose") or purpose)
        if isinstance(loaded.get("selection_rules"), dict):
            selection_rules.update(loaded["selection_rules"])
        for raw_item in loaded.get("assets", []):
            if not isinstance(raw_item, dict) or not raw_item.get("id"):
                continue
            item = _normalise_asset(raw_item)
            asset_id = str(item["id"])
            if asset_id in seen_ids:
                logging.warning("Skipping duplicate visual asset id %s from %s", asset_id, path)
                continue
            seen_ids.add(asset_id)
            assets.append(item)
    if not source_files:
        logging.warning("No readable visual asset index files were found under %s", _ROOT)
    return {
        "schema_version": "multi-file-index",
        "purpose": purpose,
        "selection_rules": selection_rules,
        "source_files": source_files,
        "assets": assets,
    }


def _minimum_display_duration_seconds(item: dict, is_gif: bool) -> float:
    """Read the asset-configured minimum display duration with sensible legacy fallbacks."""
    default = 3.0 if is_gif else _STATIC_IMAGE_MIN_DURATION_SECONDS
    try:
        duration = float(item.get("duration_seconds", default))
    except (TypeError, ValueError):
        duration = default
    return max(0.04, duration)


def _normalise_asset(item: dict) -> dict:
    """Normalize supported index layouts without exposing removed legacy fields."""
    raw_file_name = str(item.get("file_name", ""))
    file_name = _decode_file_name(raw_file_name)
    description = str(item.get("description") or "").strip()
    extension = Path(file_name).suffix.lower()
    metadata = item.get("technical_metadata") if isinstance(item.get("technical_metadata"), dict) else {}
    size_match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(item.get("size", "")))
    width = metadata.get("width") or (int(size_match.group(1)) if size_match else None)
    height = metadata.get("height") or (int(size_match.group(2)) if size_match else None)
    is_gif = extension == ".gif" or item.get("media_type") == "animated_gif"
    duration = _minimum_display_duration_seconds(item, is_gif)
    return {
        **item,
        "file_name": file_name,
        "raw_file_name": raw_file_name,
        "media_type": "animated_gif" if is_gif else str(item.get("media_type") or "image"),
        "description": description,
        "recommended_scenes": str(item.get("recommended_scenes") or ""),
        "size": str(item.get("size") or "{}x{}".format(width or "", height or "")),
        "duration_seconds": duration,
        "technical_metadata": {
            "width": width,
            "height": height,
            "frame_count": metadata.get("frame_count"),
            "has_transparency": bool(metadata.get("has_transparency")) or extension in {".png", ".gif"},
            "requires_chroma_key": bool(metadata.get("requires_chroma_key")) or (is_gif and "绿幕" in description),
        },
    }




def _configured_assets() -> list[dict]:
    return [item for item in _asset_config().get("assets", []) if isinstance(item, dict) and item.get("id")]


def resolve_visual_asset_file(asset_id: str) -> Path | None:
    if not asset_id:
        return None
    item = next((candidate for candidate in _configured_assets() if candidate.get("id") == asset_id), None)
    if not item:
        return None
    for file_name in dict.fromkeys((str(item.get("file_name", "")), str(item.get("raw_file_name", "")))):
        if not file_name or Path(file_name).name != file_name:
            continue
        path = visual_asset_directory() / file_name
        if path.is_file():
            return path
    return None


def get_visual_asset_definition(asset_id: str) -> dict:
    """Return the configured technical metadata needed by the FFmpeg renderer."""
    item = next((candidate for candidate in _configured_assets() if candidate.get("id") == asset_id), {})
    return dict(item) if isinstance(item, dict) else {}


def _available_assets() -> list[dict]:
    assets = []
    config = _asset_config()
    configured_assets = [item for item in config.get("assets", []) if isinstance(item, dict) and item.get("id")]
    for item in configured_assets:
        if resolve_visual_asset_file(str(item["id"])) is None:
            continue
        metadata = item.get("technical_metadata") if isinstance(item.get("technical_metadata"), dict) else {}
        assets.append({
            "id": item["id"],
            "file_name": item.get("file_name", ""),
            "description": item.get("description", ""),
            "media_type": item.get("media_type", "image"),
            "recommended_scenes": item.get("recommended_scenes", ""),
            "size": item.get("size", ""),
            "duration_seconds": item.get("duration_seconds", 3.0),
            "technical_metadata": {
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "frame_count": metadata.get("frame_count"),
                "has_transparency": bool(metadata.get("has_transparency")),
                "requires_chroma_key": bool(metadata.get("requires_chroma_key")),
            },
        })
    logging.warning(
        "Visual asset availability: %d/%d files matched; config_files=%s; directory=%s",
        len(assets), len(configured_assets), config.get("source_files", []), visual_asset_directory(),
    )
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


def _clean_visual_results(raw: str, sentences: list[dict], assets: list[dict]) -> list[dict]:
    """Validate one LLM response while keeping usable placements from a partial response."""
    try:
        payload = json.loads(raw.strip())
        results = payload.get("results", []) if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        results = []
    if not isinstance(results, list):
        logging.warning("Visual-asset stage rejected: model response has no results array.")
        return []

    sentence_by_id = {item["sentence_id"]: item for item in sentences}
    valid_assets = {item["id"] for item in assets}
    result_by_sentence = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        sentence_id = result.get("sentence_id")
        if isinstance(sentence_id, str) and sentence_id.isdigit():
            sentence_id = int(sentence_id)
        if sentence_id not in sentence_by_id or sentence_id in result_by_sentence:
            logging.warning("Visual-asset stage ignored an invalid or duplicate sentence_id: %r", sentence_id)
            continue
        result_by_sentence[sentence_id] = result

    if len(result_by_sentence) != len(sentences):
        logging.warning(
            "Visual-asset stage returned %d/%d sentence decisions; omitted decisions will not receive an asset.",
            len(result_by_sentence), len(sentences),
        )

    clean = []
    for sentence in sentences:
        sentence_id = sentence["sentence_id"]
        result = result_by_sentence.get(sentence_id)
        if result is None or not result.get("use_asset"):
            continue
        asset_id, target_word = result.get("asset_id"), result.get("target_word")
        sentence_words = {item["word"] for item in sentence["keywords"]}
        if asset_id not in valid_assets or not isinstance(target_word, str) or target_word not in sentence_words:
            logging.warning(
                "Visual-asset stage ignored invalid placement for sentence %d: asset=%r, word=%r.",
                sentence_id, asset_id, target_word,
            )
            continue
        clean.append({
            "sentence_id": sentence_id,
            "asset_id": asset_id,
            "target_word": target_word,
            "position": "caption_lower_left",
            "reason": str(result.get("reason", ""))[:160],
        })
    return clean


def select_visual_assets(
        highlight_srt: str, keywords_text: str, api_key: str, model: str,
        llm_call: Callable[[str, str, str, str, str], str]) -> str:
    """Run the placeholder visual-director stage and validate its placements."""
    assets, sentences = _available_assets(), build_visual_sentences(highlight_srt, keywords_text)
    if not VISUAL_ASSET_PROMPT_READY:
        return '{"placements": []}'
    if not assets or not sentences or not api_key:
        logging.warning(
            "Visual asset stage skipped: assets=%d, sentences=%d, api_key=%s.",
            len(assets), len(sentences), bool(api_key),
        )
        return '{"placements": []}'
    config = _asset_config()
    request = {
        "asset_index": {
            "schema_version": config.get("schema_version", ""),
            "purpose": config.get("purpose", ""),
            "selection_rules": config.get("selection_rules", {}),
            "rendering_policy": {
                "display_window": "Each selected asset starts at target_word and ends exactly when that subtitle cue ends.",
                "model_output": "Do not output duration_seconds or position.",
            },
            "assets": assets,
        },
        "sentences": [
            {
                "sentence_id": item["sentence_id"],
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "keywords": item["keywords"],
            }
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
    logging.warning("Visual asset stage raw response (%d chars):\n%s", len(raw), raw)
    clean = _clean_visual_results(raw, sentences, assets)
    return json.dumps({"placements": clean}, ensure_ascii=False, indent=2)
