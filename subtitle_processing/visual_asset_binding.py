"""Stage 5: select configured PNG/GIF visual assets for highlight subtitles."""

from __future__ import annotations

import json
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

# The user will replace this placeholder with the final DeepSeek visual-director prompt.
VISUAL_ASSET_PROMPT_READY = False
VISUAL_ASSET_SYSTEM_PROMPT = """
[VISUAL_ASSET_DIRECTOR_PROMPT_PLACEHOLDER]

You will receive an asset_library, sentences, and allowed_positions. Return JSON only.
Use only asset IDs from asset_library. A sentence may have at most one asset and target_word must come from that sentence's keywords.
Return this shape exactly:
{"placements":[{"sentence_id":1,"use_asset":true,"asset_id":"asset_id_from_library","target_word":"keyword_from_sentence","position":"upper_right","duration_seconds":2.0,"confidence":0.9,"reason":"short reason"}]}
Return {"placements":[]} when no visual asset is appropriate.
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
        sentences.append({
            "sentence_id": position,
            "start": match.group("start").replace(".", ","),
            "end": match.group("end").replace(".", ","),
            "text": text,
            "keywords": [keyword for keyword in keywords if keyword in text],
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
        "selection_rules": config.get("selection_rules", {}),
        "asset_library": assets,
        "sentences": [
            {"sentence_id": item["sentence_id"], "text": item["text"], "keywords": item["keywords"]}
            for item in sentences
        ],
        "allowed_positions": list(POSITION_PRESETS),
    }
    raw = llm_call(
        VISUAL_ASSET_SYSTEM_PROMPT,
        "Select only suitable configured visual assets. Return JSON only.",
        json.dumps(request, ensure_ascii=False),
        api_key,
        model,
    )
    try:
        payload = json.loads(raw.strip())
        placements = payload.get("placements", []) if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        placements = []
    sentence_by_id = {item["sentence_id"]: item for item in sentences}
    valid_assets = {item["id"] for item in assets}
    clean, used_sentences = [], set()
    for placement in placements:
        if not isinstance(placement, dict) or not placement.get("use_asset"):
            continue
        sentence_id = placement.get("sentence_id")
        if isinstance(sentence_id, str) and sentence_id.isdigit():
            sentence_id = int(sentence_id)
        sentence = sentence_by_id.get(sentence_id)
        asset_id, target_word = placement.get("asset_id"), placement.get("target_word")
        if (
            not sentence or sentence_id in used_sentences or asset_id not in valid_assets
            or not isinstance(target_word, str) or target_word not in sentence["keywords"]
        ):
            continue
        position = placement.get("position")
        if position not in POSITION_PRESETS:
            position = "upper_right"
        try:
            duration = float(placement.get("duration_seconds", 2.0))
        except (TypeError, ValueError):
            duration = 2.0
        used_sentences.add(sentence_id)
        clean.append({
            "sentence_id": sentence_id,
            "asset_id": asset_id,
            "target_word": target_word,
            "position": position,
            "duration_seconds": max(0.5, min(5.0, duration)),
            "reason": str(placement.get("reason", ""))[:160],
        })
    return json.dumps({"placements": clean}, ensure_ascii=False, indent=2)
