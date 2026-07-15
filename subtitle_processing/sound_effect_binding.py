"""Persistent sound-effect keyword bindings and fourth-stage LLM matching."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable


SOUND_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
_LOCK = threading.Lock()
_ROOT = Path(__file__).resolve().parent.parent / "sound_effects"
_BINDINGS_FILE = _ROOT / "bindings.json"

SOUND_BINDING_SYSTEM_PROMPT = """
你是一个短视频字幕关键词音效绑定器。输入包含 keywords、sound_effects，以及每个音效的用途说明和历史关键词。

必须严格遵守：
1. 每个 keyword_id 必须且只能返回一条结果。每个关键词最多绑定一个 sound_id，或返回 null；不得拆分关键词或返回多个音效。
2. sound_id 必须来自输入的 sound_effects；不得创建、修改或想象不存在的音效。
3. 关键词普通、与现有音效不匹配、音效突兀、语义无法确定、同段音效过密、或只是连接词和语气词时，应返回 null。宁可不绑定，也不要强行匹配。
4. 输入给你的关键词均未命中历史记录。必须结合关键词所在完整句子、音效 description、historical_keywords 和医疗科普视频体验判断，音效应克制准确，不得娱乐化过度。
5. 危险、禁止、严重后果和明确风险可用警示类音效；关键疾病、药物、检查和核心概念可轻度强调；数字、比例、剂量、时间可用数字提示；结论、转折、关键答案可用确认类音效。医学词汇并不必然需要音效。
6. keyword_id、keyword 必须与输入完全一致，顺序必须与输入 keywords 一致；不得遗漏或新增关键词。

只输出合法 JSON，不输出 Markdown、解释或额外文字：
{"bindings":[{"keyword_id":"kw_001","keyword":"抽烟","sound_id":"warning","confidence":0.93,"reason":"简短原因"}]}
sound_id 只能为合法音效 ID 或 null；confidence 为 0 到 1 的小数；reason 为简短说明。
""".strip()

_SRT_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")


def sound_effect_directory() -> Path:
    configured = os.environ.get("FUNCLIP_SFX_DIR") or os.environ.get("FUNCLIP_LOCAL_SFX_DIR")
    return Path(configured).expanduser() if configured else _ROOT / "audio"


def list_sound_effects() -> list[str]:
    directory = sound_effect_directory()
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SOUND_EXTENSIONS)


def resolve_sound_effect_file(name: str) -> Path | None:
    if not name or Path(name).name != name:
        return None
    path = sound_effect_directory() / name
    return path if path.is_file() and path.suffix.lower() in SOUND_EXTENSIONS else None


def _empty_data() -> dict:
    return {"version": 1, "effects": {}}


def _load_data() -> dict:
    if not _BINDINGS_FILE.is_file():
        return _empty_data()
    try:
        data = json.loads(_BINDINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_data()
    return data if isinstance(data, dict) and isinstance(data.get("effects"), dict) else _empty_data()


def _save_data(data: dict) -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)
    temporary = _BINDINGS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(_BINDINGS_FILE)


def _split_terms(value: str) -> list[str]:
    terms = []
    for part in re.split(r"[\n,，;；]+", str(value or "")):
        term = part.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def get_effect_details(effect_name: str) -> tuple[str, str]:
    with _LOCK:
        entry = _load_data()["effects"].get(effect_name, {})
    return str(entry.get("features", "")), "\n".join(entry.get("keywords", []))


def save_effect_details(effect_name: str, features: str) -> tuple[str, str]:
    """Save feature terms and make each manually chosen term uniquely owned."""
    if effect_name not in list_sound_effects():
        raise ValueError("Selected sound-effect file does not exist.")
    terms = _split_terms(features)
    with _LOCK:
        data = _load_data()
        effects = data["effects"]
        for name, entry in effects.items():
            if name != effect_name:
                entry["keywords"] = [term for term in entry.get("keywords", []) if term not in terms]
        entry = effects.setdefault(effect_name, {})
        entry["features"] = str(features or "").strip()
        entry["keywords"] = list(dict.fromkeys([*entry.get("keywords", []), *terms]))
        _save_data(data)
    return get_effect_details(effect_name)


def _history_lookup(data: dict) -> dict[str, str]:
    lookup = {}
    for effect_name, entry in data["effects"].items():
        if resolve_sound_effect_file(effect_name) is None:
            continue
        for keyword in entry.get("keywords", []):
            lookup.setdefault(keyword, effect_name)
    return lookup


def _keyword_items(keywords: list[str], highlight_srt: str) -> list[dict[str, str]]:
    lines = str(highlight_srt or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    timestamp_indexes = [index for index, line in enumerate(lines) if _SRT_TIMESTAMP_RE.match(line)]
    sentences = []
    for position, index in enumerate(timestamp_indexes):
        end = timestamp_indexes[position + 1] if position + 1 < len(timestamp_indexes) else len(lines)
        sentence = " ".join(
            line.strip()
            for line in lines[index + 1 : end]
            if line.strip() and not re.match(r"^\d+\s*(?:spk\d+)?$", line)
        )
        if sentence:
            sentences.append(sentence)
    return [
        {
            "keyword_id": "kw_{:03d}".format(index),
            "keyword": keyword,
            "sentence": next((sentence for sentence in sentences if keyword in sentence), ""),
        }
        for index, keyword in enumerate(keywords, start=1)
    ]


def bind_keywords(
    keywords_text: str,
    highlight_srt: str,
    api_key: str,
    model: str,
    llm_call: Callable[[str, str, str, str, str], str],
) -> str:
    """Use history first, then call the LLM once for only unmatched keywords."""
    keywords = _split_terms(keywords_text)
    keyword_items = _keyword_items(keywords, highlight_srt)
    with _LOCK:
        data = _load_data()
        history = _history_lookup(data)
        effects = [
            {"sound_id": name, "description": entry.get("features", ""), "historical_keywords": entry.get("keywords", [])}
            for name, entry in data["effects"].items()
            if resolve_sound_effect_file(name) is not None
        ]

    resolved = {keyword: history[keyword] for keyword in keywords if keyword in history}
    unmatched_items = [item for item in keyword_items if item["keyword"] not in resolved]
    model_results = {}
    if unmatched_items and effects and api_key:
        request = json.dumps({"keywords": unmatched_items, "sound_effects": effects}, ensure_ascii=False)
        raw = llm_call(
            SOUND_BINDING_SYSTEM_PROMPT,
            "Assign only these unmatched keywords. Return JSON only.",
            request,
            api_key,
            model,
        )
        try:
            proposed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            assignments = proposed.get("bindings", []) if isinstance(proposed, dict) else []
        except json.JSONDecodeError:
            assignments = []
        valid_effects = {item["sound_id"] for item in effects}
        expected = {item["keyword_id"]: item["keyword"] for item in unmatched_items}
        for item in assignments:
            if not isinstance(item, dict):
                continue
            keyword_id, keyword, effect = item.get("keyword_id"), item.get("keyword"), item.get("sound_id")
            if keyword_id not in expected or expected[keyword_id] != keyword or keyword_id in model_results:
                continue
            if effect is not None and effect not in valid_effects:
                continue
            model_results[keyword_id] = {
                "sound_id": effect,
                "confidence": item.get("confidence", 0),
                "reason": str(item.get("reason", ""))[:200],
            }
            if effect:
                resolved[keyword] = effect

    with _LOCK:
        data = _load_data()
        for keyword, effect in resolved.items():
            entry = data["effects"].setdefault(effect, {"features": "", "keywords": []})
            if keyword not in entry.setdefault("keywords", []):
                entry["keywords"].append(keyword)
        _save_data(data)

    bindings = []
    for item in keyword_items:
        keyword, keyword_id = item["keyword"], item["keyword_id"]
        if keyword in history:
            result = {"sound_id": history[keyword], "confidence": 1.0, "reason": "Matched historical keyword binding."}
        else:
            result = model_results.get(keyword_id, {"sound_id": None, "confidence": 0.0, "reason": "No suitable sound effect."})
        bindings.append({"keyword_id": keyword_id, "keyword": keyword, **result})
    return json.dumps({"bindings": bindings}, ensure_ascii=False, indent=2)
