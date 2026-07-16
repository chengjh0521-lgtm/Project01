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
_CONFIG_FILE = _ROOT / "medical_sound_effects_config.json"

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
_SRT_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}$")

SOUND_CUE_SYSTEM_PROMPT = """
你是一名专业的短视频音效导演（Audio Director）。你的职责不是给关键词分类，而是根据一句字幕的完整语义，决定这一句是否需要音效、使用哪个音效、音效落在哪一个关键词上。医学科普视频必须自然、克制，不能滥用音效。

输入包含 sound_effects_config 和 sentences。sound_effects_config 定义所有允许使用的音效，只能选择其中的 sound_id，绝不能创建新音效。sentences 的每条包含 sentence_id、text 和 keywords；keywords 只用于定位，判断必须依据整句语义。

逐句处理，且每个 sentence_id 必须且只能输出一次，顺序与输入一致。每句最多一个音效、一个 target_word；target_word 必须完全来自该句的 keywords，不能修改或创造新词。若不适合音效，必须 use_sound=false，sound_id 和 target_word 为 null。宁可不用，也不要强行使用。

优先考虑：危险行为（抽烟、喝酒、熬夜、自行停药）、医生最终结论（不能、必须、一定、千万不要、最好、建议）、关键数字（比例、剂量、频次、时长）、重要医学概念，以及答案揭晓或关键转折。普通连接词、口头禅、寒暄、重复表达和无传播价值的信息通常不用音效。必须结合整句理解，例如“糖尿病患者千万不要抽烟”应强调“抽烟”，而非“糖尿病”；“空腹血糖最好控制在7以下”可强调“7”。

description 表示音效适合的真实语义，semantic_tags 和 example_keywords 仅用于理解，avoid_scenes 优先遵守，strength 是强调力度而不是优先级。连续字幕属于同一知识点时，原则上只选信息量最大、传播价值最高、情绪变化最明显或结论最明确的一句，避免连续音效。

只输出合法 JSON，不输出 Markdown 或额外文字：
{"results":[{"sentence_id":15,"use_sound":true,"sound_id":"sound_id_from_config","target_word":"原始关键词","confidence":0.97,"reason":"简短原因"},{"sentence_id":16,"use_sound":false,"sound_id":null,"target_word":null,"confidence":0.99,"reason":"简短原因"}]}
""".strip()


def sound_effect_directory() -> Path:
    configured = os.environ.get("FUNCLIP_SFX_DIR") or os.environ.get("FUNCLIP_LOCAL_SFX_DIR")
    return Path(configured).expanduser() if configured else _ROOT / "audio"


def _sound_config() -> dict:
    path = Path(os.environ.get("FUNCLIP_SOUND_CONFIG", _CONFIG_FILE))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("sound_effects"), list) else {}


def _configured_effects() -> list[dict]:
    return [item for item in _sound_config().get("sound_effects", []) if isinstance(item, dict) and item.get("sound_id")]


def list_sound_effects() -> list[str]:
    configured = _configured_effects()
    if configured:
        return [str(item["sound_id"]) for item in configured if resolve_sound_effect_file(str(item["sound_id"]))]
    directory = sound_effect_directory()
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SOUND_EXTENSIONS)


def resolve_sound_effect_file(name: str) -> Path | None:
    if not name or Path(name).name != name:
        return None
    config_item = next((item for item in _configured_effects() if item.get("sound_id") == name), {})
    path = sound_effect_directory() / str(config_item.get("file_name") or name)
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
    config_item = next((item for item in _configured_effects() if item.get("sound_id") == effect_name), {})
    return str(entry.get("features") or config_item.get("description") or ""), ""


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


def _keyword_values(keywords_text: str) -> list[str]:
    values = []
    for part in re.split(r"[\n,，;；、]+", str(keywords_text or "")):
        value = re.sub(r"^\s*\d+\s*[.、)]\s*", "", part).strip(" \t\"'“”")
        if value and value not in values:
            values.append(value)
    return values


def _highlight_sentences(highlight_srt: str, keywords_text: str) -> list[dict]:
    """Build the sentence-level contract required by the sound-director prompt."""
    lines = str(highlight_srt or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    timestamp_indexes = [index for index, line in enumerate(lines) if _SRT_RANGE_RE.match(line)]
    keywords = _keyword_values(keywords_text)
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
            "text": text,
            "keywords": [keyword for keyword in keywords if keyword in text],
        })
    return sentences


def select_sound_cues(
        highlight_srt: str, keywords_text: str, api_key: str, model: str,
        llm_call: Callable[[str, str, str, str, str], str]) -> str:
    """Stage 4: let the LLM choose at most one sound cue per subtitle sentence."""
    with _LOCK:
        data = _load_data()
        overrides = data["effects"]
        config = _sound_config()
        effects = []
        for item in _configured_effects():
            sound_id = str(item["sound_id"])
            description = str(overrides.get(sound_id, {}).get("features") or item.get("description") or "").strip()
            if description and resolve_sound_effect_file(sound_id) is not None:
                effects.append({
                    "sound_id": sound_id,
                    "display_name": item.get("display_name", sound_id),
                    "file_name": item.get("file_name", ""),
                    "description": description,
                    "semantic_tags": item.get("semantic_tags", []),
                    "example_keywords": item.get("example_keywords", []),
                    "strength": item.get("strength", ""),
                    "recommended_max_per_30s": item.get("recommended_max_per_30s"),
                    "avoid_scenes": item.get("avoid_scenes", []),
                })
    sentences = _highlight_sentences(highlight_srt, keywords_text)
    if not effects or not api_key or not sentences:
        return '{"cues": []}'
    request = {
        "sound_effects_config": {
            "global_rules": config.get("global_rules", {}),
            "sound_effects": effects,
        },
        "sentences": [
            {"sentence_id": item["sentence_id"], "text": item["text"], "keywords": item["keywords"]}
            for item in sentences
        ],
    }
    raw = llm_call(
        SOUND_CUE_SYSTEM_PROMPT,
        "请逐句返回音效决策。每个 sentence_id 都必须出现一次。",
        json.dumps(request, ensure_ascii=False),
        api_key,
        model,
    )
    try:
        payload = json.loads(raw.strip())
        results = payload.get("results", []) if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        results = []
    valid, clean, used = {effect["sound_id"] for effect in effects}, [], set()
    sentence_by_id = {item["sentence_id"]: item for item in sentences}
    for result in results:
        if not isinstance(result, dict):
            continue
        sentence_id = result.get("sentence_id")
        if isinstance(sentence_id, str) and sentence_id.isdigit():
            sentence_id = int(sentence_id)
        sentence = sentence_by_id.get(sentence_id)
        if not sentence or not result.get("use_sound") or sentence_id in used:
            continue
        sound_id, target_word = result.get("sound_id"), result.get("target_word")
        if sound_id not in valid or not isinstance(target_word, str) or target_word not in sentence["keywords"]:
            continue
        used.add(sentence_id)
        clean.append({
            "sound_id": sound_id,
            "timestamp": sentence["start"],
            "text": target_word,
            "reason": str(result.get("reason", ""))[:160],
            "sentence_id": sentence_id,
        })
    return json.dumps({"cues": clean}, ensure_ascii=False, indent=2)


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
