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
You assign short-video subtitle keywords to available sound effects.
Only assign an effect when its semantic use is clearly appropriate. It is valid
to leave a keyword unbound. Never invent an effect name or alter a keyword.
Return only JSON in this exact shape:
{"bindings":[{"keyword":"original keyword","effect":"exact filename or null"}]}
""".strip()


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


def bind_keywords(
    keywords_text: str,
    api_key: str,
    model: str,
    llm_call: Callable[[str, str, str, str, str], str],
) -> str:
    """Use history first, then call the LLM once for only unmatched keywords."""
    keywords = _split_terms(keywords_text)
    with _LOCK:
        data = _load_data()
        history = _history_lookup(data)
        effects = [
            {"name": name, "features": entry.get("features", ""), "historical_keywords": entry.get("keywords", [])}
            for name, entry in data["effects"].items()
            if resolve_sound_effect_file(name) is not None
        ]

    resolved = {keyword: history[keyword] for keyword in keywords if keyword in history}
    unmatched = [keyword for keyword in keywords if keyword not in resolved]
    if unmatched and effects and api_key:
        request = json.dumps({"unmatched_keywords": unmatched, "available_effects": effects}, ensure_ascii=False)
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
        valid_effects = {item["name"] for item in effects}
        for item in assignments:
            if not isinstance(item, dict):
                continue
            keyword, effect = item.get("keyword"), item.get("effect")
            if keyword in unmatched and effect in valid_effects and keyword not in resolved:
                resolved[keyword] = effect

    with _LOCK:
        data = _load_data()
        for keyword, effect in resolved.items():
            entry = data["effects"].setdefault(effect, {"features": "", "keywords": []})
            if keyword not in entry.setdefault("keywords", []):
                entry["keywords"].append(keyword)
        _save_data(data)

    return json.dumps(
        {"bindings": [{"keyword": keyword, "effect": resolved.get(keyword)} for keyword in keywords]},
        ensure_ascii=False,
        indent=2,
    )
