"""Create a downloadable audit report for a completed video render."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


_NO_REASON = "当前步骤的模型返回格式未要求输出独立理由，因此没有可记录的模型理由。"


def _as_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bullets(values: list[str], empty: str) -> list[str]:
    return ["- {}".format(value) for value in values] or ["- {}".format(empty)]


def _duration(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip_section(index: int, clip: dict[str, Any], video_file: str | None) -> list[str]:
    ranges = clip.get("ranges") if isinstance(clip.get("ranges"), list) else []
    keywords = [line.strip() for line in _text(clip.get("keywords")).splitlines() if line.strip()]
    keyword_reasons = {
        _text(item.get("word")): _text(item.get("reason"))
        for item in clip.get("keyword_reasons", [])
        if isinstance(item, dict) and _text(item.get("word"))
    }
    sound_payload = _as_json(clip.get("sound_bindings"))
    sound_cues = sound_payload.get("cues", [])
    sound_decisions = sound_payload.get("decisions", [])
    sound_status = sound_payload.get("status", {})
    visual_assets = _as_json(clip.get("visual_bindings")).get("placements", [])
    raw_highlight = _as_json(clip.get("raw_result"))
    high_reason = _text(raw_highlight.get("reason")) or _NO_REASON

    lines = ["## 素材 {}".format(index)]
    if video_file:
        lines.extend(["", "- 输出视频：`{}`".format(Path(video_file).name)])

    lines.extend(["", "### 高光选择", ""])
    if ranges:
        for start, end in ranges:
            lines.append("- `[{} - {}]`".format(start, end))
    else:
        lines.append("- 未选择有效高光时间段。")
    lines.append("- 理由：{}".format(high_reason))

    lines.extend(["", "### 高光关键词", ""])
    if keywords:
        for keyword in keywords:
            lines.append("- “{}”：理由：{}".format(keyword, keyword_reasons.get(keyword) or _NO_REASON))
    else:
        lines.append("- 未选择关键词。")

    lines.extend(["", "### 音效选择", ""])
    if isinstance(sound_cues, list) and sound_cues:
        for cue in sound_cues:
            if not isinstance(cue, dict):
                continue
            lines.append(
                "- `{}`：关键词“{}”绑定音效 `{}`。理由：{}".format(
                    _text(cue.get("timestamp")) or "未提供时间",
                    _text(cue.get("text")) or "未提供关键词",
                    _text(cue.get("sound_id")) or "未提供音效",
                    _text(cue.get("reason")) or "模型未返回理由。",
                )
            )
    else:
        if isinstance(sound_decisions, list) and sound_decisions:
            for decision in sound_decisions:
                if not isinstance(decision, dict) or decision.get("use_sound"):
                    continue
                lines.append(
                    "- 第 {} 条字幕未添加音效。理由：{}".format(
                        _text(decision.get("sentence_id")) or "?",
                        _text(decision.get("reason")) or "模型未返回理由。",
                    )
                )
        else:
            message = _text(sound_status.get("message")) if isinstance(sound_status, dict) else ""
            lines.append("- 本素材未选择音效。理由：{}".format(message or "音效阶段未返回决策。"))

    lines.extend(["", "### 插图选择", ""])
    if isinstance(visual_assets, list) and visual_assets:
        for asset in visual_assets:
            if not isinstance(asset, dict):
                continue
            lines.append(
                "- 第 {} 条字幕：关键词“{}”绑定素材 `{}`，展示 {:.2f} 秒。理由：{}".format(
                    _text(asset.get("sentence_id")) or "?",
                    _text(asset.get("target_word")) or "未提供关键词",
                    _text(asset.get("asset_id")) or "未提供素材",
                    _duration(asset.get("duration_seconds")),
                    _text(asset.get("reason")) or "模型未返回理由。",
                )
            )
    else:
        lines.append("- 本素材未选择 GIF/PNG 插图。")

    lines.extend(["", "### 高光字幕", "", "```srt", _text(clip.get("highlight_srt")) or "（无）", "```"])
    return lines


def write_generation_report(
        output_dir: str | Path,
        corrected_srt: str,
        clips: list[dict[str, Any]],
        videos: list[str] | None = None,
) -> str:
    """Write one Markdown audit report for all videos from one render task."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    video_files = list(videos or [])
    lines = [
        "# 视频生成决策报告",
        "",
        "- 生成时间：{}".format(datetime.now().astimezone().isoformat(timespec="seconds")),
        "- 成片数量：{}".format(len(video_files)),
        "",
        "## 清洗后的字幕稿",
        "",
        "```srt",
        _text(corrected_srt) or "（未提供）",
        "```",
    ]
    for index, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            continue
        lines.extend(["", *_clip_section(index, clip, video_files[index - 1] if index <= len(video_files) else None)])

    filename = "video_generation_report_{}.md".format(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path = destination / filename
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)
