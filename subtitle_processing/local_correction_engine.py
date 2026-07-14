"""In-memory port of the user's proven DeepSeek SRT correction script.

This module deliberately keeps the source script's parsing, batching, context
payload and response validation rules.  The web app owns the API transport;
this module only owns correction semantics and never stores an API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*"
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}(?:\s+.*)?$"
)
HEADER_RE = re.compile(r"^\s*(\d+)(?:\s+.*)?$")


@dataclass
class SubtitleBlock:
    """One original SRT block; header/timestamp are never sent for editing."""

    internal_id: str
    header: str
    timestamp: str
    text_lines: list[str]
    separator_blank_lines: int

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)

    def set_text(self, value: str) -> None:
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        self.text_lines = value.split("\n") if value else [""]

    def render(self, newline: str) -> str:
        return newline.join([self.header, self.timestamp, *self.text_lines]) + newline * (
            self.separator_blank_lines + 1
        )


def parse_srt_text(srt_text: str) -> tuple[list[SubtitleBlock], str]:
    """Parse with the exact block boundaries used by the standalone script."""
    raw = str(srt_text or "")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    timestamp_positions = [i for i, line in enumerate(lines) if TIMESTAMP_RE.match(line)]
    if not timestamp_positions:
        raise ValueError("No valid SRT timestamps found.")

    blocks: list[SubtitleBlock] = []
    for pos, timestamp_index in enumerate(timestamp_positions):
        header_index = timestamp_index - 1
        if header_index < 0 or not HEADER_RE.match(lines[header_index]):
            raise ValueError("SRT timestamp has no valid header line: {!r}".format(lines[timestamp_index]))
        if pos + 1 < len(timestamp_positions):
            content_end = timestamp_positions[pos + 1] - 1
        else:
            content_end = len(lines)
        text_lines = lines[timestamp_index + 1 : content_end]
        trailing_blank_count = 0
        while text_lines and text_lines[-1] == "":
            text_lines.pop()
            trailing_blank_count += 1
        if not text_lines:
            text_lines = [""]
        blocks.append(
            SubtitleBlock(
                internal_id=str(pos + 1),
                header=lines[header_index],
                timestamp=lines[timestamp_index],
                text_lines=text_lines,
                separator_blank_lines=max(0, trailing_blank_count - 1),
            )
        )
    return blocks, newline


def render_srt(blocks: list[SubtitleBlock], newline: str) -> str:
    return "".join(block.render(newline) for block in blocks)


def make_payload(
    blocks: list[SubtitleBlock], start: int, end: int, context_size: int
) -> dict[str, Any]:
    context_start = max(0, start - context_size)
    context_end = min(len(blocks), end + context_size)
    return {
        "instruction": (
            "Only correct target_entries. context_entries are only for context. "
            "Return JSON strictly matching target_ids in count, ID, and order."
        ),
        "target_ids": [blocks[index].internal_id for index in range(start, end)],
        "context_entries": [
            {"id": blocks[index].internal_id, "speaker": blocks[index].header, "text": blocks[index].text}
            for index in range(context_start, context_end)
        ],
        "target_entries": [
            {"id": blocks[index].internal_id, "text": blocks[index].text}
            for index in range(start, end)
        ],
    }


def extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        first, last = content.find("{"), content.rfind("}")
        if first < 0 or last <= first:
            raise ValueError("API response does not contain a valid JSON object.")
        result = json.loads(content[first : last + 1])
    if not isinstance(result, dict):
        raise ValueError("API JSON root must be an object.")
    return result


def validate_entries(data: dict[str, Any], expected_ids: list[str]) -> list[dict[str, str]]:
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("API JSON is missing entries array.")
    if len(entries) != len(expected_ids):
        raise ValueError("Returned entry count does not match input.")
    clean_entries: list[dict[str, str]] = []
    returned_ids: list[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("entries[{}] is not an object.".format(position))
        entry_id, text = str(entry.get("id", "")), entry.get("text")
        if not isinstance(text, str):
            raise ValueError("Entry {} text is not a string.".format(entry_id or "?"))
        if "-->" in text or TIMESTAMP_RE.search(text):
            raise ValueError("Entry {} text contains a timestamp.".format(entry_id))
        if re.search(r"(?m)^\s*\d+\s+spk\d+\s*$", text):
            raise ValueError("Entry {} text contains a speaker header.".format(entry_id))
        returned_ids.append(entry_id)
        clean_entries.append({"id": entry_id, "text": text})
    if returned_ids != expected_ids:
        raise ValueError("Returned IDs or order do not match target_ids.")
    return clean_entries


def correct_srt(
    srt_text: str,
    call_api: Callable[[dict[str, Any], int, int], str],
    *,
    batch_size: int = 30,
    context_size: int = 15,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> str:
    """Correct batches in-place just as the standalone script does."""
    blocks, newline = parse_srt_text(srt_text)
    total_batches = (len(blocks) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(blocks), batch_size), start=1):
        end = min(start + batch_size, len(blocks))
        payload = make_payload(blocks, start, end, context_size)
        response = call_api(payload, batch_number, total_batches)
        entries = validate_entries(extract_json_object(response), payload["target_ids"])
        for index, entry in enumerate(entries, start=start):
            blocks[index].set_text(entry["text"])
        if progress:
            progress(batch_number, total_batches, end, len(blocks))
    return render_srt(blocks, newline)
