import copy
import re

from utils.subtitle_utils import str2list, time_convert


_TIMESTAMP_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$"
)


class SubtitleReplacementError(ValueError):
    pass


def _time_to_millis(value):
    hours, minutes, seconds_millis = value.replace(".", ",").split(":")
    seconds, millis = seconds_millis.split(",")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis.ljust(3, "0")[:3])
    )


def parse_srt(srt_text):
    lines = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    timestamp_indexes = [index for index, line in enumerate(lines) if _TIMESTAMP_RE.match(line)]
    if not timestamp_indexes:
        raise SubtitleReplacementError("The uploaded file contains no valid SRT timestamp ranges.")

    entries = []
    for position, timestamp_index in enumerate(timestamp_indexes):
        next_timestamp_index = (
            timestamp_indexes[position + 1]
            if position + 1 < len(timestamp_indexes)
            else len(lines)
        )
        text_end = next_timestamp_index
        if next_timestamp_index < len(lines) and re.match(r"^\s*\d+(?:\s+spk\S+)?\s*$", lines[next_timestamp_index - 1]):
            text_end = next_timestamp_index - 1
        text = "\n".join(lines[timestamp_index + 1:text_end]).strip()
        if not text:
            raise SubtitleReplacementError("Each SRT cue must contain subtitle text.")
        match = _TIMESTAMP_RE.match(lines[timestamp_index])
        start_ms = _time_to_millis(match.group("start"))
        end_ms = _time_to_millis(match.group("end"))
        if end_ms <= start_ms:
            raise SubtitleReplacementError("Every SRT cue must end after it starts.")
        prefix = lines[timestamp_index - 1] if timestamp_index else ""
        speaker_match = re.search(r"\bspk([^\s]+)", prefix, re.IGNORECASE)
        speaker = speaker_match.group(1) if speaker_match else None
        entries.append({"start": start_ms, "end": end_ms, "text": text, "speaker": speaker})
    return entries


def canonical_srt(entries):
    blocks = []
    for index, entry in enumerate(entries, start=1):
        blocks.append(
            "{}\n{} --> {}\n{}".format(
                index,
                time_convert(entry["start"]),
                time_convert(entry["end"]),
                entry["text"],
            )
        )
    return "\n\n".join(blocks) + "\n"


def _interpolate_timestamps(start_ms, end_ms, count):
    count = max(1, count)
    duration = max(count, end_ms - start_ms)
    timestamps = []
    for index in range(count):
        start = start_ms + round(duration * index / count)
        end = start_ms + round(duration * (index + 1) / count)
        timestamps.append([start, max(start + 1, end)])
    timestamps[-1][1] = max(timestamps[-1][0] + 1, end_ms)
    return timestamps


def _backup_original_subtitles(state, original_srt):
    if "_subtitle_replacement_backup" in state:
        return
    keys = ("recog_res_raw", "timestamp", "sentences", "sd_sentences")
    state["_subtitle_replacement_backup"] = {
        "srt": str(original_srt or ""),
        "values": {key: copy.deepcopy(state[key]) for key in keys if key in state},
        "missing": [key for key in keys if key not in state],
    }


def apply_replacement(state, original_srt, replacement_srt, replacement_path):
    entries = parse_srt(replacement_srt)
    if state is None:
        raise SubtitleReplacementError("Run ASR and load the video before replacing subtitles.")

    updated = dict(state)
    _backup_original_subtitles(updated, original_srt)
    sentences = []
    raw_tokens = []
    raw_timestamps = []
    has_speakers = False

    for entry in entries:
        tokens = str2list(entry["text"])
        if not tokens:
            tokens = [entry["text"]]
        timestamps = _interpolate_timestamps(entry["start"], entry["end"], len(tokens))
        sentence = {"text": tokens, "timestamp": timestamps}
        if entry["speaker"] is not None:
            sentence["spk"] = entry["speaker"]
            has_speakers = True
        sentences.append(sentence)
        raw_tokens.extend(tokens)
        raw_timestamps.extend(timestamps)

    updated["recog_res_raw"] = " ".join(raw_tokens)
    updated["timestamp"] = raw_timestamps
    updated["sentences"] = sentences
    if has_speakers:
        updated["sd_sentences"] = copy.deepcopy(sentences)
    else:
        updated.pop("sd_sentences", None)
    updated["subtitle_replacement_path"] = replacement_path
    updated["subtitle_replacement_srt"] = canonical_srt(entries)
    return updated, updated["subtitle_replacement_srt"]


def restore_original_subtitles(state):
    if state is None:
        return None, ""
    updated = dict(state)
    backup = updated.pop("_subtitle_replacement_backup", None)
    updated.pop("subtitle_replacement_path", None)
    updated.pop("subtitle_replacement_srt", None)
    if not backup:
        return updated, ""
    for key in backup.get("missing", []):
        updated.pop(key, None)
    for key, value in backup.get("values", {}).items():
        updated[key] = value
    return updated, backup.get("srt", "")
