#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import re
import os
import sys
import copy
import json
import shutil
import librosa
import logging
import argparse
import subprocess
import numpy as np
import soundfile as sf
from PIL import Image, ImageColor, ImageDraw, ImageFont

if os.environ.get("IMAGEMAGICK_BINARY") in (None, "", "unset"):
    imagemagick_binary = shutil.which("magick") or shutil.which("convert")
    if imagemagick_binary:
        os.environ["IMAGEMAGICK_BINARY"] = imagemagick_binary

from moviepy.editor import *
import moviepy.editor as mpy
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from utils.subtitle_utils import generate_srt, generate_srt_clip, str2list
from utils.argparse_tools import ArgumentParser, get_commandline_args
from utils.trans_utils import pre_proc, proc, write_state, load_state, proc_spk, convert_pcm_to_float
from llm.subtitle_correction import parse_srt_entries, _entry_timestamp_millis_range


MAX_SUBTITLE_DURATION_MS = 8000
MAX_SUBTITLE_TOKENS = 30
SENSEVOICE_TAG_RE = re.compile(r"<\|[^|>]+\|>")


def _probe_video_rotation(video_filename):
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            video_filename,
        ]
        probe = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream = (json.loads(probe.stdout).get("streams") or [{}])[0]
    except Exception:
        return 0

    rotate = (stream.get("tags") or {}).get("rotate")
    if rotate not in (None, ""):
        try:
            return int(float(rotate)) % 360
        except ValueError:
            pass

    for side_data in stream.get("side_data_list") or []:
        rotation = side_data.get("rotation")
        if rotation not in (None, ""):
            try:
                return int(float(rotation)) % 360
            except ValueError:
                pass
    return 0


def _open_video_preserving_display(video_filename):
    video = mpy.VideoFileClip(video_filename)
    rotation = int(getattr(video, "rotation", 0) or 0) % 360
    rotation = rotation or _probe_video_rotation(video_filename)
    if rotation in (90, 180, 270):
        logging.warning("Applying source video rotation metadata: %s degrees.", rotation)
        video = video.rotate(rotation, expand=True)
        video.rotation = 0
    return video


def _load_subtitle_font(font_size):
    font_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "font",
        "STHeitiMedium.ttc",
    )
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        logging.warning("Subtitle font not found, falling back to Pillow default font.")
        return ImageFont.load_default()


def _text_size(draw, text, font):
    if not text:
        return 0, 0
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, stroke_width=2)
    return right - left, bottom - top


def _wrap_subtitle_text(text, font, max_width):
    probe = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    wrapped_lines = []

    for raw_line in str(text).splitlines() or [""]:
        line = ""
        for char in raw_line:
            candidate = line + char
            width, _ = _text_size(draw, candidate, font)
            if line and width > max_width:
                wrapped_lines.append(line)
                line = char
            else:
                line = candidate
        if line:
            wrapped_lines.append(line)

    return "\n".join(wrapped_lines) if wrapped_lines else str(text)


def _split_highlight_terms(highlight_terms):
    if not highlight_terms:
        return []
    if isinstance(highlight_terms, (list, tuple, set)):
        raw_terms = [str(term).strip() for term in highlight_terms]
    else:
        text = str(highlight_terms).strip()
        separators = r"[\n,，;；、]+"
        raw_terms = [term.strip() for term in re.split(separators, text)]
        if len(raw_terms) == 1 and " " in raw_terms[0]:
            raw_terms = [term.strip() for term in raw_terms[0].split()]
    return [term for term in raw_terms if term]


def _highlight_flags(text, highlight_terms):
    flags = [False] * len(text)
    lower_text = text.lower()
    for term in _split_highlight_terms(highlight_terms):
        lower_term = term.lower()
        start = 0
        while lower_term:
            index = lower_text.find(lower_term, start)
            if index < 0:
                break
            for pos in range(index, min(index + len(term), len(flags))):
                flags[pos] = True
            start = index + max(1, len(term))
    return flags


def _wrap_colored_subtitle_text(text, flags, font, max_width):
    probe = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    lines = []
    line = []
    line_text = ""

    for char, highlighted in zip(text, flags):
        if char == "\n":
            lines.append(line)
            line = []
            line_text = ""
            continue
        candidate = line_text + char
        width, _ = _text_size(draw, candidate, font)
        if line and width > max_width:
            lines.append(line)
            line = [(char, highlighted)]
            line_text = char
        else:
            line.append((char, highlighted))
            line_text = candidate

    if line:
        lines.append(line)
    return lines or [[("", False)]]


def _subtitle_image_clip(text, font_size, font_color, video_size, highlight_terms=None, highlight_color="yellow"):
    font = _load_subtitle_font(font_size)
    max_width = max(240, int(video_size[0] * 0.9))
    text = str(text)
    highlight_terms = _split_highlight_terms(highlight_terms)

    probe = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    if highlight_terms:
        flags = _highlight_flags(text, highlight_terms)
        colored_lines = _wrap_colored_subtitle_text(text, flags, font, max_width)
        line_texts = ["".join(char for char, _ in line) for line in colored_lines]
        text_width = max((_text_size(draw, line_text, font)[0] for line_text in line_texts), default=0)
        line_height = max(_text_size(draw, "测试T", font)[1], font_size)
        text_height = len(colored_lines) * line_height + max(0, len(colored_lines) - 1) * 6
    else:
        wrapped_text = _wrap_subtitle_text(text, font, max_width)
        text_width, text_height = _text_size(draw, wrapped_text, font)
    padding_x = max(16, font_size // 2)
    padding_y = max(10, font_size // 4)
    image_size = (text_width + padding_x * 2, text_height + padding_y * 2)
    image = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    try:
        fill = ImageColor.getrgb(font_color)
    except ValueError:
        fill = (255, 255, 255)
    try:
        highlight_fill = ImageColor.getrgb(highlight_color)
    except ValueError:
        highlight_fill = (255, 255, 0)

    if highlight_terms:
        line_height = max(_text_size(draw, "测试T", font)[1], font_size)
        for line_index, line in enumerate(colored_lines):
            line_text = "".join(char for char, _ in line)
            line_width, _ = _text_size(draw, line_text, font)
            x = padding_x + max(0, text_width - line_width) / 2
            y = padding_y + line_index * (line_height + 6)
            for char, highlighted in line:
                char_fill = highlight_fill if highlighted else fill
                draw.text(
                    (x, y),
                    char,
                    font=font,
                    fill=char_fill + (255,),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 190),
                )
                char_width, _ = _text_size(draw, char, font)
                x += char_width
    else:
        draw.multiline_text(
            (padding_x, padding_y),
            wrapped_text,
            font=font,
            fill=fill + (255,),
            spacing=6,
            align="center",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 190),
        )
    return ImageClip(np.array(image), transparent=True)


def _subtitle_position(video_size, subtitle_size, subtitle_x=50, subtitle_y=88):
    video_width, video_height = video_size
    subtitle_width, subtitle_height = subtitle_size
    max_x = max(0, video_width - subtitle_width)
    max_y = max(0, video_height - subtitle_height)
    x = max_x * min(max(float(subtitle_x), 0.0), 100.0) / 100.0
    y = max_y * min(max(float(subtitle_y), 0.0), 100.0) / 100.0
    return x, y


def _subtitle_parts(sub):
    return sub[0], sub[1], sub[2] if len(sub) > 2 else None


def _corrected_subtitle_cues(subtitle_srt_text):
    if not str(subtitle_srt_text or "").strip():
        return []
    try:
        return [
            (start_ms / 1000.0, end_ms / 1000.0, entry["text"])
            for entry in parse_srt_entries(subtitle_srt_text)
            for start_ms, end_ms in [_entry_timestamp_millis_range(entry)]
        ]
    except Exception:
        logging.exception("Failed to parse corrected SRT for burned-in subtitles.")
        return []


def _subtitle_cues_for_clip(cues, clip_start, clip_end):
    subtitles = []
    for cue_start, cue_end, text in cues:
        visible_start = max(cue_start, clip_start)
        visible_end = min(cue_end, clip_end)
        if visible_end <= visible_start:
            continue
        subtitles.append(
            ((visible_start - clip_start, visible_end - clip_start), text)
        )
    return subtitles


def _format_srt_timestamp(seconds):
    millis = max(0, int(round(float(seconds) * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _subtitle_cues_to_srt(cues, begin_index=1, time_offset=0.0):
    """Render the exact subtitle cues used for the video into an SRT download."""
    blocks = []
    for index, ((start, end), text) in enumerate(cues, start=begin_index):
        blocks.append(
            "{}\n{} --> {}\n{}".format(
                index,
                _format_srt_timestamp(start + time_offset),
                _format_srt_timestamp(end + time_offset),
                text,
            )
        )
    return "\n\n".join(blocks) + ("\n\n" if blocks else ""), begin_index + len(cues)


def _with_subtitles(video_clip, subs, font_size, font_color, subtitle_x=50, subtitle_y=88, highlight_terms=None, highlight_color="yellow"):
    subtitle_clips = []
    for sub in subs:
        (start, end), text, _ = _subtitle_parts(sub)
        subtitle_clip = _subtitle_image_clip(text, font_size, font_color, video_clip.size, highlight_terms, highlight_color)
        subtitle_clip = subtitle_clip.set_start(start).set_duration(max(0, end - start))
        subtitle_clip = subtitle_clip.set_pos(
            lambda _, clip=subtitle_clip: _subtitle_position(video_clip.size, clip.size, subtitle_x, subtitle_y)
        )
        subtitle_clips.append(subtitle_clip)
    return CompositeVideoClip([video_clip, *subtitle_clips], size=video_clip.size)


def _split_trigger_terms(terms_text):
    if not terms_text:
        return []
    if isinstance(terms_text, (list, tuple, set)):
        raw_terms = terms_text
    else:
        raw_terms = re.split(r"[\n,，;；、]+", str(terms_text))
    return [str(term).strip() for term in raw_terms if str(term).strip()]


def _parse_sound_effect_rules(sound_effect_rules, sound_effect_dir=None):
    rules = []
    if not sound_effect_rules:
        return rules

    base_dir = os.path.abspath(sound_effect_dir) if sound_effect_dir else None
    project_music_dir = None
    if base_dir:
        project_music_dir = os.path.abspath(
            os.environ.get("FUNCLIP_MUSIC_DIR") or os.path.join(os.path.dirname(base_dir), "music")
        )
    allowed_dirs = []
    if base_dir:
        allowed_dirs.append(base_dir)
        allowed_dirs.append(project_music_dir)
    for line in str(sound_effect_rules).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue

        effect_path = parts[0]
        raw_effect_path = effect_path
        if base_dir and not os.path.isabs(effect_path):
            effect_path = os.path.abspath(os.path.join(base_dir, effect_path))
            if not os.path.isfile(effect_path) and raw_effect_path.replace("\\", "/").startswith("music/"):
                rel_music_path = raw_effect_path.replace("\\", "/")[len("music/"):]
                effect_path = os.path.abspath(os.path.join(project_music_dir, rel_music_path))
        else:
            effect_path = os.path.abspath(effect_path)
        if allowed_dirs and not any(
                effect_path.startswith(allowed_dir + os.sep) or effect_path == allowed_dir
                for allowed_dir in allowed_dirs):
            logging.warning("Skip sound effect outside allowed sound effect directories: %s", effect_path)
            continue
        if not os.path.isfile(effect_path):
            logging.warning("Skip missing sound effect file: %s", effect_path)
            continue

        try:
            volume = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.35
        except ValueError:
            volume = 0.35
        try:
            cooldown = max(0.0, float(parts[3])) if len(parts) >= 4 and parts[3] else 2.0
        except ValueError:
            cooldown = 2.0

        terms = _split_trigger_terms(parts[1])
        if not terms:
            continue
        rules.append({
            "path": effect_path,
            "terms": terms,
            "volume": max(0.0, min(volume, 2.0)),
            "cooldown": cooldown,
        })
    return rules


def _compact_text_with_token_map(token_times):
    compact_chars = []
    char_to_token = []
    for token_index, token in enumerate(token_times or []):
        token_text = str(token.get("text", "")).strip().lower()
        if not token_text:
            continue
        for char in token_text:
            if char.isspace():
                continue
            compact_chars.append(char)
            char_to_token.append(token_index)
    return "".join(compact_chars), char_to_token


def _sound_effect_trigger_times(start, end, text, terms, token_times=None):
    compact_text, char_to_token = _compact_text_with_token_map(token_times)
    if compact_text and char_to_token:
        triggers = []
        seen = set()
        for term in terms:
            term_compact = "".join(str(term or "").strip().lower().split())
            if not term_compact:
                continue
            search_from = 0
            while True:
                index = compact_text.find(term_compact, search_from)
                if index < 0:
                    break
                token_index = char_to_token[index]
                key = (token_index, term_compact)
                if key not in seen:
                    seen.add(key)
                    token = token_times[token_index]
                    triggers.append((float(token.get("start", start)), term_compact))
                search_from = index + max(1, len(term_compact))
        if triggers:
            return sorted(triggers, key=lambda item: item[0])

    text = str(text or "")
    text_lower = text.lower()
    if not text_lower:
        return []

    duration = max(0.0, float(end) - float(start))
    if duration <= 0:
        return []

    triggers = []
    seen = set()
    text_len = max(1, len(text_lower))
    for term in terms:
        term_lower = str(term or "").strip().lower()
        if not term_lower:
            continue

        search_from = 0
        while True:
            index = text_lower.find(term_lower, search_from)
            if index < 0:
                break

            key = (index, term_lower)
            if key not in seen:
                seen.add(key)
                ratio = max(0.0, min(1.0, index / text_len))
                triggers.append((float(start) + duration * ratio, term_lower))

            search_from = index + max(1, len(term_lower))

    return sorted(triggers, key=lambda item: item[0])


def _with_sound_effects(video_clip, subs, sound_effect_rules=None, sound_effect_dir=None):
    rules = _parse_sound_effect_rules(sound_effect_rules, sound_effect_dir)
    if not rules or not subs:
        return video_clip, 0

    audio_clips = []
    if video_clip.audio is not None:
        audio_clips.append(video_clip.audio)

    trigger_count = 0
    last_trigger_by_path = {}
    for sub in subs:
        (start, end), text, token_times = _subtitle_parts(sub)
        for rule in rules:
            for trigger_start, _ in _sound_effect_trigger_times(start, end, text, rule["terms"], token_times):
                last_trigger = last_trigger_by_path.get(rule["path"])
                if last_trigger is not None and trigger_start - last_trigger < rule["cooldown"]:
                    continue
                if trigger_start >= video_clip.duration:
                    continue

                available_duration = video_clip.duration - trigger_start
                if available_duration <= 0.05:
                    continue
                try:
                    effect_clip = AudioFileClip(rule["path"]).volumex(rule["volume"])
                except Exception:
                    logging.exception("Failed to load sound effect: %s", rule["path"])
                    continue
                if effect_clip.duration > available_duration:
                    effect_clip = effect_clip.subclip(0, available_duration)
                audio_clips.append(effect_clip.set_start(trigger_start))
                last_trigger_by_path[rule["path"]] = trigger_start
                trigger_count += 1

    if trigger_count == 0:
        return video_clip, 0
    return video_clip.set_audio(CompositeAudioClip(audio_clips)), trigger_count


def _is_valid_timestamp(timestamp):
    return (
        isinstance(timestamp, list)
        and len(timestamp) > 0
        and timestamp[0] is not None
        and timestamp[-1] is not None
    )


def _clean_recognition_text(text):
    if text is None:
        return ""
    text = SENSEVOICE_TAG_RE.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("“”")


def _split_long_sentence(sent):
    timestamp = sent.get("timestamp")
    if not _is_valid_timestamp(timestamp):
        return []

    cleaned_text = _clean_recognition_text(sent.get("text"))
    normalized = dict(sent)
    normalized["text"] = cleaned_text
    normalized["timestamp"] = timestamp

    tokens = str2list(cleaned_text)
    if len(timestamp) <= 1 or len(tokens) != len(timestamp):
        return [normalized]

    chunks = []
    start = 0
    for idx in range(len(tokens)):
        duration = timestamp[idx][1] - timestamp[start][0]
        token_count = idx - start + 1
        should_split = (
            idx > start
            and (
                duration >= MAX_SUBTITLE_DURATION_MS
                or token_count >= MAX_SUBTITLE_TOKENS
            )
        )
        if should_split:
            chunk = dict(normalized)
            chunk["text"] = tokens[start : idx + 1]
            chunk["timestamp"] = timestamp[start : idx + 1]
            chunks.append(chunk)
            start = idx + 1

    if not chunks:
        return [normalized]

    if start < len(tokens):
        chunk = dict(normalized)
        chunk["text"] = tokens[start:]
        chunk["timestamp"] = timestamp[start:]
        chunks.append(chunk)

    return chunks


def _normalize_recognition_result(result):
    text = _clean_recognition_text(
        result.get("text") or result.get("text_tn") or result.get("raw_text") or ""
    )
    raw_text = _clean_recognition_text(
        result.get("raw_text") or result.get("text_tn") or text
    )
    timestamp = result.get("timestamp") or result.get("timestamps") or []

    sentence_info = []
    for sent in result.get("sentence_info") or []:
        if _is_valid_timestamp(sent.get("timestamp")):
            sentence_info.extend(_split_long_sentence(sent))

    if not sentence_info and text and _is_valid_timestamp(timestamp):
        sentence_info = _split_long_sentence({"text": text, "timestamp": timestamp})

    return text, raw_text, timestamp, sentence_info


class VideoClipper():
    def __init__(self, funasr_model):
        logging.warning("Initializing VideoClipper.")
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0

    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input

        # Convert to float64 consistently (includes data type checking)
        data = convert_pcm_to_float(data)

        # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if sr != 16000: # resample with librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, only first channel reserved.".format(data.shape))
            data = data[:,0]
        state['audio_input'] = (sr, data)
        if sd_switch == 'Yes':
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=True,
                                                    return_raw_text=True, 
                                                    is_final=True,
                                                    output_dir=output_dir, 
                                                    hotword=hotwords, 
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_text, raw_text, timestamp, sentence_info = _normalize_recognition_result(rec_result[0])
            res_srt = generate_srt(sentence_info)
            state['sd_sentences'] = sentence_info
        else:
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=False, 
                                                    sentence_timestamp=True, 
                                                    return_raw_text=True, 
                                                    is_final=True, 
                                                    hotword=hotwords,
                                                    output_dir=output_dir,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_text, raw_text, timestamp, sentence_info = _normalize_recognition_result(rec_result[0])
            res_srt = generate_srt(sentence_info)
        state['recog_res_raw'] = raw_text
        state['timestamp'] = timestamp
        state['sentences'] = sentence_info
        return res_text, res_srt, state

    def clip(self, dest_text, start_ost, end_ost, state, dest_spk=None, output_dir=None, timestamp_list=None):
        # get from state
        audio_input = state['audio_input']
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        subtitle_overrides = state.get('subtitle_text_overrides')
        sr, data = audio_input
        data = data.astype(np.float64)

        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        log_append = ""
                        offset_b, offset_e = 0, 0
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text)
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio, \
                            offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
                log_append = ""
        else:
            all_ts = timestamp_list
        ts = all_ts
        # ts.sort()
        srt_index = 0
        clip_srt = ""
        if len(ts):
            start, end = ts[0]
            start = min(max(0, start+start_ost*16), len(data))
            end = min(max(0, end+end_ost*16), len(data))
            res_audio = data[start:end]
            start_end_info = "from {} to {}".format(start/16000, end/16000)
            srt_clip, _, srt_index = generate_srt_clip(
                sentences, start/16000.0, end/16000.0,
                begin_index=srt_index, subtitle_overrides=subtitle_overrides
            )
            clip_srt += srt_clip
            for _ts in ts[1:]:  # multiple sentence input or multiple output matched
                start, end = _ts
                start = min(max(0, start+start_ost*16), len(data))
                end = min(max(0, end+end_ost*16), len(data))
                start_end_info += ", from {} to {}".format(start, end)
                res_audio = np.concatenate([res_audio, data[start:end]], -1)
                srt_clip, _, srt_index = generate_srt_clip(
                    sentences, start/16000.0, end/16000.0,
                    begin_index=srt_index-1, subtitle_overrides=subtitle_overrides
                )
                clip_srt += srt_clip
        if len(ts):
            message = "{} periods found in the speech: ".format(len(ts)) + start_end_info + log_append
        else:
            message = "No period found in the speech, return raw speech. You may check the recognition result and try other destination text."
            res_audio = data
        return (sr, res_audio), message, clip_srt

    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        video = _open_video_preserving_display(video_filename)
        # Extract the base name, add '_clip.mp4', and 'wav'
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)
        else:
            base_name, _ = os.path.splitext(video_filename)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'

        if video.audio is None:
            logging.error("No audio information found.")
            sys.exit(1)

        video.audio.write_audiofile(audio_file)
        wav = librosa.load(audio_file, sr=16000)[0]
        # delete the audio file after processing
        if os.path.exists(audio_file):
            os.remove(audio_file)
        state = {
            'video_filename': video_filename,
            'clip_video_file': clip_video_file,
            'video': video,
        }
        # res_text, res_srt = self.recog((16000, wav), state)
        return self.recog((16000, wav), sd_switch, state, hotwords, output_dir)

    def video_clip(self, 
                   dest_text, 
                   start_ost, 
                   end_ost, 
                   state, 
                   font_size=32, 
                   font_color='white', 
                   subtitle_x=50,
                   subtitle_y=88,
                   highlight_terms=None,
                   highlight_color='yellow',
                    sound_effect_rules=None,
                    sound_effect_dir=None,
                    add_sub=False,
                    dest_spk=None,
                    output_dir=None,
                    timestamp_list=None,
                    subtitle_srt_text=None):
        # get from state
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        subtitle_overrides = state.get('subtitle_text_overrides')
        use_current_srt_for_subtitles = bool(str(subtitle_srt_text or "").strip())
        corrected_subtitle_cues = _corrected_subtitle_cues(subtitle_srt_text)
        if use_current_srt_for_subtitles and not corrected_subtitle_cues:
            raise ValueError(
                "The current SRT could not be parsed. Refusing to fall back to ASR subtitles."
            )
        if use_current_srt_for_subtitles:
            logging.warning(
                "Burned-in subtitles use the current corrected SRT exclusively (%d cues).",
                len(corrected_subtitle_cues),
            )
        video = state['video']
        clip_video_file = state['clip_video_file']
        video_filename = state['video_filename']
        
        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        offset_b, offset_e = 0, 0
                        log_append = ""
                    # import pdb; pdb.set_trace()
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text.lower())
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio, \
                            offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
        else:  # AI clip pass timestamp as input directly
            all_ts = [[i[0]*16.0, i[1]*16.0] for i in timestamp_list]
        
        srt_index = 0
        time_acc_ost = 0.0
        ts = all_ts
        # ts.sort()
        clip_srt = ""
        rendered_srt_index = 1
        sound_effect_subs = []
        if len(ts):
            if self.lang == 'en' and isinstance(sentences, str):
                sentences = sentences.split()
            start, end = ts[0][0] / 16000, ts[0][1] / 16000
            srt_clip, subs, srt_index = generate_srt_clip(
                sentences, start, end, begin_index=srt_index,
                time_acc_ost=time_acc_ost, subtitle_overrides=subtitle_overrides
            )
            start, end = start+start_ost/1000.0, end+end_ost/1000.0
            if use_current_srt_for_subtitles:
                # This is deliberately not allowed to fall back to ``subs``. Those
                # are derived from the original ASR state and would reintroduce
                # uncorrected text into the rendered video.
                rendered_subs = _subtitle_cues_for_clip(corrected_subtitle_cues, start, end)
                rendered_srt, rendered_srt_index = _subtitle_cues_to_srt(
                    rendered_subs, rendered_srt_index, time_acc_ost
                )
                clip_srt += rendered_srt
            else:
                rendered_subs = subs
                clip_srt += srt_clip
            sound_effect_subs.extend(rendered_subs)
            video_clip = video.subclip(start, end)
            start_end_info = "from {} to {}".format(start, end)
            if add_sub:
                video_clip = _with_subtitles(video_clip, rendered_subs, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color)
            concate_clip = [video_clip]
            time_acc_ost += end - start
            for _ts in ts[1:]:
                start, end = _ts[0] / 16000, _ts[1] / 16000
                srt_clip, subs, srt_index = generate_srt_clip(
                    sentences, start, end, begin_index=srt_index-1,
                    time_acc_ost=time_acc_ost, subtitle_overrides=subtitle_overrides
                )
                if not len(subs) and not corrected_subtitle_cues:
                    continue
                start, end = start+start_ost/1000.0, end+end_ost/1000.0
                if use_current_srt_for_subtitles:
                    rendered_subs = _subtitle_cues_for_clip(corrected_subtitle_cues, start, end)
                    rendered_srt, rendered_srt_index = _subtitle_cues_to_srt(
                        rendered_subs, rendered_srt_index, time_acc_ost
                    )
                    clip_srt += rendered_srt
                else:
                    rendered_subs = subs
                    clip_srt += srt_clip
                if not rendered_subs and not use_current_srt_for_subtitles:
                    continue
                sound_effect_subs.extend(
                    [
                        ((sub[0][0] + time_acc_ost, sub[0][1] + time_acc_ost), sub[1])
                        for sub in rendered_subs
                    ]
                )
                chi_subs = []
                if rendered_subs:
                    sub_starts = rendered_subs[0][0][0]
                    for sub in rendered_subs:
                        chi_subs.append(((sub[0][0]-sub_starts, sub[0][1]-sub_starts), sub[1]))
                _video_clip = video.subclip(start, end)
                start_end_info += ", from {} to {}".format(str(start)[:5], str(end)[:5])
                if add_sub and chi_subs:
                    _video_clip = _with_subtitles(_video_clip, chi_subs, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color)
                    # _video_clip.write_videofile("debug.mp4", audio_codec="aac")
                concate_clip.append(copy.copy(_video_clip))
                time_acc_ost += end - start
            message = "{} periods found in the audio: ".format(len(ts)) + start_end_info
            if use_current_srt_for_subtitles:
                message += "; burned captions source: current corrected SRT ({} cues)".format(
                    len(corrected_subtitle_cues)
                )
            logging.warning("Concating...")
            if len(concate_clip) > 1:
                video_clip = concatenate_videoclips(concate_clip)
            video_clip, sound_effect_count = _with_sound_effects(
                video_clip, sound_effect_subs, sound_effect_rules, sound_effect_dir
            )
            if sound_effect_count:
                message += "; {} sound effects mixed".format(sound_effect_count)
            # clip_video_file = clip_video_file[:-4] + '_no{}.mp4'.format(self.GLOBAL_COUNT)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                _, file_with_extension = os.path.split(clip_video_file)
                clip_video_file_name, _ = os.path.splitext(file_with_extension)
                print(output_dir, clip_video_file)
                clip_video_file = os.path.join(output_dir, "{}_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))
                temp_audio_file = os.path.join(output_dir, "{}_tempaudio_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))
            else:
                clip_video_file = clip_video_file[:-4] + '_no{}.mp4'.format(self.GLOBAL_COUNT)
                temp_audio_file = clip_video_file[:-4] + '_tempaudio_no{}.mp4'.format(self.GLOBAL_COUNT)
            video_clip.write_videofile(
                clip_video_file,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=temp_audio_file,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
            self.GLOBAL_COUNT += 1
        else:
            clip_video_file = video_filename
            message = "No period found in the audio, return raw speech. You may check the recognition result and try other destination text."
            srt_clip = ''
        return clip_video_file, message, clip_srt


def get_parser():
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file path",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Turn on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--dest_text",
        type=str,
        default=None,
        help="Destination text string for clipping",
    )
    parser.add_argument(
        "--dest_spk",
        type=str,
        default=None,
        help="Destination spk id for clipping",
    )
    parser.add_argument(
        "--start_ost",
        type=int,
        default=0,
        help="Offset time in ms at beginning for clipping"
    )
    parser.add_argument(
        "--end_ost",
        type=int,
        default=0,
        help="Offset time in ms at ending for clipping"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="language"
    )
    return parser


def runner(stage, file, sd_switch, output_dir, dest_text, dest_spk, start_ost, end_ost, output_file, config=None, lang='zh'):
    audio_suffixs = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixs = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    _,ext = os.path.splitext(file)
    if ext.lower() in audio_suffixs:
        mode = 'audio'
    elif ext.lower() in video_suffixs:
        mode = 'video'
    else:
        logging.error("Unsupported file format: {}\n\nplease choise one of the following: {}".format(file),audio_suffixs+video_suffixs)
        sys.exit(1) # exit if the file is not supported
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if stage == 1:
        from funasr import AutoModel
        # initialize funasr automodel
        logging.warning("Initializing modelscope asr pipeline.")
        if lang == 'zh':
            funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                    )
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'zh'
        elif lang == 'en':
            funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                )
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'en'
        if mode == 'audio':
            logging.warning("Recognizing audio file: {}".format(file))
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        if mode == 'video':
            logging.warning("Recognizing video file: {}".format(file))
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
            logging.warning("Write total subtitle to {}".format(total_srt_file))
        write_state(output_dir, state)
        logging.warning("Recognition successed. You can copy the text segment from below and use stage 2.")
        print(res_text)
    if stage == 2:
        audio_clipper = VideoClipper(None)
        if mode == 'audio':
            state = load_state(output_dir)
            wav, sr = librosa.load(file, sr=16000)
            state['audio_input'] = (sr, wav)
            (sr, audio), message, srt_clip = audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            if output_file is None:
                output_file = output_dir + '/result.wav'
            clip_srt_file = output_file[:-3] + 'srt'
            logging.warning(message)
            sf.write(output_file, audio, 16000)
            assert output_file.endswith('.wav'), "output_file must ends with '.wav'"
            logging.warning("Save clipped wav file to {}".format(output_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))
        if mode == 'video':
            state = load_state(output_dir)
            state['video_filename'] = file
            if output_file is None:
                state['clip_video_file'] = file[:-4] + '_clip.mp4'
            else:
                state['clip_video_file'] = output_file
            clip_srt_file = state['clip_video_file'][:-3] + 'srt'
            state['video'] = _open_video_preserving_display(file)
            clip_video_file, message, srt_clip = audio_clipper.video_clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            logging.warning("Clipping Log: {}".format(message))
            logging.warning("Save clipped mp4 file to {}".format(clip_video_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    runner(**kwargs)


if __name__ == '__main__':
    main()
