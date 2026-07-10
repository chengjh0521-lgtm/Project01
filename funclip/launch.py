#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from http import server
import json
import os
import re
import logging
import argparse
import shutil
import tempfile
import threading
import traceback
from datetime import datetime
from urllib.parse import unquote, urlparse
import gradio as gr
import requests
from PIL import Image
from funasr import AutoModel
from moviepy.editor import VideoFileClip
from videoclipper import VideoClipper, _open_video_preserving_display, _subtitle_image_clip, _subtitle_position
from llm.openai_api import openai_call
from llm.qwen_api import call_qwen_model
from llm.g4f_openai_api import g4f_openai_call
from llm.subtitle_correction import (
    DEFAULT_SUBTITLE_CORRECTION_PROMPT,
    SubtitleCorrectionError,
    correct_srt_with_llm,
    update_state_subtitles,
)
from llm.twelvelabs_api import call_twelvelabs_pegasus
from utils.trans_utils import extract_timestamps, load_state, write_state
from introduction import top_md_1, top_md_3, top_md_4


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOCAL_TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")
ASR_TASK_DIR = os.path.join(LOCAL_TMP_DIR, "asr_jobs")
LOCAL_GRADIO_TMP_DIR = os.path.join(PROJECT_ROOT, "gradio_tmp")
LOCAL_VIDEO_DIR = os.path.join(PROJECT_ROOT, "local_videos")
LOCAL_SFX_DIR = os.path.abspath(
    os.environ.get("FUNCLIP_LOCAL_SFX_DIR") or os.path.join(PROJECT_ROOT, "local_sfx")
)
LEGACY_MUSIC_DIR = os.path.abspath(
    os.environ.get("FUNCLIP_MUSIC_DIR") or os.path.join(PROJECT_ROOT, "music")
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
USER_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "user_settings.json")
ASR_TASKS = {}
ASR_TASK_LOCK = threading.Lock()
ASR_RUN_LOCK = threading.Lock()
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.makedirs(ASR_TASK_DIR, exist_ok=True)
os.makedirs(LOCAL_GRADIO_TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
os.makedirs(LOCAL_SFX_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("TMP", LOCAL_TMP_DIR)
os.environ.setdefault("TEMP", LOCAL_TMP_DIR)
os.environ.setdefault("TMPDIR", LOCAL_TMP_DIR)
os.environ.setdefault("GRADIO_TEMP_DIR", LOCAL_GRADIO_TMP_DIR)
tempfile.tempdir = LOCAL_TMP_DIR


DEFAULT_PROMPT_SYSTEM = (
    "你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，"
    "分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，"
    "注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是“-”"
)
DEFAULT_PROMPT_USER = "这是待裁剪的视频srt字幕："
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_HIGHLIGHT_PROMPT = (
    "Select subtitle keywords or short phrases that should be emphasized with color. "
    "Focus on medical conclusions, risks, diagnosis names, treatment advice, numbers, "
    "strong opinions, and sentences that make viewers want to keep watching. "
    "Return only the exact words or short phrases that appear in the SRT, one item per line. "
    "Do not explain."
)
DEFAULT_HIGHLIGHT_COUNT = 30
DEFAULT_SOUND_EFFECT_RULES = (
    "# Select a sound effect above, then enter trigger words for that sound effect.\n"
)
DEFAULT_SOUND_EFFECT_VOLUME = 0.35
DEFAULT_SOUND_EFFECT_COOLDOWN = 2


def load_user_settings():
    if not os.path.exists(USER_SETTINGS_PATH):
        return {}
    try:
        with open(USER_SETTINGS_PATH, "r", encoding="utf-8") as settings_file:
            return json.load(settings_file)
    except Exception:
        logging.exception("Failed to load user settings.")
        return {}


def save_user_settings(settings):
    with open(USER_SETTINGS_PATH, "w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, ensure_ascii=False, indent=2)
    if os.name != "nt":
        os.chmod(USER_SETTINGS_PATH, 0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--lang', '-l', type=str, default = "zh", help="language")
    parser.add_argument('--model', '-m', type=str, default="paraformer", choices=["paraformer", "fun-asr-nano", "sensevoice"], help="ASR model: paraformer, fun-asr-nano, or sensevoice")
    parser.add_argument('--share', '-s', action='store_true', help="if to establish gradio share link")
    parser.add_argument('--port', '-p', type=int, default=7860, help='port number')
    parser.add_argument('--listen', action='store_true', help="if to listen to all hosts")
    parser.add_argument('--with-examples', action='store_true', help="load remote demo examples")
    args = parser.parse_args()
    if not args.with_examples:
        gr.Examples = lambda *example_args, **example_kwargs: None
    
    if args.lang == 'zh':
        if hasattr(args, 'model') and args.model == 'fun-asr-nano':
            funasr_model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512",
                                    trust_remote_code=True,
                                    remote_code="./model.py",
                                    vad_model="fsmn-vad",
                                    vad_kwargs={"max_single_segment_time": 30000},
                                    spk_model="cam++",
                                    hub="hf",
                                    )
        elif hasattr(args, 'model') and args.model == 'sensevoice':
            funasr_model = AutoModel(model="iic/SenseVoiceSmall",
                                    vad_model="fsmn-vad",
                                    vad_kwargs={"max_single_segment_time": 30000},
                                    spk_model="cam++",
                                    )
        else:
            funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                    )
    else:
        funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                )
    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = args.lang
    user_settings = load_user_settings()

    def setting_float(key, default):
        try:
            return float(user_settings.get(key, default))
        except (TypeError, ValueError):
            return default

    def setting_int(key, default):
        try:
            return int(float(user_settings.get(key, default)))
        except (TypeError, ValueError):
            return default

    subtitle_font_color_value = user_settings.get("subtitle_font_color") or "white"
    if subtitle_font_color_value not in ["black", "white", "green", "red"]:
        subtitle_font_color_value = "white"

    VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
    SFX_EXTENSIONS = (".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg")

    def list_local_videos():
        choices = []
        for root, _, files in os.walk(LOCAL_VIDEO_DIR):
            for name in files:
                if not name.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, LOCAL_VIDEO_DIR).replace(os.sep, "/")
                choices.append(rel_path)
        return sorted(choices)

    def sound_effect_scan_roots():
        search_roots = [(LOCAL_SFX_DIR, "")]
        if os.path.isdir(LEGACY_MUSIC_DIR):
            search_roots.append((LEGACY_MUSIC_DIR, "music/"))
        return search_roots

    def sound_effect_folder_hint():
        roots = [root for root, _ in sound_effect_scan_roots()]
        return "\n".join(roots)

    def list_local_sfx():
        choices = []
        search_roots = sound_effect_scan_roots()
        for search_root, prefix in search_roots:
            for root, _, files in os.walk(search_root):
                for name in files:
                    if not name.lower().endswith(SFX_EXTENSIONS):
                        continue
                    path = os.path.join(root, name)
                    rel_path = os.path.relpath(path, search_root).replace(os.sep, "/")
                    choices.append(prefix + rel_path)
        return sorted(choices)

    def resolve_local_video(local_video):
        if not local_video:
            return None
        candidate = os.path.abspath(os.path.join(LOCAL_VIDEO_DIR, local_video))
        local_root = os.path.abspath(LOCAL_VIDEO_DIR)
        if not candidate.startswith(local_root + os.sep) and candidate != local_root:
            raise ValueError("Invalid local video path.")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Local video not found: {local_video}")
        return candidate
    
    server_name='127.0.0.1'
    if args.listen:
        server_name = '0.0.0.0'
        
    def save_text_to_file(content, extension, output_dir=None):
        if not content:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.{extension}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
        else:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def audio_recog(audio_input, sd_switch, hotwords, output_dir):
        return audio_clipper.recog(audio_input, sd_switch, None, hotwords, output_dir=output_dir)

    def video_recog(video_input, sd_switch, hotwords, output_dir):
        return audio_clipper.video_recog(video_input, sd_switch, hotwords, output_dir=output_dir)

    def video_clip(dest_text, video_spk_input, start_ost, end_ost, state, output_dir, sound_effect_rules=None):
        return audio_clipper.video_clip(
            dest_text, start_ost, end_ost, state, dest_spk=video_spk_input, output_dir=output_dir,
            sound_effect_rules=sound_effect_rules, sound_effect_dir=LOCAL_SFX_DIR
            )

    def refresh_local_videos():
        return gr.update(choices=list_local_videos())

    def refresh_local_sfx(sound_effect_rules):
        choices = list_local_sfx()
        selected = choices[0] if choices else None
        return gr.update(choices=choices, value=selected), load_sound_effect_terms(selected, sound_effect_rules)

    def _parse_sound_effect_bindings(sound_effect_rules):
        bindings = {}
        for line in str(sound_effect_rules or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2 or not parts[0]:
                continue
            try:
                volume = float(parts[2]) if len(parts) >= 3 and parts[2] else DEFAULT_SOUND_EFFECT_VOLUME
            except ValueError:
                volume = DEFAULT_SOUND_EFFECT_VOLUME
            try:
                cooldown = float(parts[3]) if len(parts) >= 4 and parts[3] else DEFAULT_SOUND_EFFECT_COOLDOWN
            except ValueError:
                cooldown = DEFAULT_SOUND_EFFECT_COOLDOWN
            bindings[parts[0]] = {
                "terms": parts[1],
                "volume": max(0.0, min(volume, 2.0)),
                "cooldown": max(0.0, cooldown),
            }
        return bindings

    def _format_sound_effect_bindings(bindings):
        lines = []
        for sound_file in sorted(bindings):
            binding = bindings[sound_file]
            terms = str(binding.get("terms") or "").strip()
            if not terms:
                continue
            volume = binding.get("volume", DEFAULT_SOUND_EFFECT_VOLUME)
            cooldown = binding.get("cooldown", DEFAULT_SOUND_EFFECT_COOLDOWN)
            lines.append(f"{sound_file} | {terms} | {volume:g} | {cooldown:g}")
        return "\n".join(lines)

    def _sync_sound_effect_binding(sound_effect_rules, selected_sfx, selected_terms):
        bindings = _parse_sound_effect_bindings(sound_effect_rules)
        selected_sfx = (selected_sfx or "").strip()
        selected_terms = (selected_terms or "").strip()
        if selected_sfx:
            if selected_terms:
                current = bindings.get(selected_sfx) or {}
                bindings[selected_sfx] = {
                    "terms": selected_terms,
                    "volume": current.get("volume", DEFAULT_SOUND_EFFECT_VOLUME),
                    "cooldown": current.get("cooldown", DEFAULT_SOUND_EFFECT_COOLDOWN),
                }
            elif selected_sfx in bindings:
                del bindings[selected_sfx]
        return _format_sound_effect_bindings(bindings)

    def load_sound_effect_terms(selected_sfx, sound_effect_rules):
        bindings = _parse_sound_effect_bindings(sound_effect_rules)
        binding = bindings.get((selected_sfx or "").strip()) or {}
        return binding.get("terms", "")

    def update_sound_effect_terms(selected_sfx, selected_terms, sound_effect_rules):
        return _sync_sound_effect_binding(sound_effect_rules, selected_sfx, selected_terms)

    def _srt_time_to_millis(time_text):
        time_text = (time_text or "").strip().replace(",", ".")
        parts = time_text.split(":")
        if len(parts) != 3:
            raise ValueError(f"Unsupported SRT timestamp: {time_text}")
        hours = int(parts[0])
        minutes = int(parts[1])
        second_parts = parts[2].split(".")
        seconds = int(second_parts[0])
        millis = int((second_parts[1] if len(second_parts) > 1 else "0").ljust(3, "0")[:3])
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis

    def _filter_srt_by_ranges(srt_text, timestamp_ranges):
        if not timestamp_ranges:
            return ""
        range_pairs = [(int(start), int(end)) for start, end in timestamp_ranges if end > start]
        if not range_pairs:
            return ""

        def overlaps(start_ms, end_ms):
            return any(start_ms < range_end and end_ms > range_start for range_start, range_end in range_pairs)

        lines = str(srt_text or "").splitlines()
        selected_blocks = []
        index = 0
        time_text = r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
        arrow_time_line = re.compile(rf"({time_text})\s*-->\s*({time_text})")
        bracket_time_line = re.compile(
            rf"\[\s*({time_text})\s*(?:-->|-|~|至|到)\s*({time_text})\s*\]"
        )

        while index < len(lines):
            block = []
            while index < len(lines) and lines[index].strip():
                block.append(lines[index])
                index += 1
            index += 1
            if not block:
                continue

            match = None
            for line in block:
                match = arrow_time_line.search(line) or bracket_time_line.search(line)
                if match:
                    break
            if not match:
                continue

            start_ms = _srt_time_to_millis(match.group(1))
            end_ms = _srt_time_to_millis(match.group(2))
            if overlaps(start_ms, end_ms):
                selected_blocks.append("\n".join(block))

        if selected_blocks:
            return "\n\n".join(selected_blocks)

        # Some ASR/LLM outputs are line based, for example:
        # 1. [00:06:04,460-00:06:15,370] subtitle text
        selected_line_blocks = []
        line_index = 0
        while line_index < len(lines):
            line = lines[line_index]
            match = arrow_time_line.search(line) or bracket_time_line.search(line)
            if not match:
                line_index += 1
                continue
            start_ms = _srt_time_to_millis(match.group(1))
            end_ms = _srt_time_to_millis(match.group(2))
            if overlaps(start_ms, end_ms):
                block = [line]
                next_index = line_index + 1
                while next_index < len(lines):
                    next_line = lines[next_index]
                    if not next_line.strip():
                        break
                    if arrow_time_line.search(next_line) or bracket_time_line.search(next_line):
                        break
                    block.append(next_line)
                    next_index += 1
                selected_line_blocks.append("\n".join(block))
            line_index += 1
        return "\n\n".join(selected_line_blocks)

    def _safe_video_filename_from_url(video_url):
        parsed = urlparse(video_url)
        filename = os.path.basename(unquote(parsed.path)).strip()
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        if not filename:
            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        name, ext = os.path.splitext(filename)
        if ext.lower() not in VIDEO_EXTENSIONS:
            filename = f"{name or 'video'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        return filename

    def download_video_from_url(video_url):
        video_url = (video_url or "").strip()
        parsed = urlparse(video_url)
        if parsed.scheme not in ("http", "https"):
            return "Only http/https video URLs are supported.", gr.update(choices=list_local_videos())

        filename = _safe_video_filename_from_url(video_url)
        output_path = os.path.join(LOCAL_VIDEO_DIR, filename)
        if os.path.exists(output_path):
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            output_path = os.path.join(LOCAL_VIDEO_DIR, filename)
        part_path = output_path + ".part"

        try:
            with requests.get(video_url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with open(part_path, "wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            file_obj.write(chunk)
            os.replace(part_path, output_path)
        except Exception as exc:
            if os.path.exists(part_path):
                os.remove(part_path)
            return f"Download failed: {exc}", gr.update(choices=list_local_videos())

        choices = list_local_videos()
        return f"Downloaded to local_videos/{filename}", gr.update(choices=choices, value=filename)

    def save_llm_settings(
            prompt_system, prompt_user, model, apikey, subtitle_correction_prompt, highlight_prompt,
            highlight_count, font_size, font_color, subtitle_x, subtitle_y, highlight_color,
            sound_effect_rules, selected_sfx, selected_sfx_terms):
        settings = load_user_settings()
        sound_effect_rules = _sync_sound_effect_binding(sound_effect_rules, selected_sfx, selected_sfx_terms)
        try:
            saved_highlight_count = max(1, int(float(highlight…7334 tokens truncated…xtbox(label="Download Status", interactive=False)
                with gr.Row():
                    video_input = gr.Video(label="视频输入 | Video Input", height=640, elem_classes=["video-preserve"])
                    audio_input = gr.Audio(label="音频输入 | Audio Input")
                with gr.Column():
                    gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%A4%9A%E8%AF%BB%E4%B9%A6%EF%BC%9F%E8%BF%99%E6%98%AF%E6%88%91%E5%90%AC%E8%BF%87%E6%9C%80%E5%A5%BD%E7%9A%84%E7%AD%94%E6%A1%88-%E7%89%87%E6%AE%B5.mp4', 
                                 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/2022%E4%BA%91%E6%A0%96%E5%A4%A7%E4%BC%9A_%E7%89%87%E6%AE%B52.mp4', 
                                 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%BD%BF%E7%94%A8chatgpt_%E7%89%87%E6%AE%B5.mp4'],
                                [video_input],
                                label='示例视频 | Demo Video')
                    gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E8%AE%BF%E8%B0%88.mp4'],
                                [video_input],
                                label='多说话人示例视频 | Multi-speaker Demo Video')
                    gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E9%B2%81%E8%82%83%E9%87%87%E8%AE%BF%E7%89%87%E6%AE%B51.wav'],
                                [audio_input],
                                label="示例音频 | Demo Audio")
                    with gr.Column():
                        # with gr.Row():
                            # video_sd_switch = gr.Radio(["No", "Yes"], label="👥区分说话人 Get Speakers", value='No')
                        hotwords_input = gr.Textbox(label="🚒 热词 | Hotwords(可以为空，多个热词使用空格分隔，仅支持中文热词)")
                        output_dir = gr.Textbox(label="📁 文件输出路径 | File Output Dir (可以为空，Linux, mac系统可以稳定使用)", value=DEFAULT_OUTPUT_DIR)
                        with gr.Row():
                            recog_button = gr.Button("👂 识别 | ASR", variant="primary")
                            recog_button2 = gr.Button("👂👫 识别+区分说话人 | ASR+SD")
                            query_asr_button = gr.Button("Query ASR Task")
                        asr_job_id = gr.Textbox(label="ASR Job ID", interactive=True)
                        asr_task_status = gr.Textbox(label="后台识别任务状态 | ASR Background Task Status", interactive=False)
                video_text_output = gr.Textbox(label="✏️ 识别结果 | Recognition Result")
                video_srt_output = gr.Textbox(label="📖 SRT字幕内容 | RST Subtitles")
                with gr.Row():
                    video_text_file = gr.File(label="⬇️ 下载识别结果 | Download Recognition Result", interactive=False)
                    video_srt_file = gr.File(label="⬇️ 下载SRT字幕 | Download SRT Subtitles", interactive=False)
                with gr.Accordion("DeepSeek 字幕修正 | Subtitle Correction", open=True):
                    subtitle_correction_prompt = gr.Textbox(
                        label="字幕修正提示词 | Correction Prompt",
                        value=(
                            user_settings.get("subtitle_correction_prompt")
                            or DEFAULT_SUBTITLE_CORRECTION_PROMPT
                        ),
                        lines=5,
                    )
                    subtitle_correction_button = gr.Button(
                        "使用 DeepSeek 修正字幕 | Correct Subtitles",
                        variant="primary",
                    )
                    subtitle_correction_status = gr.Textbox(
                        label="字幕修正状态 | Correction Status",
                        interactive=False,
                    )
                    corrected_srt_file = gr.File(
                        label="下载修正后的 SRT | Download Corrected SRT",
                        interactive=False,
                    )
            with gr.Column():
                with gr.Tab("🧠 LLM智能裁剪 | LLM Clipping"):
                    with gr.Column():
                        prompt_head = gr.Textbox(
                            label="Prompt System (按需更改，最好不要变动主体和要求)",
                            value=user_settings.get("prompt_system") or DEFAULT_PROMPT_SYSTEM,
                        )
                        prompt_head2 = gr.Textbox(
                            label="Prompt User（不需要修改，会自动拼接左下角的srt字幕）",
                            value=user_settings.get("prompt_user") or DEFAULT_PROMPT_USER,
                        )
                        with gr.Column():
                            with gr.Row():
                                llm_model = gr.Dropdown(
                                    choices=[
                                        "deepseek-v4-flash",
                                        "deepseek-v4-pro",
                                        "deepseek-chat",
                                        "deepseek-reasoner",
                                        "qwen-plus",
                                             "gpt-3.5-turbo", 
                                             "gpt-3.5-turbo-0125", 
                                             "gpt-4-turbo",
                                             "g4f-gpt-3.5-turbo",
                                             "pegasus1.5"],
                                    value=user_settings.get("llm_model") or DEFAULT_LLM_MODEL,
                                    label="LLM Model Name",
                                    allow_custom_value=True)
                                apikey_input = gr.Textbox(
                                    label="DeepSeek API Key / APIKEY",
                                    type="password",
                                    value=user_settings.get("apikey") or "",
                                )
                            with gr.Row():
                                save_settings_button = gr.Button("保存提示词/API | Save Settings")
                                save_settings_status = gr.Textbox(label="Settings Status", interactive=False)
                            llm_button =  gr.Button("LLM推理 | LLM Inference（首先进行识别，非g4f需配置对应apikey）", variant="primary")
                        llm_result = gr.Textbox(label="LLM Clipper Result")
                        with gr.Row():
                            llm_clip_button = gr.Button("🧠 LLM智能裁剪 | AI Clip", variant="primary")
                            llm_clip_subti_button = gr.Button("🧠 LLM智能裁剪+字幕 | AI Clip+Subtitles")
                with gr.Tab("✂️ 根据文本/说话人裁剪 | Text/Speaker Clipping"):
                    video_text_input = gr.Textbox(label="✏️ 待裁剪文本 | Text to Clip (多段文本使用'#'连接)")
                    video_spk_input = gr.Textbox(label="✏️ 待裁剪说话人 | Speaker to Clip (多个说话人使用'#'连接)")
                    with gr.Row():
                        clip_button = gr.Button("✂️ 裁剪 | Clip", variant="primary")
                        clip_subti_button = gr.Button("✂️ 裁剪+字幕 | Clip+Subtitles")
                    with gr.Row():
                        video_start_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50, label="⏪ 开始位置偏移 | Start Offset (ms)")
                        video_end_ost = gr.Slider(minimum=-500, maximum=1000, value=100, step=50, label="⏩ 结束位置偏移 | End Offset (ms)")
                with gr.Row():
                    font_size = gr.Slider(minimum=10, maximum=100, value=setting_float("subtitle_font_size", 32), step=2, label="🔠 字幕字体大小 | Subtitle Font Size")
                    font_color = gr.Radio(["black", "white", "green", "red"], label="🌈 字幕颜色 | Subtitle Color", value=subtitle_font_color_value)
                    # font = gr.Radio(["黑体", "Alibaba Sans"], label="字体 Font")
                with gr.Row():
                    subtitle_x = gr.Slider(minimum=0, maximum=100, value=setting_float("subtitle_x", 50), step=1, label="Subtitle X")
                    subtitle_y = gr.Slider(minimum=0, maximum=100, value=setting_float("subtitle_y", 88), step=1, label="Subtitle Y")
                subtitle_sample_text = gr.Textbox(
                    label="Subtitle Preview Text",
                    value="这里是样例字幕，可调整大小和位置",
                )
                highlight_prompt = gr.Textbox(
                    label="Subtitle Highlight Prompt",
                    value=user_settings.get("highlight_prompt") or DEFAULT_HIGHLIGHT_PROMPT,
                    lines=4,
                )
                with gr.Row():
                    highlight_count = gr.Number(
                        label="预期高亮字幕数量 | Expected Highlight Count",
                        value=setting_int("highlight_count", DEFAULT_HIGHLIGHT_COUNT),
                        precision=0,
                    )
                    highlight_color = gr.Textbox(label="Subtitle Highlight Color", value=user_settings.get("highlight_color") or "yellow")
                    llm_highlight_button = gr.Button("LLM Pick Subtitle Highlights")
                highlight_terms = gr.Textbox(
                    label="Subtitle Highlight Terms",
                    placeholder="One term per line. You can also separate terms with commas.",
                    lines=4,
                )
                sound_effect_choices = list_local_sfx()
                selected_sound_effect = sound_effect_choices[0] if sound_effect_choices else None
                sound_effect_dirs = gr.Textbox(
                    label="Sound Effect Folders",
                    value=sound_effect_folder_hint(),
                    interactive=False,
                    lines=2,
                )
                with gr.Row():
                    local_sfx_list = gr.Dropdown(
                        choices=sound_effect_choices,
                        value=selected_sound_effect,
                        label="Server Sound Effects | local_sfx/ and music/",
                        interactive=True,
                    )
                    refresh_local_sfx_button = gr.Button("Refresh Sound Effects")
                selected_sfx_terms = gr.Textbox(
                    label="Selected Sound Effect Trigger Words",
                    value=load_sound_effect_terms(selected_sound_effect, user_settings.get("sound_effect_rules") or ""),
                    placeholder="糖尿病,戒烟\n心梗脑梗",
                    lines=4,
                )
                sound_effect_rules = gr.Textbox(
                    label="Sound Effect Word Bindings",
                    value=user_settings.get("sound_effect_rules") or DEFAULT_SOUND_EFFECT_RULES,
                    visible=False,
                )
                subtitle_preview_button = gr.Button("Preview Subtitle")
                subtitle_preview_image = gr.Image(label="Subtitle Position Preview", interactive=False)
                video_output = gr.Video(label="裁剪结果 | Video Clipped", height=640, elem_classes=["video-preserve"])
                audio_output = gr.Audio(label="裁剪结果 | Audio Clipped")
                clip_message = gr.Textbox(label="⚠️ 裁剪信息 | Clipping Log")
                srt_clipped = gr.Textbox(label="📖 裁剪部分SRT字幕内容 | Clipped RST Subtitles")            
                
        refresh_local_video_button.click(
                            refresh_local_videos,
                            inputs=[],
                            outputs=[local_video_input])
        refresh_local_sfx_button.click(
                            refresh_local_sfx,
                            inputs=[sound_effect_rules],
                            outputs=[local_sfx_list, selected_sfx_terms])
        local_sfx_list.change(
                            load_sound_effect_terms,
                            inputs=[local_sfx_list, sound_effect_rules],
                            outputs=[selected_sfx_terms])
        selected_sfx_terms.change(
                            update_sound_effect_terms,
                            inputs=[local_sfx_list, selected_sfx_terms, sound_effect_rules],
                            outputs=[sound_effect_rules])
        download_video_button.click(
                            download_video_from_url,
                            inputs=[video_url_input],
                            outputs=[download_video_status, local_video_input])
        save_settings_button.click(
                            save_llm_settings,
                            inputs=[prompt_head, prompt_head2, llm_model, apikey_input,
                                    subtitle_correction_prompt, highlight_prompt,
                                    highlight_count, font_size, font_color, subtitle_x, subtitle_y, highlight_color,
                                    sound_effect_rules, local_sfx_list, selected_sfx_terms],
                            outputs=[save_settings_status])
        recog_button.click(start_asr_task,
                            inputs=[local_video_input,
                                    video_input, 
                                    audio_input, 
                                    hotwords_input, 
                                    output_dir,
                                    ], 
                            outputs=[asr_task_status, video_text_output, video_srt_output, video_state, audio_state, video_text_file, video_srt_file, asr_job_id])
        recog_button2.click(start_asr_speaker_task,
                            inputs=[local_video_input,
                                    video_input, 
                                    audio_input, 
                                    hotwords_input, 
                                    output_dir,
                                    ], 
                            outputs=[asr_task_status, video_text_output, video_srt_output, video_state, audio_state, video_text_file, video_srt_file, asr_job_id])
        query_asr_button.click(
                            query_asr_task,
                            inputs=[asr_job_id],
                            outputs=[asr_task_status, video_text_output, video_srt_output, video_state, audio_state, video_text_file, video_srt_file, asr_job_id])
        asr_task_timer.tick(
                            query_asr_task,
                            inputs=[asr_job_id],
                            outputs=[asr_task_status, video_text_output, video_srt_output, video_state, audio_state, video_text_file, video_srt_file, asr_job_id])
        subtitle_correction_button.click(
                            correct_subtitles_with_deepseek,
                            inputs=[video_srt_output, subtitle_correction_prompt, llm_model,
                                    apikey_input, video_state, audio_state, output_dir],
                            outputs=[video_srt_output, video_state, audio_state,
                                     corrected_srt_file, subtitle_correction_status])
        subtitle_preview_button.click(
                            preview_subtitle,
                            inputs=[local_video_input, video_input, subtitle_sample_text, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color],
                            outputs=[subtitle_preview_image])
        clip_button.click(mix_clip, 
                           inputs=[video_text_input, 
                                   video_spk_input, 
                                   video_start_ost, 
                                   video_end_ost, 
                                   video_state, 
                                   audio_state, 
                                   output_dir,
                                   sound_effect_rules,
                                   local_sfx_list,
                                   selected_sfx_terms
                                   ],
                           outputs=[video_output, audio_output, clip_message, srt_clipped])
        clip_subti_button.click(video_clip_addsub, 
                           inputs=[video_text_input, 
                                   video_spk_input, 
                                   video_start_ost, 
                                   video_end_ost, 
                                   video_state, 
                                   output_dir, 
                                   font_size, 
                                   font_color,
                                   subtitle_x,
                                   subtitle_y,
                                   highlight_terms,
                                   highlight_color,
                                   sound_effect_rules,
                                   local_sfx_list,
                                   selected_sfx_terms,
                                   ], 
                           outputs=[video_output, clip_message, srt_clipped])
        llm_button.click(llm_inference,
                         inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input, video_input],
                         outputs=[llm_result])
        llm_highlight_button.click(llm_subtitle_highlights,
                         inputs=[llm_result, video_srt_output, llm_model, apikey_input, highlight_prompt, highlight_count],
                         outputs=[highlight_terms])
        llm_clip_button.click(AI_clip, 
                           inputs=[llm_result,
                                   video_text_input, 
                                   video_spk_input, 
                                   video_start_ost, 
                                   video_end_ost, 
                                   video_state, 
                                   audio_state, 
                                   output_dir,
                                   sound_effect_rules,
                                   local_sfx_list,
                                   selected_sfx_terms,
                                   ],
                           outputs=[video_output, audio_output, clip_message, srt_clipped])
        llm_clip_subti_button.click(AI_clip_subti, 
                           inputs=[llm_result,
                                   video_text_input, 
                                   video_spk_input, 
                                   video_start_ost, 
                                   video_end_ost, 
                                   video_state, 
                                   audio_state, 
                                   output_dir,
                                   font_size,
                                   font_color,
                                   subtitle_x,
                                   subtitle_y,
                                   highlight_terms,
                                   highlight_color,
                                   sound_effect_rules,
                                   local_sfx_list,
                                   selected_sfx_terms,
                                   ],
                           outputs=[video_output, audio_output, clip_message, srt_clipped])
    
    # start gradio service in local or share
    if args.listen:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name, inbrowser=False)
    else:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name)

