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
from videoclipper import VideoClipper, _subtitle_image_clip, _subtitle_position
from llm.openai_api import openai_call
from llm.qwen_api import call_qwen_model
from llm.g4f_openai_api import g4f_openai_call
from llm.twelvelabs_api import call_twelvelabs_pegasus
from utils.trans_utils import extract_timestamps
from introduction import top_md_1, top_md_3, top_md_4


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOCAL_TMP_DIR = os.path.join(PROJECT_ROOT, "tmp")
LOCAL_GRADIO_TMP_DIR = os.path.join(PROJECT_ROOT, "gradio_tmp")
LOCAL_VIDEO_DIR = os.path.join(PROJECT_ROOT, "local_videos")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
USER_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "user_settings.json")
ASR_TASKS = {}
ASR_TASK_LOCK = threading.Lock()
ASR_RUN_LOCK = threading.Lock()
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_GRADIO_TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
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
    "Return only the exact words or short phrases that appear in the SRT, one item per line, up to 20 items. "
    "Do not explain."
)


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

    subtitle_font_color_value = user_settings.get("subtitle_font_color") or "white"
    if subtitle_font_color_value not in ["black", "white", "green", "red"]:
        subtitle_font_color_value = "white"

    VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")

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

    def video_clip(dest_text, video_spk_input, start_ost, end_ost, state, output_dir):
        return audio_clipper.video_clip(
            dest_text, start_ost, end_ost, state, dest_spk=video_spk_input, output_dir=output_dir
            )

    def refresh_local_videos():
        return gr.update(choices=list_local_videos())

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
            prompt_system, prompt_user, model, apikey, highlight_prompt,
            font_size, font_color, subtitle_x, subtitle_y, highlight_color):
        settings = load_user_settings()
        settings.update({
            "prompt_system": prompt_system or "",
            "prompt_user": prompt_user or "",
            "llm_model": model or DEFAULT_LLM_MODEL,
            "apikey": apikey or "",
            "highlight_prompt": highlight_prompt or DEFAULT_HIGHLIGHT_PROMPT,
            "subtitle_font_size": font_size,
            "subtitle_font_color": font_color or "white",
            "subtitle_x": subtitle_x,
            "subtitle_y": subtitle_y,
            "highlight_color": highlight_color or "yellow",
        })
        save_user_settings(settings)
        return "Saved. These settings will be loaded automatically next time."

    def mix_recog(local_video, video_input, audio_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        audio_state, video_state = None, None
        local_video_path = resolve_local_video(local_video)
        if local_video_path is not None:
            res_text, res_srt, video_state = video_recog(
                local_video_path, 'No', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, video_state, None, text_file, srt_file
        if video_input is not None:
            res_text, res_srt, video_state = video_recog(
                video_input, 'No', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, video_state, None, text_file, srt_file
        if audio_input is not None:
            res_text, res_srt, audio_state = audio_recog(
                audio_input, 'No', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, None, audio_state, text_file, srt_file
    
    def mix_recog_speaker(local_video, video_input, audio_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        audio_state, video_state = None, None
        local_video_path = resolve_local_video(local_video)
        if local_video_path is not None:
            res_text, res_srt, video_state = video_recog(
                local_video_path, 'Yes', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, video_state, None, text_file, srt_file
        if video_input is not None:
            res_text, res_srt, video_state = video_recog(
                video_input, 'Yes', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, video_state, None, text_file, srt_file
        if audio_input is not None:
            res_text, res_srt, audio_state = audio_recog(
                audio_input, 'Yes', hotwords, output_dir=output_dir)
            text_file = save_text_to_file(res_text, 'txt', output_dir)
            srt_file = save_text_to_file(res_srt, 'srt', output_dir)
            return res_text, res_srt, None, audio_state, text_file, srt_file

    def _copy_video_input_for_background(video_input, job_id):
        if not video_input or not isinstance(video_input, str) or not os.path.isfile(video_input):
            return video_input
        _, ext = os.path.splitext(video_input)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", job_id)
        copied_path = os.path.join(LOCAL_TMP_DIR, f"asr_{safe_name}{ext or '.mp4'}")
        shutil.copy2(video_input, copied_path)
        return copied_path

    def _set_asr_task(job_id, **updates):
        with ASR_TASK_LOCK:
            task = ASR_TASKS.setdefault(job_id, {})
            task.update(updates)

    def _get_asr_task(job_id):
        with ASR_TASK_LOCK:
            return dict(ASR_TASKS.get(job_id) or {})

    def _run_asr_task(job_id, speaker_mode, local_video, video_input, audio_input, hotwords, output_dir):
        _set_asr_task(job_id, status="queued", message="Queued. Waiting for the ASR worker.")
        try:
            with ASR_RUN_LOCK:
                _set_asr_task(job_id, status="running", message="Running ASR in the background. This may take several minutes.")
                result = (
                    mix_recog_speaker(local_video, video_input, audio_input, hotwords, output_dir)
                    if speaker_mode
                    else mix_recog(local_video, video_input, audio_input, hotwords, output_dir)
                )
            if result is None:
                raise ValueError("No local video, uploaded video, or audio input was provided.")
            _set_asr_task(
                job_id,
                status="done",
                message="ASR completed. Click Query ASR Task to load results if they are not shown yet.",
                result=result,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as exc:
            logging.exception("ASR background task failed: %s", job_id)
            _set_asr_task(
                job_id,
                status="failed",
                message=f"ASR failed: {exc}",
                traceback=traceback.format_exc(),
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _start_asr_task(local_video, video_input, audio_input, hotwords, output_dir, speaker_mode=False):
        if not local_video and video_input is None and audio_input is None:
            return "Please choose a server local video, uploaded video, or audio first.", "", "", None, None, None, None, ""

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        copied_video_input = _copy_video_input_for_background(video_input, job_id)
        _set_asr_task(
            job_id,
            status="starting",
            message="Starting ASR background task.",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        thread = threading.Thread(
            target=_run_asr_task,
            args=(job_id, speaker_mode, local_video, copied_video_input, audio_input, hotwords, output_dir),
            daemon=True,
        )
        thread.start()
        mode_name = "ASR+SD" if speaker_mode else "ASR"
        return (
            f"{mode_name} task started. Job ID: {job_id}\nThe backend will continue running. Use Query ASR Task to load the result.",
            "",
            "",
            None,
            None,
            None,
            None,
            job_id,
        )

    def start_asr_task(local_video, video_input, audio_input, hotwords, output_dir):
        return _start_asr_task(local_video, video_input, audio_input, hotwords, output_dir, speaker_mode=False)

    def start_asr_speaker_task(local_video, video_input, audio_input, hotwords, output_dir):
        return _start_asr_task(local_video, video_input, audio_input, hotwords, output_dir, speaker_mode=True)

    def query_asr_task(job_id):
        job_id = (job_id or "").strip()
        if not job_id:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), job_id

        task = _get_asr_task(job_id)
        if not task:
            return f"ASR task not found: {job_id}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), job_id

        status = task.get("status", "unknown")
        message = task.get("message", "")
        if status == "done":
            res_text, res_srt, video_state, audio_state, text_file, srt_file = task.get("result")
            return f"Done. Job ID: {job_id}\n{message}", res_text, res_srt, video_state, audio_state, text_file, srt_file, job_id
        if status == "failed":
            detail = task.get("traceback") or ""
            return f"Failed. Job ID: {job_id}\n{message}\n{detail[-2000:]}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), job_id
        return f"{status.title()}. Job ID: {job_id}\n{message}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), job_id
    
    def mix_clip(dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state, dest_spk=video_spk_input, output_dir=output_dir)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state, dest_spk=video_spk_input, output_dir=output_dir)
            return None, (sr, res_audio), message, clip_srt
    
    def _preview_background_frame(local_video, video_input):
        video_path = None
        try:
            video_path = resolve_local_video(local_video)
        except Exception:
            video_path = None
        if video_path is None and isinstance(video_input, str) and os.path.isfile(video_input):
            video_path = video_input

        if video_path:
            clip = None
            try:
                clip = VideoFileClip(video_path)
                frame = clip.get_frame(min(1.0, max(0.0, (clip.duration or 1.0) * 0.1)))
                image = Image.fromarray(frame).convert("RGBA")
            finally:
                if clip is not None:
                    clip.close()
        else:
            image = Image.new("RGBA", (720, 1280), (0, 0, 0, 255))

        max_height = 960
        if image.height > max_height:
            width = int(image.width * max_height / image.height)
            image = image.resize((width, max_height), Image.LANCZOS)
        return image

    def preview_subtitle(local_video, video_input, sample_text, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color):
        background = _preview_background_frame(local_video, video_input)
        sample_text = (sample_text or "").strip() or "这里是样例字幕，可调整大小和位置"
        subtitle_clip = _subtitle_image_clip(sample_text, int(font_size), font_color, background.size, highlight_terms, highlight_color)
        subtitle_frame = subtitle_clip.get_frame(0)
        subtitle_image = Image.fromarray(subtitle_frame).convert("RGBA")
        if subtitle_clip.mask is not None:
            alpha = subtitle_clip.mask.get_frame(0)
            alpha_image = Image.fromarray((alpha * 255).astype("uint8"), mode="L")
            subtitle_image.putalpha(alpha_image)
        x, y = _subtitle_position(background.size, subtitle_image.size, subtitle_x, subtitle_y)
        composed = background.copy()
        composed.alpha_composite(subtitle_image, (int(x), int(y)))
        return composed.convert("RGB")

    def video_clip_addsub(dest_text, video_spk_input, start_ost, end_ost, state, output_dir, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        return audio_clipper.video_clip(
            dest_text, start_ost, end_ost, state, 
            font_size=font_size, font_color=font_color,
            subtitle_x=subtitle_x, subtitle_y=subtitle_y,
            highlight_terms=highlight_terms, highlight_color=highlight_color,
            add_sub=True, dest_spk=video_spk_input, output_dir=output_dir
            )
        
    def llm_inference(system_content, user_content, srt_text, model, apikey, video_input=None):
        SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot', 'deepseek', 'pegasus']
        format_instruction = (
            "\n\nSelect subtitle-aligned highlight material by content quality, not by a fixed duration. "
            "The final clip may be shorter or longer than 60 seconds when the story needs it; do not pad or force an exact length. "
            "Prefer complete, coherent statements. Remove filler, repeated phrases, pauses, greetings, transitions, and weak sentences when possible. "
            "If a strong highlight contains one or two weak sentences in the middle, split it into multiple timestamp ranges so those weak sentences are omitted; "
            "the clipping tool will concatenate the selected ranges in order. "
            "Output 1 to 6 ranges total, ordered by time, each as: N. [HH:MM:SS-HH:MM:SS] short reason/title. "
            "Do not output clips without a timestamp range."
        )
        system_content = (system_content or "") + format_instruction
        if model.startswith('pegasus'):
            # TwelveLabs Pegasus reasons over the actual video (visuals + audio)
            # rather than the ASR transcript, so it needs the video source.
            if video_input is None:
                logging.error("Pegasus requires a video input; please upload a video first.")
                return "Please upload a video before running Pegasus inference."
            return call_twelvelabs_pegasus(apikey, video_input, model=model, prompt=system_content)
        if model.startswith('qwen'):
            return call_qwen_model(apikey, model, user_content+'\n'+srt_text, system_content)
        if model.startswith('gpt') or model.startswith('moonshot') or model.startswith('deepseek'):
            return openai_call(apikey, model, user_content+'\n'+srt_text, system_content)
        elif model.startswith('g4f'):
            model = "-".join(model.split('-')[1:])
            return g4f_openai_call(model, system_content, user_content+'\n'+srt_text)
        else:
            logging.error("LLM name error, only {} are supported as LLM name prefix."
                          .format(SUPPORT_LLM_PREFIX))

    def llm_subtitle_highlights(srt_text, model, apikey, highlight_prompt):
        srt_text = (srt_text or "").strip()
        if not srt_text:
            return "Please run ASR first so the SRT subtitles are available."
        system_content = (highlight_prompt or DEFAULT_HIGHLIGHT_PROMPT).strip()
        user_content = (
            "Use the instruction above to select subtitle highlight terms from this SRT. "
            "Return one exact term or short phrase per line only.\n\n"
            + srt_text
        )
        if model.startswith('qwen'):
            return call_qwen_model(apikey, model, user_content, system_content)
        if model.startswith('gpt') or model.startswith('moonshot') or model.startswith('deepseek'):
            return openai_call(apikey, model, user_content, system_content)
        if model.startswith('g4f'):
            return g4f_openai_call("-".join(model.split('-')[1:]), system_content, user_content)
        return "Please choose a deepseek, qwen, gpt, moonshot, or g4f model to generate subtitle highlights."

    def AI_clip(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir):
        timestamp_list = extract_timestamps(LLM_res)
        if not timestamp_list:
            message = "No timestamps found in LLM result. Please make sure the LLM output contains ranges like [00:01:20-00:02:20]."
            return None, None, message, ""
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state, 
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=False)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state, 
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list)
            return None, (sr, res_audio), message, clip_srt
    
    def AI_clip_subti(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir, font_size, font_color, subtitle_x, subtitle_y, highlight_terms, highlight_color):
        timestamp_list = extract_timestamps(LLM_res)
        if not timestamp_list:
            message = "No timestamps found in LLM result. Please make sure the LLM output contains ranges like [00:01:20-00:02:20]."
            return None, None, message, ""
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state, 
                font_size=font_size, font_color=font_color,
                subtitle_x=subtitle_x, subtitle_y=subtitle_y,
                highlight_terms=highlight_terms, highlight_color=highlight_color,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=True)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state, 
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list)
            return None, (sr, res_audio), message, clip_srt
    
    # gradio interface
    app_css = """
    .video-preserve video {
        object-fit: contain !important;
        width: 100% !important;
        height: auto !important;
        max-height: 78vh !important;
        background: #000 !important;
    }
    .video-preserve [data-testid="video"] {
        background: #000 !important;
    }
    """
    theme = gr.Theme.load("funclip/utils/theme.json")
    with gr.Blocks(theme=theme, css=app_css) as funclip_service:
        gr.Markdown(top_md_1)
        # gr.Markdown(top_md_2)
        gr.Markdown(top_md_3)
        gr.Markdown(top_md_4)
        video_state, audio_state = gr.State(), gr.State()
        asr_task_timer = gr.Timer(value=10)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    local_video_input = gr.Dropdown(
                        choices=list_local_videos(),
                        label="服务器本地视频 | Server Local Video",
                        allow_custom_value=False,
                        interactive=True,
                    )
                    refresh_local_video_button = gr.Button("Refresh Local Videos")
                with gr.Row():
                    video_url_input = gr.Textbox(label="视频 URL 下载到服务器 | Download URL to Server")
                    download_video_button = gr.Button("Download URL")
                download_video_status = gr.Textbox(label="Download Status", interactive=False)
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
                    highlight_color = gr.Textbox(label="Subtitle Highlight Color", value=user_settings.get("highlight_color") or "yellow")
                    llm_highlight_button = gr.Button("LLM Pick Subtitle Highlights")
                highlight_terms = gr.Textbox(
                    label="Subtitle Highlight Terms",
                    placeholder="One term per line. You can also separate terms with commas.",
                    lines=4,
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
        download_video_button.click(
                            download_video_from_url,
                            inputs=[video_url_input],
                            outputs=[download_video_status, local_video_input])
        save_settings_button.click(
                            save_llm_settings,
                            inputs=[prompt_head, prompt_head2, llm_model, apikey_input, highlight_prompt,
                                    font_size, font_color, subtitle_x, subtitle_y, highlight_color],
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
                                   output_dir
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
                                   ], 
                           outputs=[video_output, clip_message, srt_clipped])
        llm_button.click(llm_inference,
                         inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input, video_input],
                         outputs=[llm_result])
        llm_highlight_button.click(llm_subtitle_highlights,
                         inputs=[video_srt_output, llm_model, apikey_input, highlight_prompt],
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
                                   ],
                           outputs=[video_output, audio_output, clip_message, srt_clipped])
    
    # start gradio service in local or share
    if args.listen:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name, inbrowser=False)
    else:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name)
