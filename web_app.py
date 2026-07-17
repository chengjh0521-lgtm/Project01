"""Minimal three-step web UI for the external FunClip modules."""

from __future__ import annotations

import json
import os
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr


OUTPUT_VIDEO_CSS = """
#generated-video { max-width: 640px; margin-left: auto; margin-right: auto; }
#generated-video video { max-height: 360px; object-fit: contain; }
.task-progress { min-width: 0; }
#clip-count-toggle {
  min-width: 24px !important;
  width: 24px !important;
  height: 24px !important;
  min-height: 24px !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  color: #777 !important;
  font-size: 13px !important;
  line-height: 1 !important;
  opacity: 0.22;
}
#clip-count-toggle:hover, #clip-count-toggle:focus-visible {
  background: #eceff1 !important;
  opacity: 0.9;
}
@media (max-width: 640px) {
  #generated-video video { max-height: 280px; }
}
"""


def patch_gradio_boolean_schema() -> None:
    """Work around Gradio 4.x API-info generation for boolean JSON schemas."""
    try:
        from gradio_client import utils as client_utils
    except ImportError:
        return

    original = getattr(client_utils, "_json_schema_to_python_type", None)
    if original is None or getattr(original, "_funclip_safe_boolean_schema", False):
        return

    def safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            schema = {}
        return original(schema, defs)

    safe_json_schema_to_python_type._funclip_safe_boolean_schema = True
    client_utils._json_schema_to_python_type = safe_json_schema_to_python_type


patch_gradio_boolean_schema()

from subtitle_generation import generate_subtitles
from subtitle_processing import process_from_corrected_subtitles, process_multiple_subtitles
from subtitle_processing.pipeline import build_corrected_video_state
from subtitle_processing.sound_effect_binding import (
    get_effect_details,
    list_sound_effects,
    save_effect_details,
)
from video_generation import (
    describe_sound_effect_events,
    describe_visual_asset_events,
    render_highlight_video,
    write_generation_report,
)
from background_jobs import JOBS


VIDEO_LIBRARY_DIR = Path(__file__).resolve().parent / "subtitle_generation" / "pending_videos"
SOUND_LOGIC_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "sound_effect_logic"
RENDER_REPORT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "generation_reports"
SAVED_CORRECTED_SUBTITLE_DIR = Path(__file__).resolve().parent / "subtitle_processing" / "saved_corrected_subtitles"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
VIDEO_SOURCE_CONTEXT_FILE = Path(__file__).resolve().parent / "subtitle_generation" / ".latest_video_source.json"
_VIDEO_STATE_LOCK = threading.RLock()
_LATEST_VIDEO_STATE = None
_LATEST_VIDEO_SOURCE = None


def _remember_video_source(video_path) -> str | None:
    """Persist the source path so rendering can rebuild FunClip state after a web refresh."""
    global _LATEST_VIDEO_SOURCE
    if not video_path:
        return None
    path = Path(str(video_path)).expanduser().resolve()
    if not path.is_file():
        logging.warning("Refusing to cache a missing source video: %s", path)
        return None
    with _VIDEO_STATE_LOCK:
        _LATEST_VIDEO_SOURCE = str(path)
    try:
        VIDEO_SOURCE_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        VIDEO_SOURCE_CONTEXT_FILE.write_text(
            json.dumps({"video_path": str(path)}, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logging.warning("Unable to persist the source video path: %s", exc)
    logging.warning("Source video cached: %s", path)
    return str(path)


def _resolve_video_source() -> str | None:
    with _VIDEO_STATE_LOCK:
        cached = _LATEST_VIDEO_SOURCE
    if cached and Path(cached).is_file():
        return cached
    try:
        saved = json.loads(VIDEO_SOURCE_CONTEXT_FILE.read_text(encoding="utf-8"))
        path = Path(str(saved.get("video_path", ""))).expanduser().resolve()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return _remember_video_source(path) if path.is_file() else None


def _remember_video_state(video_state, source: str):
    """Keep the upstream state server-side when Gradio State is lost by the browser."""
    global _LATEST_VIDEO_STATE
    if video_state is None:
        return None
    with _VIDEO_STATE_LOCK:
        _LATEST_VIDEO_STATE = video_state
    if isinstance(video_state, dict):
        _remember_video_source(video_state.get("video_filename"))
    logging.warning("Video state cached from %s.", source)
    return video_state


def _resolve_video_state(video_state, source: str):
    if video_state is not None:
        return _remember_video_state(video_state, source)
    with _VIDEO_STATE_LOCK:
        cached = _LATEST_VIDEO_STATE
    if cached is not None:
        logging.warning("Video state missing from browser; using server-side cache for %s.", source)
        return cached
    logging.warning("Video state is unavailable for %s.", source)
    return None


def _corrected_srt_from_plan(plan) -> str:
    if not isinstance(plan, dict):
        return ""
    corrected_srt = plan.get("corrected_srt")
    if isinstance(corrected_srt, str) and corrected_srt.strip():
        return corrected_srt
    clips = plan.get("clips")
    if isinstance(clips, list) and clips:
        # Compatibility for plans created before corrected_srt was stored at the root.
        return str(clips[0].get("highlight_srt", "")) if isinstance(clips[0], dict) else ""
    return ""


def _rebuild_video_state_for_render(plan):
    """Recreate the small upstream state contract from a persisted video path and SRT."""
    video_path = _resolve_video_source()
    corrected_srt = _corrected_srt_from_plan(plan)
    if not video_path or not corrected_srt.strip():
        logging.warning(
            "Cannot rebuild video state: source_video=%s, corrected_srt=%s.",
            bool(video_path), bool(corrected_srt.strip()),
        )
        return None
    try:
        from moviepy.editor import VideoFileClip

        source = Path(video_path)
        state = {
            "video_filename": str(source),
            "clip_video_file": str(source.with_suffix("")) + "_clip.mp4",
            "video": VideoFileClip(str(source)),
        }
        rebuilt = build_corrected_video_state(state, corrected_srt)
    except Exception as exc:
        logging.exception("Video-state rebuild failed.")
        logging.warning("Unable to rebuild video state: %s", exc)
        return None
    logging.warning("Video state rebuilt from persisted source video and corrected SRT.")
    return _remember_video_state(rebuilt, "render reconstruction")


def list_library_videos():
    """Return safe, readable paths relative to the server-side video library."""
    VIDEO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        str(path.relative_to(VIDEO_LIBRARY_DIR))
        for path in VIDEO_LIBRARY_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def refresh_library_videos():
    return gr.Dropdown(choices=list_library_videos(), value=None)


def refresh_sound_effects():
    choices = list_sound_effects()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def load_sound_effect(effect_name):
    if not effect_name:
        return ""
    return get_effect_details(effect_name)[0]


def save_sound_effect(effect_name, features):
    if not effect_name:
        raise gr.Error("Please select a sound-effect file first.")
    return save_effect_details(effect_name, features)[0]


def resolve_library_video(selected_video):
    if not selected_video:
        return None
    library_root = VIDEO_LIBRARY_DIR.resolve()
    candidate = (library_root / selected_video).resolve()
    if library_root not in candidate.parents or not candidate.is_file():
        raise ValueError("所选服务器视频不存在或不在待处理视频目录内。")
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("所选文件不是支持的视频格式。")
    return str(candidate)


def _remember_render_source(library_video, uploaded_video_path):
    """Use the currently selected original video when a render needs state reconstruction."""
    try:
        source = resolve_library_video(library_video) or uploaded_video_path
    except ValueError as exc:
        logging.warning("Selected render source is invalid: %s", exc)
        return None
    return _remember_video_source(source)


def list_saved_corrected_subtitles():
    SAVED_CORRECTED_SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted((path.name for path in SAVED_CORRECTED_SUBTITLE_DIR.glob("*.srt")), reverse=True)


def refresh_saved_corrected_subtitles():
    choices = list_saved_corrected_subtitles()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _saved_corrected_subtitle_path(file_name):
    if not file_name:
        raise ValueError("请先选择一份已保存的字幕2文件。")
    root = SAVED_CORRECTED_SUBTITLE_DIR.resolve()
    path = (root / str(file_name)).resolve()
    if root not in path.parents or not path.is_file() or path.suffix.lower() != ".srt":
        raise ValueError("所选字幕2文件不存在或不在保存目录内。")
    return path


def save_corrected_subtitle(srt_text):
    """Persist subtitle 2 once so later tests can skip correction entirely."""
    SAVED_CORRECTED_SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(srt_text).encode("utf-8")).hexdigest()[:10]
    filename = "subtitle2_{}_{}.srt".format(datetime.now().strftime("%Y%m%d_%H%M%S"), digest)
    path = SAVED_CORRECTED_SUBTITLE_DIR / filename
    path.write_text(str(srt_text).rstrip() + "\n", encoding="utf-8")
    return str(path)


def write_sound_effect_logic(video_path, clip_srt, sound_bindings, clip_id, ranges=None, visual_bindings=None):
    """Persist the exact sound and visual placements used by the renderer."""
    SOUND_LOGIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(video_path)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video_file": source.name,
        "clip_id": str(clip_id),
        "highlight_ranges": ranges or [],
        "sound_effect_events": describe_sound_effect_events(clip_srt, sound_bindings),
        "visual_asset_events": describe_visual_asset_events(clip_srt, visual_bindings),
    }
    filename = "{}_{}_sound_effect_logic.json".format(
        source.stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    path = SOUND_LOGIC_OUTPUT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _skipped_outputs(count: int):
    return tuple(gr.skip() for _ in range(count))

def _progress_updates(kind: str | None = None, value: int | None = None):
    value = 0 if value is None else max(0, min(100, int(value)))
    mapping = {
        "asr": (value, gr.skip(), gr.skip()),
        "process": (gr.skip(), value, gr.skip()),
        "render": (gr.skip(), gr.skip(), value),
    }
    return mapping.get(kind, (gr.skip(), gr.skip(), gr.skip()))


def toggle_clip_count_settings(is_visible):
    """Keep multi-video output as an advanced, opt-in setting."""
    visible = not bool(is_visible)
    return gr.update(visible=visible), visible


def submit_generate(uploaded_video_path, library_video, hotwords):
    try:
        video_path = resolve_library_video(library_video) or uploaded_video_path
    except ValueError as exc:
        return ("字幕生成任务未提交：{}".format(exc), *_skipped_outputs(11), "", None, *_progress_updates("asr", 0))
    if not video_path:
        return ("请上传视频，或从服务器待处理视频中选择一个文件。", *_skipped_outputs(11), "", None, *_progress_updates("asr", 0))
    _remember_video_source(video_path)

    def worker(report):
        report("阶段 1/1：正在进行 ASR 字幕识别。", 5)
        text, srt, video_state, _, _, _ = generate_subtitles(
            video_path, hotwords=hotwords or ""
        )
        _remember_video_state(video_state, "ASR")
        report("阶段 1/1：ASR 字幕识别完成。", 100)
        return {"subtitle": srt, "video_state": video_state}

    job_id = JOBS.submit("asr", worker)
    return ("字幕生成已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(11), job_id, {"id": job_id}, *_progress_updates("asr", 0))


def submit_process(srt_text, api_key, keyword_count, clip_count, video_state):
    if not srt_text:
        return ("请先生成字幕。", *_skipped_outputs(11), "", None, *_progress_updates("process", 0))
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    resolved_video_state = _resolve_video_state(video_state, "subtitle processing")

    def worker(report):
        corrected_file = None

        def save_after_correction(corrected_srt):
            nonlocal corrected_file
            try:
                corrected_file = save_corrected_subtitle(corrected_srt)
                report("阶段 1/5：字幕2已保存，可用于后续直接测试。")
            except OSError as exc:
                logging.warning("保存字幕2失败：%s", exc)

        result = process_multiple_subtitles(
            srt_text, key, keyword_count, clip_count, resolved_video_state,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash"),
            status_callback=report,
            on_corrected=save_after_correction,
        )
        _remember_video_state(result[-1], "subtitle processing")
        return {"pipeline": result, "corrected_file": corrected_file or save_corrected_subtitle(result[0])}

    job_id = JOBS.submit("process", worker)
    return ("字幕处理已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(11), job_id, {"id": job_id}, *_progress_updates("process", 0))


def submit_process_from_saved(saved_file, api_key, keyword_count, clip_count, video_state):
    try:
        path = _saved_corrected_subtitle_path(saved_file)
        corrected_srt = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return ("保存字幕任务未提交：{}".format(exc), *_skipped_outputs(11), "", None, *_progress_updates("process", 0))
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    resolved_video_state = _resolve_video_state(video_state, "saved subtitle processing")

    def worker(report):
        result = process_from_corrected_subtitles(
            corrected_srt, key, keyword_count, clip_count, resolved_video_state,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash"),
            status_callback=report,
        )
        _remember_video_state(result[-1], "saved subtitle processing")
        return {"pipeline": result, "corrected_file": str(path)}

    job_id = JOBS.submit("process", worker)
    return ("已保存字幕2处理任务已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(11), job_id, {"id": job_id}, *_progress_updates("process", 0))


def submit_render(llm_result, video_state, keywords, sound_bindings, library_video, uploaded_video_path):
    resolved_video_state = _resolve_video_state(video_state, "rendering")
    if resolved_video_state is None:
        _remember_render_source(library_video, uploaded_video_path)
        resolved_video_state = _rebuild_video_state_for_render(llm_result)
    logging.warning(
        "Render submission received: browser_video_state=%s, resolved_video_state=%s, plan=%s, keywords=%s, sound_bindings=%s.",
        video_state is not None,
        resolved_video_state is not None,
        bool(llm_result),
        bool(keywords),
        bool(sound_bindings),
    )
    if resolved_video_state is None:
        return (
            "无法获得原视频状态。请在第一步选择服务器视频或重新上传原视频后，再点击生成视频。",
            *_skipped_outputs(11), "", None, *_progress_updates("render", 0),
        )
    if not llm_result:
        return ("请先完成第二步高光提取。", *_skipped_outputs(11), "", None, *_progress_updates("render", 0))

    def worker(report):
        report("阶段 1/1：正在剪辑视频并烧录字幕。", 5)
        if isinstance(llm_result, dict) and llm_result.get("clips"):
            videos, logic_files, render_messages = [], [], []
            for index, clip in enumerate(llm_result["clips"], start=1):
                ranges = "\n".join("[{}-{}]".format(start, end) for start, end in clip["ranges"])
                video, _, render_message, clip_srt = render_highlight_video(
                    ranges,
                    resolved_video_state,
                    keywords=clip["keywords"],
                    sound_bindings=clip["sound_bindings"],
                    visual_bindings=clip.get("visual_bindings"),
                )
                if video:
                    videos.append(video)
                    render_messages.append("素材{}：{}；输出={}".format(index, render_message, video))
                    logic_files.append(write_sound_effect_logic(
                        video,
                        clip_srt,
                        clip["sound_bindings"],
                        clip["id"],
                        clip["ranges"],
                        clip.get("visual_bindings"),
                    ))
                report("阶段 1/1：已完成 {}/{} 条视频。".format(index, len(llm_result["clips"])), round(index * 100 / len(llm_result["clips"])))
            report_file = write_generation_report(
                RENDER_REPORT_OUTPUT_DIR,
                _corrected_srt_from_plan(llm_result),
                llm_result["clips"],
                videos,
            )
            return {
                "video": videos,
                "sound_logic": logic_files,
                "report": report_file,
                "render_message": "\n".join(render_messages),
            }
        video, _, render_message, clip_srt = render_highlight_video(
            llm_result, resolved_video_state, keywords=keywords, sound_bindings=sound_bindings
        )
        report("阶段 1/1：视频生成完成。", 100)
        logic_files = [write_sound_effect_logic(video, clip_srt, sound_bindings, "clip_01")] if video else []
        report_file = write_generation_report(
            RENDER_REPORT_OUTPUT_DIR,
            "",
            [{
                "id": "clip_01",
                "ranges": [],
                "highlight_srt": clip_srt,
                "keywords": keywords,
                "sound_bindings": sound_bindings,
                "visual_bindings": "{\"placements\": []}",
            }],
            [video] if video else [],
        )
        return {
            "video": video,
            "sound_logic": logic_files,
            "report": report_file,
            "render_message": "{}；输出={}".format(render_message, video),
        }

    job_id = JOBS.submit("render", worker)
    return ("视频生成已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(11), job_id, {"id": job_id}, *_progress_updates("render", 0))


def poll_job(job_ref):
    job_id = job_ref.get("id") if isinstance(job_ref, dict) else None
    job = JOBS.get(job_id)
    if job is None:
        return ("后台任务不存在，可能服务刚刚重启。", *_skipped_outputs(11), "", None, *_progress_updates())
    status = job["status"]
    prefix = "后台任务 {}：".format(job["kind"])
    if status in {"queued", "running"}:
        return (prefix + job["message"], *_skipped_outputs(11), job_id, job_ref, *_progress_updates(job["kind"], job["progress"]))
    if status == "failed":
        return (prefix + "失败：{}".format(job["error"] or job["message"]), *_skipped_outputs(11), job_id, None, *_progress_updates(job["kind"], job["progress"]))

    result = job["result"]
    if job["kind"] == "asr":
        return (
            prefix + "完成。",
            result["subtitle"],
            result["video_state"],
            "",
            None,
            "",
            "",
            None,
            None,
            None,
            None,
            None,
            job_id,
            None,
            *_progress_updates("asr", 100),
        )
    if job["kind"] == "process":
        corrected_srt, highlight_display, plan, updated_video_state = result["pipeline"]
        keywords = "\n\n".join("素材{}：\n{}".format(i, clip["keywords"]) for i, clip in enumerate(plan["clips"], 1))
        sound_bindings = "\n\n".join("素材{}：\n{}".format(i, clip["sound_bindings"]) for i, clip in enumerate(plan["clips"], 1))
        return (
            prefix + "完成。",
            gr.skip(),
            updated_video_state,
            corrected_srt,
            result["corrected_file"],
            highlight_display,
            keywords,
            sound_bindings,
            plan,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            job_id,
            None,
            *_progress_updates("process", 100),
        )
    if job["kind"] == "render":
        return (
            prefix + "完成。" + str(result.get("render_message") or ""),
            *_skipped_outputs(8),
            result["video"],
            result["sound_logic"],
            result.get("report"),
            job_id,
            None,
            *_progress_updates("render", 100),
        )
    return (prefix + "完成。", *_skipped_outputs(11), job_id, None, *_progress_updates())


def resume_job(job_id):
    """Resume polling after a browser refresh using the visible job ID."""
    logging.warning("Job query received: %s", str(job_id or "").strip()[:16])
    if not str(job_id or "").strip():
        return ("请输入需要恢复的后台任务 ID。", *_skipped_outputs(11), "", None, *_progress_updates())
    return poll_job({"id": str(job_id).strip()})


with gr.Blocks(title="FunClip 三模块", css=OUTPUT_VIDEO_CSS) as app:
    video_input = gr.File(label="本地上传视频（可选）", file_types=["video"], type="filepath")
    library_video_input = gr.Dropdown(
        label="服务器待处理视频", choices=list_library_videos(), value=None
    )
    hotwords_input = gr.Textbox(label="热词（可留空）", value="")
    api_key_input = gr.Textbox(label="DeepSeek API Key", type="password")
    job_status = gr.Textbox(label="后台任务状态", interactive=False)
    job_id_input = gr.Textbox(label="后台任务 ID（刷新后可粘贴恢复）")

    subtitle_output = gr.Textbox(label="字幕1：ASR 原始字幕", lines=12)
    corrected_output = gr.Textbox(label="字幕2：DeepSeek 洗稿字幕", lines=12)
    corrected_file_output = gr.File(label="字幕2：已保存 SRT（可下载）")
    highlight_output = gr.Textbox(label="字幕3：高光时间戳与对应字幕", lines=12)
    keyword_count_input = gr.Number(label="预期关键词数量", value=8, precision=0, minimum=1)
    clip_count_input = gr.Number(
        label="最大输出视频数量", value=1, precision=0, minimum=1, maximum=8, visible=False
    )
    keyword_output = gr.Textbox(label="高光关键词", lines=6)
    sound_bindings_output = gr.Textbox(label="音效选择与理由", lines=10, interactive=False)
    video_output = gr.File(label="输出视频（可多条下载）", file_count="multiple")
    sound_logic_output = gr.File(label="音效与 GIF/PNG 添加逻辑（JSON，可下载）", file_count="multiple")
    generation_report_output = gr.File(label="视频生成决策报告（Markdown，可下载）")
    video_state = gr.State()
    llm_result_state = gr.State()
    job_state = gr.State()
    clip_count_settings_visible = gr.State(False)

    with gr.Row():
        subtitle_button = gr.Button("1. 生成字幕", variant="primary")
        highlight_button = gr.Button("2. 洗稿、提取高光与关键词")
        video_button = gr.Button("3. 生成视频")
        clip_count_toggle_button = gr.Button("...", size="sm", elem_id="clip-count-toggle")
        resume_button = gr.Button("恢复/查询任务")
    with gr.Row():
        asr_progress = gr.Slider(label="字幕生成进度", minimum=0, maximum=100, value=0, step=1, interactive=False, elem_classes="task-progress")
        process_progress = gr.Slider(label="字幕处理进度", minimum=0, maximum=100, value=0, step=1, interactive=False, elem_classes="task-progress")
        render_progress = gr.Slider(label="视频生成进度", minimum=0, maximum=100, value=0, step=1, interactive=False, elem_classes="task-progress")

    task_outputs = [
        job_status,
        subtitle_output,
        video_state,
        corrected_output,
        corrected_file_output,
        highlight_output,
        keyword_output,
        sound_bindings_output,
        llm_result_state,
        video_output,
        sound_logic_output,
        generation_report_output,
        job_id_input,
        job_state,
        asr_progress,
        process_progress,
        render_progress,
    ]

    with gr.Accordion("音效绑定库", open=False):
        sound_effect_input = gr.Dropdown(label="音效文件", choices=list_sound_effects())
        sound_feature_input = gr.Textbox(
            label="该音效应绑定的关键词特征", lines=3,
            placeholder="例如：警告, 禁止, 风险；用逗号或换行分隔",
        )
        with gr.Row():
            sound_refresh_button = gr.Button("刷新音效列表")
            sound_save_button = gr.Button("保存该音效绑定")

    with gr.Accordion("已保存的字幕2", open=False):
        saved_corrected_input = gr.Dropdown(
            label="服务器保存的洗稿字幕", choices=list_saved_corrected_subtitles()
        )
        with gr.Row():
            saved_corrected_refresh_button = gr.Button("刷新保存字幕")
            saved_corrected_continue_button = gr.Button("使用已保存字幕继续处理")

    subtitle_button.click(
        submit_generate,
        inputs=[video_input, library_video_input, hotwords_input],
        outputs=task_outputs,
        show_progress="hidden",
        concurrency_limit=4,
    )
    clip_count_toggle_button.click(
        toggle_clip_count_settings,
        inputs=[clip_count_settings_visible],
        outputs=[clip_count_input, clip_count_settings_visible],
        show_progress="hidden",
    )
    highlight_button.click(
        submit_process,
        inputs=[subtitle_output, api_key_input, keyword_count_input, clip_count_input, video_state],
        outputs=task_outputs,
        show_progress="hidden",
        concurrency_limit=4,
    )
    saved_corrected_refresh_button.click(
        refresh_saved_corrected_subtitles,
        outputs=[saved_corrected_input],
        show_progress="hidden",
    )
    saved_corrected_continue_button.click(
        submit_process_from_saved,
        inputs=[saved_corrected_input, api_key_input, keyword_count_input, clip_count_input, video_state],
        outputs=task_outputs,
        show_progress="hidden",
        concurrency_limit=4,
    )
    video_button.click(
        submit_render,
        inputs=[
            llm_result_state,
            video_state,
            keyword_output,
            sound_bindings_output,
            library_video_input,
            video_input,
        ],
        outputs=task_outputs,
        show_progress="hidden",
        queue=False,
    )
    resume_button.click(
        resume_job,
        inputs=[job_id_input],
        outputs=task_outputs,
        show_progress="hidden",
        queue=False,
    )
    sound_refresh_button.click(
        refresh_sound_effects,
        outputs=[sound_effect_input],
        show_progress="hidden",
    )
    sound_effect_input.change(
        load_sound_effect,
        inputs=[sound_effect_input],
        outputs=[sound_feature_input],
        show_progress="hidden",
    )
    sound_save_button.click(
        save_sound_effect,
        inputs=[sound_effect_input, sound_feature_input],
        outputs=[sound_feature_input],
        show_progress="hidden",
    )
    app.load(refresh_library_videos, outputs=[library_video_input])
    app.load(refresh_saved_corrected_subtitles, outputs=[saved_corrected_input])
    job_timer = gr.Timer(value=2)
    job_timer.tick(
        poll_job,
        inputs=[job_state],
        outputs=task_outputs,
        show_progress="hidden",
        queue=False,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    app.queue(default_concurrency_limit=4, max_size=20).launch(
        server_name="0.0.0.0", server_port=port, inbrowser=False
    )
