"""Minimal three-step web UI for the external FunClip modules."""

from __future__ import annotations

import json
import os
import hashlib
import logging
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
from subtitle_processing.sound_effect_binding import (
    get_effect_details,
    list_sound_effects,
    save_effect_details,
)
from video_generation import describe_sound_effect_events, describe_visual_asset_events, render_highlight_video
from background_jobs import JOBS


VIDEO_LIBRARY_DIR = Path(__file__).resolve().parent / "subtitle_generation" / "pending_videos"
SOUND_LOGIC_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "sound_effect_logic"
SAVED_CORRECTED_SUBTITLE_DIR = Path(__file__).resolve().parent / "subtitle_processing" / "saved_corrected_subtitles"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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
        return ("字幕生成任务未提交：{}".format(exc), *_skipped_outputs(9), "", None, *_progress_updates("asr", 0))
    if not video_path:
        return ("请上传视频，或从服务器待处理视频中选择一个文件。", *_skipped_outputs(9), "", None, *_progress_updates("asr", 0))

    def worker(report):
        report("阶段 1/1：正在进行 ASR 字幕识别。", 5)
        text, srt, video_state, _, _, _ = generate_subtitles(
            video_path, hotwords=hotwords or ""
        )
        report("阶段 1/1：ASR 字幕识别完成。", 100)
        return {"subtitle": srt, "video_state": video_state}

    job_id = JOBS.submit("asr", worker)
    return ("字幕生成已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(9), job_id, {"id": job_id}, *_progress_updates("asr", 0))


def submit_process(srt_text, api_key, keyword_count, clip_count, video_state):
    if not srt_text:
        return ("请先生成字幕。", *_skipped_outputs(9), "", None, *_progress_updates("process", 0))
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

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
            srt_text, key, keyword_count, clip_count, video_state,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash"),
            status_callback=report,
            on_corrected=save_after_correction,
        )
        return {"pipeline": result, "corrected_file": corrected_file or save_corrected_subtitle(result[0])}

    job_id = JOBS.submit("process", worker)
    return ("字幕处理已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(9), job_id, {"id": job_id}, *_progress_updates("process", 0))


def submit_process_from_saved(saved_file, api_key, keyword_count, clip_count, video_state):
    try:
        path = _saved_corrected_subtitle_path(saved_file)
        corrected_srt = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return ("保存字幕任务未提交：{}".format(exc), *_skipped_outputs(9), "", None, *_progress_updates("process", 0))
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    def worker(report):
        result = process_from_corrected_subtitles(
            corrected_srt, key, keyword_count, clip_count, video_state,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-v4-flash"),
            status_callback=report,
        )
        return {"pipeline": result, "corrected_file": str(path)}

    job_id = JOBS.submit("process", worker)
    return ("已保存字幕2处理任务已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(9), job_id, {"id": job_id}, *_progress_updates("process", 0))


def submit_render(llm_result, video_state, keywords, sound_bindings):
    logging.warning(
        "Render submission received: video_state=%s, plan=%s, keywords=%s, sound_bindings=%s.",
        video_state is not None,
        bool(llm_result),
        bool(keywords),
        bool(sound_bindings),
    )
    if video_state is None:
        return ("请先完成字幕生成和第二步处理。", *_skipped_outputs(9), "", None, *_progress_updates("render", 0))
    if not llm_result:
        return ("请先完成第二步高光提取。", *_skipped_outputs(9), "", None, *_progress_updates("render", 0))

    def worker(report):
        report("阶段 1/1：正在剪辑视频并烧录字幕。", 5)
        if isinstance(llm_result, dict) and llm_result.get("clips"):
            videos, logic_files = [], []
            for index, clip in enumerate(llm_result["clips"], start=1):
                ranges = "\n".join("[{}-{}]".format(start, end) for start, end in clip["ranges"])
                video, _, _, clip_srt = render_highlight_video(
                    ranges,
                    video_state,
                    keywords=clip["keywords"],
                    sound_bindings=clip["sound_bindings"],
                    visual_bindings=clip.get("visual_bindings"),
                )
                if video:
                    videos.append(video)
                    logic_files.append(write_sound_effect_logic(
                        video,
                        clip_srt,
                        clip["sound_bindings"],
                        clip["id"],
                        clip["ranges"],
                        clip.get("visual_bindings"),
                    ))
                report("阶段 1/1：已完成 {}/{} 条视频。".format(index, len(llm_result["clips"])), round(index * 100 / len(llm_result["clips"])))
            return {"video": videos, "sound_logic": logic_files}
        video, _, _, clip_srt = render_highlight_video(llm_result, video_state, keywords=keywords, sound_bindings=sound_bindings)
        report("阶段 1/1：视频生成完成。", 100)
        logic_files = [write_sound_effect_logic(video, clip_srt, sound_bindings, "clip_01")] if video else []
        return {"video": video, "sound_logic": logic_files}

    job_id = JOBS.submit("render", worker)
    return ("视频生成已进入后台队列：{}".format(job_id[:8]), *_skipped_outputs(9), job_id, {"id": job_id}, *_progress_updates("render", 0))


def poll_job(job_ref):
    job_id = job_ref.get("id") if isinstance(job_ref, dict) else None
    job = JOBS.get(job_id)
    if job is None:
        return ("后台任务不存在，可能服务刚刚重启。", *_skipped_outputs(10), "", None, *_progress_updates())
    status = job["status"]
    prefix = "后台任务 {}：".format(job["kind"])
    if status in {"queued", "running"}:
        return (prefix + job["message"], *_skipped_outputs(10), job_id, job_ref, *_progress_updates(job["kind"], job["progress"]))
    if status == "failed":
        return (prefix + "失败：{}".format(job["error"] or job["message"]), *_skipped_outputs(10), job_id, None, *_progress_updates(job["kind"], job["progress"]))

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
            job_id,
            None,
            *_progress_updates("process", 100),
        )
    if job["kind"] == "render":
        return (prefix + "完成。", *_skipped_outputs(8), result["video"], result["sound_logic"], job_id, None, *_progress_updates("render", 100))
    return (prefix + "完成。", *_skipped_outputs(10), job_id, None, *_progress_updates())


def resume_job(job_id):
    """Resume polling after a browser refresh using the visible job ID."""
    logging.warning("Job query received: %s", str(job_id or "").strip()[:16])
    if not str(job_id or "").strip():
        return ("请输入需要恢复的后台任务 ID。", *_skipped_outputs(10), "", None, *_progress_updates())
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
    sound_bindings_output = gr.State()
    video_output = gr.File(label="输出视频（可多条下载）", file_count="multiple")
    sound_logic_output = gr.File(label="音效与 GIF/PNG 添加逻辑（JSON，可下载）", file_count="multiple")
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
        outputs=[
            job_status,
            subtitle_output,
            video_state,
            corrected_output,
            highlight_output,
            keyword_output,
            sound_bindings_output,
            llm_result_state,
            video_output,
            sound_logic_output,
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
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
        outputs=[
            job_status,
            subtitle_output,
            video_state,
            corrected_output,
            highlight_output,
            keyword_output,
            sound_bindings_output,
            llm_result_state,
            video_output,
            sound_logic_output,
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
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
        outputs=[
            job_status,
            subtitle_output,
            video_state,
            corrected_output,
            highlight_output,
            keyword_output,
            sound_bindings_output,
            llm_result_state,
            video_output,
            sound_logic_output,
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
        show_progress="hidden",
        concurrency_limit=4,
    )
    video_button.click(
        submit_render,
        inputs=[llm_result_state, video_state, keyword_output, sound_bindings_output],
        outputs=[
            job_status,
            subtitle_output,
            video_state,
            corrected_output,
            highlight_output,
            keyword_output,
            sound_bindings_output,
            llm_result_state,
            video_output,
            sound_logic_output,
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
        show_progress="hidden",
        queue=False,
    )
    resume_button.click(
        resume_job,
        inputs=[job_id_input],
        outputs=[
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
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
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
        outputs=[
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
            job_id_input,
            job_state,
            asr_progress,
            process_progress,
            render_progress,
        ],
        show_progress="hidden",
        queue=False,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    app.queue(default_concurrency_limit=4, max_size=20).launch(
        server_name="0.0.0.0", server_port=port, inbrowser=False
    )
