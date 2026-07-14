"""Minimal three-step web UI for the external FunClip modules."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr


OUTPUT_VIDEO_CSS = """
#generated-video { max-width: 640px; margin-left: auto; margin-right: auto; }
#generated-video video { max-height: 360px; object-fit: contain; }
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
from subtitle_processing import process_subtitles
from video_generation import render_highlight_video


VIDEO_LIBRARY_DIR = Path(__file__).resolve().parent / "subtitle_generation" / "pending_videos"
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


def generate(uploaded_video_path, library_video, hotwords):
    try:
        video_path = resolve_library_video(library_video) or uploaded_video_path
    except ValueError as exc:
        return "字幕生成失败：{}".format(exc), None, "", "", "", None, None
    if not video_path:
        return "请上传视频，或从服务器待处理视频中选择一个文件。", None, "", "", "", None, None
    try:
        text, srt, video_state, _, _, _ = generate_subtitles(
            video_path, hotwords=hotwords or ""
        )
        return srt, video_state, "", "", "", None, None
    except Exception as exc:
        return "字幕生成失败：{}".format(exc), None, "", "", "", None, None


def process(srt_text, api_key, keyword_count, video_state):
    if not srt_text:
        return "请先生成字幕。", "", "", None, video_state
    try:
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        return process_subtitles(
            srt_text,
            key,
            keyword_count,
            video_state,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-chat"),
        )
    except Exception as exc:
        return "字幕处理失败：{}".format(exc), "", "", None, video_state


def render(llm_result, video_state, keywords):
    if video_state is None:
        return None
    if not llm_result:
        return None
    try:
        video, _, _, _ = render_highlight_video(llm_result, video_state, keywords=keywords)
        return video
    except Exception as exc:
        raise gr.Error("视频生成失败：{}".format(exc)) from exc


with gr.Blocks(title="FunClip 三模块", css=OUTPUT_VIDEO_CSS) as app:
    video_input = gr.File(label="本地上传视频（可选）", file_types=["video"], type="filepath")
    library_video_input = gr.Dropdown(
        label="服务器待处理视频", choices=list_library_videos(), value=None
    )
    hotwords_input = gr.Textbox(label="热词（可留空）", value="")
    api_key_input = gr.Textbox(label="DeepSeek API Key", type="password")

    subtitle_output = gr.Textbox(label="字幕1：ASR 原始字幕", lines=12)
    corrected_output = gr.Textbox(label="字幕2：DeepSeek 洗稿字幕", lines=12)
    highlight_output = gr.Textbox(label="字幕3：高光时间戳与对应字幕", lines=12)
    keyword_count_input = gr.Number(label="预期关键词数量", value=8, precision=0, minimum=1)
    keyword_output = gr.Textbox(label="高光关键词", lines=6)
    video_output = gr.Video(label="视频输出", height=360, elem_id="generated-video")
    video_state = gr.State()
    llm_result_state = gr.State()

    with gr.Row():
        subtitle_button = gr.Button("1. 生成字幕", variant="primary")
        highlight_button = gr.Button("2. 洗稿、提取高光与关键词")
        video_button = gr.Button("3. 生成视频")

    subtitle_button.click(
        generate,
        inputs=[video_input, library_video_input, hotwords_input],
        outputs=[
            subtitle_output,
            video_state,
            corrected_output,
            highlight_output,
            keyword_output,
            llm_result_state,
            video_output,
        ],
    )
    highlight_button.click(
        process,
        inputs=[subtitle_output, api_key_input, keyword_count_input, video_state],
        outputs=[corrected_output, highlight_output, keyword_output, llm_result_state, video_state],
    )
    video_button.click(
        render,
        inputs=[llm_result_state, video_state, keyword_output],
        outputs=[video_output],
    )
    app.load(refresh_library_videos, outputs=[library_video_input])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    app.queue().launch(server_name="0.0.0.0", server_port=port, inbrowser=False)
