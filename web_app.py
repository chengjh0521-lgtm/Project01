"""Minimal three-step web UI for the external FunClip modules."""

from __future__ import annotations

import os

import gradio as gr

from 字幕生成 import generate_subtitles
from 字幕处理 import choose_highlights
from 视频生成 import render_highlight_video


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert short-form video editor. Select coherent, compelling "
    "highlights from the supplied SRT. Return timestamp ranges in the exact "
    "format [HH:MM:SS,mmm-HH:MM:SS,mmm]."
)
DEFAULT_USER_PROMPT = "Select the most engaging clips."


def generate(video_path, hotwords):
    if not video_path:
        return "请先上传视频。", None, "", None
    try:
        text, srt, video_state, _, _, _ = generate_subtitles(
            video_path, hotwords=hotwords or ""
        )
        return srt, video_state, "", None
    except Exception as exc:
        return "字幕生成失败：{}".format(exc), None, "", None


def choose(srt_text, api_key):
    if not srt_text:
        return "请先生成字幕。"
    try:
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        return choose_highlights(
            srt_text,
            key,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_USER_PROMPT,
            model=os.environ.get("FUNCLIP_LLM_MODEL", "deepseek-chat"),
        )
    except Exception as exc:
        return "高光提取失败：{}".format(exc)


def render(llm_result, video_state):
    if video_state is None:
        return None
    if not llm_result:
        return None
    try:
        video, _, _, _ = render_highlight_video(llm_result, video_state)
        return video
    except Exception as exc:
        raise gr.Error("视频生成失败：{}".format(exc)) from exc


with gr.Blocks(title="FunClip 三模块") as app:
    video_input = gr.File(label="视频", file_types=["video"], type="filepath")
    hotwords_input = gr.Textbox(label="热词（可留空）", value="")
    api_key_input = gr.Textbox(label="DeepSeek API Key", type="password")

    subtitle_output = gr.Textbox(label="字幕输出", lines=16)
    highlight_output = gr.Textbox(label="高光输出", lines=12)
    video_output = gr.Video(label="视频输出")
    video_state = gr.State()

    with gr.Row():
        subtitle_button = gr.Button("1. 生成字幕", variant="primary")
        highlight_button = gr.Button("2. 选择高光")
        video_button = gr.Button("3. 生成视频")

    subtitle_button.click(
        generate,
        inputs=[video_input, hotwords_input],
        outputs=[subtitle_output, video_state, highlight_output, video_output],
    )
    highlight_button.click(
        choose,
        inputs=[subtitle_output, api_key_input],
        outputs=[highlight_output],
    )
    video_button.click(
        render,
        inputs=[highlight_output, video_state],
        outputs=[video_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    app.queue().launch(server_name="0.0.0.0", server_port=port, inbrowser=False)
