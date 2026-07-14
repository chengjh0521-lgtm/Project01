"""Video rendering entry point backed by the upstream FunClip callback."""

from __future__ import annotations

from pathlib import Path

from funclip_loader import get_launch


def render_highlight_video(
        llm_result: str,
        video_state,
        output_dir: str | Path = "",
        burn_subtitles: bool = True,
        start_offset_ms: int = 0,
        end_offset_ms: int = 100):
    """Render LLM timestamp ranges. Returns video, audio, message, and clip SRT."""
    launch = get_launch()
    if burn_subtitles:
        return launch.AI_clip_subti(
            llm_result, "", "", start_offset_ms, end_offset_ms,
            video_state, None, str(output_dir),
        )
    return launch.AI_clip(
        llm_result, "", "", start_offset_ms, end_offset_ms,
        video_state, None, str(output_dir),
    )
