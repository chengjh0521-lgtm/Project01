"""LLM highlight selection using the upstream FunClip callback."""

from __future__ import annotations

from funclip_loader import get_launch


def choose_highlights(
        srt_text: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        model: str = "deepseek-chat",
        video_path: str | None = None) -> str:
    """Return the original FunClip LLM result with highlight timestamps."""
    launch = get_launch()
    return launch.llm_inference(
        system_prompt,
        user_prompt,
        srt_text,
        model,
        api_key,
        video_path,
    )
