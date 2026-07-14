"""ASR entry point backed by the upstream FunClip runtime."""

from __future__ import annotations

from pathlib import Path

from funclip_loader import get_launch


def generate_subtitles(
        video_path: str | Path,
        output_dir: str | Path = "",
        hotwords: str = "",
        speaker_diarization: bool = False):
    """Return text, SRT, media state, downloadable files, and optional audio state."""
    launch = get_launch()
    recognizer = launch.mix_recog_speaker if speaker_diarization else launch.mix_recog
    return recognizer(str(video_path), None, hotwords, str(output_dir))
