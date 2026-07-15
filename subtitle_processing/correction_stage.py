"""Stage 1 contract: correct one SRT and return one corrected SRT."""

from __future__ import annotations

from typing import Callable


def run(srt_text: str, corrector: Callable[[str], str]) -> str:
    """Standardized stage boundary kept independent from orchestration/UI."""
    corrected = corrector(srt_text)
    if not isinstance(corrected, str) or not corrected.strip():
        raise ValueError("Subtitle correction stage returned no SRT.")
    return corrected
