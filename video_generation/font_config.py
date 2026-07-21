"""Shared configurable font for subtitles and question cards."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_FONT_FILE = Path(__file__).with_name("fonts") / "unified_font.ttf"
FALLBACK_FONT_FAMILY = "STHeiti"


def unified_font_file() -> Path:
    configured = os.environ.get("FUNCLIP_UNIFIED_FONT_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_FONT_FILE


def unified_font_family() -> str:
    """Read the installed font's family so libass can resolve it through fontsdir."""
    font_file = unified_font_file()
    if not font_file.is_file():
        return FALLBACK_FONT_FAMILY
    fc_scan = shutil.which("fc-scan")
    if not fc_scan:
        return font_file.stem
    completed = subprocess.run(
        [fc_scan, "--format=%{family}", str(font_file)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    family = completed.stdout.strip().split(",", 1)[0].strip()
    return family or font_file.stem


def subtitle_fonts_directory() -> Path | None:
    font_file = unified_font_file()
    return font_file.parent.resolve() if font_file.is_file() else None
