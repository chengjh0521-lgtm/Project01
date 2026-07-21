"""Reference layout measured from the approved 1080x1920 portrait example."""

from __future__ import annotations


REFERENCE_WIDTH = 1080
REFERENCE_HEIGHT = 1920

TITLE_LINE_ONE = (540, 230)
TITLE_LINE_TWO = (540, 335)
CAPTION_CENTER = (540, 1220)
DISCLAIMER_CENTER = (540, 1825)
DOCTOR_LABEL_CENTER = (166, 765)

TITLE_FONT_SIZE = 88
CAPTION_FONT_SIZE = 70
DISCLAIMER_FONT_SIZE = 32
DOCTOR_LABEL_SCALE = 0.36


def scaled_position(position: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return (
        round(position[0] * width / REFERENCE_WIDTH),
        round(position[1] * height / REFERENCE_HEIGHT),
    )


def scaled_font_size(reference_size: int, width: int, height: int) -> int:
    return max(1, round(reference_size * min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)))


def overlay_center_expression(position: tuple[int, int]) -> tuple[str, str]:
    """Return FFmpeg expressions for a reference-canvas centre anchor."""
    x_ratio = position[0] / REFERENCE_WIDTH
    y_ratio = position[1] / REFERENCE_HEIGHT
    return "W*{:.9f}-w/2".format(x_ratio), "H*{:.9f}-h/2".format(y_ratio)
