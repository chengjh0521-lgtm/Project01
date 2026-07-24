"""Reference layout measured from the approved 1080x1920 portrait example."""

from __future__ import annotations


REFERENCE_WIDTH = 1080
REFERENCE_HEIGHT = 1920

# The approved reference uses a translucent title band occupying the upper
# portion of the portrait canvas. Keep these values in the 1080x1920 design
# space and scale them only at render time.
TITLE_BACKGROUND_TOP = 160
TITLE_BACKGROUND_HEIGHT = 248
TITLE_BACKGROUND_COLOR = "white@0.56"
TITLE_BACKGROUND_BORDER_COLOR = "black@0.22"

TITLE_LINE_ONE = (540, 222)
TITLE_LINE_TWO = (540, 322)

CAPTION_CENTER = (540, 1450)

DISCLAIMER_CENTER = (540, 1825)
DOCTOR_LABEL_CENTER = (146, 800)

TITLE_FONT_SIZE = 110
SINGLE_LINE_TITLE_FONT_SIZE = 138
CAPTION_FONT_SIZE = 86
DISCLAIMER_FONT_SIZE = 32
DOCTOR_LABEL_SCALE = 0.24

# The source clinic footage is usually flatter and darker than the approved
# portrait reference. Apply this once, before caption rendering, so text and
# overlays keep their intended colours.
VIDEO_BRIGHTNESS = 0.08
VIDEO_CONTRAST = 1.03
VIDEO_SATURATION = 1.04


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
