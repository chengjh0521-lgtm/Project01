"""Compose a complete highlight from the four approved video sections."""

from __future__ import annotations

from pathlib import Path

from .question_intro import concat_video_segments, create_question_intro, create_title_cover_frame
from .render import render_highlight_video


def _ranges_as_llm_result(ranges) -> str:
    return "\n".join("[{}-{}]".format(start, end) for start, end in (ranges or []))


def render_four_part_highlight(clip: dict, video_state):
    """Render cover, unchanged question card, direct answer, then full highlight."""
    ranges = _ranges_as_llm_result(clip.get("ranges"))
    question = str(clip.get("question") or "").strip()
    question_lines = clip.get("question_lines")

    # Part 4 keeps the existing layout, visual assets, sound effects, and
    # doctor label. The intro is assembled here so it appears only once.
    main_video, _, main_message, clip_srt = render_highlight_video(
        ranges,
        video_state,
        keywords=clip["keywords"],
        impact_keywords=clip.get("impact_keywords"),
        sound_bindings=clip["sound_bindings"],
        visual_bindings=clip.get("visual_bindings"),
        question=question,
        question_lines=question_lines,
        caption_srt=clip.get("highlight_srt"),
        prepend_question=False,
    )
    if not main_video or not question:
        return main_video, main_message, clip_srt

    # Part 3 is deliberately rendered from unmodified SRT cues selected by
    # the highlighter. It is not semantic-captioned or summarized again.
    answer_ranges = clip.get("doctor_answer_ranges") or list(clip.get("ranges") or [])[:1]
    answer_srt = clip.get("doctor_answer_srt") or clip.get("highlight_srt")
    answer_video, _, answer_message, _ = render_highlight_video(
        _ranges_as_llm_result(answer_ranges),
        video_state,
        caption_srt=answer_srt,
        question=question,
        question_lines=question_lines,
        prepend_question=False,
    )
    if not answer_video:
        raise RuntimeError("医生原话回答片段没有成功生成。")

    main_path = Path(main_video)
    cover_video = create_title_cover_frame(
        question,
        question_lines=question_lines,
        output_path=main_path.with_name("{}_title_cover.mp4".format(main_path.stem)),
    )
    intro_video = create_question_intro(
        question,
        question_lines=question_lines,
        output_path=main_path.with_name("{}_question_intro.mp4".format(main_path.stem)),
    )
    final_video = concat_video_segments(
        [cover_video, intro_video, answer_video, main_video],
        main_path.with_name("{}_four_part.mp4".format(main_path.stem)),
    )
    message = "{}; four-part sequence=True (cover, question intro, doctor answer, main highlight); doctor answer: {}".format(
        main_message, answer_message
    )
    return final_video, message, clip_srt
