import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.four_part import render_four_part_highlight


class FourPartHighlightRenderTests(unittest.TestCase):
    def test_renders_cover_intro_doctor_answer_then_main_highlight(self):
        clip = {
            "id": "clip_01",
            "question": "糖尿病能喝酒吗？",
            "question_lines": ["糖尿病能", "喝酒吗？"],
            "ranges": [("00:00:01,000", "00:00:42,000")],
            "doctor_answer_ranges": [("00:00:10,000", "00:00:12,000")],
            "doctor_answer_srt": "1\n00:00:10,000 --> 00:00:12,000\n不建议喝酒\n",
            "highlight_srt": "1\n00:00:01,000 --> 00:00:42,000\n高光内容\n",
            "keywords": "喝酒",
            "impact_keywords": [],
            "sound_bindings": "{}",
            "visual_bindings": "{}",
        }
        render_results = [
            ("/tmp/main.mp4", None, "main", "main srt"),
            ("/tmp/answer.mp4", None, "answer", "answer srt"),
        ]
        with patch("video_generation.four_part.render_highlight_video", side_effect=render_results) as render, patch(
            "video_generation.four_part.create_title_cover_frame", return_value="/tmp/cover.mp4"
        ) as cover, patch("video_generation.four_part.create_question_intro", return_value="/tmp/intro.mp4") as intro, patch(
            "video_generation.four_part.concat_video_segments", return_value="/tmp/final.mp4"
        ) as concat:
            video, message, clip_srt = render_four_part_highlight(clip, {"video": "state"})

        self.assertEqual(video, "/tmp/final.mp4")
        self.assertEqual(clip_srt, "main srt")
        self.assertIn("four-part sequence=True", message)
        self.assertEqual(render.call_count, 2)
        self.assertFalse(render.call_args_list[0].kwargs["prepend_question"])
        self.assertEqual(render.call_args_list[1].kwargs["question"], "糖尿病能喝酒吗？")
        self.assertEqual(render.call_args_list[1].kwargs["question_lines"], ["糖尿病能", "喝酒吗？"])
        self.assertFalse(render.call_args_list[1].kwargs["prepend_question"])
        cover.assert_called_once()
        intro.assert_called_once()
        concat.assert_called_once_with(
            ["/tmp/cover.mp4", "/tmp/intro.mp4", "/tmp/answer.mp4", "/tmp/main.mp4"],
            Path("/tmp/main_four_part.mp4"),
        )


if __name__ == "__main__":
    unittest.main()
