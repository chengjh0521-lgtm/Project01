import unittest
from unittest.mock import patch

from video_generation.render import render_highlight_video


class QuestionIntroRenderTests(unittest.TestCase):
    def test_renderer_prepends_the_saved_highlight_question(self):
        launch = type("Launch", (), {
            "AI_clip": staticmethod(lambda *_args: ("raw.mp4", "raw.mp3", "done", "clip srt"))
        })()
        with patch("video_generation.render.get_launch", return_value=launch), patch(
            "video_generation.render._burn_srt_with_ffmpeg", return_value="captioned.mp4"
        ), patch("video_generation.render._overlay_visual_assets", return_value=("visual.mp4", 1)), patch(
            "video_generation.render._mix_sound_effects", return_value=("mixed.mp4", 2)
        ), patch("video_generation.render.prepend_question_intro", return_value="with_intro.mp4") as prepend, patch(
            "video_generation.render.apply_doctor_label", return_value="final.mp4"
        ) as label:
            video, _, message, _ = render_highlight_video(
                "[00:00:01,000-00:00:02,000]", {}, question="糖尿病能喝酒吗？"
            )

        self.assertEqual(video, "final.mp4")
        self.assertIn("question intro=True", message)
        prepend.assert_called_once_with("mixed.mp4", "糖尿病能喝酒吗？")
        label.assert_called_once_with("with_intro.mp4")


if __name__ == "__main__":
    unittest.main()
