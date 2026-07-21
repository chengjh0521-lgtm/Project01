import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.question_intro import (
    MAX_QUESTION_INTRO_SECONDS,
    QUESTION_TEXT_ASS_COLOR,
    _wrap_question_text,
    _write_question_ass,
    create_question_intro,
    prepend_question_intro,
)


class QuestionIntroTests(unittest.TestCase):
    def test_question_text_wraps_to_at_most_six_characters_per_line(self):
        wrapped = _wrap_question_text("糖尿病患者能不能喝酒呢？")

        lines = wrapped.split("\\N")
        self.assertEqual("".join(lines), "糖尿病患者能不能喝酒呢？")
        self.assertTrue(all(len(line) <= 6 for line in lines))

    def test_question_mark_is_never_orphaned_on_its_own_line(self):
        wrapped = _wrap_question_text("糖前期应该吃粗粮吗？")

        self.assertNotIn("\\N？", wrapped)
        self.assertNotIn("\\N?", wrapped)

    def test_question_text_is_double_sized_and_starts_at_half_height(self):
        with tempfile.TemporaryDirectory() as temporary:
            ass_path = Path(temporary) / "question.ass"
            _write_question_ass("糖尿病能喝酒吗？", ass_path, 1080, 1920)
            ass = ass_path.read_text(encoding="utf-8")

        self.assertIn("Style: Question,STHeiti,152,", ass)
        self.assertIn(",{},".format(QUESTION_TEXT_ASS_COLOR), ass)
        self.assertIn(",1,2,1,8,80,80,1344,1", ass)

    def test_creates_a_short_static_video_with_the_question_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.png"
            output = root / "question.mp4"
            background.write_bytes(b"png")

            def render(command, **_kwargs):
                Path(command[-1]).write_bytes(b"video")
                class Completed:
                    returncode = 0
                    stderr = ""
                return Completed()

            with patch("video_generation.question_intro._synthesize_question_audio"), patch(
                "video_generation.question_intro._audio_duration_seconds", return_value=2.4
            ), patch("video_generation.question_intro.shutil.which", return_value="ffmpeg"), patch(
                "video_generation.question_intro.subprocess.run", side_effect=render
            ) as run:
                result = create_question_intro("糖尿病能喝酒吗？", background_path=background, output_path=output)

        command = run.call_args.args[0]
        self.assertEqual(Path(result).name, "question.mp4")
        self.assertIn("-loop", command)
        self.assertIn("subtitles=filename=", command[command.index("-filter:v") + 1])
        self.assertIn("-t", command)
        self.assertLessEqual(float(command[command.index("-t") + 1]), MAX_QUESTION_INTRO_SECONDS)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")

    def test_rejects_audio_that_cannot_fit_the_three_second_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.png"
            background.write_bytes(b"png")
            with patch("video_generation.question_intro._synthesize_question_audio"), patch(
                "video_generation.question_intro._audio_duration_seconds", side_effect=[3.4, 3.1]
            ):
                with self.assertRaisesRegex(ValueError, "above the 3.0s limit"):
                    create_question_intro("糖尿病能喝酒吗？", background_path=background)

    def test_prepends_the_intro_before_the_main_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "clip.mp4"
            intro = root / "clip_question_intro.mp4"
            output = root / "clip_with_question_intro.mp4"
            source.write_bytes(b"video")
            intro.write_bytes(b"intro")

            def concat(command, **_kwargs):
                Path(command[-1]).write_bytes(b"combined")
                class Completed:
                    returncode = 0
                    stderr = ""
                return Completed()

            with patch("video_generation.question_intro._video_dimensions", return_value=(1080, 1920)), patch(
                "video_generation.question_intro.create_question_intro", return_value=str(intro)
            ), patch("video_generation.question_intro.shutil.which", return_value="ffmpeg"), patch(
                "video_generation.question_intro.subprocess.run", side_effect=concat
            ) as run:
                result = prepend_question_intro(source, "糖尿病能喝酒吗？")

        command = run.call_args.args[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertEqual(Path(result).name, "clip_with_question_intro.mp4")
        self.assertIn(str(intro), command)
        self.assertIn("setsar=1", filters)
        self.assertIn("concat=n=2:v=1:a=1", filters)


if __name__ == "__main__":
    unittest.main()
