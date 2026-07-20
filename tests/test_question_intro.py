import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.question_intro import MAX_QUESTION_INTRO_SECONDS, create_question_intro


class QuestionIntroTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
