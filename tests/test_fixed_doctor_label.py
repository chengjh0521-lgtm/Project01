import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.doctor_label import apply_doctor_label


class FixedDoctorLabelTests(unittest.TestCase):
    def test_label_stays_at_visible_upper_left(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "clip.mp4"
            label = root / "label.png"
            source.write_bytes(b"video")
            label.write_bytes(b"png")
            with patch("video_generation.doctor_label.shutil.which", return_value="ffmpeg"), patch(
                "video_generation.doctor_label.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                run.return_value.stderr = ""
                output = root / "clip_label.mp4"
                output.write_bytes(b"labeled video")
                result = apply_doctor_label(source, label)

        command = run.call_args.args[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertEqual(Path(result).name, "clip_label.mp4")
        self.assertIn("scale=67:-1", filters)
        self.assertIn("overlay=x=20:y=20", filters)
        self.assertIn("format=auto:alpha=straight", filters)
        self.assertIn(str(label), command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")

    def test_label_failure_is_not_silently_returned_as_a_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "clip.mp4"
            label = root / "label.png"
            source.write_bytes(b"video")
            label.write_bytes(b"png")
            with patch("video_generation.doctor_label.shutil.which", return_value="ffmpeg"), patch(
                "video_generation.doctor_label.subprocess.run"
            ) as run:
                run.return_value.returncode = 1
                run.return_value.stderr = "FFmpeg label failure"
                with self.assertRaisesRegex(RuntimeError, "Doctor-label overlay failed"):
                    apply_doctor_label(source, label)


if __name__ == "__main__":
    unittest.main()
