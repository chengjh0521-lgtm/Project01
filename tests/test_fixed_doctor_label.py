import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.render import _overlay_doctor_label


class FixedDoctorLabelTests(unittest.TestCase):
    def test_label_stays_at_visible_upper_left(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "clip.mp4"
            label = root / "label.png"
            source.write_bytes(b"video")
            label.write_bytes(b"png")
            with patch("video_generation.render._DOCTOR_LABEL_FILE", label), patch(
                "video_generation.render.shutil.which", return_value="ffmpeg"
            ), patch("video_generation.render.subprocess.run") as run:
                run.return_value.returncode = 0
                result, applied = _overlay_doctor_label(source)

        command = run.call_args.args[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertTrue(applied)
        self.assertEqual(Path(result).name, "clip_label.mp4")
        self.assertIn("scale=200:-1", filters)
        self.assertIn("overlay=x=20:y=20", filters)
        self.assertIn(str(label), command)


if __name__ == "__main__":
    unittest.main()
