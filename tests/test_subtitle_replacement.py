import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "funclip"))

from subtitle_replacement import apply_replacement, restore_original_subtitles
from utils.subtitle_utils import generate_srt_clip


class TestSubtitleReplacement(unittest.TestCase):
    def test_replacement_rebuilds_caption_state_and_restores_asr_state(self):
        original_state = {
            "recog_res_raw": "old words",
            "timestamp": [[0, 1000]],
            "sentences": [{"text": ["old"], "timestamp": [[0, 1000]]}],
            "video": object(),
        }
        replacement = """1
00:00:01,000 --> 00:00:03,000
new caption

2
00:00:03,000 --> 00:00:04,000
second cue
"""

        updated, canonical = apply_replacement(
            original_state, "original srt", replacement, "/tmp/replacement.srt"
        )

        self.assertIn("new caption", canonical)
        self.assertEqual(updated["video"], original_state["video"])
        self.assertIn("new", updated["recog_res_raw"])
        self.assertEqual(updated["sentences"][0]["timestamp"][0][0], 1000)
        self.assertEqual(updated["sentences"][1]["timestamp"][-1][1], 4000)
        clipped_srt, _, _ = generate_srt_clip(updated["sentences"], 1.0, 3.0)
        self.assertIn("new caption", clipped_srt)
        self.assertNotIn("old words", clipped_srt)

        restored, original_srt = restore_original_subtitles(updated)
        self.assertEqual(original_srt, "original srt")
        self.assertEqual(restored["recog_res_raw"], "old words")
        self.assertEqual(restored["video"], original_state["video"])


if __name__ == "__main__":
    unittest.main()
