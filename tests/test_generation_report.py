import tempfile
import unittest
from pathlib import Path

from video_generation.report import write_generation_report


class GenerationReportTests(unittest.TestCase):
    def test_report_contains_all_render_decisions(self):
        clip = {
            "ranges": [("00:00:01,000", "00:00:05,000")],
            "raw_result": '{"ranges": [], "reason": "知识点完整"}',
            "highlight_srt": "1\n00:00:01,000 --> 00:00:05,000\n不要抽烟。",
            "keywords": "不要\n抽烟",
            "keyword_reasons": [{"word": "抽烟", "reason": "危险行为"}],
            "sound_bindings": (
                '{"cues":[{"timestamp":"00:00:01,000","text":"抽烟",'
                '"sound_id":"warning","reason":"危险行为"}]}'
            ),
            "visual_bindings": (
                '{"placements":[{"sentence_id":1,"target_word":"抽烟",'
                '"asset_id":"no_smoking","duration_seconds":1.2,"reason":"帮助理解"}]}'
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(write_generation_report(
                temporary,
                "1\n00:00:01,000 --> 00:00:05,000\n不要抽烟。",
                [clip],
                ["/tmp/result.mp4"],
            ))
            content = path.read_text(encoding="utf-8")

        self.assertEqual(path.suffix, ".md")
        self.assertIn("清洗后的字幕稿", content)
        self.assertIn("知识点完整", content)
        self.assertIn("抽烟", content)
        self.assertIn("危险行为", content)
        self.assertIn("帮助理解", content)


if __name__ == "__main__":
    unittest.main()
