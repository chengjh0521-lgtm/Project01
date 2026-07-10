import json
import sys
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "funclip"))

from llm.subtitle_correction import (
    SubtitleCorrectionError,
    correct_srt_with_llm,
    parse_srt_entries,
    update_state_subtitles,
)


SRT = """1
00:00:01,000 --> 00:00:03,000
这个病人血糖有点高

2  spk0
00:00:03,200 --> 00:00:05,000
需要检察糖化血红蛋白
"""


class TestSubtitleCorrection(unittest.TestCase):
    def test_corrects_text_and_preserves_timeline_and_prefix(self):
        def fake_call(user_content, _system_content):
            payload = json.loads(user_content.split("\n", 1)[1])
            self.assertNotIn("id", payload["subtitles"][0])
            payload["subtitles"][1]["text"] = "需要检查糖化血红蛋白"
            # DeepSeek sometimes adds a repeated id. Order, rather than that id,
            # must remain the source of truth for a whole subtitle batch.
            payload["subtitles"][0]["id"] = 1
            payload["subtitles"][1]["id"] = 1
            return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

        corrected, changed, total = correct_srt_with_llm(SRT, "校对医学术语", fake_call)

        self.assertEqual(changed, 1)
        self.assertEqual(total, 2)
        self.assertIn("2  spk0", corrected)
        self.assertIn("00:00:03,200 --> 00:00:05,000", corrected)
        self.assertIn("需要检查糖化血红蛋白", corrected)
        self.assertNotIn("需要检察糖化血红蛋白", corrected)

    def test_rejects_response_that_changes_subtitle_count(self):
        response = json.dumps(
            {"subtitles": [{"id": 1, "text": "这个病人血糖有点高"}]},
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(SubtitleCorrectionError, "omitted, added, or merged"):
            correct_srt_with_llm(SRT, "校对", lambda *_args: response)

    def test_updates_rendering_state_without_changing_timestamps(self):
        video_handle = threading.Lock()
        state = {
            "recog_res_raw": "raw text",
            "timestamp": [[1000, 3000], [3200, 5000]],
            "video": video_handle,
            "sentences": [
                {"text": "旧字幕一", "timestamp": [[1000, 3000]]},
                {"text": "旧字幕二", "timestamp": [[3200, 5000]]},
            ],
        }
        corrected = SRT.replace("需要检察糖化血红蛋白", "需要检查糖化血红蛋白")

        updated = update_state_subtitles(state, corrected)

        self.assertEqual(updated["sentences"][1]["text"], "需要检查糖化血红蛋白")
        self.assertEqual(updated["sentences"][1]["timestamp"], [[3200, 5000]])
        self.assertIs(updated["video"], video_handle)
        self.assertEqual(state["sentences"][1]["text"], "旧字幕二")

    def test_parse_rejects_non_srt_text(self):
        with self.assertRaises(SubtitleCorrectionError):
            parse_srt_entries("plain transcript without timestamps")


if __name__ == "__main__":
    unittest.main()
