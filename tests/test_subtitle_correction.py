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
from utils.subtitle_utils import generate_srt
from utils.subtitle_utils import generate_srt_clip
from videoclipper import _subtitle_cues_for_clip, _subtitle_cues_to_srt


SRT = """1
00:00:01,000 --> 00:00:03,000
这个病人血糖有点高

2  spk0
00:00:03,200 --> 00:00:05,000
需要检察糖化血红蛋白
"""

CONTINUOUS_SPEAKER_SRT = """1  spk0
00:00:00,150 --> 00:00:04,490
第一句
2  spk1
00:00:04,490 --> 00:00:07,170
第二句
"""


class TestSubtitleCorrection(unittest.TestCase):
    def test_corrects_text_and_preserves_timeline_and_prefix(self):
        def fake_call(user_content, _system_content):
            self.assertIn("[00:00:01,000-00:00:03,000]", user_content)
            self.assertIn("[00:00:03,200-00:00:05,000]", user_content)
            return (
                "1. [00:00:01,000-00:00:03,000] 这个病人血糖有点高\n"
                "2. [00:00:03,200-00:00:05,000] 需要检查糖化血红蛋白"
            )

        corrected, changed, total, matched = correct_srt_with_llm(SRT, "校对医学术语", fake_call)

        self.assertEqual(changed, 1)
        self.assertEqual(total, 2)
        self.assertEqual(matched, 2)
        self.assertIn("2  spk0", corrected)
        self.assertIn("00:00:03,200 --> 00:00:05,000", corrected)
        self.assertIn("需要检查糖化血红蛋白", corrected)
        self.assertNotIn("需要检察糖化血红蛋白", corrected)

    def test_keeps_unmatched_subtitles_unchanged(self):
        response = "1. [00:00:03,200-00:00:05,000] 需要检查糖化血红蛋白"
        corrected, changed, total, matched = correct_srt_with_llm(
            SRT, "校对", lambda *_args: response
        )
        self.assertEqual(changed, 1)
        self.assertEqual(total, 2)
        self.assertEqual(matched, 1)
        self.assertIn("这个病人血糖有点高", corrected)

    def test_parses_speaker_srt_without_blank_lines_between_cues(self):
        entries = parse_srt_entries(CONTINUOUS_SPEAKER_SRT)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["prefix"], ["1  spk0"])
        self.assertEqual(entries[0]["text"], "第一句")
        self.assertEqual(entries[1]["prefix"], ["2  spk1"])
        self.assertEqual(entries[1]["text"], "第二句")

    def test_speaker_srt_output_has_standard_blank_cue_separator(self):
        srt = generate_srt(
            [
                {"text": ["第", "一", "句"], "timestamp": [[0, 500], [500, 1000], [1000, 1500]], "spk": 0},
                {"text": ["第", "二", "句"], "timestamp": [[1500, 2000], [2000, 2500], [2500, 3000]], "spk": 1},
            ]
        )

        self.assertIn("第一句\n\n2  spk1", srt)

    def test_clip_subtitles_use_timestamp_matched_correction_override(self):
        sentences = [
            {"text": ["原", "始", "文", "案"], "timestamp": [[1000, 1500], [1500, 2000], [2000, 2500], [2500, 3000]]},
        ]

        srt, subs, _ = generate_srt_clip(
            sentences,
            1.0,
            3.0,
            subtitle_overrides={"1000-3000": "DeepSeek 修正文案"},
        )

        self.assertIn("DeepSeek 修正文案", srt)
        self.assertEqual(subs[0][1], "DeepSeek 修正文案")

    def test_partial_clip_subtitles_use_correction_override(self):
        sentences = [
            {"text": ["原", "始", "文", "案"], "timestamp": [[1000, 1500], [1500, 2000], [2000, 2500], [2500, 3000]]},
        ]

        srt, subs, _ = generate_srt_clip(
            sentences,
            1.6,
            3.0,
            subtitle_overrides={"1000-3000": "DeepSeek 修正文案"},
        )

        self.assertIn("DeepSeek 修正文案", srt)
        self.assertEqual(subs[0][1], "DeepSeek 修正文案")

    def test_direct_srt_cues_are_trimmed_for_the_clip_window(self):
        subs = _subtitle_cues_for_clip(
            [(1.0, 3.0, "DeepSeek 修正文案")],
            1.6,
            2.5,
        )

        self.assertEqual(subs[0][0][0], 0.0)
        self.assertAlmostEqual(subs[0][0][1], 0.9)
        self.assertEqual(subs[0][1], "DeepSeek 修正文案")

    def test_direct_srt_cues_are_written_to_the_downloaded_clip_srt(self):
        srt, next_index = _subtitle_cues_to_srt(
            [((0.0, 0.9), "DeepSeek 修正文案")],
            begin_index=3,
            time_offset=12.1,
        )

        self.assertIn("3\n00:00:12,100 --> 00:00:13,000", srt)
        self.assertIn("DeepSeek 修正文案", srt)
        self.assertEqual(next_index, 4)

    def test_rebuilds_all_text_state_from_corrected_srt(self):
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

        updated, synced = update_state_subtitles(state, corrected)

        self.assertEqual(synced, 2)
        self.assertEqual(updated["sentences"][1]["text"], "需要检查糖化血红蛋白")
        self.assertEqual(updated["sentences"][1]["timestamp"][0][0], 3200)
        self.assertEqual(updated["sentences"][1]["timestamp"][-1][1], 5000)
        self.assertEqual(
            updated["subtitle_text_overrides"]["3200-5000"],
            "需要检查糖化血红蛋白",
        )
        self.assertIn("需 要 检 查 糖 化 血 红 蛋 白", updated["recog_res_raw"])
        self.assertNotIn("raw text", updated["recog_res_raw"])
        self.assertIn("需要检查糖化血红蛋白", updated["canonical_subtitle_srt"])
        self.assertIs(updated["video"], video_handle)
        self.assertEqual(state["sentences"][1]["text"], "旧字幕二")

        clip_srt, clip_subs, _ = generate_srt_clip(
            updated["sentences"], 3.2, 5.0
        )
        self.assertIn("需要检查糖化血红蛋白", clip_srt)
        self.assertEqual(clip_subs[0][1], "需要检查糖化血红蛋白")

    def test_removes_extra_original_asr_sentences(self):
        state = {
            "sentences": [
                {"text": "旧字幕一", "timestamp": [[1000, 3000]]},
                {"text": "旧字幕二", "timestamp": [[3200, 5000]]},
                {"text": "内部额外句子", "timestamp": [[6000, 7000]]},
            ],
        }
        corrected = SRT.replace("需要检察糖化血红蛋白", "需要检查糖化血红蛋白")

        updated, synced = update_state_subtitles(state, corrected)

        self.assertEqual(synced, 2)
        self.assertEqual(updated["sentences"][1]["text"], "需要检查糖化血红蛋白")
        self.assertEqual(len(updated["sentences"]), 2)
        self.assertNotIn("内部额外句子", str(updated))

    def test_parse_rejects_non_srt_text(self):
        with self.assertRaises(SubtitleCorrectionError):
            parse_srt_entries("plain transcript without timestamps")


if __name__ == "__main__":
    unittest.main()

