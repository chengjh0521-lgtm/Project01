import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "funclip"))

from llm.subtitle_correction import render_llm_highlight_srt
from utils.trans_utils import extract_timestamps
from videoclipper import VideoClipper


class TestLlmClipRanges(unittest.TestCase):
    def test_extracts_only_the_llm_timestamp_ranges(self):
        result = (
            "1. [00:01:02,300-00:01:05,800] first highlight\n"
            "2. [00:02:10-00:02:15] second highlight"
        )

        self.assertEqual(
            extract_timestamps(result),
            [[62_300, 65_800], [130_000, 135_000]],
        )

    def test_audio_llm_ranges_are_converted_from_millis_to_samples(self):
        audio = np.arange(16000 * 3, dtype=np.float64)
        state = {
            "audio_input": (16000, audio),
            "recog_res_raw": "测 试",
            "timestamp": [[1000, 1500], [1500, 2000]],
            "sentences": [
                {"text": "测试", "timestamp": [[1000, 1500], [1500, 2000]]}
            ],
        }

        (_, clipped), _, _ = VideoClipper(None).clip(
            "ignored text",
            0,
            0,
            state,
            timestamp_list=[[1000, 2000]],
        )

        self.assertEqual(len(clipped), 16000)
        self.assertEqual(clipped[0], audio[16000])
        self.assertEqual(clipped[-1], audio[31999])

    def test_step_three_text_and_ranges_become_caption_srt(self):
        result = (
            "1. [00:01:02,300-00:01:05,800] 第三步高光字幕一\n"
            "2. [00:02:10-00:02:15] 第三步高光字幕二"
        )

        highlight_srt, ranges = render_llm_highlight_srt(result)

        self.assertEqual(ranges, [[62_300, 65_800], [130_000, 135_000]])
        self.assertIn("00:01:02,300 --> 00:01:05,800", highlight_srt)
        self.assertIn("第三步高光字幕一", highlight_srt)
        self.assertIn("第三步高光字幕二", highlight_srt)


if __name__ == "__main__":
    unittest.main()
