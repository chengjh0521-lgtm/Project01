import unittest

from video_generation.render import _rebase_caption_srt


class CaptionTimelineTests(unittest.TestCase):
    def test_rebases_absolute_source_timestamps_to_clip_timeline(self):
        source_srt = (
            "1\n00:06:04,000 --> 00:06:07,000\n糖尿病患者\n\n"
            "2\n00:06:07,000 --> 00:06:10,000\n需要控制饮食\n"
        )
        result = _rebase_caption_srt(
            source_srt,
            "[00:06:04,000-00:06:10,000]",
        )

        self.assertIn("00:00:00,000 --> 00:00:03,000", result)
        self.assertIn("00:00:03,000 --> 00:00:06,000", result)
        self.assertIn("糖尿病患者", result)

    def test_keeps_the_accumulated_offset_for_discontinuous_ranges(self):
        source_srt = (
            "1\n00:00:10,000 --> 00:00:12,000\n第一段\n\n"
            "2\n00:00:20,000 --> 00:00:22,000\n第二段\n"
        )
        result = _rebase_caption_srt(
            source_srt,
            "[00:00:10,000-00:00:12,000]\n[00:00:20,000-00:00:22,000]",
        )

        self.assertIn("00:00:02,100 --> 00:00:04,100", result)
