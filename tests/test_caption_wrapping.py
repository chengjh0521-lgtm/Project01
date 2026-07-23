import tempfile
import unittest
from pathlib import Path

from video_generation.render import (
    _caption_font_size,
    _caption_display_events,
    _impact_caption_display_events,
    _strip_caption_fillers,
    _title_lines,
    _write_ass_subtitles,
    _write_reference_layout_ass,
    _wrap_caption_two_lines,
)


class CaptionWrappingTests(unittest.TestCase):
    def test_keeps_short_captions_on_one_line(self):
        self.assertEqual(_wrap_caption_two_lines("糖尿病能喝酒吗？"), "糖尿病能喝酒吗？")

    def test_caption_font_size_matches_the_reference_layout(self):
        self.assertEqual(_caption_font_size("任意长度的字幕"), 86)

    def test_splits_captions_longer_than_fifteen_characters_at_punctuation(self):
        text = "糖尿病患者控制血糖很重要，但是不能因此过度焦虑。"

        wrapped = _wrap_caption_two_lines(text)

        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertIn("\n", wrapped)
        self.assertTrue(wrapped.split("\n")[0].endswith("，"))

    def test_uses_a_connector_when_no_punctuation_is_available(self):
        text = "血糖控制得很好但是仍然需要规律复查"

        wrapped = _wrap_caption_two_lines(text)

        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertIn("\n", wrapped)
        self.assertTrue(wrapped.split("\n")[1].startswith("但是"))

    def test_removes_spoken_fillers_only_from_display_text(self):
        self.assertEqual(
            _strip_caption_fillers("嗯，呃，糖尿病患者啊，不能喝酒呢。"),
            "糖尿病患者，不能喝酒。",
        )

    def test_long_caption_becomes_sequential_single_line_events(self):
        events = _caption_display_events(
            "00:00:00,000",
            "00:00:04,000",
            "糖尿病患者控制血糖很重要，但是不能因此过度焦虑。",
        )

        self.assertGreater(len(events), 1)
        self.assertTrue(all("\n" not in line for _, _, line in events))
        self.assertEqual("".join(line for _, _, line in events), "糖尿病患者控制血糖很重要，但是不能因此过度焦虑。")

    def test_first_impact_keyword_becomes_a_separate_title_size_event(self):
        events = _impact_caption_display_events(
            "00:00:00,000", "00:00:04,000", "糖前期就会发展成糖尿病", ["糖尿病"], set()
        )

        self.assertEqual([item[2] for item in events], ["糖前期就会发展成", "糖尿病"])
        self.assertEqual([item[3] for item in events], [False, True])

    def test_impact_keyword_is_enlarged_only_at_its_first_appearance(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:03,000\n糖前期就会发展成糖尿病\n\n"
            "2\n00:00:03,000 --> 00:00:06,000\n糖尿病需要长期管理\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            ass_path = Path(temporary) / "captions.ass"
            _write_ass_subtitles(srt, ass_path, ["糖尿病"], impact_keywords=["糖尿病"])
            rendered = ass_path.read_text(encoding="utf-8")

        self.assertIn("Style: Impact,STHeiti,110,", rendered)
        self.assertEqual(rendered.count(",Impact,,0,0,0,,"), 1)
        self.assertIn("糖前期就会发展成", rendered)
        self.assertIn("糖尿病需要长期管理", rendered)

    def test_ass_caption_uses_cleaned_display_text(self):
        srt = "1\n00:00:00,000 --> 00:00:03,000\n嗯，糖尿病患者啊，不能喝酒。\n"
        with tempfile.TemporaryDirectory() as temporary:
            ass_path = Path(temporary) / "captions.ass"
            count = _write_ass_subtitles(srt, ass_path, [])
            rendered = ass_path.read_text(encoding="utf-8")

        self.assertEqual(count, 1)
        self.assertIn("糖尿病患者，不能喝酒。", rendered)
        self.assertNotIn("嗯，", rendered)
        self.assertIn("Style: Default,STHeiti,86,", rendered)
        self.assertIn(",1,2,4,5,48,48,0,1", rendered)
        self.assertIn(r"\pos(540,1250)", rendered)

    def test_reference_layout_adds_two_colour_title_and_disclaimer(self):
        with tempfile.TemporaryDirectory() as temporary:
            ass_path = Path(temporary) / "reference.ass"
            _write_reference_layout_ass("合理饮食才能更好控制血糖", ass_path, 1080, 1920)
            rendered = ass_path.read_text(encoding="utf-8")

        self.assertEqual(_title_lines("合理饮食才能更好控制血糖"), ("合理饮食才能", "更好控制血糖"))
        self.assertIn(r"\pos(540,222)\fs110\c&H00FFFFFF&", rendered)
        self.assertIn(r"\pos(540,322)\c&H006AF2FF&", rendered)
        self.assertIn(r"\pos(540,1825)", rendered)
        self.assertIn("科学科普 仅供参考\\N身体如有不适请线下就医", rendered)


if __name__ == "__main__":
    unittest.main()
