import unittest

from video_generation.render import _wrap_caption_two_lines


class CaptionWrappingTests(unittest.TestCase):
    def test_keeps_short_captions_on_one_line(self):
        self.assertEqual(_wrap_caption_two_lines("糖尿病能喝酒吗？"), "糖尿病能喝酒吗？")

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


if __name__ == "__main__":
    unittest.main()
