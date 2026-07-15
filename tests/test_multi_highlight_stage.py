import unittest

from subtitle_processing.multi_highlight_stage import parse_ranges, select_multiple


class MultiHighlightStageTests(unittest.TestCase):
    def test_parse_ranges_accepts_json_wrapped_by_text(self):
        response = 'Result: {"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}]}'
        self.assertEqual(parse_ranges(response), [("00:00:01,000", "00:00:42,000")])

    def test_parse_ranges_accepts_chinese_dash(self):
        self.assertEqual(
            parse_ranges("[00:00:01,000\u201400:00:42,000]"),
            [("00:00:01,000", "00:00:42,000")],
        )

    def test_select_multiple_retries_after_invalid_model_response(self):
        responses = iter([
            "I cannot choose a clip.",
            '{"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}]}',
        ])

        selected = select_multiple("1\n00:00:01,000 --> 00:00:42,000\ntext\n", 1, lambda *_: next(responses))

        self.assertEqual(selected[0]["ranges"], [("00:00:01,000", "00:00:42,000")])


if __name__ == "__main__":
    unittest.main()
