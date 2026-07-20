import unittest

from subtitle_processing.multi_highlight_stage import parse_highlight_selection, parse_ranges, select_multiple


class MultiHighlightStageTests(unittest.TestCase):
    def test_parse_ranges_accepts_json_wrapped_by_text(self):
        response = 'Result: {"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}]}'
        self.assertEqual(parse_ranges(response), [("00:00:01,000", "00:00:42,000")])

    def test_parse_ranges_accepts_chinese_dash(self):
        self.assertEqual(
            parse_ranges("[00:00:01,000\u201400:00:42,000]"),
            [("00:00:01,000", "00:00:42,000")],
        )

    def test_parse_highlight_selection_keeps_the_question_and_reason(self):
        response = (
            '{"question":"Can diabetes patients drink alcohol?",'
            '"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}],'
            '"reason":"Clear recommendation."}'
        )
        self.assertEqual(
            parse_highlight_selection(response),
            {
                "question": "Can diabetes patients drink alcohol?",
                "ranges": [("00:00:01,000", "00:00:42,000")],
                "reason": "Clear recommendation.",
            },
        )

    def test_select_multiple_retries_after_invalid_model_response(self):
        responses = iter([
            "I cannot choose a clip.",
            '{"question":"Can diabetes patients drink alcohol?",'
            '"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}],"reason":"Clear answer."}',
        ])

        selected = select_multiple("1\n00:00:01,000 --> 00:00:42,000\ntext\n", 1, lambda *_: next(responses))

        self.assertEqual(selected[0]["ranges"], [("00:00:01,000", "00:00:42,000")])
        self.assertEqual(selected[0]["question"], "Can diabetes patients drink alcohol?")

    def test_select_multiple_retries_when_a_question_is_missing(self):
        responses = iter([
            '{"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}]}',
            '{"question":"What should patients do?",'
            '"ranges":[{"start":"00:00:01,000","end":"00:00:42,000"}],"reason":"Clear answer."}',
        ])

        selected = select_multiple("1\n00:00:01,000 --> 00:00:42,000\ntext\n", 1, lambda *_: next(responses))

        self.assertEqual(selected[0]["question"], "What should patients do?")


if __name__ == "__main__":
    unittest.main()
