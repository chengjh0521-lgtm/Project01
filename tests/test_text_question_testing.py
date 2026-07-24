import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text_question_testing.pipeline import TextQuestionTestError, _ranges, run_text_question_test


SRT = (
    "1\n00:00:00,000 --> 00:00:45,000\n第一段完整回答。\n\n"
    "2\n00:00:45,000 --> 00:01:30,000\n第二段完整回答。\n"
)


class TextQuestionTestingTests(unittest.TestCase):
    def test_rejects_answer_duration_outside_the_strict_window(self):
        raw = json.dumps({"ranges": [{"start": "00:00:00,000", "end": "00:00:30,000"}]})
        with self.assertRaises(TextQuestionTestError):
            _ranges(raw, {"00:00:00,000"}, {"00:00:30,000"})

    def test_writes_one_markdown_report_with_ten_questions_and_answers(self):
        questions = {
            "questions": [
                {"id": "q{:02d}".format(index), "question": "问题{}？".format(index), "reason": "有吸引力"}
                for index in range(1, 11)
            ]
        }
        answer = {
            "ranges": [{"start": "00:00:00,000", "end": "00:00:45,000"}],
            "answer_summary": "完整回答。",
            "reason": "结论清晰。",
        }

        responses = iter([json.dumps(questions, ensure_ascii=False)] + [
            json.dumps(answer, ensure_ascii=False) for _ in range(10)
        ])
        with tempfile.TemporaryDirectory() as temporary, patch(
            "text_question_testing.pipeline._call_deepseek", side_effect=lambda *_args, **_kwargs: next(responses)
        ):
            path, markdown = run_text_question_test(SRT, "saved.srt", "key", "model", temporary)

            self.assertTrue(Path(path).is_file())
            self.assertEqual(markdown.count("\n## "), 10)
            self.assertIn("回答片段总时长：45.00 秒", markdown)


if __name__ == "__main__":
    unittest.main()
