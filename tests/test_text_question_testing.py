import json
import tempfile
import threading
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

    def test_preserves_model_narrative_order_instead_of_sorting_timestamps(self):
        raw = json.dumps({"ranges": [
            {"start": "00:00:45,000", "end": "00:01:30,000"},
            {"start": "00:00:00,000", "end": "00:00:45,000"},
        ]})

        ranges, _, _ = _ranges(
            raw,
            {"00:00:00,000", "00:00:45,000"},
            {"00:00:45,000", "00:01:30,000"},
        )

        self.assertEqual(ranges[0], ("00:00:45,000", "00:01:30,000"))

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

    def test_answer_requests_run_in_parallel(self):
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
        lock = threading.Lock()
        answer_calls = 0
        second_answer_started = threading.Event()

        def call_llm(*args, **_kwargs):
            nonlocal answer_calls
            stage = str(args[5])
            if stage.startswith("text-question"):
                return json.dumps(questions, ensure_ascii=False)
            with lock:
                answer_calls += 1
                first_answer = answer_calls == 1
                if answer_calls == 2:
                    second_answer_started.set()
            if first_answer and not second_answer_started.wait(timeout=2):
                raise AssertionError("回答请求没有并行执行。")
            return json.dumps(answer, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "text_question_testing.pipeline._call_deepseek", side_effect=call_llm
        ):
            _, markdown = run_text_question_test(SRT, "saved.srt", "key", "model", temporary)

        self.assertEqual(markdown.count("\n## "), 10)


if __name__ == "__main__":
    unittest.main()
