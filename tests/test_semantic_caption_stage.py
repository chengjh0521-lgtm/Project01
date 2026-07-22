import unittest

from subtitle_processing.semantic_caption_stage import SemanticCaptionError, segment_highlight_cues


class SemanticCaptionStageTests(unittest.TestCase):
    def setUp(self):
        self.cues = [{
            "start": "00:00:01,000",
            "end": "00:00:05,000",
            "text": "糖尿病患者一定要合理控制饮食",
        }]

    def test_splits_at_llm_selected_semantic_boundaries(self):
        result = segment_highlight_cues(
            self.cues,
            lambda _system, _user: '{"entries":[{"id":"1","segments":["糖尿病患者", "一定要合理", "控制饮食"]}]}'
        )

        self.assertEqual([item["text"] for item in result], ["糖尿病患者", "一定要合理", "控制饮食"])
        self.assertEqual(result[0]["start"], "00:00:01,000")
        self.assertEqual(result[-1]["end"], "00:00:05,000")
        self.assertTrue(all(len(item["text"]) <= 14 for item in result))

    def test_rejects_rewritten_subtitle_text(self):
        with self.assertRaises(SemanticCaptionError):
            segment_highlight_cues(
                self.cues,
                lambda _system, _user: '{"entries":[{"id":"1","segments":["糖尿病患者", "必须控制饮食"]}]}'
            )


if __name__ == "__main__":
    unittest.main()
