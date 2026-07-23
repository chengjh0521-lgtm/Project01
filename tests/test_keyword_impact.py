import json
import unittest

from subtitle_processing.keyword_stage import select_keywords


class KeywordImpactTests(unittest.TestCase):
    def test_keeps_only_explicit_impact_keywords(self):
        raw = json.dumps({"keywords": [
            {"word": "糖尿病", "reason": "疾病诊断", "impact": True},
            {"word": "控制血糖", "reason": "普通建议", "impact": False},
        ]}, ensure_ascii=False)

        selected, details = select_keywords(
            "糖前期会发展成糖尿病，需要控制血糖。",
            8,
            lambda _system, _user: raw,
            include_reasons=True,
        )

        self.assertEqual(selected.splitlines(), ["糖尿病", "控制血糖"])
        self.assertEqual([item["impact"] for item in details], [True, False])


if __name__ == "__main__":
    unittest.main()
