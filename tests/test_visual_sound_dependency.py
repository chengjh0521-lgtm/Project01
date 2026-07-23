import json
import unittest

from subtitle_processing.sound_effect_binding import _visual_bound_sentences


class VisualSoundDependencyTests(unittest.TestCase):
    def test_only_visual_bound_keywords_are_sound_candidates(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,000\nDo not smoke.\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nContinue the examination.\n"
        )
        visual_bindings = json.dumps({"placements": [{
            "sentence_id": 1,
            "asset_id": "warning_card",
            "target_word": "smoke",
        }]})

        self.assertEqual(
            _visual_bound_sentences(srt, "smoke\nexamination", visual_bindings),
            [{
                "sentence_id": 1,
                "start": "00:00:01,000",
                "text": "Do not smoke.",
                "keywords": ["smoke"],
            }],
        )


if __name__ == "__main__":
    unittest.main()
