import unittest

from subtitle_processing.sound_effect_binding import _highlight_sentences


class SoundEffectBindingTests(unittest.TestCase):
    def test_sentences_only_receive_keywords_present_in_their_text(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:03,000\n"
            "Do not smoke.\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\n"
            "Continue the examination.\n"
        )

        sentences = _highlight_sentences(srt, "smoke\nexamination\nmissing")

        self.assertEqual(
            sentences,
            [
                {
                    "sentence_id": 1,
                    "start": "00:00:01,000",
                    "text": "Do not smoke.",
                    "keywords": ["smoke"],
                },
                {
                    "sentence_id": 2,
                    "start": "00:00:03,000",
                    "text": "Continue the examination.",
                    "keywords": ["examination"],
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
