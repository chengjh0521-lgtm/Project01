import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.render import describe_sound_effect_events


class SoundEffectLogicTests(unittest.TestCase):
    def test_exported_event_uses_the_target_word_position(self):
        srt = "1\n00:00:10,000 --> 00:00:14,000\nDo not smoke.\n"
        bindings = '{"cues":[{"sound_id":"alert.wav","text":"smoke","reason":"risk","sentence_id":1}]}'

        with patch("video_generation.render.resolve_sound_effect_file", return_value=Path("alert.wav")):
            events = describe_sound_effect_events(srt, bindings)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["offset_ms"], 12_154)
        self.assertEqual(events[0]["timestamp"], "00:00:12,154")
        self.assertEqual(events[0]["target_word"], "smoke")
        self.assertEqual(events[0]["sound_file"], "alert.wav")


if __name__ == "__main__":
    unittest.main()
