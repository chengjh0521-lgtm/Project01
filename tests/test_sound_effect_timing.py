import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "funclip"))

from videoclipper import _sound_effect_trigger_times


class TestSoundEffectTiming(unittest.TestCase):
    def test_trigger_uses_term_position_inside_subtitle(self):
        triggers = _sound_effect_trigger_times(
            10.0,
            20.0,
            "before keyword after",
            ["keyword"],
        )

        self.assertEqual(len(triggers), 1)
        self.assertGreater(triggers[0][0], 13.0)
        self.assertLess(triggers[0][0], 15.0)

    def test_multiple_occurrences_can_trigger_inside_one_subtitle(self):
        triggers = _sound_effect_trigger_times(
            0.0,
            8.0,
            "risk text risk text",
            ["risk"],
        )

        self.assertEqual(len(triggers), 2)
        self.assertEqual(triggers[0][0], 0.0)
        self.assertGreater(triggers[1][0], 3.0)

    def test_token_times_take_priority_over_subtitle_estimate(self):
        triggers = _sound_effect_trigger_times(
            0.0,
            10.0,
            "糖尿病要戒烟",
            ["戒烟"],
            [
                {"text": "糖", "start": 0.0, "end": 0.2},
                {"text": "尿", "start": 0.2, "end": 0.4},
                {"text": "病", "start": 0.4, "end": 0.6},
                {"text": "要", "start": 4.0, "end": 4.2},
                {"text": "戒", "start": 7.0, "end": 7.2},
                {"text": "烟", "start": 7.2, "end": 7.4},
            ],
        )

        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0][0], 7.0)


if __name__ == "__main__":
    unittest.main()
