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


if __name__ == "__main__":
    unittest.main()
