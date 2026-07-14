import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_launch_importable import MARKER, transform_source


SOURCE = '''import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if args.listen:
        funclip_service.launch()
    else:
        funclip_service.launch()
'''


class TestMakeLaunchImportable(unittest.TestCase):
    def test_transform_is_idempotent_and_guards_launch(self):
        transformed, changed = transform_source(SOURCE)
        self.assertTrue(changed)
        self.assertIn(MARKER, transformed)
        self.assertIn('parser.parse_args() if __name__ == "__main__"', transformed)
        self.assertIn('if __name__ == "__main__":\n        if args.listen:', transformed)
        self.assertEqual(transform_source(transformed), (transformed, False))


if __name__ == "__main__":
    unittest.main()
