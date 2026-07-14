import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_launch_importable import PATCH_MARKER, transform_source


SOURCE = '''import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    funclip_service = object()
    if args.listen:
        funclip_service.launch(server_name="0.0.0.0")
    else:
        funclip_service.launch()
'''


class TestMakeLaunchImportable(unittest.TestCase):
    def test_makes_callbacks_importable_and_keeps_launch_as_script_only(self):
        transformed, changed = transform_source(SOURCE)

        self.assertTrue(changed)
        self.assertIn(PATCH_MARKER, transformed)
        self.assertIn('parser.parse_args() if __name__ == "__main__"', transformed)
        self.assertIn('if __name__ == "__main__":\n        if args.listen:', transformed)

        transformed_again, changed_again = transform_source(transformed)
        self.assertFalse(changed_again)
        self.assertEqual(transformed_again, transformed)


if __name__ == "__main__":
    unittest.main()
