import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_generation.render import _overlay_visual_assets


class VisualAssetRenderingTests(unittest.TestCase):
    def test_transparent_assets_are_composited_in_rgba_before_final_yuv_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "clip.mp4"
            asset = root / "transparent.png"
            output = root / "clip_visual.mp4"
            source.write_bytes(b"video")
            asset.write_bytes(b"png")

            events = [{
                "asset_id": "transparent_asset",
                "duration_seconds": 1.0,
                "media_type": "image",
                "offset_ms": 0,
                "requires_chroma_key": False,
                "position": "caption_lower_left",
            }]
            with patch("video_generation.render._visual_asset_events", return_value=events), patch(
                "video_generation.render.resolve_visual_asset_file", return_value=asset
            ), patch("video_generation.render._asset_pixel_format", return_value="rgba"
            ), patch("video_generation.render.shutil.which", return_value="ffmpeg"), patch(
                "video_generation.render.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                run.return_value.stderr = ""
                result, count = _overlay_visual_assets(source, "unused", "{}")

        filters = run.call_args.args[0][run.call_args.args[0].index("-filter_complex") + 1]
        self.assertEqual(Path(result).name, output.name)
        self.assertEqual(count, 1)
        self.assertIn("[0:v]format=rgba[base_rgba]", filters)
        self.assertIn("[1:v]format=rgba,split=2[asset_rgb_src1][asset_alpha_src1]", filters)
        self.assertIn("[asset_alpha_src1]alphaextract,scale=260:-1:flags=lanczos[asset_alpha1]", filters)
        self.assertIn("[asset_rgb1][asset_alpha1]alphamerge,format=rgba,setsar=1", filters)
        self.assertIn("overlay=x=W*0.22:y=H*0.70:eof_action=pass:shortest=0:format=rgb:alpha=straight", filters)
        self.assertIn("format=yuv420p[visual_yuv]", filters)


if __name__ == "__main__":
    unittest.main()
