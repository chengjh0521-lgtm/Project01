import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_processing.visual_asset_binding import _available_assets, build_visual_sentences, select_visual_assets
from video_generation.render import describe_visual_asset_events


class VisualAssetBindingTests(unittest.TestCase):
    def test_updated_array_index_decodes_file_name_and_green_screen_gif(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_directory = root / "assets"
            asset_directory.mkdir()
            (asset_directory / "7月16日.gif").write_bytes(b"GIF89a")
            config_path = root / "picture_assets_index.json"
            config_path.write_text(json.dumps([{
                "id": "asset_001",
                "file_name": "7#U670816#U65e5.gif",
                "description": "绿幕动图，显示注意提示。",
                "main_content": "注意文字。",
                "recommended_scenes": "风险提醒。",
                "disabled_scenes": "普通过渡。",
                "size": "426x240",
                "duration_seconds": 5.2,
            }], ensure_ascii=False), encoding="utf-8")
            with patch.dict(os.environ, {
                "FUNCLIP_VISUAL_ASSET_CONFIG": str(config_path),
                "FUNCLIP_VISUAL_ASSET_DIR": str(asset_directory),
            }, clear=False):
                assets = _available_assets()

        self.assertEqual(assets[0]["file_name"], "7月16日.gif")
        self.assertEqual(assets[0]["media_type"], "animated_gif")
        self.assertEqual(assets[0]["duration_seconds"], 5.2)
        self.assertTrue(assets[0]["technical_metadata"]["requires_chroma_key"])

    def test_static_images_keep_the_configured_point_two_second_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_directory = root / "assets"
            asset_directory.mkdir()
            (asset_directory / "meal.png").write_bytes(b"PNG")
            config_path = root / "picture_assets_index.json"
            config_path.write_text(json.dumps([{
                "id": "asset_image",
                "file_name": "meal.png",
                "description": "A balanced meal.",
                "main_content": "Meal.",
                "recommended_scenes": "Diet.",
                "disabled_scenes": "None.",
                "size": "100x100",
                "duration_seconds": 0.2,
            }]), encoding="utf-8")
            with patch.dict(os.environ, {
                "FUNCLIP_VISUAL_ASSET_CONFIG": str(config_path),
                "FUNCLIP_VISUAL_ASSET_DIR": str(asset_directory),
            }, clear=False):
                assets = _available_assets()

        self.assertEqual(assets[0]["media_type"], "image")
        self.assertEqual(assets[0]["duration_seconds"], 0.2)

    def test_visual_sentences_only_expose_present_keywords(self):
        srt = "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n"
        self.assertEqual(
            build_visual_sentences(srt, "smoking\nmissing"),
            [{
                "sentence_id": 1,
                "start": "00:00:01,000",
                "end": "00:00:04,000",
                "text": "Avoid smoking.",
                "keywords": [{"word": "smoking", "start": 2.286, "end": 3.786}],
            }],
        )

    def test_selection_keeps_only_valid_asset_and_keyword(self):
        available = [{"id": "warning", "file_name": "warning.png", "media_type": "image"}]

        def call_llm(_system, _user, content, _key, _model):
            request = json.loads(content)
            self.assertEqual(
                request["sentences"][0]["keywords"],
                [{"word": "smoking", "start": 2.286, "end": 3.786}],
            )
            return json.dumps({"results": [{
                "sentence_id": 1,
                "use_asset": True,
                "asset_id": "warning",
                "target_word": "smoking",
                "duration_seconds": 1.2,
                "reason": "risk",
            }]})

        with patch("subtitle_processing.visual_asset_binding.VISUAL_ASSET_PROMPT_READY", True), patch(
            "subtitle_processing.visual_asset_binding._available_assets", return_value=available
        ), patch(
            "subtitle_processing.visual_asset_binding._asset_config", return_value={"selection_rules": {}}
        ), patch(
            "subtitle_processing.visual_asset_binding.get_visual_asset_definition", return_value={"media_type": "image"}
        ):
            result = json.loads(select_visual_assets(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n",
                "smoking", "key", "model", call_llm,
            ))
        self.assertEqual(result["placements"][0]["asset_id"], "warning")
        self.assertEqual(result["placements"][0]["duration_seconds"], 1.2)

    def test_selection_keeps_valid_placement_when_model_omits_other_sentences(self):
        available = [{"id": "warning", "file_name": "warning.png", "media_type": "image"}]

        def call_llm(_system, _user, _content, _key, _model):
            return json.dumps({"results": [{
                "sentence_id": 1,
                "use_asset": True,
                "asset_id": "warning",
                "target_word": "smoking",
                "duration_seconds": 1.0,
                "reason": "risk",
            }]})

        with patch("subtitle_processing.visual_asset_binding._available_assets", return_value=available), patch(
            "subtitle_processing.visual_asset_binding._asset_config", return_value={"selection_rules": {}}
        ), patch(
            "subtitle_processing.visual_asset_binding.get_visual_asset_definition", return_value={"duration_seconds": 2}
        ):
            result = json.loads(select_visual_assets(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n\n"
                "2\n00:00:04,000 --> 00:00:07,000\nGeneral advice.\n",
                "smoking", "key", "model", call_llm,
            ))

        self.assertEqual(len(result["placements"]), 1)
        self.assertEqual(result["placements"][0]["sentence_id"], 1)

    def test_selection_retries_until_it_meets_global_visual_coverage_targets(self):
        available = [
            {"id": "warning", "file_name": "warning.png", "media_type": "image"},
            {"id": "food", "file_name": "food.png", "media_type": "image"},
        ]
        responses = iter([
            {"results": [{
                "sentence_id": 1,
                "use_asset": True,
                "asset_id": "warning",
                "target_word": "smoking",
                "duration_seconds": 1.0,
            }]},
            {"results": [
                {
                    "sentence_id": 1,
                    "use_asset": True,
                    "asset_id": "warning",
                    "target_word": "smoking",
                    "duration_seconds": 2.7,
                },
                {
                    "sentence_id": 2,
                    "use_asset": True,
                    "asset_id": "food",
                    "target_word": "food",
                    "duration_seconds": 2.7,
                },
            ]},
        ])

        def call_llm(_system, _user, _content, _key, _model):
            return json.dumps(next(responses))

        with patch("subtitle_processing.visual_asset_binding._available_assets", return_value=available), patch(
            "subtitle_processing.visual_asset_binding._asset_config", return_value={"selection_rules": {}}
        ), patch(
            "subtitle_processing.visual_asset_binding.get_visual_asset_definition", return_value={"media_type": "image", "duration_seconds": 0.2}
        ):
            result = json.loads(select_visual_assets(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n\n"
                "2\n00:00:04,000 --> 00:00:07,000\nChoose food.\n",
                "smoking, food", "key", "model", call_llm,
            ))

        self.assertEqual(len(result["placements"]), 2)
        self.assertGreater(sum(item["duration_seconds"] for item in result["placements"]), 5.0)

    def test_render_event_uses_selected_sentence_and_asset_metadata(self):
        bindings = json.dumps({"placements": [{
            "sentence_id": 1,
            "asset_id": "warning",
            "target_word": "smoking",
            "position": "upper_right",
            "duration_seconds": 2,
            "reason": "risk",
        }]})
        with patch("video_generation.render.resolve_visual_asset_file", return_value=Path("warning.png")), patch(
            "video_generation.render.get_visual_asset_definition",
            return_value={
                "media_type": "image",
                "duration_seconds": 0.2,
                "technical_metadata": {"requires_chroma_key": False},
            },
        ):
            events = describe_visual_asset_events(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n", bindings
            )
        self.assertEqual(events[0]["asset_id"], "warning")
        self.assertEqual(events[0]["asset_file"], "warning.png")
        self.assertEqual(events[0]["target_word"], "smoking")
        self.assertEqual(events[0]["duration_seconds"], 2.0)

    def test_render_event_never_uses_less_than_the_asset_minimum_duration(self):
        bindings = json.dumps({"placements": [{
            "sentence_id": 1,
            "asset_id": "warning",
            "target_word": "smoking",
            "duration_seconds": 0.1,
        }]})
        with patch("video_generation.render.resolve_visual_asset_file", return_value=Path("warning.png")), patch(
            "video_generation.render.get_visual_asset_definition",
            return_value={"media_type": "image", "duration_seconds": 0.2},
        ):
            events = describe_visual_asset_events(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n", bindings
            )
        self.assertEqual(events[0]["duration_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
