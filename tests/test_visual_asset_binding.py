import json
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_processing.visual_asset_binding import build_visual_sentences, select_visual_assets
from video_generation.render import describe_visual_asset_events


class VisualAssetBindingTests(unittest.TestCase):
    def test_visual_sentences_only_expose_present_keywords(self):
        srt = "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n"
        self.assertEqual(
            build_visual_sentences(srt, "smoking\nmissing"),
            [{
                "sentence_id": 1,
                "start": "00:00:01,000",
                "end": "00:00:04,000",
                "text": "Avoid smoking.",
                "keywords": ["smoking"],
            }],
        )

    def test_selection_keeps_only_valid_asset_and_keyword(self):
        available = [{"id": "warning", "file_name": "warning.png", "media_type": "image"}]

        def call_llm(_system, _user, content, _key, _model):
            request = json.loads(content)
            self.assertEqual(request["sentences"][0]["keywords"], ["smoking"])
            return json.dumps({"placements": [{
                "sentence_id": 1,
                "use_asset": True,
                "asset_id": "warning",
                "target_word": "smoking",
                "position": "upper_right",
                "duration_seconds": 2,
                "reason": "risk",
            }]})

        with patch("subtitle_processing.visual_asset_binding.VISUAL_ASSET_PROMPT_READY", True), patch(
            "subtitle_processing.visual_asset_binding._available_assets", return_value=available
        ), patch(
            "subtitle_processing.visual_asset_binding._asset_config", return_value={"selection_rules": {}}
        ):
            result = json.loads(select_visual_assets(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n",
                "smoking", "key", "model", call_llm,
            ))
        self.assertEqual(result["placements"][0]["asset_id"], "warning")

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
            return_value={"media_type": "image", "technical_metadata": {"requires_chroma_key": False}},
        ):
            events = describe_visual_asset_events(
                "1\n00:00:01,000 --> 00:00:04,000\nAvoid smoking.\n", bindings
            )
        self.assertEqual(events[0]["asset_id"], "warning")
        self.assertEqual(events[0]["asset_file"], "warning.png")
        self.assertEqual(events[0]["target_word"], "smoking")


if __name__ == "__main__":
    unittest.main()
