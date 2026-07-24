import unittest
import threading
from unittest.mock import patch

from subtitle_processing.pipeline import process_from_corrected_subtitles


class SavedSubtitlePipelineTests(unittest.TestCase):
    def test_saved_subtitle_skips_correction_and_reuses_downstream_pipeline(self):
        corrected_srt = "1\n00:00:01,000 --> 00:00:04,000\nCorrected text.\n"
        candidate = {
            "id": "clip_01",
            "question": "What does the corrected text explain?",
            "ranges": [("00:00:01,000", "00:00:04,000")],
            "raw_result": "selected",
        }
        status = []

        with patch("subtitle_processing.pipeline.select_multiple", return_value=[candidate]), patch(
            "subtitle_processing.pipeline.build_highlight_srt", return_value=corrected_srt
        ), patch(
            "subtitle_processing.pipeline.build_semantic_highlight_srt", return_value=corrected_srt
        ), patch("subtitle_processing.pipeline.select_keywords_for_clip", return_value="Corrected"), patch(
            "subtitle_processing.pipeline.select_sound_cues", return_value='{"cues": []}'
        ), patch("subtitle_processing.pipeline.select_visual_assets", return_value='{"placements": []}'):
            result = process_from_corrected_subtitles(
                corrected_srt, "key", 1, 1, status_callback=lambda message: status.append(message)
            )

        self.assertEqual(result[0], corrected_srt)
        self.assertEqual(result[2]["clips"][0]["keywords"], "Corrected")
        self.assertIn("What does the corrected text explain?", result[1])
        self.assertIn("跳过 ASR 与洗稿", status[0])

    def test_independent_post_highlight_stages_run_in_parallel(self):
        corrected_srt = "1\n00:00:01,000 --> 00:00:45,000\nCorrected text.\n"
        candidates = [
            {"id": "clip_01", "question": "Question one?", "ranges": [("00:00:01,000", "00:00:45,000")]},
            {"id": "clip_02", "question": "Question two?", "ranges": [("00:00:01,000", "00:00:45,000")]},
        ]
        barrier = threading.Barrier(2)

        def semantic_stage(*_args, **_kwargs):
            barrier.wait(timeout=2)
            return corrected_srt

        with patch("subtitle_processing.pipeline.select_multiple", return_value=candidates), patch(
            "subtitle_processing.pipeline.build_highlight_srt", return_value=corrected_srt
        ), patch("subtitle_processing.pipeline.build_semantic_highlight_srt", side_effect=semantic_stage), patch(
            "subtitle_processing.pipeline.select_keywords_for_clip", return_value="Corrected"
        ), patch("subtitle_processing.pipeline.select_sound_cues", return_value='{"cues": []}'), patch(
            "subtitle_processing.pipeline.select_visual_assets", return_value='{"placements": []}'
        ):
            result = process_from_corrected_subtitles(corrected_srt, "key", 1, 2)

        self.assertEqual(len(result[2]["clips"]), 2)
        self.assertEqual(result[2]["clips"][0]["keywords"], "Corrected")
        self.assertEqual(result[2]["clips"][1]["keywords"], "Corrected")


if __name__ == "__main__":
    unittest.main()
