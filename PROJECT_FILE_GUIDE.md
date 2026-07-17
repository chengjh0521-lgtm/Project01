# Project File Guide

This document describes the maintained files in the module gateway. Generated
directories such as `.venv/`, `runtime/`, `output/`, `__pycache__/`, and user
media directories are deliberately excluded.

## Root Files

| File | Purpose |
| --- | --- |
| `.env.example` | Server environment-variable template. It contains paths and placeholders only; real API keys belong in the ignored `.env`. |
| `.gitignore` | Keeps virtual environments, generated runtime, model output, user videos, local settings, and generated asset bindings out of Git. |
| `.gitmodules` | Declares `upstream_funclip` as the official FunClip source submodule. |
| `README.md` | Deployment, runtime preparation, environment configuration, and Python API usage guide. |
| `SECOND_MODULE_LLM_PROMPTS.md` | Human-readable copy of the current subtitle-correction, highlight, keyword, sound, and visual-asset prompts. |
| `background_jobs.py` | Single-worker background job manager used by the web UI for ASR, processing, rendering, and label jobs. |
| `funclip_loader.py` | Imports the patched runtime copy of upstream `launch.py` once per process, without launching the upstream Gradio UI. |
| `make_launch_importable.py` | Applies the reusable import-safety patch to a stock upstream `launch.py`. |
| `prepare_funclip_runtime.py` | Copies the upstream submodule into ignored `runtime/` and applies the import-safety patch. |
| `requirements-web.txt` | Web-layer dependency pins, including Gradio, Hugging Face Hub compatibility, and NumPy compatibility. |
| `start_module_web.sh` | Linux service entry point. Loads `.env`, exports runtime settings, and starts `web_app.py`. |
| `web_app.py` | Gradio application, browser state recovery, server video selection, background-job polling, and the four visible workflow buttons. |

## Subtitle Generation

| File | Purpose |
| --- | --- |
| `subtitle_generation/__init__.py` | Public export for subtitle generation. |
| `subtitle_generation/asr.py` | Calls the upstream ASR callback and returns ASR text, SRT, and video state. |
| `subtitle_generation/pending_videos/.gitkeep` | Keeps the otherwise empty server-side input-video directory in Git. |

## Subtitle Processing

| File | Purpose |
| --- | --- |
| `subtitle_processing/__init__.py` | Public exports for processing a fresh or previously corrected SRT. |
| `subtitle_processing/correction_stage.py` | Small stage contract that validates a correction result before later stages use it. |
| `subtitle_processing/local_correction_engine.py` | Parser, batching, context construction, and strict response validation copied from the proven standalone correction workflow. It never stores an API key. |
| `subtitle_processing/pipeline.py` | Main five-stage orchestration: correction, low-overlap highlights, keywords, sound decisions, visual-asset decisions, and corrected video-state construction. |
| `subtitle_processing/multi_highlight_stage.py` | Iteratively chooses publishable highlight clips while enforcing a maximum 30 percent overlap with prior clips. |
| `subtitle_processing/keyword_stage.py` | Selects and validates display keywords from one highlight SRT, with optional reasons. |
| `subtitle_processing/sound_effect_binding.py` | Loads sound configuration, persists operator descriptions, calls the sound director prompt, validates cues, and resolves sound files. |
| `subtitle_processing/visual_asset_binding.py` | Loads the GIF/PNG asset index, calls the visual director prompt, validates placements, and resolves source files. |

## Video Generation

| File | Purpose |
| --- | --- |
| `video_generation/__init__.py` | Public exports for rendering, reports, sound/visual inspection, and fixed doctor labels. |
| `video_generation/render.py` | Clips upstream video ranges, burns SRT captions and keyword highlights, places GIF/PNG assets, mixes sound effects, and invokes the final doctor-label step. |
| `video_generation/doctor_label.py` | Independent fourth-module implementation that permanently overlays `label.png` above the final video. |
| `video_generation/label.png` | The doctor/expert label image used by the fixed label module. |
| `video_generation/report.py` | Creates the downloadable Markdown audit report containing corrected captions and LLM decisions. |

## Visual Asset Index

| File | Purpose |
| --- | --- |
| `visual_assets/picture_assets_index.json` | Describes permitted GIF/PNG assets, their meaning, recommended and forbidden scenes, and technical metadata. Asset media files themselves are server-local and ignored by Git. |

## Tests

| File | Purpose |
| --- | --- |
| `tests/test_fixed_doctor_label.py` | Verifies fixed-label FFmpeg construction and explicit failure behavior. |
| `tests/test_generation_report.py` | Verifies the Markdown report includes correction, highlight, keyword, sound, and visual decisions. |
| `tests/test_multi_highlight_stage.py` | Verifies retry handling and overlap limits for multi-clip selection. |
| `tests/test_saved_subtitle_pipeline.py` | Verifies resuming from a saved corrected SRT bypasses another correction call. |
| `tests/test_sound_effect_binding.py` | Verifies sound configuration, LLM decision parsing, and persisted descriptions. |
| `tests/test_sound_effect_logic.py` | Verifies render-time sound-effect placement logic. |
| `tests/test_visual_asset_binding.py` | Verifies visual asset selection, validation, file resolution, and minimum coverage retry behavior. |

## Upstream FunClip Submodule

`upstream_funclip/` is not old gateway code. It is the unmodified official
FunClip source used to rebuild `runtime/` after an upstream update. Do not
delete individual files inside it; update it through Git submodule commands.

| File or directory | Purpose |
| --- | --- |
| `LICENSE`, `README.md`, `README_zh.md`, `requirements.txt` | Upstream license, documentation, and original dependency list. |
| `font/STHeitiMedium.ttc` | Upstream Chinese caption font copied into the runtime when available. |
| `funclip/__init__.py`, `introduction.py`, `launch.py`, `videoclipper.py` | Upstream package entry points, interface text, Gradio callbacks, and core clipping logic. |
| `funclip/llm/demo_prompt.py` | Upstream prompt example. |
| `funclip/llm/g4f_openai_api.py`, `openai_api.py`, `qwen_api.py`, `twelvelabs_api.py` | Upstream optional LLM-provider adapters. |
| `funclip/utils/argparse_tools.py`, `subtitle_utils.py`, `theme.json`, `trans_utils.py` | Upstream argument parsing, subtitle helpers, UI theme, and translation helpers. |
| `funclip/test/imagemagick_test.py`, `funclip/test/test.sh` | Upstream environment checks. |
| `tests/test_recognition_result_compat.py`, `tests/test_twelvelabs_pegasus.py` | Upstream regression tests. |
| `docs/images/demo.png`, `demo_en.png`, `dingding.png`, `guide.jpg`, `interface.jpg`, `LLM_guide.png`, `wechat.png` | Upstream README/documentation images. |

## Removed Legacy File

`subtitle_processing/highlights.py` was removed. It only made the former
single-call upstream LLM highlight request and is superseded by
`multi_highlight_stage.py` plus `pipeline.py`.
