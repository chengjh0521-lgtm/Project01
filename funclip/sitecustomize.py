"""Runtime compatibility patch for sound-effect trigger timing.

This file is imported automatically when the app is started with
`python funclip/launch.py`. It keeps older deployed branches able to time sound
effects to the trigger word position without replacing the large videoclipper.py
file through GitHub's contents API.
"""

import importlib.abc
import importlib.machinery
import logging
import sys


def _sound_effect_trigger_times(start, end, text, terms):
    text = str(text or "")
    text_lower = text.lower()
    if not text_lower:
        return []

    duration = max(0.0, float(end) - float(start))
    if duration <= 0:
        return []

    triggers = []
    seen = set()
    text_len = max(1, len(text_lower))
    for term in terms:
        term_lower = str(term or "").strip().lower()
        if not term_lower:
            continue

        search_from = 0
        while True:
            index = text_lower.find(term_lower, search_from)
            if index < 0:
                break

            key = (index, term_lower)
            if key not in seen:
                seen.add(key)
                ratio = max(0.0, min(1.0, index / text_len))
                triggers.append((float(start) + duration * ratio, term_lower))

            search_from = index + max(1, len(term_lower))

    return sorted(triggers, key=lambda item: item[0])


def _patch_videoclipper(module):
    if getattr(module, "_sound_effect_word_timing_patch", False):
        return
    if not hasattr(module, "_parse_sound_effect_rules"):
        return

    def _with_sound_effects(video_clip, subs, sound_effect_rules=None, sound_effect_dir=None):
        rules = module._parse_sound_effect_rules(sound_effect_rules, sound_effect_dir)
        if not rules or not subs:
            return video_clip, 0

        audio_clips = []
        if video_clip.audio is not None:
            audio_clips.append(video_clip.audio)

        trigger_count = 0
        last_trigger_by_path = {}
        for (start, end), text in subs:
            for rule in rules:
                for trigger_start, _ in _sound_effect_trigger_times(start, end, text, rule["terms"]):
                    last_trigger = last_trigger_by_path.get(rule["path"])
                    if last_trigger is not None and trigger_start - last_trigger < rule["cooldown"]:
                        continue
                    if trigger_start >= video_clip.duration:
                        continue

                    available_duration = video_clip.duration - trigger_start
                    if available_duration <= 0.05:
                        continue
                    try:
                        effect_clip = module.AudioFileClip(rule["path"]).volumex(rule["volume"])
                    except Exception:
                        logging.exception("Failed to load sound effect: %s", rule["path"])
                        continue
                    if effect_clip.duration > available_duration:
                        effect_clip = effect_clip.subclip(0, available_duration)
                    audio_clips.append(effect_clip.set_start(trigger_start))
                    last_trigger_by_path[rule["path"]] = trigger_start
                    trigger_count += 1

        if trigger_count == 0:
            return video_clip, 0
        return video_clip.set_audio(module.CompositeAudioClip(audio_clips)), trigger_count

    module._sound_effect_trigger_times = _sound_effect_trigger_times
    module._with_sound_effects = _with_sound_effects
    module._sound_effect_word_timing_patch = True


class _VideoClipperPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_videoclipper(module)


class _VideoClipperPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "videoclipper":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _VideoClipperPatchLoader(spec.loader)
        return spec


if "videoclipper" in sys.modules:
    _patch_videoclipper(sys.modules["videoclipper"])
else:
    sys.meta_path.insert(0, _VideoClipperPatchFinder())
