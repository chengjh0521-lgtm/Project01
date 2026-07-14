"""Load an importable upstream FunClip launch module once per process."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path


_LOCK = threading.Lock()
_MODULE = None


def get_launch(launch_dir: str | Path | None = None):
    """Import the patched launch.py without starting its Gradio server."""
    global _MODULE
    with _LOCK:
        if _MODULE is not None:
            return _MODULE

        root = Path(launch_dir or os.environ.get("FUNCLIP_LAUNCH_DIR", "")).expanduser()
        if not root.is_dir() or not (root / "launch.py").is_file():
            raise RuntimeError(
                "Set FUNCLIP_LAUNCH_DIR to the funclip directory containing the "
                "patched upstream launch.py."
            )

        root_text = str(root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        # Upstream launch.py uses paths such as ``funclip/utils/theme.json``.
        # It therefore has to be imported from the project root, while its own
        # directory still needs to be first on sys.path for sibling imports.
        project_root = root.parent
        previous_directory = Path.cwd()
        try:
            os.chdir(project_root)
            _MODULE = importlib.import_module("launch")
        finally:
            os.chdir(previous_directory)

        for name in ("mix_recog", "llm_inference", "AI_clip", "AI_clip_subti"):
            if not hasattr(_MODULE, name):
                raise RuntimeError(
                    "launch.py is not importable yet. Run make_launch_importable.py first."
                )
        return _MODULE
