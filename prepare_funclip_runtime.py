#!/usr/bin/env python3
"""Copy upstream FunClip into a private import runtime and patch launch.py.

The upstream Git submodule remains unmodified.  This makes future upstream
updates a simple submodule update followed by rebuilding ``runtime``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from make_launch_importable import transform_source


ROOT = Path(__file__).resolve().parent


def copy_runtime(source: Path, runtime: Path, replace: bool) -> Path:
    source = source.resolve()
    source_funclip = source / "funclip"
    source_launch = source_funclip / "launch.py"
    if not source_launch.is_file():
        raise RuntimeError("Upstream launch.py was not found: {}".format(source_launch))

    if runtime.exists():
        if not replace:
            raise RuntimeError(
                "Runtime already exists: {}. Re-run with --replace after stopping "
                "the service.".format(runtime)
            )
        shutil.rmtree(runtime)

    shutil.copytree(source_funclip, runtime / "funclip")
    source_font = source / "font"
    if source_font.is_dir():
        shutil.copytree(source_font, runtime / "font")

    launch = runtime / "funclip" / "launch.py"
    original = launch.read_text(encoding="utf-8")
    patched, changed = transform_source(original)
    if not changed:
        raise RuntimeError("The copied upstream launch.py was already patched.")
    launch.with_name("launch.py.before-importable").write_text(original, encoding="utf-8")
    launch.write_text(patched, encoding="utf-8")
    return launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "upstream_funclip")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    launch = copy_runtime(args.source, args.runtime, args.replace)
    print("Runtime prepared: {}".format(launch))
    print("Set FUNCLIP_LAUNCH_DIR={}".format(launch.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
