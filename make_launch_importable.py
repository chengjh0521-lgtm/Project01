#!/usr/bin/env python3
"""Patch a stock upstream FunClip launch.py for safe cross-file imports."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MARKER = "# external-import patch: processing callbacks are importable"


def transform_source(source: str) -> tuple[str, bool]:
    if MARKER in source:
        return source, False
    lines = source.splitlines(keepends=True)
    outer = next((i for i, line in enumerate(lines) if re.match(
        r'^if __name__ == ["\']__main__["\']:\s*$', line.strip())), None)
    parse_args = next((i for i, line in enumerate(lines) if re.match(
        r'^\s*args = parser\.parse_args\(\)\s*$', line)), None)
    ui_start = next((i for i, line in enumerate(lines) if re.match(
        r'^\s*# gradio interface\s*$', line)), None)
    if outer is None or parse_args is None or ui_start is None:
        raise RuntimeError("Unsupported launch.py structure; update this patch script first.")

    outer_indent = lines[outer][:-len(lines[outer].lstrip())]
    lines[outer] = f"{outer_indent}{MARKER}\n{outer_indent}if True:\n"
    parse_indent = lines[parse_args][:-len(lines[parse_args].lstrip())]
    lines[parse_args] = (
        f'{parse_indent}args = (parser.parse_args() if __name__ == "__main__" '
        f"else parser.parse_args([]))\n"
    )
    # The callback functions above this line stay importable.  The upstream
    # Blocks UI is only needed when launch.py is executed as a program; building
    # it during an import triggers component-version failures and wastes memory.
    ui_indent = lines[ui_start][:-len(lines[ui_start].lstrip())]
    lines.insert(ui_start, f'{ui_indent}if __name__ == "__main__":\n')
    for i in range(ui_start + 1, len(lines)):
        if lines[i].strip():
            lines[i] = "    " + lines[i]
    return "".join(lines), True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("launch", type=Path, help="Path to upstream funclip/launch.py")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    source = args.launch.read_text(encoding="utf-8")
    transformed, changed = transform_source(source)
    if not changed:
        print("launch.py already importable")
        return 0
    if not args.no_backup:
        shutil.copy2(args.launch, args.launch.with_name("launch.py.before-importable"))
    args.launch.write_text(transformed, encoding="utf-8")
    print("launch.py patched for import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
