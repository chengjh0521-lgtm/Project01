#!/usr/bin/env python3
"""Make the upstream FunClip launch module importable without launching Gradio."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MARKER = "# external-import patch: processing callbacks are importable"
DEFAULT_LAUNCH = Path(__file__).resolve().parent / "funclip" / "launch.py"


class LaunchPatchError(RuntimeError):
    pass


def transform_source(source: str) -> tuple[str, bool]:
    """Patch a stock launch.py. Return transformed source and changed status."""
    if MARKER in source:
        return source, False

    lines = source.splitlines(keepends=True)
    outer = next(
        (i for i, line in enumerate(lines)
         if re.match(r'^if __name__ == ["\']__main__["\']:\s*$', line.strip())),
        None,
    )
    parse_args = next(
        (i for i, line in enumerate(lines)
         if re.match(r'^\s*args = parser\.parse_args\(\)\s*$', line)),
        None,
    )
    launch_calls = [i for i, line in enumerate(lines) if "funclip_service.launch(" in line]
    if outer is None or parse_args is None or not launch_calls:
        raise LaunchPatchError("Unsupported launch.py structure; update the patch script first.")

    footer = next(
        (i for i in range(launch_calls[-1], -1, -1)
         if re.match(r'^\s*if args\.listen:\s*$', lines[i])),
        None,
    )
    if footer is None:
        raise LaunchPatchError("Could not find the final Gradio launch block.")

    outer_indent = lines[outer][:-len(lines[outer].lstrip())]
    lines[outer] = f"{outer_indent}{MARKER}\n{outer_indent}if True:\n"

    parse_indent = lines[parse_args][:-len(lines[parse_args].lstrip())]
    lines[parse_args] = (
        f'{parse_indent}args = (parser.parse_args() if __name__ == "__main__" '
        f"else parser.parse_args([]))\n"
    )

    footer_indent = lines[footer][:-len(lines[footer].lstrip())]
    lines.insert(footer, f'{footer_indent}if __name__ == "__main__":\n')
    for i in range(footer + 1, len(lines)):
        if lines[i].strip():
            lines[i] = "    " + lines[i]
    return "".join(lines), True


def patch(launch_path: Path, backup: bool) -> bool:
    source = launch_path.read_text(encoding="utf-8")
    transformed, changed = transform_source(source)
    if not changed:
        return False
    if backup:
        shutil.copy2(launch_path, launch_path.with_name("launch.py.before-importable"))
    launch_path.write_text(transformed, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    if not args.launch.is_file():
        raise SystemExit(f"launch.py not found: {args.launch}")
    if args.check:
        _, changed = transform_source(args.launch.read_text(encoding="utf-8"))
        print("needs patch" if changed else "already importable")
        return 0
    print("launch.py patched" if patch(args.launch, not args.no_backup) else "already importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
