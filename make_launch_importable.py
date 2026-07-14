#!/usr/bin/env python3
"""Make FunClip's launch.py importable without starting Gradio on import.

Run this after every upstream update that replaces funclip/launch.py.
The script is idempotent and fails loudly if the expected launch structure has
changed, rather than silently producing a partially patched file.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PATCH_MARKER = "# external-import patch: processing callbacks are importable"
DEFAULT_LAUNCH = Path(__file__).resolve().parent / "funclip" / "launch.py"


class LaunchPatchError(RuntimeError):
    pass


def transform_source(source: str) -> tuple[str, bool]:
    """Return an import-safe launch module and whether a change was needed."""
    if PATCH_MARKER in source:
        return source, False

    lines = source.splitlines(keepends=True)
    outer_guard_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r'^if __name__ == ["\']__main__["\']:\s*$', line.strip())
        ),
        None,
    )
    if outer_guard_index is None:
        raise LaunchPatchError("Could not find the outer __main__ guard in launch.py.")

    parse_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r'^\s*args = parser\.parse_args\(\)\s*$', line)
        ),
        None,
    )
    if parse_index is None:
        raise LaunchPatchError("Could not find parser.parse_args() in launch.py.")

    launch_indexes = [
        index for index, line in enumerate(lines) if "funclip_service.launch(" in line
    ]
    if not launch_indexes:
        raise LaunchPatchError("Could not find the Gradio launch call in launch.py.")
    final_launch_index = launch_indexes[-1]
    footer_index = next(
        (
            index
            for index in range(final_launch_index, -1, -1)
            if re.match(r'^\s*if args\.listen:\s*$', lines[index])
        ),
        None,
    )
    if footer_index is None:
        raise LaunchPatchError("Could not find the final if args.listen launch block.")

    outer_indent = lines[outer_guard_index][:-len(lines[outer_guard_index].lstrip())]
    lines[outer_guard_index] = (
        f"{outer_indent}{PATCH_MARKER}\n"
        f"{outer_indent}if True:\n"
    )

    parse_indent = lines[parse_index][:-len(lines[parse_index].lstrip())]
    lines[parse_index] = (
        f'{parse_indent}args = (parser.parse_args() if __name__ == "__main__" '
        f"else parser.parse_args([]))\n"
    )

    footer_indent = lines[footer_index][:-len(lines[footer_index].lstrip())]
    lines.insert(footer_index, f'{footer_indent}if __name__ == "__main__":\n')
    for index in range(footer_index + 1, len(lines)):
        if lines[index].strip():
            lines[index] = "    " + lines[index]

    return "".join(lines), True


def patch_launch(launch_path: Path, create_backup: bool = True) -> bool:
    source = launch_path.read_text(encoding="utf-8")
    transformed, changed = transform_source(source)
    if not changed:
        return False
    if create_backup:
        backup_path = launch_path.with_name("launch.py.before-importable")
        shutil.copy2(launch_path, backup_path)
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
    source = args.launch.read_text(encoding="utf-8")
    _, changed = transform_source(source)
    if args.check:
        print("needs patch" if changed else "already importable")
        return 0
    changed = patch_launch(args.launch, create_backup=not args.no_backup)
    print("launch.py patched for import" if changed else "launch.py already importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
