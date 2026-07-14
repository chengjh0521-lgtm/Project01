#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCLIP_LAUNCH_DIR="${FUNCLIP_LAUNCH_DIR:-$ROOT/runtime/funclip}"
PORT="${PORT:-7861}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export FUNCLIP_LAUNCH_DIR
export PORT
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
exec "$PYTHON_BIN" "$ROOT/web_app.py"
