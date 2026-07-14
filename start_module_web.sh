#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCLIP_LAUNCH_DIR="${FUNCLIP_LAUNCH_DIR:-$ROOT/runtime/funclip}"
PORT="${PORT:-7861}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FUNCLIP_LLM_MODEL="${FUNCLIP_LLM_MODEL:-deepseek-v4-flash}"

export FUNCLIP_LAUNCH_DIR
export PORT
export FUNCLIP_LLM_MODEL
export GRADIO_ANALYTICS_ENABLED="False"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
exec "$PYTHON_BIN" "$ROOT/web_app.py"
