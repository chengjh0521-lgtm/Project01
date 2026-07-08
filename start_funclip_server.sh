#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export GRADIO_ANALYTICS_ENABLED=False
export TMP="$PWD/tmp"
export TEMP="$PWD/tmp"
export TMPDIR="$PWD/tmp"
export GRADIO_TEMP_DIR="$PWD/gradio_tmp"
mkdir -p "$TMP" "$GRADIO_TEMP_DIR" output
./funclip-env/bin/python funclip/launch.py -m paraformer -p "${PORT:-7860}" --listen
