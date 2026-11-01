#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ".venv/bin/python" ]]; then
  echo "Create the virtual environment first: python3.12 -m venv .venv" >&2
  exit 1
fi
exec .venv/bin/python -B app.py "$@"
