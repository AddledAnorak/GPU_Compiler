#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/mesa/build}"
if [ "$#" -gt 0 ]; then
    shift
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "Python 3.11+ required; create .venv or install python3." >&2
    exit 1
fi

exec "$PYTHON" "$ROOT/tools/validate_frontend.py" --build-dir "$BUILD_DIR" "$@"
