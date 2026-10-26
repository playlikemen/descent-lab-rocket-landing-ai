#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
echo "Descent Lab: http://localhost:${PORT}"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
