#!/usr/bin/env bash
# Launch JD inside the shared magenta-rt venv (which provides magenta-rt[mlx]).
# Pass through any flags, e.g.:  ./run.sh --port 9000 --webcam 1
set -euo pipefail

# Run from the directory this script lives in, so `python -m jd` finds the package.
cd "$(dirname "$0")"

exec /Users/twinkle/Desktop/JY/magenta-rt/.venv/bin/python -m jd "$@"
