#!/usr/bin/env bash
# Source this file: `source scripts/env.sh`
# Same layout as ../speedy_weather/scripts/env.sh — Julia 1.10.8 module
# plus the shared micromamba `speedy_weather` environment (which has
# xarray, cartopy, ffmpeg, matplotlib, etc.).

module load apps/julia/1.10.8 2>/dev/null || echo "WARN: 'module' not available or julia module missing"

if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate speedy_weather 2>/dev/null || true
fi

export JULIA_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "JULIA_PROJECT=$JULIA_PROJECT"
echo "julia: $(julia --version 2>&1)"
