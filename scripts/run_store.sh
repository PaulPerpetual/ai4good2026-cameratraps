#!/usr/bin/env bash
# Fill the feature store for the protocol's two backbones. Launch through when_gpu_free.sh:
#   nohup bash scripts/when_gpu_free.sh bash scripts/run_store.sh > logs/store.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source /zfs/ai4good/groups/camera-traps/env.sh
X="uv run python scripts/extract_features.py"
SMOKE=$(mktemp -d)
$X --backbone bioclip2 --cameras trainval --sample 512 --out "$SMOKE/bioclip2"   # check before the long runs
rm -rf "$SMOKE"
$X --backbone bioclip2 --cameras trainval
$X --backbone bioclip2 --cameras testempty
$X --backbone bioclip2 --cameras test
$X --backbone dinov2l  --cameras testempty
