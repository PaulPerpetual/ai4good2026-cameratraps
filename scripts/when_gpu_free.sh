#!/usr/bin/env bash
# Run a command once nobody else has a process on the shared GPU (checked every 5 min).
#   bash scripts/when_gpu_free.sh uv run python scripts/extract_features.py --backbone bioclip2
set -euo pipefail
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  echo "$(date '+%F %T') GPU busy: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"
  sleep 300
done
echo "$(date '+%F %T') GPU free, starting: $*"
exec "$@"
