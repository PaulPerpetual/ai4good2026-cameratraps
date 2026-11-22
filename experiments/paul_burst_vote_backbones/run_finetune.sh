#!/bin/bash
# Overnight queue: LP-FT of BioCLIP on crops, 3 seeds per split, one run after another.
# Each run waits until nobody else is on the GPU. Logged to logs/finetune-queue.log.
set -uo pipefail
source /zfs/ai4good/groups/camera-traps/env.sh
cd "$G/users/paul/camera-trap"
export HF_HOME="$G/users/paul/.cache/huggingface" OMP_NUM_THREADS=16
exec > >(tee -a logs/finetune-queue.log) 2>&1

wait_for_gpu() {
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
        echo "$(date +%H:%M) GPU busy, waiting"; sleep 300
    done
}

for split in official_ood random_burst; do
    for seed in 0 1 2; do
        wait_for_gpu
        echo "== $(date) $split seed $seed"
        uv run python finetune.py --split "$split" --seed "$seed" \
            2>&1 | grep --line-buffered -v -i -E "warning|x_all, y_all" | tee "logs/finetune-${split}-s${seed}.log" \
            || echo "!! $split seed $seed failed, continuing"
    done
done

echo "== $(date) all runs done"
uv run python - <<'EOF'
import json
import pandas as pd
rows = [json.loads(l) for l in open("results/finetune.jsonl")]
df = pd.DataFrame([{"split": r["split"], "seed": r["seed"], "epoch": r["best_epoch"],
                    "alpha": r["wise_alpha"], "test_f1_image": r["test"]["f1_image"],
                    "test_f1_burst": r["test"]["f1_burst"],
                    "plain_ft_burst": r["by_alpha"]["1.0"]["test"]["f1_burst"]} for r in rows if not r["limit"]])
print(df.round(3).to_string(index=False))
print(df.groupby("split")[["test_f1_image", "test_f1_burst", "plain_ft_burst"]].agg(["mean", "std"]).round(3))
EOF
