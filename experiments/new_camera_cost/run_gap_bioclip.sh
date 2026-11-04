#!/usr/bin/env bash
# Run 7: the new-camera cost on Paul's BioCLIP features (whole frames and MegaDetector crops),
# read only, used with his permission. First the import into our cache format (CPU), then
# gap_measure.py version 2 on both (GPU), with Run 3's arguments.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-bioclip ./run_gap_bioclip.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
P=$G/users/paul/camera-trap/feats
cd "$W/ai4good2026-cameratraps"
PY=.venv/bin/python
LOG=$W/pilot_a/logs/bioclip_import_$(date +%F_%H%M%S).log
# The order file must list the source rows in the task order this repo produces.
IWILDCAM_DATA_ROOT=$G/camera-trap/data $PY - "$P/md.csv" <<'PYEOF' 2>&1 | tee "$LOG"
import sys
import pandas as pd
from iwildcam.data import image_dir_for, load_task
order = pd.read_csv(sys.argv[1], usecols=["file_name"])["file_name"].to_numpy()
task = load_task(image_dir=image_dir_for(448))
assert len(order) == len(task.df) and (order == task.df["file_name"].to_numpy()).all(), "md.csv is not in task order"
print("md.csv order equals load_task order:", len(order), "rows")
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "order check failed"; exit 1; }
for view in whole crop; do
    src=$P/bioclip.npz; [ "$view" = crop ] && src=$P/bioclip-crop.npz
    $PY "$W/pilot_a/import_features.py" --npz "$src" --order-csv "$P/md.csv" \
        --index "$W/pilot_a/features/dinov2_vitb14_reg_448/index.csv.gz" --out "$W/pilot_a/features/bioclip_${view}_paul" \
        --expect-rows 127681 --expect-dim 512 \
        --note "BioCLIP (hf-hub:imageomics/bioclip) image embeddings from Paul's linprobe.py, view=$view, read with his permission" \
        2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "import of $view failed"; exit 1; }
done
for view in whole crop; do
    NAME=gap_bioclip_${view}_v1
    if [ -e "$W/pilot_a/results/$NAME" ]; then echo "results/$NAME exists; not overwriting"; exit 9; fi
    LOG=$W/pilot_a/logs/${NAME}_$(date +%F_%H%M%S).log
    $PY "$W/pilot_a/gap_measure.py" --cache "$W/pilot_a/features/bioclip_${view}_paul" --repo . \
        --out "$W/pilot_a/results/$NAME" --threads 32 --device cuda --max-iter 2000 --max-minutes 30 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
    echo "EXIT=$status" | tee -a "$LOG"
    if [ "$status" -ne 0 ]; then echo "run on $view failed; not starting the next one"; exit "$status"; fi
done
