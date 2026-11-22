#!/usr/bin/env bash
# Run from an isolated snapshot with its own .venv and shared read-only image data.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p results logs
trap 'echo "failed: line $LINENO" > results/sweep-status.txt' ERR
PY=.venv/bin/python
echo 'whole-image training' > results/sweep-status.txt
$PY -m iwildcam.train --config configs/background-full.yaml --background-mode whole --seed 0     

echo 'installing preprocessing dependencies' > results/sweep-status.txt
python3 -m venv .preprocess
cp .venv/lib/python3.12/site-packages/shared_cuda.pth .preprocess/lib/python3.12/site-packages/shared_cuda.pth
PRE=.preprocess/bin/python
$PRE -m pip install 'torch==2.6.0' 'torchvision==0.21.0' 'numpy<2.3' megadetector 'segment-anything @ git+https://github.com/facebookresearch/segment-anything.git'
mkdir -p data/weights
if [ ! -f data/weights/sam_vit_b_01ec64.pth ]; then
    $PRE -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 'data/weights/sam_vit_b_01ec64.pth')"
fi
$PRE -m pip freeze > results/preprocessing-environment.txt
echo 'MegaDetector inference' > results/sweep-status.txt
if [ ! -f data/md-v5a.json ]; then
    resume_args=()
    if [ -f data/md-checkpoint.json ]; then
        resume_args=(--resume_from_checkpoint data/md-checkpoint.json)
    fi
    $PRE -m megadetector.detection.run_detector_batch MDV5A "$IWILDCAM_DATA_ROOT/images_h448" data/md-v5a.json --recursive --output_relative_filenames --checkpoint_frequency 5000 --checkpoint_path data/md-checkpoint.json "${resume_args[@]}"
fi
echo 'SAM foreground preparation' > results/sweep-status.txt
if [ ! -f data/foregrounds-v1/manifest.json ]; then
    $PRE -m iwildcam.prepare_foregrounds --detections data/md-v5a.json --image-dir "$IWILDCAM_DATA_ROOT/images_h448" --output data/foregrounds-v1 --sam-checkpoint data/weights/sam_vit_b_01ec64.pth
fi
for mode in crop neutral source target; do
    echo "training $mode" > results/sweep-status.txt
    $PY -m iwildcam.train --config configs/background-full.yaml --background-mode "$mode" --seed 0     
done
$PY -m iwildcam.summarise
echo complete > results/sweep-status.txt
