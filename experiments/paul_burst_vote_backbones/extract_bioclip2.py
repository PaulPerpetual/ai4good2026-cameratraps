"""BioCLIP 2 features on the MegaDetector crops, same crop rule and 224 squash as bioclip-crop.

    python extract_bioclip2.py     # writes feats/bioclip2-crop.npz, every labelled image in task.df order
"""

from __future__ import annotations

import time

import numpy as np
import open_clip
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import linprobe as lp
from iwildcam.data import image_dir_for, load_task

MODEL = "hf-hub:imageomics/bioclip-2"
OUT = lp.FEATS / "bioclip2-crop.npz"

task = load_task(image_dir=image_dir_for(448))
model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
model = model.eval().cuda()
norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
size = next(t for t in preprocess.transforms if isinstance(t, T.CenterCrop)).size
print("image size", size, "normalise", norm, flush=True)
transform = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])
loader = DataLoader(lp.Files(task, transform, lp.crop_boxes(task), lp.ORIGINALS), batch_size=128,
                    num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
out, t0, done = [], time.time(), 0
with torch.no_grad():
    for i, xb in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.encode_image(xb.cuda(non_blocking=True)).float().cpu())
        done += len(xb)
        if i % 50 == 0:
            print(f"  {done:,}/{len(task.df):,} images, {done / (time.time() - t0):.0f} img/s, "
                  f"~{(len(task.df) - done) / (done / (time.time() - t0)) / 60:.0f} min left", flush=True)
x = torch.cat(out).numpy()
assert len(x) == len(task.df) and np.isfinite(x).all()
np.savez(OUT, x=x.astype(np.float16), file_name=task.df["file_name"].to_numpy())
print(f"wrote {OUT} {x.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
