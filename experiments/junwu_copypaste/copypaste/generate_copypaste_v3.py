"""Camera-trap cross-camera Copy-Paste v3 (GrabCut prototype, CPU).

Run from any folder:
    python generate_copypaste_v3.py

Requires: numpy, pandas, Pillow, opencv-python (cv2)
Reads existing candidate JSON and camera-trap metadata; writes only to users/junwu/copypaste/generated_v3.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/zfs/ai4good/groups/camera-traps')
WORK = ROOT / 'users/junwu/copypaste'
DATA = ROOT / 'camera-trap/data'
OUT = WORK / 'generated_v3'
OUT.mkdir(parents=True, exist_ok=True)

N_IMAGES = 20
MAX_ATTEMPTS = 300
random.seed(42)

with open(WORK / 'train_animal_candidates.json') as f:
    animals = json.load(f)

meta = pd.read_csv(DATA / 'metadata.csv.gz')
meta['image_id'] = meta['image_id'].astype(str)
meta = meta.drop_duplicates(subset='image_id').set_index('image_id')

# Only genuine empty frames as stated in metadata (not just detector non-detections).
backgrounds = meta[meta['is_empty'].astype(str).str.lower().eq('true')].copy()
# Prefer allowed target-domain empty frames when they are present in this metadata.
if 'official_ood' in backgrounds.columns:
    target = backgrounds[backgrounds['official_ood'].astype(str).eq('test')]
    if not target.empty:
        backgrounds = target

background_ids = backgrounds.index.tolist()
print('Animal candidates:', len(animals))
print('Empty background candidates:', len(background_ids))

# File names generally match image IDs; adapt only if needed.
IMAGE_DIR = DATA / 'images_h448'

def image_path(image_id):
    p = IMAGE_DIR / (str(image_id) + '.jpg')
    if p.is_file():
        return p
    # Metadata may contain an alternate filename or nested paths.
    if image_id in meta.index and 'file_name' in meta.columns:
        name = str(meta.at[image_id, 'file_name'])
        q = IMAGE_DIR / name
        if q.is_file():
            return q
    return None


def segment_animal(image_rgb, normalized_box):
    """Approximate silhouette inside a detector box using GrabCut.
    Returns RGB crop and feathered L-mode alpha mask, or None.
    """
    W, H = image_rgb.size
    x, y, w, h = normalized_box
    x0 = max(0, min(W - 1, round(x * W)))
    y0 = max(0, min(H - 1, round(y * H)))
    x1 = max(x0 + 1, min(W, round((x + w) * W)))
    y1 = max(y0 + 1, min(H, round((y + h) * H)))
    bw, bh = x1 - x0, y1 - y0
    if bw < 20 or bh < 20:
        return None

    arr = np.asarray(image_rgb)
    # Expand around box so GrabCut has some certain background pixels.
    pad_x = max(10, bw // 5)
    pad_y = max(10, bh // 5)
    l, t = max(0, x0 - pad_x), max(0, y0 - pad_y)
    r, b = min(W, x1 + pad_x), min(H, y1 + pad_y)
    patch = arr[t:b, l:r].copy()
    ph, pw = patch.shape[:2]
    if min(ph, pw) < 24:
        return None

    mask = np.full((ph, pw), cv2.GC_PR_BGD, dtype=np.uint8)
    # Bounding box inside the expanded patch is probable foreground.
    bx0, by0, bx1, by1 = x0 - l, y0 - t, x1 - l, y1 - t
    mask[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    # Small center seed ensures a foreground class is present.
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    rr = max(2, min(bw, bh) // 12)
    mask[max(0, cy - rr):min(ph, cy + rr),
         max(0, cx - rr):min(pw, cx + rr)] = cv2.GC_FGD
    mask[0, :] = mask[-1, :] = cv2.GC_BGD
    mask[:, 0] = mask[:, -1] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR), mask,
                    None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    binary = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
    count, comps, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return None
    # Keep the foreground component nearest the detection box center.
    k = min(range(1, count), key=lambda j:
            (centroids[j][0] - cx) ** 2 + (centroids[j][1] - cy) ** 2)
    binary = ((comps == k) * 255).astype(np.uint8)
    if np.count_nonzero(binary) < 100:
        return None
    alpha = cv2.GaussianBlur(binary, (5, 5), 1.0)
    ys, xs = np.where(binary > 0)
    xl, xr, yt, yb = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (Image.fromarray(patch[yt:yb, xl:xr], 'RGB'),
            Image.fromarray(alpha[yt:yb, xl:xr], 'L'))


results = []
for attempt in range(MAX_ATTEMPTS):
    if len(results) >= N_IMAGES:
        break
    animal = random.choice(animals)
    source_id = str(animal['image_id'])
    if source_id not in meta.index or not background_ids:
        continue
    source_camera = str(meta.at[source_id, 'location'])
    target_id = random.choice(background_ids)
    target_camera = str(meta.at[target_id, 'location'])
    if target_camera == source_camera:
        continue
    src_path, dst_path = image_path(source_id), image_path(target_id)
    if src_path is None or dst_path is None:
        continue

    boxes = [d for d in animal['boxes'] if str(d.get('category')) == '1']
    if not boxes:
        continue
    detector_box = max(boxes, key=lambda d: float(d.get('conf', 0)))['bbox']
    with Image.open(src_path) as im:
        src = im.convert('RGB')
    with Image.open(dst_path) as im:
        dst = im.convert('RGB')
    segmented = segment_animal(src, detector_box)
    if segmented is None:
        continue
    crop, alpha = segmented

    # Reduce excessive size; keep proportions.
    crop.thumbnail((max(1, int(dst.width * .40)), max(1, int(dst.height * .45))))
    alpha = alpha.resize(crop.size, Image.Resampling.BILINEAR)
    if crop.width > dst.width or crop.height > dst.height:
        continue
    px = random.randint(0, dst.width - crop.width)
    # Bias position toward the lower part of the scene (still heuristic).
    ylo = max(0, int(dst.height * .45) - crop.height)
    py = random.randint(ylo, dst.height - crop.height)
    dst.paste(crop, (px, py), alpha)

    filename = f'segmented_{len(results):03d}.jpg'
    dst.save(OUT / filename, quality=92)
    results.append(dict(filename=filename, species=animal['species'],
                        source_image=source_id, background_image=target_id,
                        source_camera=source_camera, target_camera=target_camera))

with open(OUT / 'labels.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('Generated:', len(results), 'after', min(MAX_ATTEMPTS, attempt + 1), 'attempts')
print('Saved to:', OUT)