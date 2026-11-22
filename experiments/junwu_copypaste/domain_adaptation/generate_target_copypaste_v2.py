"""Quality-controlled paired target-background Copy-Paste augmentation.

Reuses the user's working SAM2/MD cutout function. Random and matched arms
use the same source image and identical admissibility/QC criteria.
Only unlabelled target-camera empty frames are read; no val animal labels.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps

import generate_target_copypaste as old

ROOT = old.ROOT
USER = old.USER
CP = old.CP


def appearance(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB').resize((64, 64)), dtype=np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    chroma = float(np.abs(a[:, :, 0]-a[:, :, 1]).mean() +
                   np.abs(a[:, :, 1]-a[:, :, 2]).mean())
    contrast = float(lum.std())
    # Light-weight image-texture gate, not semantic habitat detection.
    edges = float(np.abs(np.diff(lum, axis=0)).mean() +
                  np.abs(np.diff(lum, axis=1)).mean())
    hist = np.histogram(lum, bins=8, range=(0, 1))[0].astype(np.float32)
    hist /= max(1, hist.sum())
    vec = np.concatenate([2*hist, a.mean((0, 1)), a.std((0, 1)),
                          [float(lum.mean()), contrast, chroma]]).astype(np.float32)
    return vec, float(lum.mean()), chroma, contrast, edges


def mode(chroma):
    if chroma < 0.045:
        return 'mono'
    if chroma > 0.075:
        return 'color'
    return 'uncertain'


def compatible(source, bg):
    # Same rules for both random and appearance-matched arms.
    _, lum_s, chr_s, _, _ = source
    _, lum_b, chr_b, contrast_b, edge_b = bg
    if mode(chr_s) != 'uncertain' and mode(chr_b) != 'uncertain':
        if mode(chr_s) != mode(chr_b):
            return False
    if abs(lum_s - lum_b) > 0.35:
        return False
    if contrast_b < 0.045 or edge_b < 0.012:
        return False
    return True


def source_mask_ok(cut, source_file, image_id, candidates):
    """Conservative geometric checks; cannot guarantee semantic segmentation."""
    alpha = np.asarray(cut.getchannel('A'))
    pixels = int((alpha > 128).sum())
    coverage = pixels / alpha.size
    if not 0.17 <= coverage <= 0.97:
        return False, f'mask coverage {coverage:.3f}'
    if min(cut.size) < 25:
        return False, f'cutout dimensions {cut.size}'
    with Image.open(ROOT / 'raw/train' / str(source_file)) as im:
        w0, h0 = im.size
        shrink = min(1.0, 1024.0 / max(w0, h0))
        W, H = round(w0*shrink), round(h0*shrink)
    if pixels / (W*H) < 0.003:
        return False, 'mask too small in source image'
    rec = candidates.get(str(image_id).removesuffix('.jpg'))
    if rec and rec.get('boxes'):
        detection = max(rec['boxes'], key=lambda b: float(b.get('conf', 0)))
        x, y, bw, bh = map(float, detection['bbox'])
        box_pixels = max(1.0, bw*bh*W*H)
        fraction = pixels / box_pixels
        if fraction < 0.18 or fraction > 1.5:
            return False, f'mask/detector area ratio {fraction:.2f}'
    return True, 'ok'


def paste_low(cut, bg, rng):
    # Use realistic foreground size and place the lower edge in lower image.
    W, H = bg.size
    ratio = min(1.0, (0.66*W)/cut.width, (0.62*H)/cut.height)
    if ratio <= 0:
        raise ValueError('invalid size')
    cut = cut.resize((max(1, round(cut.width*ratio)),
                      max(1, round(cut.height*ratio))), Image.Resampling.LANCZOS)
    if max(cut.width/W, cut.height/H) < 0.10:
        raise ValueError('foreground too small for target scene')
    # Animal's lowest pixel should be at roughly 74-91% of image height.
    y_bottom = int(rng.uniform(0.74, 0.90) * H)
    y = min(H-cut.height, max(0, y_bottom-cut.height))
    x = int(rng.integers(0, W-cut.width+1))
    out = bg.convert('RGBA')
    out.alpha_composite(cut, (x, y))
    return out.convert('RGB')


def contact_sheet(records, dest, max_pairs=12):
    by_pair = {}
    for rec in records:
        by_pair.setdefault(rec['pair_id'], {})[rec['method']] = rec
    pairs = [(pid, imgs) for pid, imgs in sorted(by_pair.items())
             if {'random_target','matched_target'}.issubset(imgs)][:max_pairs]
    if not pairs:
        return
    tw, th = 340, 235
    sheet = Image.new('RGB', (tw*2, th*len(pairs)), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (pid, items) in enumerate(pairs):
        for col, method in enumerate(('random_target','matched_target')):
            rec = items[method]
            with Image.open(rec['image_file']) as im:
                small = ImageOps.contain(im.convert('RGB'), (tw-12, th-40))
            x, y = col*tw, n*th
            sheet.paste(small, (x+(tw-small.width)//2, y))
            draw.text((x+5, y+th-33), f"{method} / pair={pid}", fill='black')
            draw.text((x+5, y+th-17), str(rec['species'])[:35], fill='black')
    sheet.save(dest, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--backgrounds', type=int, default=1600)
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()
    if args.limit <= 0 or args.backgrounds <= 0:
        ap.error('--limit and --backgrounds must be positive')

    src_manifest = USER / 'domain_adaptation' / f'pairs_{args.split}' / 'random_target.csv'
    pairs = pd.read_csv(src_manifest).head(args.limit)
    needed = {'pair_id', 'source_image', 'source_file', 'species', 'y'}
    if not needed.issubset(pairs.columns):
        raise ValueError(f'Columns missing: {needed - set(pairs.columns)}')
    meta = pd.read_csv(ROOT / 'camera-trap/data/metadata.csv.gz',
                       usecols=['image_id', 'file_name', 'official_ood', 'is_empty'])
    is_empty = meta['is_empty'].astype(str).str.lower().eq('true')
    pool = meta.loc[is_empty & meta['official_ood'].eq(args.split),
                    ['image_id', 'file_name']].dropna().copy()
    pool = pool.drop_duplicates('image_id')
    pool = pool.sample(n=min(len(pool), args.backgrounds), random_state=args.seed)
    pool['path'] = pool['file_name'].map(lambda x: ROOT / 'raw/train' / str(x))
    pool = pool.loc[pool['path'].map(lambda p: p.is_file())].reset_index(drop=True)
    if pool.empty:
        raise RuntimeError('No readable target empty frames in raw/train')

    # Cache color/illumination features, no need to rerun SAM2 for backgrounds.
    target = []
    for row in pool.itertuples(index=False):
        try:
            target.append((str(row.image_id), row.file_name, appearance(row.path)))
        except (OSError, ValueError):
            pass
    if not target:
        raise RuntimeError('No readable target backgrounds')
    print(f'Read {len(target)} target backgrounds; choosing mode-compatible random and matched backgrounds', flush=True)

    with (CP / 'train_animal_candidates.json').open() as f:
        import json
        raw = json.load(f)
    candidates = {str(a['image_id']).removesuffix('.jpg'): a for a in raw}
    if not old.CHECKPOINT.is_file():
        raise FileNotFoundError(old.CHECKPOINT)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('SAM2 device:', device, flush=True)
    model = build_sam2(old.CONFIG, str(old.CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(model)

    out = USER / 'domain_adaptation' / f'synthetic_{args.split}_v2'
    for method in ('random_target', 'matched_target'):
        (out / method).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cache = {}
    records, failures = [], []

    for k, row in enumerate(pairs.itertuples(index=False), 1):
        image_id = str(row.source_image)
        try:
            if image_id not in cache:
                cut = old.animal_cutout(image_id, row.source_file, candidates, predictor)
                ok, why = source_mask_ok(cut, row.source_file, image_id, candidates)
                if not ok:
                    raise ValueError(why)
                cache[image_id] = cut
            cut = cache[image_id]
            source = appearance(ROOT / 'raw/train' / str(row.source_file))
            candidates_ok = [(id_, fn, st) for id_, fn, st in target if compatible(source, st)]
            if len(candidates_ok) < 2:
                raise ValueError('fewer than 2 compatible target backgrounds')
            # Both methods draw from the SAME style/day-night admissible pool.
            r = candidates_ok[int(rng.integers(len(candidates_ok)))]
            # Nearest by exposure, coarse RGB statistics and luminance histogram.
            dist = np.array([np.sum((source[0]-b[2][0])**2) for b in candidates_ok])
            top = np.argsort(dist)[:min(20, len(dist))]
            # Avoid sampling the identical frame as random arm if possible.
            alternatives = [int(i) for i in top if candidates_ok[int(i)][0] != r[0]]
            matched = candidates_ok[int(rng.choice(alternatives))] if alternatives else candidates_ok[int(top[0])]
            staged = []
            for method, bginfo in [('random_target', r), ('matched_target', matched)]:
                b_id, b_name, _ = bginfo
                bg = old.load_image(b_name, args.split)
                image = paste_low(cut, bg, rng)
                dest = out / method / f'{method}_{int(row.pair_id):05d}.jpg'
                staged.append((method, b_id, image, dest))
            # Save both arms for this source only if both succeeded.
            for method, b_id, image, dest in staged:
                image.save(dest, quality=93)
                records.append({'pair_id': int(row.pair_id), 'method': method,
                                'image_file': str(dest), 'source_image': image_id,
                                'background_image': b_id, 'species': str(row.species),
                                'y': int(row.y)})
        except (ValueError, OSError, RuntimeError, KeyError, FileNotFoundError) as exc:
            failures.append({'pair_id': int(row.pair_id), 'source_image': image_id,
                             'reason': str(exc)})
        if k % 25 == 0 or k == len(pairs):
            print(f'{k}/{len(pairs)} sources; {len(records)//2} complete pairs; '
                  f'{len(failures)} rejected', flush=True)

    record_cols = ['pair_id','method','image_file','source_image','background_image','species','y']
    pd.DataFrame(records, columns=record_cols).to_csv(out / 'labels.csv', index=False)
    pd.DataFrame(failures, columns=['pair_id','source_image','reason']).to_csv(out / 'rejected.csv', index=False)
    contact_sheet(records, out / 'review.jpg')
    print('DONE:', len(records)//2, 'paired examples =', len(records), 'images;',
          len(failures), 'source pairs rejected', flush=True)
    print('Files:', out / 'labels.csv', 'and', out / 'review.jpg', flush=True)
    print('QC is heuristic: manually inspect review.jpg before training.', flush=True)
    if failures:
        print('Reasons:', Counter(r['reason'] for r in failures).most_common(6), flush=True)


if __name__ == '__main__':
    main()