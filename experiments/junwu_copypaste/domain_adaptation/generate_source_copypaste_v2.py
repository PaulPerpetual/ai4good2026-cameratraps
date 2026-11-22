"""E1/E2 source-domain Copy-Paste, paired to successful E3/E4 sources.

Run from the domain_adaptation/scripts directory or by absolute path.
Uses the existing SAM2 cutout and V2 quality/placement code.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import generate_target_copypaste as old
from generate_target_copypaste_v2 import appearance, compatible, source_mask_ok, paste_low

ROOT = old.ROOT
WORK = old.USER / 'domain_adaptation'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val'])
    ap.add_argument('--backgrounds', type=int, default=1600)
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()
    if args.backgrounds <= 0:
        ap.error('--backgrounds must be positive')

    # Exactly the successful E3/E4 animal sources, not all original 500.
    old_labels = pd.read_csv(WORK / 'synthetic_val_v2' / 'labels.csv')
    required = {'pair_id', 'method', 'source_image', 'species', 'y'}
    if not required.issubset(old_labels.columns):
        raise ValueError(f'Missing columns in V2 labels: {required - set(old_labels.columns)}')
    paired = old_labels.groupby('pair_id')['method'].agg(set)
    ids = paired[paired.map(lambda x: {'random_target', 'matched_target'} <= x)].index
    sources = old_labels[old_labels['pair_id'].isin(ids)].drop_duplicates('pair_id').sort_values('pair_id')
    if len(sources) != 477:
        raise RuntimeError(f'Expected 477 successful source pairs, found {len(sources)}')

    manifest = pd.read_csv(WORK / 'pairs_val' / 'random_target.csv')
    if not {'pair_id', 'source_image', 'source_file', 'species', 'y'} <= set(manifest.columns):
        raise ValueError('Missing columns in pairs_val/random_target.csv')
    pairs = sources[['pair_id', 'source_image', 'species', 'y']].merge(
        manifest[['pair_id', 'source_image', 'source_file', 'species', 'y']],
        on='pair_id', how='left', validate='one_to_one', suffixes=('_v2', '')
    )
    for field in ('source_image', 'species', 'y'):
        if not (pairs[field + '_v2'].astype(str).to_numpy() == pairs[field].astype(str).to_numpy()).all():
            raise RuntimeError(f'V2 source does not match original manifest: {field}')

    meta = pd.read_csv(ROOT / 'camera-trap/data/metadata.csv.gz',
                       usecols=['image_id', 'file_name', 'location', 'category', 'official_ood', 'is_empty'])
    train = meta[meta['official_ood'].eq('train')].copy()
    empty = train[train['is_empty'].astype(str).str.lower().eq('true')].copy()
    animals = train[train['is_empty'].astype(str).str.lower().eq('false')].copy()
    species_locations = animals.groupby('category')['location'].agg(set).to_dict()

    # Equal-sized background sampling pool to E3/E4; deterministic.
    pool = empty[['image_id', 'file_name', 'location']].dropna().drop_duplicates('image_id')
    pool = pool.sample(n=min(args.backgrounds, len(pool)), random_state=args.seed)
    pool['path'] = pool['file_name'].map(lambda fn: ROOT / 'raw/train' / str(fn))
    pool = pool[pool['path'].map(Path.is_file)]
    bg = []
    for rec in pool.itertuples(index=False):
        try:
            bg.append((str(rec.image_id), str(rec.file_name), rec.location, appearance(rec.path)))
        except (OSError, ValueError):
            pass
    if not bg:
        raise RuntimeError('No readable source empty backgrounds')
    print(f'Background pool: {len(bg)} source empty frames', flush=True)

    with (old.CP / 'train_animal_candidates.json').open() as f:
        candidates = {str(a['image_id']).removesuffix('.jpg'): a for a in json.load(f)}
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('SAM2 device:', device, flush=True)
    predictor = SAM2ImagePredictor(build_sam2(old.CONFIG, str(old.CHECKPOINT), device=device))

    output = WORK / 'synthetic_source_v2'
    for method in ('random_source', 'same_y_source'):
        (output / method).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records, rejected = [], []
    fallbacks = Counter()

    for k, row in enumerate(pairs.itertuples(index=False), 1):
        pair_id = int(row.pair_id)
        source_id = str(row.source_image)
        try:
            cut = old.animal_cutout(source_id, row.source_file, candidates, predictor)
            ok, reason = source_mask_ok(cut, row.source_file, source_id, candidates)
            if not ok:
                raise ValueError(reason)
            source = appearance(ROOT / 'raw/train' / str(row.source_file))
            admissible = [b for b in bg if compatible(source, b[3])]
            if not admissible:
                raise ValueError('No compatible source backgrounds')
            random_bg = admissible[int(rng.integers(len(admissible)))]
            allowed_locations = species_locations.get(str(row.species), set())
            same_y = [b for b in admissible if b[2] in allowed_locations and b[0] != random_bg[0]]
            fallback = not bool(same_y)
            if fallback:
                alternatives = [b for b in admissible if b[0] != random_bg[0]]
                same_bg = alternatives[int(rng.integers(len(alternatives)))] if alternatives else random_bg
                fallbacks[str(row.species)] += 1
            else:
                same_bg = same_y[int(rng.integers(len(same_y)))]

            staged = []
            for method, b, used_fallback in [
                ('random_source', random_bg, False),
                ('same_y_source', same_bg, fallback),
            ]:
                # Source frames are physically in raw/train.
                background = old.load_image(b[1], 'train')
                im = paste_low(cut, background, rng)
                path = output / method / f'{method}_{pair_id:05d}.jpg'
                staged.append((method, b[0], used_fallback, im, path))
            for method, bg_id, used_fallback, im, path in staged:
                im.save(path, quality=93)
                records.append({
                    'pair_id': pair_id, 'method': method, 'image_file': str(path),
                    'source_image': source_id, 'background_image': bg_id,
                    'species': str(row.species), 'y': int(row.y),
                    'same_y_fallback': used_fallback,
                })
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append({'pair_id': pair_id, 'source_image': source_id, 'reason': str(exc)})
        if k % 25 == 0 or k == len(pairs):
            print(f'{k}/{len(pairs)} sources; complete pairs={len(records)//2}; rejected={len(rejected)}', flush=True)

    pd.DataFrame(records).to_csv(output / 'labels.csv', index=False)
    pd.DataFrame(rejected, columns=['pair_id', 'source_image', 'reason']).to_csv(output / 'rejected.csv', index=False)
    print('DONE:', len(records)//2, 'paired sources,', len(rejected), 'rejected', flush=True)
    print('Same-Y fallbacks:', dict(fallbacks), flush=True)
    print('Saved:', output / 'labels.csv', flush=True)
    if len(rejected):
        print('IMPORTANT: Compare only pair_id intersection across all four arms.', flush=True)

if __name__ == '__main__':
    main()