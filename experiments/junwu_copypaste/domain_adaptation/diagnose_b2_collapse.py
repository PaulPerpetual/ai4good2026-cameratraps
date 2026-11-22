"""B2 post-hoc feature-collapse diagnosis. Inference only; no training.

Uses the original paired animal/augmentation features and the saved B1/B2
AdapterClassifier checkpoints. Compares within-animal, within-species and
between-species cosine similarity on exactly the same examples.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

METHODS = ('random_source', 'same_y_source', 'random_target', 'matched_target')


def load_pairs(root: Path, work: Path):
    orig = root / 'features/dinov2_vitl14_reg_448'
    idx = pd.read_csv(orig / 'index.csv.gz', usecols=['file_name', 'official_ood', 'y'])
    status = np.load(orig / 'status.npy', mmap_mode='r')
    features = np.load(orig / 'features.npy', mmap_mode='r')
    if len(idx) != len(features) or len(idx) != len(status):
        raise ValueError('Original feature index/array/status length mismatch')

    train_mask = idx['official_ood'].eq('train').to_numpy() & (idx['y'].to_numpy() >= 0) & (status == 1)
    train_rows = np.flatnonzero(train_mask)
    source_ids = idx.iloc[train_rows]['file_name'].astype(str).str.removesuffix('.jpg')
    if source_ids.duplicated().any():
        raise ValueError('Duplicate original training image IDs')
    source_to_row = dict(zip(source_ids, train_rows))

    indexes = {}
    vectors = {}
    for method in METHODS:
        parent = 'features_source_v2' if method.endswith('_source') else 'features_v2'
        folder = work / parent / method
        df = pd.read_csv(folder / 'index.csv')
        x = np.load(folder / 'features.npy', mmap_mode='r')
        if len(df) != len(x) or df['pair_id'].duplicated().any():
            raise ValueError(f'Invalid synthetic index for {method}')
        indexes[method] = df.set_index('pair_id', drop=False)
        vectors[method] = x

    common = sorted(set.intersection(*(set(indexes[m].index) for m in METHODS)))
    if not common:
        raise ValueError('No shared pair_id across all four methods')

    anchor_ids = None
    ys = None
    x_syn_parts = []
    for method in METHODS:
        df = indexes[method].loc[common]
        ids = df['source_image'].astype(str).str.removesuffix('.jpg').to_numpy()
        labels = df['y'].to_numpy(dtype=np.int64)
        if anchor_ids is None:
            anchor_ids, ys = ids, labels
        elif not np.array_equal(ids, anchor_ids) or not np.array_equal(labels, ys):
            raise ValueError(f'Pair IDs do not match source identity and label for {method}')
        # Keep original ordering by selecting rows corresponding to sorted common IDs.
        positions = indexes[method].index.get_indexer(common)
        x = np.array(vectors[method][positions], dtype=np.float32)
        normalize(x, copy=False)
        x_syn_parts.append(x)

    original_rows = []
    for source, y in zip(anchor_ids, ys):
        if source not in source_to_row:
            raise ValueError(f'Missing original training source: {source}')
        row = source_to_row[source]
        if int(idx.iloc[row]['y']) != int(y):
            raise ValueError(f'Original source label mismatch: {source}')
        original_rows.append(row)

    x_anchor = np.array(features[original_rows], dtype=np.float32)
    normalize(x_anchor, copy=False)
    x_synth = np.concatenate(x_syn_parts, axis=0)
    n = len(common)
    print(f'Common pairs: {n}; synthetic views: {len(x_synth)}', flush=True)

    # Identical anchors appear once for each synthetic arm.
    x_anchor_repeated = np.tile(x_anchor, (len(METHODS), 1))
    # Avoid counting the exact same original image more than once.
    first_occurrence = {}
    for i, source in enumerate(anchor_ids):
        first_occurrence.setdefault(source, i)
    unique_positions = np.array(sorted(first_occurrence.values()), dtype=int)
    return x_anchor_repeated, x_synth, x_anchor[unique_positions], ys[unique_positions], n


@torch.inference_mode()
def embeddings(model, array: np.ndarray, batch_size: int):
    model.eval()
    pieces = []
    for start in range(0, len(array), batch_size):
        part = torch.from_numpy(np.ascontiguousarray(array[start:start + batch_size]))
        z = model.represent(part).cpu().numpy().astype(np.float32, copy=False)
        pieces.append(z)
    return np.concatenate(pieces, axis=0)


def statistics(model, x_anchor_rep, x_synth, x_unique, y_unique, batch_size):
    a = embeddings(model, x_anchor_rep, batch_size)
    b = embeddings(model, x_synth, batch_size)
    u = embeddings(model, x_unique, batch_size)

    paired = np.sum(a * b, axis=1)
    # Dot product gives cosine because represent() L2-normalizes its output.
    sim = u @ u.T
    iu = np.triu_indices(len(u), k=1)
    s = sim[iu]
    equal = (y_unique[iu[0]] == y_unique[iu[1]])
    same = s[equal]
    diff = s[~equal]
    if len(diff) == 0:
        raise ValueError('At least two distinct animal species needed')
    return {
        'paired_cosine': float(paired.mean()),
        'different_species_cosine': float(diff.mean()),
        'same_species_different_image_cosine': float(same.mean()) if len(same) else float('nan'),
        'paired_minus_different': float(paired.mean() - diff.mean()),
        'same_species_minus_different': float(same.mean() - diff.mean()) if len(same) else float('nan'),
        'mean_embedding_norm': float(np.linalg.norm(u.mean(axis=0))),
        'feature_spread': float(u.std(axis=0).mean()),
        'unique_sources': int(len(u)),
        'between_species_pairs': int(len(diff)),
        'same_species_pairs': int(len(same)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/zfs/ai4good/groups/camera-traps'))
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    args.work = args.work or args.root / 'users/junwu/domain_adaptation'
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')

    # Import the exact network implementation used to train the checkpoints.
    sys.path.insert(0, str((args.work / 'scripts').resolve()))
    from train_b2_consistency import AdapterClassifier

    x_anchor_rep, x_synth, x_unique, y_unique, n = load_pairs(args.root, args.work)
    output = []
    for name in ('B1_augmented', 'B2_consistency'):
        path = args.work / 'results_b2' / (name + '.pt')
        # This is safe only for checkpoints created by you; never load untrusted .pt files.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        hidden = int(checkpoint['config']['hidden'])
        classes = np.asarray(checkpoint['classes'])
        model = AdapterClassifier(x_unique.shape[1], len(classes), hidden)
        model.load_state_dict(checkpoint['model'], strict=True)
        row = statistics(model, x_anchor_rep, x_synth, x_unique, y_unique, args.batch_size)
        output.append({'experiment': name, **row})
        print(name, 'done', flush=True)

    frame = pd.DataFrame(output)
    print('\n===== B2 FEATURE DIAGNOSIS =====', flush=True)
    cols = ['experiment', 'paired_cosine', 'same_species_different_image_cosine',
            'different_species_cosine', 'paired_minus_different',
            'same_species_minus_different', 'mean_embedding_norm', 'feature_spread']
    print(frame[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)
    print(f'Unique source animals: {len(x_unique)}; common source pairs: {n}', flush=True)
    out = args.work / 'results_b2' / 'feature_diagnosis.csv'
    frame.to_csv(out, index=False)
    print('Saved:', out, flush=True)
    print('Interpretation: higher cosine for unrelated species AND near-1 mean_embedding_norm suggest collapse.', flush=True)


if __name__ == '__main__':
    main()
