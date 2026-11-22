"""B2 pilot: learn background-consistent representations from frozen DINOv2 features.

Compares three identically initialized neural models:
  B0: original train images only
  B1: original + 4 paired Copy-Paste arms (ordinary supervised training)
  B2: exactly B1 + cosine consistency for original/synthetic views of same animal

The OOD validation set is evaluated once per model, after fixed training epochs.
This is exploratory because prior work has already used this validation set.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

ROOT_DEFAULT = Path('/zfs/ai4good/groups/camera-traps')
METHODS = ('random_source', 'same_y_source', 'random_target', 'matched_target')


class AdapterClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, hidden)
        self.up = nn.Linear(hidden, feature_dim)
        # Starts as the identity on normalized frozen DINOv2 features.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.logit_scale = 20.0

    def represent(self, features):
        features = F.normalize(features, dim=-1)
        residual = self.up(F.gelu(self.down(self.norm(features))))
        representation = features + residual / (features.shape[-1] ** 0.5)
        return F.normalize(representation, dim=-1)

    def forward(self, features):
        z = self.represent(features)
        return self.logit_scale * self.classifier(z), z


def load_data(root, work):
    original = root / 'features/dinov2_vitl14_reg_448'
    idx = pd.read_csv(original / 'index.csv.gz')
    raw = np.load(original / 'features.npy', mmap_mode='r')
    status = np.load(original / 'status.npy')
    if len(idx) != len(raw) or len(idx) != len(status):
        raise ValueError('Original index/features/status lengths do not match')

    valid = (idx['y'] >= 0) & (status == 1)
    train_ids = np.flatnonzero((valid & idx['official_ood'].eq('train')).to_numpy())
    val_ids = np.flatnonzero((valid & idx['official_ood'].eq('val')).to_numpy())
    if not len(train_ids) or not len(val_ids):
        raise ValueError('Empty train or validation split')

    x_train = np.asarray(raw[train_ids], dtype=np.float32).copy()
    x_val = np.asarray(raw[val_ids], dtype=np.float32).copy()
    normalize(x_train, norm='l2', copy=False)
    normalize(x_val, norm='l2', copy=False)
    y_train = idx.iloc[train_ids]['y'].to_numpy(dtype=np.int64)
    y_val = idx.iloc[val_ids]['y'].to_numpy(dtype=np.int64)

    classes = np.unique(y_train)
    class_to_idx = {int(y): i for i, y in enumerate(classes)}
    y_train_idx = np.array([class_to_idx[int(y)] for y in y_train], dtype=np.int64)

    # Training-source verification, and mapping source image to its frozen feature.
    source_keys = idx.iloc[train_ids]['file_name'].astype(str).str.removesuffix('.jpg')
    if source_keys.duplicated().any():
        raise ValueError('Duplicate train source image IDs in original index')
    source_to_row = dict(zip(source_keys, range(len(train_ids))))

    arms = {}
    for method in METHODS:
        feature_dir = work / ('features_source_v2' if method.endswith('_source') else 'features_v2') / method
        arm_idx = pd.read_csv(feature_dir / 'index.csv')
        arm_x = np.asarray(np.load(feature_dir / 'features.npy'), dtype=np.float32).copy()
        if len(arm_idx) != len(arm_x) or arm_x.shape[1] != x_train.shape[1]:
            raise ValueError(f'Bad synthetic array shape for {method}')
        if arm_idx['pair_id'].duplicated().any():
            raise ValueError(f'Duplicate pair IDs for {method}')
        normalize(arm_x, norm='l2', copy=False)
        arms[method] = (arm_idx, arm_x)

    common_pairs = sorted(set.intersection(*[set(arms[m][0]['pair_id']) for m in METHODS]))
    if not common_pairs:
        raise ValueError('No shared pair_id across the four synthetic arms')

    all_synthetic, all_targets, all_anchors = [], [], []
    identities = None
    for method in METHODS:
        arm_idx, arm_x = arms[method]
        locations = arm_idx.set_index('pair_id').index.get_indexer(common_pairs)
        if np.any(locations < 0):
            raise ValueError(f'Missing pair: {method}')
        aligned_idx = arm_idx.iloc[locations]
        aligned_x = arm_x[locations]
        current_identities = list(zip(
            aligned_idx['source_image'].astype(str).str.removesuffix('.jpg'),
            aligned_idx['y'].astype(int)))
        if identities is None:
            identities = current_identities
        elif identities != current_identities:
            raise ValueError(f'Pair animal identity/label mismatch in {method}')

        source_rows = []
        target_ids = []
        for source, y in current_identities:
            if source not in source_to_row:
                raise ValueError(f'{method}: source {source} is not in the train split')
            pos = source_to_row[source]
            if int(y_train[pos]) != y:
                raise ValueError(f'{method}: source label mismatch: {source}')
            source_rows.append(pos)
            if int(y) not in class_to_idx:
                raise ValueError(f'Unknown source class {y}')
            target_ids.append(class_to_idx[int(y)])
        all_synthetic.append(aligned_x)
        all_targets.extend(target_ids)
        all_anchors.append(x_train[source_rows])

    x_synth = np.concatenate(all_synthetic, axis=0)
    x_anchor = np.concatenate(all_anchors, axis=0)
    y_synth_idx = np.asarray(all_targets, dtype=np.int64)

    for name, arr in [('train', x_train), ('val', x_val), ('synthetic', x_synth), ('anchors', x_anchor)]:
        if not np.isfinite(arr).all():
            raise ValueError(f'Non-finite feature values in {name}')

    print(f'Original train: {len(x_train)}; OOD val: {len(x_val)}; '
          f'classes: {len(classes)}; common pairs: {len(common_pairs)}; '
          f'synthetic views: {len(x_synth)}', flush=True)

    return {
        'x_train': torch.from_numpy(x_train),
        'y_train_idx': torch.from_numpy(y_train_idx),
        'x_val': torch.from_numpy(x_val),
        'y_val': y_val,
        'classes': classes,
        'x_synth': torch.from_numpy(x_synth),
        'y_synth_idx': torch.from_numpy(y_synth_idx),
        'x_anchor': torch.from_numpy(x_anchor),
        'n_pairs': len(common_pairs),
        'dimension': x_train.shape[1],
    }


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    preds = []
    x_val = data['x_val']
    for s in range(0, len(x_val), batch_size):
        logits, _ = model(x_val[s:s + batch_size].to(device))
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred_y = data['classes'][np.concatenate(preds)]
    true_y = data['y_val']
    return {
        'accuracy': float(accuracy_score(true_y, pred_y)),
        'macro_f1': float(f1_score(true_y, pred_y, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(true_y, pred_y, average='weighted', zero_division=0)),
    }


@torch.no_grad()
def paired_cosine(model, data, device, batch_size):
    model.eval()
    similarities = []
    x_syn, x_anc = data['x_synth'], data['x_anchor']
    for s in range(0, len(x_syn), batch_size):
        a = model.represent(x_syn[s:s + batch_size].to(device))
        b = model.represent(x_anc[s:s + batch_size].to(device))
        similarities.append((a * b).sum(dim=1).cpu().numpy())
    return float(np.concatenate(similarities).mean())


def fit_one(name, data, args, device):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = AdapterClassifier(data['dimension'], len(data['classes']), args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    add_synthetic = (name != 'B0_original_only')
    consistency_weight = args.lambda_consistency if name == 'B2_consistency' else 0.0

    train_x = data['x_train']
    train_y = data['y_train_idx']
    syn_x = data['x_synth']
    syn_y = data['y_synth_idx']
    anc_x = data['x_anchor']

    print(f'\n=== {name} | lambda={consistency_weight} ===', flush=True)
    for epoch in range(args.epochs):
        # Identical real train batches and identical synthetic draws across B1/B2.
        real_rng = np.random.default_rng(args.seed + epoch)
        syn_rng = np.random.default_rng(args.seed + 10000 + epoch)
        order = real_rng.permutation(len(train_x))
        model.train()
        loss_total = loss_cons_total = steps = 0
        for begin in range(0, len(order), args.batch_real):
            positions = order[begin:begin + args.batch_real]
            x_real = train_x[positions].to(device)
            y_real = train_y[positions].to(device)

            if add_synthetic:
                sel = syn_rng.integers(0, len(syn_x), size=args.batch_synthetic)
                x_syn = syn_x[sel].to(device)
                y_syn = syn_y[sel].to(device)
                logits, z = model(torch.cat([x_real, x_syn], dim=0))
                labels = torch.cat([y_real, y_syn], dim=0)
                classification_loss = F.cross_entropy(logits, labels)
                if consistency_weight > 0:
                    z_syn = z[len(x_real):]
                    z_anchor = model.represent(anc_x[sel].to(device))
                    consistency_loss = (1 - (z_syn * z_anchor).sum(dim=1)).mean()
                else:
                    consistency_loss = classification_loss.new_zeros(())
            else:
                logits, _ = model(x_real)
                classification_loss = F.cross_entropy(logits, y_real)
                consistency_loss = classification_loss.new_zeros(())

            loss = classification_loss + consistency_weight * consistency_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += classification_loss.item()
            loss_cons_total += consistency_loss.item()
            steps += 1

        print(f'Epoch {epoch+1}/{args.epochs} | classification loss {loss_total/steps:.4f} '
              f'| paired consistency loss {loss_cons_total/steps:.4f}', flush=True)

    metrics = evaluate(model, data, device, args.eval_batch)
    metrics.update({
        'experiment': name,
        'consistency_weight': consistency_weight,
        'paired_cosine': paired_cosine(model, data, device, args.eval_batch),
        'epochs': args.epochs,
        'seed': args.seed,
        'synthetic_views': len(syn_x) if add_synthetic else 0,
    })
    print('FINAL:', json.dumps(metrics, ensure_ascii=False), flush=True)
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    ap.add_argument('--work', type=Path, default=None)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-real', type=int, default=512)
    ap.add_argument('--batch-synthetic', type=int, default=64)
    ap.add_argument('--eval-batch', type=int, default=1024)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--lambda-consistency', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    args.work = args.work or args.root / 'users/junwu/domain_adaptation'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a GPU allocation, or --device cpu')
    if min(args.epochs, args.batch_real, args.batch_synthetic, args.eval_batch, args.hidden) <= 0:
        ap.error('Epochs, batch sizes, and hidden width must be positive')
    if args.lambda_consistency < 0:
        ap.error('Consistency weight must not be negative')
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = load_data(args.root, args.work)
    out = args.work / 'results_b2'
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name in ('B0_original_only', 'B1_augmented', 'B2_consistency'):
        model, row = fit_one(name, data, args, device)
        torch.save({'model': model.state_dict(), 'classes': data['classes'],
                    'config': vars(args)}, out / f'{name}.pt')
        results.append(row)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    frame = pd.DataFrame(results)
    frame['delta_vs_B1_macro_f1'] = frame['macro_f1'] - frame.loc[
        frame['experiment'].eq('B1_augmented'), 'macro_f1'].iloc[0]
    frame.to_csv(out / 'comparison.csv', index=False)
    print('\n====== B2 COMPARISON ======', flush=True)
    print(frame.to_string(index=False, float_format=lambda z: f'{z:.4f}'), flush=True)
    print('Saved:', out / 'comparison.csv', flush=True)
    print('Note: OOD validation has been explored previously; treat scores as exploratory.', flush=True)


if __name__ == '__main__':
    main()