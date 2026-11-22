
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path('/zfs/ai4good/groups/camera-traps')
USER = ROOT / 'users/junwu'
CP = USER / 'copypaste'
CHECKPOINT = CP / 'checkpoints/sam2.1_hiera_tiny.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'


def load_image(name, split):
    path = ROOT / 'raw' / 'train' / str(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return im.copy()


def animal_cutout(image_id, file_name, candidates, predictor):
    record = candidates.get(str(image_id).removesuffix('.jpg'))
    if record is None or not record.get('boxes'):
        raise ValueError(f'No detection boxes: {image_id}')

    detection = max(
        record['boxes'],
        key=lambda b: float(b.get('conf', 0))
    )
    x, y, w, h = map(float, detection['bbox'])

    image = load_image(file_name, 'train')
    W, H = image.size
    box = np.array([
        np.clip(x * W, 0, W - 2),
        np.clip(y * H, 0, H - 2),
        np.clip((x + w) * W, 1, W - 1),
        np.clip((y + h) * H, 1, H - 1)
    ], dtype=np.float32)

    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Invalid bounding box')

    predictor.set_image(np.asarray(image))

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

    choices = [
        i for i in np.argsort(scores)[::-1]
        if 0.001 < masks[i].mean() < 0.65
    ]
    if not choices:
        raise ValueError('No usable SAM2 mask')

    mask = Image.fromarray(
        (masks[choices[0]] * 255).astype('uint8')
    )

    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError('Empty mask')

    l, t, r, b = bounds
    bounds = (
        max(0, l - 3),
        max(0, t - 3),
        min(W, r + 3),
        min(H, b + 3)
    )

    cut = image.crop(bounds).convert('RGBA')
    cut.putalpha(
        mask.crop(bounds).filter(
            ImageFilter.GaussianBlur(1.3)
        )
    )
    return cut


def compose(cut, bg, rng):
    scale = min(
        float(rng.uniform(0.72, 1.15)),
        0.72 * bg.width / cut.width,
        0.72 * bg.height / cut.height
    )

    if scale <= 0.03:
        raise ValueError('Animal too large')

    cut = cut.resize(
        (
            max(1, round(cut.width * scale)),
            max(1, round(cut.height * scale))
        ),
        Image.Resampling.LANCZOS
    )

    x = int(rng.integers(
        0, bg.width - cut.width + 1
    ))

    y_min = max(
        0, int(bg.height * 0.36) - cut.height
    )

    y = int(rng.integers(
        y_min, bg.height - cut.height + 1
    ))

    out = bg.convert('RGBA')
    out.alpha_composite(cut, (x, y))

    return out.convert('RGB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--split',
        choices=('val', 'test'),
        default='val'
    )
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.limit <= 0:
        ap.error('--limit must be positive')

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f'SAM2 checkpoint missing: {CHECKPOINT}'
        )

    manifests = (
        USER / 'domain_adaptation' / f'pairs_{args.split}'
    )
    output = (
        USER / 'domain_adaptation' / f'synthetic_{args.split}'
    )

    methods = ('random_target', 'matched_target')
    frames = {}

    required = {
        'pair_id',
        'source_image',
        'source_file',
        'background_file',
        'background_image',
        'species',
        'y',
        'target_split'
    }

    for method in methods:
        frames[method] = pd.read_csv(
            manifests / f'{method}.csv'
        ).head(args.limit)

        missing = required - set(frames[method].columns)
        if missing:
            raise ValueError(
                f'{method}: missing columns {missing}'
            )

        if not frames[method]['target_split'].eq(
            args.split
        ).all():
            raise ValueError(
                f'{method}: unexpected target split'
            )

        (output / method).mkdir(
            parents=True, exist_ok=True
        )

    with (
        CP / 'train_animal_candidates.json'
    ).open() as f:
        candidates = json.load(f)

    candidates = {
        str(a['image_id']).removesuffix('.jpg'): a
        for a in candidates
    }

    device = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(
        'SAM2 device:', device,
        '| candidate images:', len(candidates),
        flush=True
    )

    model = build_sam2(
        CONFIG,
        str(CHECKPOINT),
        device=device
    )
    predictor = SAM2ImagePredictor(model)

    rng = np.random.default_rng(args.seed)
    cached_cuts = {}
    records, failures = [], []

    for method in methods:
        df = frames[method]

        for index, row in enumerate(
            df.itertuples(index=False), 1
        ):
            image_id = str(row.source_image)

            try:
                if image_id not in cached_cuts:
                    cached_cuts[image_id] = animal_cutout(
                        image_id,
                        row.source_file,
                        candidates,
                        predictor
                    )

                bg = load_image(
                    row.background_file, args.split
                )

                composite = compose(
                    cached_cuts[image_id],
                    bg,
                    rng
                )

                path = (
                    output / method /
                    f'{method}_{int(row.pair_id):05d}.jpg'
                )
                composite.save(path, quality=92)

                records.append(dict(
                    method=method,
                    image_file=str(path),
                    source_image=image_id,
                    background_image=str(
                        row.background_image
                    ),
                    species=str(row.species),
                    y=int(row.y)
                ))

            except (
                ValueError,
                OSError,
                RuntimeError,
                FileNotFoundError
            ) as exc:
                failures.append(dict(
                    method=method,
                    source_image=image_id,
                    error=str(exc)
                ))

            if index % 25 == 0 or index == len(df):
                print(
                    f'{method}: {index}/{len(df)} '
                    f'processed; {len(records)} saved, '
                    f'{len(failures)} failures',
                    flush=True
                )

    pd.DataFrame(
        records,
        columns=[
            'method', 'image_file', 'source_image',
            'background_image', 'species', 'y'
        ]
    ).to_csv(
        output / 'labels.csv', index=False
    )

    pd.DataFrame(
        failures,
        columns=[
            'method', 'source_image', 'error'
        ]
    ).to_csv(
        output / 'errors.csv', index=False
    )

    print(
        f'DONE: {len(records)} synthetic images; '
        f'{len(failures)} failures',
        flush=True
    )
    print('Output directory:', output, flush=True)

    if failures:
        print('First errors:', failures[:3], flush=True)


if __name__ == '__main__':
    main()
