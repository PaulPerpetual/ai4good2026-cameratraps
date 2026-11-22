#!/usr/bin/env python3
"""AI4Good first target-domain Copy-Paste experiment: reproducible pair manifests.

Build TWO equally sized pairing manifests for OOD validation cameras:
  random_target.csv: train animal -> random validation-camera empty background
  matched_target.csv: train animal -> validation-camera empty background whose
                      CAMERA APPEARANCE resembles source-species camera backgrounds.

This computes candidate pairings, NOT foreground masks or pasted images.
It never uses val/test species labels and never writes under shared directories.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/zfs/ai4good/groups/camera-traps')
USER = ROOT / 'users/junwu'
M = ROOT / 'camera-trap/data/metadata.csv.gz'
OLD = USER / 'copypaste/synthetic_features_v6/index.csv'


def image_dir():
    paths = [ROOT/'camera-trap/data/images_h448', ROOT/'camera-trap/data/images_h224',
             USER/'data/images_h448', USER/'data/images_h224']
    for path in paths:
        if path.is_dir():
            return path
    raise FileNotFoundError('Prepared images directory not found. Tried: ' + ', '.join(map(str, paths)))


def feature(path):
    """Small, deterministic APPEARANCE vector, not a semantic habitat embedding."""
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert('RGB').resize((32,32)), np.float32)/255.
        lum = arr @ np.array([.299,.587,.114],dtype=np.float32)
        hist = np.histogram(lum, bins=8, range=(0,1), density=False)[0].astype(np.float32)
        hist /= max(hist.sum(),1)
        # grayscale fraction: infrared images often have near-equal channels
        grayness = np.abs(arr[:,:,0]-arr[:,:,1]).mean()+np.abs(arr[:,:,1]-arr[:,:,2]).mean()
        # exposure/contrast, channel means/stds, color/gray behavior
        return np.concatenate([hist*2, arr.mean((0,1)), arr.std((0,1)),
                               [lum.mean(),lum.std(),grayness]]).astype(np.float32)
    except (OSError,ValueError):
        return None

def image_dir():
    return ROOT / "raw" / "train"
    candidates = [
        ROOT / "camera-trap/data/images",
        ROOT / "camera-trap/data",
        ROOT / "raw",
    ]

    for path in candidates:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        "Cannot find image directory. "
        "Please check where the camera-trap JPEG files are stored."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val','test'], default='val')
    ap.add_argument('--pairs', type=int, default=500)
    ap.add_argument('--max-train-bg', type=int, default=1200)
    ap.add_argument('--max-target-bg', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.pairs<=0: ap.error('--pairs must be positive')
    rng = np.random.default_rng(args.seed)
    out = USER / 'domain_adaptation' / f'pairs_{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    if not M.exists() or not OLD.exists():
        raise FileNotFoundError(f'Missing metadata {M} or old synthetic index {OLD}')
    pic_dir=image_dir()
    cols=['image_id','file_name','location','category','official_ood','is_empty']
    meta=pd.read_csv(M,usecols=cols)
    meta['image_id']=meta['image_id'].astype(str)
    meta['is_empty']=meta['is_empty'].astype(str).str.lower().eq('true')
    old=pd.read_csv(OLD)
    required={'source_image','y'}
    if not required.issubset(old.columns):
        raise ValueError('Expected source_image and y in original synthetic index')
    old['source_image']=old['source_image'].astype(str)
    old=old.drop_duplicates('source_image')
    src=old[['source_image','y']].merge(meta,left_on='source_image',right_on='image_id',how='inner')
    src=src[(src.official_ood=='train') & (~src.is_empty)].copy()
    src=src[src['file_name'].map(lambda x:(pic_dir/x).is_file())]
    if src.empty: raise RuntimeError('No original train animal images could be matched')
    # Unlike test/val animal labels, source labels are allowed in training.
    train_cam_by_species=src.groupby('y')['location'].agg(lambda x:set(x)).to_dict()
    bg=meta[meta.is_empty].copy()
    bg=bg[bg['file_name'].map(lambda x:(pic_dir/x).is_file())]
    train_bg=bg[bg.official_ood=='train'].copy()
    target_bg=bg[bg.official_ood==args.split].copy()
    if len(train_bg)==0 or len(target_bg)==0:
        raise RuntimeError(f'No train or {args.split} empty background JPEGs found')
    train_bg=train_bg.sample(n=min(len(train_bg),args.max_train_bg),random_state=args.seed)
    target_bg=target_bg.sample(n=min(len(target_bg),args.max_target_bg),random_state=args.seed)
    print('Source foreground candidates:', len(src),'species:',src.y.nunique(), flush=True)
    print('Train empty candidates:',len(train_bg),'target empty candidates:',len(target_bg), flush=True)

    def embed(df):
        good=[]; feats=[]
        for row in df.itertuples(index=False):
            z=feature(pic_dir/getattr(row,'file_name'))
            if z is not None:
                good.append(row); feats.append(z)
        if not good: raise RuntimeError('All background images failed to load')
        return pd.DataFrame(good,columns=df.columns),np.stack(feats)
    tr,Ztr=embed(train_bg); tar,Ztar=embed(target_bg)
    print('Readable backgrounds:',len(tr),'train,',len(tar),args.split,flush=True)
    global_center=Ztr.mean(axis=0)
    target_modes=np.where(Ztar[:,-1]<0.025,0,1) # approximate gray/color
    summaries={}
    for y,cams in train_cam_by_species.items():
        sel=tr.location.isin(cams).to_numpy()
        summaries[int(y)]=(Ztr[sel].mean(0) if sel.sum()>=3 else global_center, int(sel.sum()))
    # Balanced over source images so frequent species do not dominate the output.
    groups={int(y):g.reset_index(drop=True) for y,g in src.groupby('y')}
    keys=sorted(groups)
    random_rows=[]; matched_rows=[]
    for j in range(args.pairs):
        y=keys[j % len(keys)]
        g=groups[y]
        s=g.iloc[int(rng.integers(len(g)))]
        center,n_ref=summaries[y]
        d=((Ztar-center)**2).sum(axis=1)
        # prevent night/day mismatch with the SOURCE CAMERA appearance when possible
        mode=0 if center[-1]<0.025 else 1
        valid=np.flatnonzero(target_modes==mode)
        if len(valid)<10: valid=np.arange(len(tar))
        nearest=valid[np.argsort(d[valid])[:min(30,len(valid))]]
        matched_i=int(rng.choice(nearest))
        random_i=int(rng.choice(valid))
        base={'pair_id':j,'source_image':s.source_image,'source_file':s.file_name,
              'species':s.category,'y':y,'source_camera':s.location,
              'reference_empty_count':n_ref,'target_split':args.split}
        for method,idx,container in [('matched_target',matched_i,matched_rows),
                                      ('random_target',random_i,random_rows)]:
            b=tar.iloc[idx]
            container.append({**base,'method':method,'background_image':b.image_id,
                              'background_file':b.file_name,'background_camera':b.location,
                              'appearance_distance':round(float(d[idx]),6)})
    for method,rows in [('random_target',random_rows),('matched_target',matched_rows)]:
        dest=out/f'{method}.csv'
        pd.DataFrame(rows).to_csv(dest,index=False)
        print(method, 'pairs=',len(rows),'background cameras=',len(set(r['background_camera'] for r in rows)),
              '->',dest,flush=True)
    print('DONE: Pair manifests only, no synthetic JPEGs or model training yet.',flush=True)
    print('These similarity scores capture COLOR/EXPOSURE, not semantic habitat. Validate images before training.',flush=True)

if __name__=='__main__': main()