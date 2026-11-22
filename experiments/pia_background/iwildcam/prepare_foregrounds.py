"""Cache frozen MegaDetector boxes and optional SAM cutouts. No species labels used.

Input: modern MegaDetector JSON with relative XYWH boxes and category '1' = animal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .data import load_task


def animal_box(record, threshold):
    candidates = []
    for detection in record.get("detections") or []:
        if str(detection["category"]) != "1" or detection["conf"] < threshold:
            continue
        box = np.asarray(detection["bbox"], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                (box[:2] < 0).any() or (box[2:] <= 0).any() or
                (box[:2] + box[2:] > 1.00001).any()):
            raise ValueError("Expected normalized XYWH MegaDetector boxes (not absolute pixels)")
        box[2:] = np.minimum(box[2:], 1 - box[:2])
        candidates.append((float(detection["conf"]), box.tolist()))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output", required=True, help="New directory for immutable preprocessing artifacts")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--sam-checkpoint", help="Omit for crop-only experiments")
    ap.add_argument("--sam-model", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error("threshold must be between 0 and 1")
    raw = Path(args.detections).read_bytes()
    detections = json.loads(raw)
    if detections.get("detection_categories", {}).get("1") != "animal":
        raise ValueError("Expected modern MegaDetector JSON with category '1' named 'animal'")
    records = {}
    for item in detections["images"]:
        name = Path(item["file"].replace("\\", "/")).name
        if name in records:
            raise ValueError(f"Ambiguous duplicate image basename: {name}")
        if item.get("failure"):
            raise ValueError(f"Detector failed for {name}: {item['failure']}; rerun detection")
        records[name] = item
    task = load_task(image_dir=args.image_dir)
    missing = set(task.df.file_name) - records.keys()
    if missing:
        raise ValueError(f"Detections missing for {len(missing)} available task images")
    predictor = None
    checkpoint_hash = None
    if args.sam_checkpoint:
        from segment_anything import SamPredictor, sam_model_registry

        checkpoint_hash = hashlib.sha256(Path(args.sam_checkpoint).read_bytes()).hexdigest()
        sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(args.device)
        sam.eval()
        predictor = SamPredictor(sam)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    images = {}
    for name in tqdm(task.df.file_name, desc="Foregrounds"):
        box = animal_box(records[name], args.threshold)
        entry = {"box": box, "foreground": None,
                 "status": "no_detection" if box is None else "box_only"}
        if box is not None and predictor is not None:
            with Image.open(task.image_dir / name) as image:
                rgb = np.array(image.convert("RGB"))
            height, width = rgb.shape[:2]
            x, y, w, h = box
            prompt = np.array([x * width, y * height, (x + w) * width, (y + h) * height])
            with torch.inference_mode():
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(box=prompt, multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            # Restrict SAM output to the detector box to avoid unrelated objects.
            left, top = int(prompt[0]), int(prompt[1])
            right, bottom = max(left + 1, int(prompt[2])), max(top + 1, int(prompt[3]))
            alpha = mask[top:bottom, left:right].astype(np.uint8) * 255
            if alpha.any():
                rgba = Image.fromarray(rgb[top:bottom, left:right]).convert("RGBA")
                rgba.putalpha(Image.fromarray(alpha))
                filename = f"{Path(name).stem}.png"
                rgba.save(output / filename)
                entry.update(foreground=filename, status="segmented",
                             sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest())
            else:
                entry["status"] = "empty_mask"
        images[name] = entry
    from collections import Counter

    manifest = {"format_version": 1, "detector_info": detections.get("info", {}),
                "detections_sha256": hashlib.sha256(raw).hexdigest(),
                "threshold": args.threshold, "sam_model": args.sam_model if predictor else None,
                "sam_checkpoint_sha256": checkpoint_hash,
                "counts": dict(Counter(item["status"] for item in images.values())),
                "images": images}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
