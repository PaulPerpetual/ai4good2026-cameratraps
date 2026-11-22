"""Check split isolation and compositing behavior without downloading model weights."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from iwildcam.backgrounds import BackgroundProcessor, background_pool, coverage_scores
from iwildcam.data import Task, make_loaders
from iwildcam.prepare_foregrounds import animal_box


@pytest.fixture
def scene(tmp_path):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        name = f"animal{i}.png"
        Image.new("RGB", (8, 8), "red").save(tmp_path / name)
        rows.append(dict(file_name=name, location=i, y=0, official_ood=split))
    empties = []
    for i, split in enumerate(("train", "val", "test")):
        name = f"empty{i}.png"
        Image.new("RGB", (8, 8), (0, 0, (i + 1) * 50)).save(tmp_path / name)
        empties.append(dict(file_name=name, location=i, official_ood=split))
    foreground = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    foreground.putpixel((0, 0), (0, 255, 0, 0))
    foreground.save(tmp_path / "fg.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(sam_model="vit_b", images={
        row["file_name"]: dict(box=[0.25, 0.25, 0.5, 0.5], foreground="fg.png", status="segmented")
        for row in rows})))
    return Task(pd.DataFrame(rows), pd.DataFrame(empties), ["animal"], tmp_path), manifest


def test_background_split_isolation(scene):
    task, _ = scene
    assert set(background_pool(task, "official_ood", "source").location) == {0}
    assert set(background_pool(task, "official_ood", "target").location) == {2}
    with pytest.raises(ValueError):
        background_pool(task, "random_burst", "target")


def test_alpha_composite_and_evaluation_unchanged(scene):
    task, manifest = scene
    method = dict(mode="target", manifest=str(manifest), probability=1)
    image = Image.open(task.image_dir / "animal0.png").convert("RGB")
    train = BackgroundProcessor(task, "official_ood", method, True)
    torch.manual_seed(5)
    result = np.array(train("animal0.png", image))
    assert (result == [0, 0, 150]).all(axis=-1).sum() == 49  # 15 opaque foreground pixels
    assert (result == [255, 0, 0]).all(axis=-1).sum() == 15
    evaluation = BackgroundProcessor(task, "official_ood", method, False)
    assert np.array_equal(evaluation("animal0.png", image), image)


def test_no_detection_fallback_and_empty_pool_fail(scene):
    task, manifest = scene
    content = json.loads(manifest.read_text())
    content["images"]["animal0.png"] = dict(box=None, foreground=None, status="no_detection")
    manifest.write_text(json.dumps(content))
    image = Image.open(task.image_dir / "animal0.png").convert("RGB")
    processor = BackgroundProcessor(task, "official_ood", dict(mode="neutral", manifest=str(manifest)), True)
    assert np.array_equal(processor("animal0.png", image), image)
    task.empty = task.empty[task.empty.location != 2]
    with pytest.raises(ValueError, match="No available"):
        BackgroundProcessor(task, "official_ood", dict(mode="target", manifest=str(manifest)), True)


def test_loader_and_coverage(scene):
    task, manifest = scene
    fold = {s: np.array([i]) for i, s in enumerate(("train", "val", "test"))}
    processors = {s: BackgroundProcessor(task, "official_ood", dict(mode="neutral", manifest=str(manifest)), s == "train") for s in fold}
    loaders = make_loaders(task, fold, batch_size=1, size=8, num_workers=0, processors=processors)
    xb, yb, domain = next(iter(loaders["train"]))
    assert xb.shape == (1, 3, 8, 8)
    assert yb.tolist() == [0]
    assert domain.tolist() == [0]
    report = coverage_scores(task, fold, np.array([0]), np.array([0]))
    assert report["with_empty"]["macro_f1_present"] == 1
    assert report["without_empty"]["n"] == 0


def test_detection_selection_and_coordinate_validation():
    record = {"detections": [dict(category="1", conf=0.8, bbox=[0.1, 0.1, 0.3, 0.3]),
                              dict(category="2", conf=0.99, bbox=[0, 0, 1, 1])]}
    assert animal_box(record, 0.2) == [0.1, 0.1, 0.3, 0.3]
    assert animal_box(record, 0.9) is None
    record["detections"][0]["bbox"] = [10, 10, 30, 30]
    with pytest.raises(ValueError, match="normalized"):
        animal_box(record, 0.2)


def test_masked_crop_matches_baseline_geometry_and_foreground(scene):
    task, manifest = scene
    content = json.loads(manifest.read_text())
    content["images"]["animal0.png"]["box"] = [0.125, 0.25, 0.75, 0.5]
    manifest.write_text(json.dumps(content))
    foreground = Image.new("RGBA", (6, 4), (255, 0, 0, 255))
    foreground.putpixel((0, 0), (255, 0, 0, 0))
    foreground.save(task.image_dir / "fg.png")
    image = Image.new("RGB", (8, 8), "red")
    method = dict(mode="masked_crop", manifest=str(manifest))
    crop = BackgroundProcessor(task, "official_ood", dict(method, mode="crop"), False)("animal0.png", image)
    for training in (True, False):
        result = BackgroundProcessor(task, "official_ood", method, training)("animal0.png", image)
        assert result.size == crop.size == (6, 4)
        assert result.getpixel((0, 0)) == (127, 127, 127)
        assert np.array_equal(np.array(result)[1:], np.array(crop)[1:])
    content["images"]["animal0.png"]["foreground"] = None
    manifest.write_text(json.dumps(content))
    assert np.array_equal(BackgroundProcessor(task, "official_ood", method, False)("animal0.png", image), crop)
    content["images"]["animal0.png"]["box"] = None
    manifest.write_text(json.dumps(content))
    assert np.array_equal(BackgroundProcessor(task, "official_ood", method, False)("animal0.png", image), image)


def test_training_smoke_without_pretrained_download(scene, monkeypatch):
    from iwildcam import train

    task, manifest = scene
    task.df["burst_id"] = ["train_burst", "val_burst", "test_burst"]
    monkeypatch.setattr(train, "load_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs:
                        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 1)))
    cfg = dict(seed=0, device="cpu", amp=False, height=8, size=8, split="official_ood",
               batch_size=1, num_workers=0, model="resnet18", pretrained=False,
               epochs=1, lr=0.001, verbose=False, wandb=False,
               pred_dir=str(task.image_dir / "preds"),
               background=dict(mode="target", manifest=str(manifest), probability=1))
    result = train.run_fold(cfg)
    assert result["n_train"] == 1
    assert result["test_macro_f1_present"] == 1
    assert result["background"]["background_cameras"] == [2]
    assert result["test_empty_coverage"]["with_empty"]["n"] == 1
    assert len(list((task.image_dir / "preds").glob("*.npz"))) == 1
