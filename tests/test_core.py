"""Checks of the shared core that run without features or a GPU."""
import json

import numpy as np
import pytest

from iwildcam import ledger, metrics, store
from iwildcam.crops import square_box
from iwildcam.folds import camera_folds, make_folds


def test_macro_f1_only_present_species():
    true = np.array([0, 0, 1, 1])
    assert metrics.macro_f1(true, true) == 1.0
    assert metrics.macro_f1(true, np.array([0, 0, 0, 0])) == pytest.approx((2 / 3 + 0) / 2)


def test_burst_vote_weights_by_detector_confidence():
    predicted = np.array([1, 2, 2, 5])
    bursts = np.array(["a", "a", "a", "b"])
    conf = np.array([0.9, 0.1, 0.2, 0.0])          # one confident photo outvotes two weak ones
    assert metrics.burst_vote(predicted, bursts, conf).tolist() == [1, 1, 1, 5]


def test_burst_vote_without_detections_is_equal_vote():
    predicted = np.array([1, 2, 2])
    assert metrics.burst_vote(predicted, np.zeros(3), np.zeros(3)).tolist() == [2, 2, 2]


def test_square_box():
    assert square_box(None, 100, 50) == (0.0, 0.0, 1.0, 1.0)
    assert square_box((0.1, 0.4, 0.4, 0.2, 0.2), 100, 50) == (0.0, 0.0, 1.0, 1.0)   # below 0.2
    left, top, right, bottom = square_box((0.9, 0.4, 0.4, 0.1, 0.1), 200, 100)
    assert (right - left) * 200 == pytest.approx((bottom - top) * 100)             # square in pixels
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1


def test_committed_folds_match_the_rule():
    made = make_folds()
    assert dict(zip(made.location, made.fold)) == camera_folds()
    assert made.location.is_unique and set(made.fold) == set(range(5))


def test_store_refuses_labelled_test():
    with pytest.raises(PermissionError):
        store.load("dinov2l", cameras="test")


def test_ledger_scores_a_config_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "TEST_LEDGER", tmp_path / "ledger.jsonl")
    final = tmp_path / "final"
    final.mkdir()
    config = final / "x.json"
    config.write_text(json.dumps({"backbone": "dinov2l", "shrinkage": 0.1}))
    monkeypatch.setattr(ledger.store, "_read", lambda *a, **k: ("rows", "x"))
    assert ledger.load_test("dinov2l", config) == ("rows", "x")
    ledger.record(config, {"per_image": 0.5})
    renamed = final / "y.json"                     # renaming does not reset the rule
    renamed.write_text(config.read_text())
    with pytest.raises(PermissionError):
        ledger.load_test("dinov2l", renamed)
    not_final = tmp_path / "z.json"
    not_final.write_text("{}")
    with pytest.raises(PermissionError):
        ledger.load_test("dinov2l", not_final)
