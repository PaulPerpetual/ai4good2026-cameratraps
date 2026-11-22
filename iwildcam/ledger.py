"""The test ledger: the only way to read labelled test features or score the test cameras.

Rule (PROTOCOL.md, "Test access"): the test cameras are scored once per final configuration.
A configuration is a JSON file under configs/final/ whose settings were all chosen on the
camera-wise CV folds. Scoring appends one line to results/test_ledger.jsonl (commit it) and
refuses to score the same configuration twice. Ablations report CV and official val only.

    from iwildcam import ledger
    rows, x = ledger.load_test("dinov2l", config="configs/final/dinov2l_lda_cam.json")
    ...
    ledger.record(config, scores)        # appends the result next to the access line
"""
from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from iwildcam import store
from iwildcam.protocol import REPO, TEST_LEDGER


def config_hash(config: Path) -> str:
    """Hash of the configuration's content, so renaming the file does not reset the rule."""
    canonical = json.dumps(json.loads(Path(config).read_text()), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def entries() -> list[dict]:
    if not TEST_LEDGER.exists():
        return []
    return [json.loads(line) for line in TEST_LEDGER.read_text().splitlines() if line.strip()]


def _append(entry: dict) -> None:
    TEST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_test(backbone: str, config: Path, view: str = "crop") -> tuple[pd.DataFrame, np.ndarray]:
    """Labelled test features, after logging the access. Refuses a configuration that was
    already scored, and one that is not under configs/final/."""
    config = Path(config).resolve()
    if config.parent.name != "final":
        raise PermissionError(f"{config} is not a frozen final configuration (configs/final/)")
    h = config_hash(config)
    done = [e for e in entries() if e["config_hash"] == h and e["event"] == "scored"]
    if done:
        raise PermissionError(f"{config.name} was already scored on test on {done[0]['date']} "
                              f"by {done[0]['who']}: {done[0]['scores']}")
    _append({"event": "opened", "config": config.name, "config_hash": h, "backbone": backbone,
             "who": getpass.getuser(), "date": dt.datetime.now().isoformat(timespec="seconds"),
             "commit": _commit()})
    return store._read(store.folder(backbone, "test", view), empty=False)


def record(config: Path, scores: dict) -> None:
    """Append the test scores of a configuration; call right after scoring."""
    config = Path(config).resolve()
    _append({"event": "scored", "config": config.name, "config_hash": config_hash(config),
             "who": getpass.getuser(), "date": dt.datetime.now().isoformat(timespec="seconds"),
             "commit": _commit(), "scores": scores})
